"""Le moteur : fait tourner l'expérience complète d'une famille de scénarios.

    python -m transfer_forest.run S1          # depuis la racine du dépôt

Pour chaque scénario de la famille :
    1. découpage test / réserve, et caractérisation du décalage (sur la réserve) ;
    2. pour chaque graine, en parallèle : la forêt source, puis chaque méthode à chaque
       budget (Src et Oracle une seule fois) : apprentissage, prédiction sur le test, notation ;
    3. gains, Wilcoxon-Holm, budget de croisement, économie d'annotation, figures.

Sorties dans results/ :
    <S>.csv            une ligne par (graine, budget, méthode) : notes, temps, version du code
    <S>_summary.csv    par (budget, méthode) : gains, IC, victoires, p-valeurs, synthèse
    <S>_shift.csv      le décalage du scénario
    <S>_scores.png, <S>_gains.png
"""
import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

from transfer_forest.contracts import Context, Scenario, make_forest
from transfer_forest.data import load_scenarios
from transfer_forest.evaluation import evaluate
from transfer_forest.methods import METHODS, make_method
from transfer_forest.plots import plot_gains, plot_scores
from transfer_forest.scoring import (annotation_saving, compute_gains, crossover_budget, summarize,
                                     timing_table)
from transfer_forest.shift import characterize
from transfer_forest.splits import draws, split_target

RESULTS = Path("results")
# Le protocole définit G_ref avec min(Src, Tgt, Union). On y ajoute Union-P, la meilleure
# référence sur S1 : un gain ne doit jamais être déclaré face à une simple repondération.
G_REF = ("Src", "Tgt", "Union", "Union-P")


def measure_storage(method) -> tuple[int, float]:
    """Taille du modèle sauvegardé (octets) et temps pour le recharger (s)."""
    buf = io.BytesIO()
    joblib.dump(method, buf)
    size = buf.getbuffer().nbytes
    buf.seek(0)
    t0 = time.perf_counter()
    joblib.load(buf)
    return size, time.perf_counter() - t0


def run_seed(scn: Scenario, reserve, test, seed: int, draw: dict) -> tuple[list, list]:
    """Toutes les expériences d'une graine. Renvoie (lignes de résultats, échecs)."""
    rf_src = make_forest(seed).fit(scn.Xs, scn.ys)
    none = np.array([], dtype=int)
    # sans budget (Src, Oracle) : aucun exemple étiqueté, mais la réserve entière pour Oracle
    no_budget = Context(scn.Xs, scn.ys, scn.Xt[none], scn.yt[none], scn.Xt[reserve], rf_src, scn.K,
                        seed, X_reserve=scn.Xt[reserve], y_reserve=scn.yt[reserve])
    jobs = [(name, None, no_budget) for name, m in METHODS.items() if not m.uses_budget]
    for p, (lab, unl) in draw.items():
        ctx = Context(scn.Xs, scn.ys, scn.Xt[lab], scn.yt[lab], scn.Xt[unl], rf_src, scn.K, seed,
                      rare_class=scn.critical_class)
        jobs += [(name, p, ctx) for name, m in METHODS.items() if m.uses_budget]

    rows, failures = [], []
    for name, p, ctx in jobs:
        try:                                    # une méthode qui plante n'arrête pas la campagne
            t0 = time.perf_counter()
            method = make_method(name).fit(ctx)
            t1 = time.perf_counter()
            P = method.predict_proba(scn.Xt[test])
            t2 = time.perf_counter()
            notes = evaluate(scn.yt[test], P, scn.critical_class)
            size_bytes, t_load = measure_storage(method)
        except Exception as e:
            failures.append(f"{scn.name}, graine {seed}, {name}, p = {p} : {e!r}")
            continue
        rows.append({"scenario": scn.name, "family": scn.family, "seed": seed, "p": p,
                     "n_T": len(ctx.yl) if p else None, "method": name, **notes,
                     "t_fit": t1 - t0, "t_predict": t2 - t1,
                     "t_predict_per_1000": (t2 - t1) / len(test) * 1000,
                     "size_bytes": size_bytes, "t_load": t_load})
    return rows, failures


def analyze(scn: Scenario, results: pd.DataFrame) -> pd.DataFrame:
    """Gains, significativité, indicateurs de synthèse et figures d'un scénario."""
    summary = summarize(compute_gains(results, metric=scn.metric, refs=G_REF))
    summary["saving"] = annotation_saving(summary)
    summary["crossover_n_T"] = summary["method"].map(crossover_budget(summary))
    summary.to_csv(RESULTS / f"{scn.name}_summary.csv", index=False)
    timing_table(results).to_csv(RESULTS / f"{scn.name}_timing.csv")
    plot_scores(results, scn.metric, RESULTS / f"{scn.name}_scores.png")
    plot_gains(summary, RESULTS / f"{scn.name}_gains.png")
    return summary


def code_version() -> str:
    """Empreinte Git du code, suffixée de -dirty s'il y a des modifications non commitées."""
    git = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True)
    return git.stdout.strip() or "inconnue"


def run(family: str, n_jobs: int = -1) -> pd.DataFrame:
    """Fait tourner l'expérience de chaque scénario de la famille ; renvoie les résultats."""
    RESULTS.mkdir(exist_ok=True)
    version, all_results = code_version(), []
    for scn in load_scenarios(family):
        start = time.perf_counter()
        reserve, test = split_target(scn)
        pd.DataFrame([characterize(scn, reserve)]).to_csv(RESULTS / f"{scn.name}_shift.csv", index=False)
        out = Parallel(n_jobs=n_jobs)(delayed(run_seed)(scn, reserve, test, seed, draw)
                                      for seed, draw in draws(scn, reserve))
        for failure in (f for _, failures in out for f in failures):
            print("ÉCHEC :", failure, file=sys.stderr)
        results = pd.DataFrame([row for rows, _ in out for row in rows]).assign(code_version=version)
        results.to_csv(RESULTS / f"{scn.name}.csv", index=False)
        analyze(scn, results)
        print(f"{scn.name} : {len(results)} expériences en {time.perf_counter() - start:.0f} s, "
              f"résultats dans {RESULTS}/")
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    for family in sys.argv[1:] or ["S1"]:
        run(family)
