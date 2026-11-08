"""Gains du transfert, graine par graine (protocole, section 6 : gain du transfert).

Entrée : le tableau de résultats des notebooks, une ligne par
(graine, budget, méthode) :

    seed, n_T, method, <notes de evaluate : error, ber, ...>

* ``n_T`` vaut 0 ou est vide pour les méthodes sans budget (Src, Oracle) : leur
  ligne est recopiée sur tous les budgets de la même graine.
* L'ancienne colonne ``err`` des premiers notebooks est lue comme ``error``.
* Si le tableau couvre plusieurs scénarios, ajouter une colonne ``scenario`` :
  les gains sont alors calculés scénario par scénario.

Gains, pour une graine s et un budget n_T (note m : error ou ber) :

    G_Tgt(s, n_T) = m_Tgt(s, n_T) - m_méthode(s, n_T)
    G_ref(s, n_T) = min(m_Src, m_Tgt, m_Union)(s, n_T) - m_méthode(s, n_T)

G > 0 : la méthode fait mieux que la référence. Gains dans l'unité de la note
(fraction : 0,02 = 2 points d'erreur).

    from transfer_forest.scoring import compute_gains, summarize
    gains = compute_gains(results, metric=scn.metric)
    summary = summarize(gains)   # par budget et méthode : gain moyen, IC 95 %, taux de victoire
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

REFS_PROTOCOL = ("Src", "Tgt", "Union")   # les trois références de G_ref (protocole)
N_BOOT = 10_000


def _prepare(results: pd.DataFrame, budget_col: str) -> tuple[pd.DataFrame, list[str]]:
    df = results.copy()
    if "error" not in df.columns and "err" in df.columns:
        df = df.rename(columns={"err": "error"})
    groups = ["scenario"] if "scenario" in df.columns else []
    missing = {"seed", budget_col, "method"} - set(df.columns)
    if missing:
        raise ValueError(f"colonnes manquantes : {sorted(missing)}")

    # méthodes sans budget (n_T = 0 ou vide) : recopiées sur chaque budget de la graine
    no_budget = df[budget_col].isna() | (df[budget_col] == 0)
    budgets = df.loc[~no_budget, groups + ["seed", budget_col]].drop_duplicates()
    once = df[no_budget].drop(columns=budget_col).merge(budgets, on=groups + ["seed"])
    df = pd.concat([df[~no_budget], once], ignore_index=True)

    keys = groups + ["seed", budget_col]
    dup = df.duplicated(keys + ["method"], keep=False)
    if dup.any():
        raise ValueError("plusieurs résultats pour une même (graine, budget, méthode) :\n"
                         f"{df.loc[dup, keys + ['method']].drop_duplicates().head()}")
    return df, keys


def compute_gains(results: pd.DataFrame, metric: str = "error", budget_col: str = "n_T",
                  refs: tuple[str, ...] = REFS_PROTOCOL) -> pd.DataFrame:
    """G_Tgt et G_ref pour chaque (graine, budget, méthode).

    metric : la note comparée, ``Scenario.metric`` en général (« error » ou « ber »).
    refs : références dont le minimum définit G_ref ; par défaut celles du
        protocole. Pour inclure Union-P : ``refs=("Src", "Tgt", "Union", "Union-P")``.

    Toutes les méthodes reçoivent des gains, références comprises (G_Tgt de Tgt
    vaut 0). Si une référence manque pour une graine, les gains correspondants
    valent NaN avec un avertissement : on ne compare jamais deux tirages différents.
    """
    df, keys = _prepare(results, budget_col)
    if metric not in df.columns:
        raise ValueError(f"note {metric!r} absente du tableau (colonnes : {list(df.columns)})")
    if "Tgt" not in refs:
        refs = tuple(refs) + ("Tgt",)

    wide = df.pivot_table(index=keys, columns="method", values=metric, aggfunc="first")
    wide = wide.reindex(columns=sorted(set(wide.columns) | set(refs)))
    lacking = wide[list(refs)].isna()
    if lacking.any(axis=None):
        rows = lacking.any(axis=1)
        detail = [f"{k} : manque {', '.join(c for c in refs if lacking.loc[k, c])}"
                  for k in lacking.index[rows][:10]]
        warnings.warn("références manquantes, gains mis à NaN pour :\n  " + "\n  ".join(detail),
                      stacklevel=2)
    ref_tgt = wide["Tgt"]
    ref_min = wide[list(refs)].min(axis=1, skipna=False)

    out = df[keys + ["method", metric]].rename(columns={metric: "score"})
    out = out.merge(pd.DataFrame({"score_tgt": ref_tgt, "score_ref": ref_min}).reset_index(),
                    on=keys, how="left")
    out.insert(len(keys) + 1, "metric", metric)
    out["g_tgt"] = out["score_tgt"] - out["score"]
    out["g_ref"] = out["score_ref"] - out["score"]
    return out.sort_values(keys + ["method"]).reset_index(drop=True)


def bootstrap_ci(values, n_boot: int = N_BOOT, level: float = 0.95, rng=None):
    """IC de la moyenne par percentiles, en rééchantillonnant les graines avec remise."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(0) if rng is None else rng
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def summarize(gains: pd.DataFrame, budget_col: str = "n_T", n_boot: int = N_BOOT,
              level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Synthèse sur les graines, par budget et méthode (protocole, section 6).

    Pour G_Tgt et G_ref : gain moyen, IC à 95 % (bootstrap des graines) et taux de
    victoire, la part des graines où G > 0 (une égalité n'est pas une victoire).
    Les graines aux gains NaN sont écartées ; ``n_seeds`` compte celles retenues.
    """
    rng = np.random.default_rng(seed)
    by = (["scenario"] if "scenario" in gains.columns else []) + [budget_col, "method"]
    rows = []
    for key, grp in gains.groupby(by, sort=True):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["metric"] = grp["metric"].iloc[0]
        row["score_mean"] = grp["score"].mean()
        row["n_seeds"] = int(grp["g_tgt"].notna().sum())
        for g in ("g_tgt", "g_ref"):
            v = grp[g].dropna().to_numpy()
            row[f"{g}_mean"] = v.mean() if v.size else np.nan
            row[f"{g}_ci_low"], row[f"{g}_ci_high"] = bootstrap_ci(v, n_boot, level, rng)
            row[f"{g}_win_rate"] = np.mean(v > 0) if v.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
