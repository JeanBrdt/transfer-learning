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
    summary = summarize(gains)   # par budget et méthode : gain moyen, IC 95 %, victoires, Wilcoxon

Puis, pour un scénario, les deux indicateurs de synthèse du protocole :
crossover_budget (budget de croisement) et annotation_saving (économie d'annotation).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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

    Pour G_Tgt et G_ref : gain moyen, IC à 95 % (bootstrap des graines), taux de
    victoire (part des graines où G > 0 ; une égalité n'est pas une victoire) et
    p-valeur du test de Wilcoxon apparié, brute (``_p``) puis ajustée par Holm sur
    les budgets d'une même méthode (``_p_holm``).
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
            row[f"{g}_p"] = wilcoxon_p(v)
        rows.append(row)
    out = pd.DataFrame(rows)
    same_method = [c for c in by if c != budget_col]
    for g in ("g_tgt", "g_ref"):
        out[f"{g}_p_holm"] = out.groupby(same_method)[f"{g}_p"].transform(holm)
    return out


def wilcoxon_p(gains) -> float:
    """p-valeur du test de Wilcoxon apparié, H0 : le gain est nul ; NaN si tous les gains sont nuls."""
    v = np.asarray(gains, dtype=float)
    return float(wilcoxon(v).pvalue) if np.any(v != 0) else np.nan


def holm(p) -> np.ndarray:
    """p-valeurs ajustées par la méthode de Holm : la i-ème plus petite est multipliée
    par (m - i + 1), puis on impose qu'elles croissent. Les NaN sont ignorés."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)                                    # NaN en dernier
    m = np.count_nonzero(~np.isnan(p))
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((m - np.arange(len(p))) * p[order]))
    return adjusted


def crossover_budget(summary: pd.DataFrame, gain: str = "g_tgt", alpha: float = 0.05,
                     budget_col: str = "n_T") -> pd.Series:
    """Budget de croisement de chaque méthode, pour un scénario.

    Le plus petit budget à partir duquel la méthode n'a plus de gain significatif
    (p de Holm < alpha et gain moyen > 0) : au-delà, autant apprendre sur la cible
    seule. NaN si le gain est encore significatif au plus grand budget.
    """
    out = {}
    for method, s in summary.sort_values(budget_col).groupby("method"):
        significant = ((s[f"{gain}_p_holm"] < alpha) & (s[f"{gain}_mean"] > 0)).to_numpy()
        first = np.flatnonzero(significant)[-1] + 1 if significant.any() else 0
        out[method] = s[budget_col].iloc[first] if first < len(s) else np.nan
    return pd.Series(out, name=f"crossover_{budget_col}")


def annotation_saving(summary: pd.DataFrame, budget_col: str = "n_T") -> pd.Series:
    """Économie d'annotation E = ñ / n_T, pour un scénario.

    ñ : nombre d'exemples dont Tgt aurait besoin pour atteindre la note moyenne de la
    méthode, interpolé sur la courbe moyenne de Tgt (en log n_T). E = 4 : la méthode
    fait avec n_T exemples ce que Tgt fait avec 4 n_T. NaN hors des budgets testés.
    """
    tgt = summary[summary.method == "Tgt"].sort_values(budget_col)
    score, log_n = tgt["score_mean"].to_numpy()[::-1], np.log(tgt[budget_col].to_numpy())[::-1]
    n_tilde = np.exp(np.interp(summary["score_mean"], score, log_n, left=np.nan, right=np.nan))
    return pd.Series(n_tilde / summary[budget_col].to_numpy(), index=summary.index, name="saving")
