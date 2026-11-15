"""Caractérisation du décalage entre source et cible (protocole, section 6).

Calculée une fois par scénario, sur la réserve : jamais sur le test.

    auc_domain       AUC d'une forêt qui devine si un exemple vient de la source ou de
                     la cible (0,5 : domaines indiscernables ; 1 : séparables),
                     moyenne et écart type sur 5 graines ;
    auc_domain_excl  la même chose sans la variable de coupe (NaN s'il n'y en a pas) ;
    prop_gap         écart des proportions de classes : ½ Σ_k |P_S(k) - P_T(k)| ;
    domain_gap       erreur de la forêt source sur la cible moins son erreur (hors sac)
                     sur la source.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from transfer_forest.contracts import Scenario, make_forest

N_SEEDS = 5


def domain_auc(Xs: np.ndarray, Xt: np.ndarray) -> tuple[float, float]:
    """AUC en validation croisée à 5 plis d'une forêt source / cible ; moyenne et écart type."""
    X = np.vstack([Xs, Xt])
    domain = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    aucs = [cross_val_score(make_forest(s), X, domain, scoring="roc_auc", n_jobs=-1,
                            cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
            for s in range(N_SEEDS)]
    return float(np.mean(aucs)), float(np.std(aucs))


def characterize(scn: Scenario, reserve: np.ndarray) -> dict:
    """Une ligne du tableau de décalage pour le scénario."""
    Xr, yr = scn.Xt[reserve], scn.yt[reserve]
    auc, auc_sd = domain_auc(scn.Xs, Xr)
    j = scn.params.get("split_feature_idx")
    auc_excl = np.nan if j is None else domain_auc(np.delete(scn.Xs, j, 1), np.delete(Xr, j, 1))[0]
    p_s, p_t = (np.bincount(y, minlength=scn.K) / len(y) for y in (scn.ys, yr))
    rf = make_forest(0).set_params(oob_score=True).fit(scn.Xs, scn.ys)
    return {"scenario": scn.name, "auc_domain": auc, "auc_domain_sd": auc_sd,
            "auc_domain_excl": auc_excl, "prop_gap": 0.5 * np.abs(p_s - p_t).sum(),
            "domain_gap": np.mean(rf.predict(Xr) != yr) - (1 - rf.oob_score_),
            "n_S": scn.n_S, "n_target": len(scn.yt), "K": scn.K, "d": scn.d}
