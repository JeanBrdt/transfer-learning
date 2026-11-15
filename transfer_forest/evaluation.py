"""Notation d'une méthode sur le test cible (protocole, section 6 : mesures).

Formats du contrat : y entier dans 0..K-1, P probabilités (n, K).

    from transfer_forest.evaluation import evaluate
    P = make_method(name).fit(ctx).predict_proba(X_test)
    notes = evaluate(y_test, P, critical_class=scn.critical_class)
    # {"error": ..., "ber": ..., "f1_macro": ..., "fnr_critical": ..., "log_loss": ...}

Note principale (comme Segev et al.), donnée par ``Scenario.metric`` :
    error : taux d'erreur (S0, S1, S3, S6) ;
    ber   : erreur équilibrée, moyenne des erreurs par classe (S2, S4, S5).
Notes secondaires : F1 macro, taux de faux négatifs de la classe critique (ou de
la classe la plus manquée), perte logarithmique.
"""
from __future__ import annotations

import numpy as np

from transfer_forest.contracts import check_proba

NOTES = ("error", "ber", "f1_macro", "fnr_critical", "log_loss")
LOG_LOSS_EPS = 1e-15


def _labels(y_true, K: int) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_true doit être un vecteur non vide")
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= K:
        raise ValueError(f"y_true doit être entier dans 0..{K - 1}")
    return y


def per_class_error(y_true, y_pred, K: int) -> np.ndarray:
    """e_k = part des exemples de la classe k mal classés ; NaN si k est absente du test."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    e = np.full(K, np.nan)
    for k in np.unique(y_true):
        e[k] = np.mean(y_pred[y_true == k] != k)
    return e


def error_rate(y_true, y_pred) -> float:
    """Part des exemples du test mal classés."""
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def balanced_error_rate(y_true, y_pred, K: int) -> float:
    """BER = moyenne des e_k sur les classes présentes dans le test.

    Chaque classe pèse autant, quelle que soit sa taille. BER = 1 - exactitude
    équilibrée. Le test étant stratifié (splits.py), toutes les classes y sont.
    """
    return float(np.nanmean(per_class_error(y_true, y_pred, K)))


def f1_macro(y_true, y_pred, K: int) -> float:
    """Moyenne non pondérée des F1 des classes présentes dans le test.

    F1_k = 2 TP_k / (2 TP_k + FP_k + FN_k) ; vaut 0 si la classe n'est jamais
    prédite correctement.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = []
    for k in np.unique(y_true):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def fnr_critical(y_true, y_pred, K: int, critical_class: int | None = None) -> float:
    """Taux de faux négatifs de la classe critique, ou de la pire classe si None.

    Pour une classe k, c'est e_k : la part de ses exemples que la méthode manque.
    """
    e = per_class_error(y_true, y_pred, K)
    if critical_class is None:
        return float(np.nanmax(e))
    if not 0 <= critical_class < K or np.isnan(e[critical_class]):
        raise ValueError(f"classe critique {critical_class} absente du test")
    return float(e[critical_class])


def log_loss(y_true, P: np.ndarray, eps: float = LOG_LOSS_EPS) -> float:
    """-(1/n) somme log P[i, y_i].

    Une probabilité nulle sur la vraie classe (feuille qui ne l'a jamais vue) est
    remontée à eps, pour garder une perte finie : elle compte alors log(1/eps) ≈ 34,5.
    """
    y = np.asarray(y_true)
    p = np.clip(P[np.arange(y.size), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def evaluate(y_true, P, critical_class: int | None = None) -> dict:
    """Toutes les notes d'une méthode sur le test cible.

    y_true : vraies classes du test, entières dans 0..K-1.
    P : probabilités (n, K) renvoyées par ``Method.predict_proba``.
    critical_class : ``Scenario.critical_class`` ; None = la classe la plus manquée.

    La classe prédite est ``P.argmax(axis=1)`` (en cas d'égalité, la plus petite).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"P doit être une matrice (n, K) ; forme reçue {P.shape}")
    K = P.shape[1]
    y = _labels(y_true, K)
    check_proba(P, len(y), K)
    y_pred = P.argmax(axis=1)
    return {
        "error": error_rate(y, y_pred),
        "ber": balanced_error_rate(y, y_pred, K),
        "f1_macro": f1_macro(y, y_pred, K),
        "fnr_critical": fnr_critical(y, y_pred, K, critical_class),
        "log_loss": log_loss(y, P),
    }

