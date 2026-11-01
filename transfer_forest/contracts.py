"""Le contrat du pipeline : formats et règles communs à tous les maillons.

On importe depuis ce fichier, on ne recopie jamais une constante ou un format,
et on ne le modifie pas sans l'accord du groupe.

Formats :
    X : tableau float64 (n, d), sans valeur manquante ;
    y : tableau int64 (n,), valeurs 0, ..., K-1 ;
    P : probabilités (n, K), toujours K colonnes, chaque ligne somme à 1.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

CONTRACT_VERSION = "1.0"
N_TREES = 50


def make_forest(seed: int, n_trees: int = N_TREES) -> RandomForestClassifier:
    """La forêt aléatoire commune à toutes les méthodes (n_jobs=1 : le moteur parallélise)."""
    return RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=1)


@dataclass(frozen=True)
class Context:
    """Tout ce qu'une méthode a le droit d'utiliser pour apprendre. Le test n'y est pas.

    On ne modifie jamais le contexte, et surtout pas rf_src : la même forêt
    source sert à toutes les méthodes d'une graine.
    """
    Xs: np.ndarray                          # source, entièrement étiquetée
    ys: np.ndarray
    Xl: np.ndarray                          # exemples cibles étiquetés du budget
    yl: np.ndarray
    Xu: np.ndarray                          # reste de la réserve, sans étiquettes
    rf_src: RandomForestClassifier          # forêt apprise sur la source pour cette graine
    K: int
    seed: int
    n_trees: int = N_TREES
    rare_class: int | None = None           # classe à protéger (SER*)
    X_reserve: np.ndarray | None = None     # toute la réserve étiquetée : réservé à Oracle
    y_reserve: np.ndarray | None = None


class Method:
    """Forme commune de toutes les méthodes, références comme transfert.

    Une classe fille fixe name (nom officiel) et uses_budget (False si la
    méthode n'utilise pas les exemples étiquetés : le moteur ne la fait alors
    tourner qu'une fois par graine). Elle doit pouvoir se créer sans argument.
    """
    name: str
    uses_budget: bool

    def fit(self, ctx: Context) -> "Method":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def full_proba(model, X: np.ndarray, K: int) -> np.ndarray:
    """Probabilités (n, K) d'un modèle scikit-learn, en complétant par des zéros
    les classes qu'il n'a pas vues à l'apprentissage."""
    P = np.zeros((len(X), K))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def check_proba(P: np.ndarray, n: int, K: int) -> None:
    """Vérifie que P a la forme (n, K) et que chaque ligne est une distribution."""
    if P.shape != (n, K):
        raise ValueError(f"probabilités de forme {P.shape}, attendu {(n, K)}")
    if (P < 0).any() or not np.allclose(P.sum(axis=1), 1):
        raise ValueError("chaque ligne de probabilités doit être positive et sommer à 1")
