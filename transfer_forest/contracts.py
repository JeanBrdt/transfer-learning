"""Le contrat du pipeline : formats et règles communs à tous les maillons.

On importe depuis ce fichier, on ne recopie jamais une constante ou un format,
et on ne le modifie pas sans l'accord du groupe.

Formats :
    X : tableau float64 (n, d), sans valeur manquante ;
    y : tableau int64 (n,), valeurs 0, ..., K-1 ;
    P : probabilités (n, K), toujours K colonnes, chaque ligne somme à 1.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier

CONTRACT_VERSION = "1.0"
N_TREES = 50
METRICS = ("error", "ber")


def make_forest(seed: int, n_trees: int = N_TREES) -> RandomForestClassifier:
    """La forêt aléatoire commune à toutes les méthodes (n_jobs=1 : le moteur parallélise)."""
    return RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=1)


@dataclass(frozen=True)
class Scenario:
    """Les données d'un scénario : la source et la cible entière.

    La cible n'est pas découpée : mettre le test de côté est le travail du protocole.
    """
    name: str                               # identifiant unique, par exemple "S1"
    family: str
    Xs: np.ndarray                          # source, entièrement étiquetée
    ys: np.ndarray
    Xt: np.ndarray                          # cible complète
    yt: np.ndarray
    class_names: tuple[str, ...]            # class_names[k] = nom de la classe k
    feature_names: tuple[str, ...]
    metric: str                             # note principale, l'une de METRICS
    critical_class: int | None = None       # classe à surveiller ; None = la pire classe
    test_frac: float = 0.5                  # part de la cible gardée pour le test
    groups_t: np.ndarray | None = None      # groupes dans la cible (C-MAPSS, HHAR)
    params: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.class_names)

    @property
    def n_S(self) -> int:
        return len(self.ys)

    @property
    def d(self) -> int:
        return self.Xs.shape[1]


def check_scenario(scn: Scenario) -> None:
    """Vérifie les formats d'un scénario ; s'arrête au premier écart."""
    if scn.metric not in METRICS:
        raise ValueError(f"{scn.name} : metric doit être l'une de {METRICS}")
    if len(scn.feature_names) != scn.d:
        raise ValueError(f"{scn.name} : {len(scn.feature_names)} noms pour {scn.d} variables")
    for part, X, y in (("source", scn.Xs, scn.ys), ("cible", scn.Xt, scn.yt)):
        if X.dtype != np.float64 or X.shape != (len(y), scn.d) or np.isnan(X).any():
            raise ValueError(f"{scn.name}, {part} : X doit être float64 ({len(y)}, {scn.d}), sans NaN")
        if y.dtype != np.int64 or not np.array_equal(np.unique(y), np.arange(scn.K)):
            raise ValueError(f"{scn.name}, {part} : y doit être int64 et contenir chaque classe 0..{scn.K - 1}")


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
