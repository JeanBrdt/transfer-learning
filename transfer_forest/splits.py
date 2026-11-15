"""Découpage de la cible et tirages emboîtés (protocole, partie 2).

    reserve, test = split_target(scn)                 # une fois par scénario
    for seed, tirage in draws(scn, reserve):          # 20 graines
        for p, (lab, unl) in tirage.items():          # 5 budgets, emboîtés
            ...                                       # lab : annotés, unl : reste de la réserve

Tout est en indices dans scn.Xt / scn.yt. Le test ne sert qu'à l'évaluation.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from transfer_forest.contracts import Scenario

TEST_SEED = 12345                          # graine du test : fixe, indépendante des SEEDS
SEEDS = range(20)                          # les R = 20 graines
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)   # n_T = p * n_S


def split_target(scn: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Coupe la cible en (réserve, test), stratifié sur les classes.

    Toujours le même découpage pour un scénario donné : le test ne dépend ni
    de la graine ni du budget.
    """
    return train_test_split(np.arange(len(scn.yt)), test_size=scn.test_frac,
                            stratify=scn.yt, random_state=TEST_SEED)


def reserve_order(y_reserve, seed):
    """Ordre de la réserve pour une graine (positions dans la réserve).

    Les K premiers éléments couvrent les K classes. Un budget prend les n_T
    premiers éléments : les tirages sont donc emboîtés.
    """
    perm = np.random.default_rng(seed).permutation(len(y_reserve))
    first = np.sort(np.unique(y_reserve[perm], return_index=True)[1])  # 1re apparition de chaque classe
    return np.concatenate([perm[first], np.delete(perm, first)])


def budgets(scn, n_reserve):
    """{p: n_T} avec n_T = round(p * n_S), et au moins K pour couvrir toutes les classes."""
    n_T = {p: max(round(p * scn.n_S), scn.K) for p in BUDGETS}
    if max(n_T.values()) > n_reserve:
        raise ValueError(f"{scn.name} : n_T = {max(n_T.values())} dépasse la réserve ({n_reserve})")
    return n_T


def draws(scn, reserve):
    """Pour chaque graine : (seed, {p: (lab, unl)}).

    lab : indices cibles des n_T exemples annotés du budget p ;
    unl : indices cibles du reste de la réserve, sans étiquettes.
    """
    n_T = budgets(scn, len(reserve))
    for seed in SEEDS:
        order = reserve[reserve_order(scn.yt[reserve], seed)]   # positions → indices cibles
        yield seed, {p: (order[:n], order[n:]) for p, n in n_T.items()}
