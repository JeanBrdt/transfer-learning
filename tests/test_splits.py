"""Tests du découpage cible et des tirages emboîtés (transfer_forest.splits)."""
from types import SimpleNamespace

import numpy as np
import pytest

from transfer_forest.splits import BUDGETS, SEEDS, budgets, draws, reserve_order, split_target


@pytest.fixture
def scn():
    """Faux scénario : source de 1000 exemples, cible de 600 sur K = 3 classes déséquilibrées."""
    yt = np.random.default_rng(0).choice(3, 600, p=[0.80, 0.15, 0.05])
    return SimpleNamespace(name="test", yt=yt, test_frac=0.5, n_S=1000, K=3)


def test_test_et_reserve(scn):
    reserve, test = split_target(scn)
    assert len(np.intersect1d(reserve, test)) == 0                 # disjoints
    assert len(reserve) + len(test) == len(scn.yt)                 # toute la cible
    for k in range(3):                                             # stratifié
        assert abs((scn.yt[reserve] == k).sum() - (scn.yt[test] == k).sum()) <= 1
    np.testing.assert_array_equal(split_target(scn)[1], test)      # toujours le même test


def test_ordre_de_la_reserve(scn):
    reserve, _ = split_target(scn)
    yr = scn.yt[reserve]
    for seed in SEEDS:
        order = reserve_order(yr, seed)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(yr)))  # une permutation
        assert set(yr[order[:3]]) == {0, 1, 2}                             # couvre les K classes
    np.testing.assert_array_equal(reserve_order(yr, 3), reserve_order(yr, 3))
    assert not np.array_equal(reserve_order(yr, 3), reserve_order(yr, 4))


def test_budgets(scn):
    reserve, _ = split_target(scn)
    assert list(budgets(scn, len(reserve)).values()) == [10, 20, 50, 100, 200]
    assert budgets(SimpleNamespace(name="petit", n_S=100, K=3), 300)[0.01] == 3  # au moins K
    with pytest.raises(ValueError):
        budgets(SimpleNamespace(name="gros", n_S=10_000, K=3), 300)


def test_tirages_emboites_sans_le_test(scn):
    reserve, test = split_target(scn)
    n_T = budgets(scn, len(reserve))
    seeds = []
    for seed, tirage in draws(scn, reserve):
        seeds.append(seed)
        assert list(tirage) == list(BUDGETS)
        prev = set()
        for p, (lab, unl) in tirage.items():
            assert len(lab) == n_T[p]
            assert set(lab) | set(unl) == set(reserve) and not set(lab) & set(unl)
            assert prev <= set(lab) and not set(lab) & set(test)   # emboîtés, jamais le test
            prev = set(lab)
    assert seeds == list(SEEDS)
