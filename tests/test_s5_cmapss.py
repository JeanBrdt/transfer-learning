"""Tests du scénario S5 (C-MAPSS) et de son découpage par moteur."""
import numpy as np

from transfer_forest.data import load_scenarios
from transfer_forest.splits import budgets, split_target


def test_scenario_s5():
    [scn] = load_scenarios("S5")
    assert (scn.n_S, len(scn.yt), scn.K, scn.d) == (20631, 53759, 2, 24)
    assert scn.metric == "ber" and scn.class_names[scn.critical_class] == "panne"
    assert 0.14 < scn.ys.mean() < 0.16 and 0.14 < scn.yt.mean() < 0.16    # 15 % de « panne »
    assert len(np.unique(scn.groups_t)) == 260                             # 260 moteurs cibles


def test_un_moteur_est_entier_dans_le_test_ou_la_reserve():
    [scn] = load_scenarios("S5")
    reserve, test = split_target(scn)
    assert not set(scn.groups_t[reserve]) & set(scn.groups_t[test])
    assert max(budgets(scn, len(reserve)).values()) == 4126                # 20 % de n_S
