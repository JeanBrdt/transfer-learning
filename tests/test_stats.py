"""Tests de la significativité et des indicateurs de synthèse (transfer_forest.scoring)."""
import numpy as np
import pandas as pd
import pytest

from transfer_forest.scoring import annotation_saving, crossover_budget, holm, wilcoxon_p


def test_holm_by_hand():
    # 3 p-valeurs : 0,01 x 3 = 0,03 ; 0,03 x 2 = 0,06 ; 0,04 x 1 = 0,04 -> 0,06 (croissance imposée)
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, np.nan]), [0.03, 0.06, 0.06, np.nan])


def test_wilcoxon():
    gains = 0.05 + np.random.default_rng(0).normal(0, 0.01, 20)
    assert wilcoxon_p(gains) < 1e-3                          # 20 gains positifs : significatif
    assert np.isnan(wilcoxon_p(np.zeros(20)))                # Tgt contre lui-même


def test_crossover_budget():
    s = pd.DataFrame({
        "method": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        "n_T": [108, 216, 541] * 3,
        "g_tgt_mean": [0.05, 0.03, 0.01, 0.05, 0.03, 0.02, -0.01, 0.0, 0.0],
        "g_tgt_p_holm": [0.001, 0.01, 0.5, 0.001, 0.01, 0.02, 0.01, 0.9, 0.9],
    })
    cross = crossover_budget(s)
    assert cross["A"] == 541                                 # plus de gain à partir de 541
    assert np.isnan(cross["B"])                              # encore un gain au plus grand budget
    assert cross["C"] == 108                                 # jamais de gain (C perd à 108)


def test_annotation_saving():
    s = pd.DataFrame({"method": ["Tgt"] * 3 + ["M", "M"], "n_T": [100, 200, 400, 100, 100],
                      "score_mean": [0.4, 0.3, 0.2, 0.3, 0.1]})
    saving = annotation_saving(s)
    assert saving.iloc[3] == pytest.approx(2.0)              # M avec 100 = Tgt avec 200
    assert np.isnan(saving.iloc[4])                          # mieux que Tgt à tous les budgets testés
    assert saving.iloc[1] == pytest.approx(1.0)              # Tgt contre lui-même
