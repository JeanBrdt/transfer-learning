"""Tests de la notation (evaluation.py) et des gains (scoring.py).

Valeurs calculées à la main, et recoupées avec scikit-learn quand il existe un
équivalent.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import log_loss as sk_log_loss

from transfer_forest.contracts import full_proba, make_forest
from transfer_forest.evaluation import NOTES, evaluate
from transfer_forest.scoring import bootstrap_ci, compute_gains, summarize

# 10 exemples : 8 de la classe 0, 2 de la classe 1 (déséquilibré, comme S2).
Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
# Une erreur dans chaque classe : l'exemple 7 (classe 0) et l'exemple 8 (classe 1).
P = np.array([[0.9, 0.1]] * 7 + [[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])


# ------------------------------------------------------------------ evaluate
def test_notes_by_hand():
    n = evaluate(Y, P)
    assert set(n) == set(NOTES)
    assert n["error"] == pytest.approx(2 / 10)
    # BER = (1/8 + 1/2) / 2 : la classe rare pèse autant que la classe majoritaire.
    assert n["ber"] == pytest.approx(0.3125)
    # F1 : classe 0, TP=7 FP=1 FN=1 -> 14/16 ; classe 1, TP=1 FP=1 FN=1 -> 2/4.
    assert n["f1_macro"] == pytest.approx((14 / 16 + 2 / 4) / 2)
    assert n["fnr_critical"] == pytest.approx(1 / 2)          # pire classe : la 1
    expected_ll = -np.mean(np.log([0.9] * 7 + [0.3, 0.4, 0.8]))
    assert n["log_loss"] == pytest.approx(expected_ll)


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=200)
    P = rng.dirichlet(np.ones(4), size=200)
    pred = P.argmax(axis=1)
    n = evaluate(y, P)
    assert n["ber"] == pytest.approx(1 - balanced_accuracy_score(y, pred))
    assert n["f1_macro"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert n["log_loss"] == pytest.approx(sk_log_loss(y, P, labels=range(4)))


def test_trivial_classifier_low_error_but_high_ber():
    # Tout prédire « 0 » (aucune mine détectée) : erreur 0,2 mais BER 0,5.
    n = evaluate(Y, np.tile([1.0, 0.0], (10, 1)))
    assert n["error"] == pytest.approx(0.2)
    assert n["ber"] == pytest.approx(0.5)
    assert n["fnr_critical"] == pytest.approx(1.0)
    assert np.isfinite(n["log_loss"])            # P = 0 sur la vraie classe : perte finie


def test_critical_class():
    assert evaluate(Y, P, critical_class=0)["fnr_critical"] == pytest.approx(1 / 8)
    assert evaluate(Y, P, critical_class=1)["fnr_critical"] == pytest.approx(1 / 2)
    with pytest.raises(ValueError):
        evaluate(Y, P, critical_class=5)


def test_class_absent_from_test_is_ignored():
    # 3 classes au contrat, mais la classe 2 n'est pas dans ce test : BER et F1
    # ne moyennent que sur les classes 0 et 1, même si la classe 2 est prédite.
    P3 = np.hstack([P, np.zeros((10, 1))])
    P3[0] = [0.1, 0.1, 0.8]
    n = evaluate(Y, P3)
    assert n["ber"] == pytest.approx((2 / 8 + 1 / 2) / 2)
    assert n["f1_macro"] == pytest.approx(
        f1_score(Y, P3.argmax(1), labels=[0, 1], average="macro"))


def test_contract_checks():
    with pytest.raises(ValueError):
        evaluate(Y, P[:5])                       # mauvais nombre de lignes
    with pytest.raises(ValueError):
        evaluate(Y, P * 2)                       # lignes qui ne somment pas à 1
    with pytest.raises(ValueError):
        evaluate(Y + 1, P)                       # classe 2 hors de 0..K-1
    with pytest.raises(ValueError):
        evaluate(Y.astype(float), P)             # étiquettes non entières


def test_with_a_real_forest():
    # Chaîne complète du contrat : forêt -> full_proba -> evaluate.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(np.int64)
    rf = make_forest(0).fit(X[:200], y[:200])
    n = evaluate(y[200:], full_proba(rf, X[200:], K=2))
    assert n["error"] == pytest.approx(np.mean(rf.predict(X[200:]) != y[200:]))
    assert n["error"] < 0.2


# ---------------------------------------------------------------------- gains
def _results(err_col="error"):
    """Le format du notebook : seed, n_T, method, note ; Src et Oracle avec n_T = 0."""
    rows = [
        (0, 0, "Src", 0.60), (0, 0, "Oracle", 0.06),
        (0, 108, "Tgt", 0.40), (0, 108, "Union", 0.38), (0, 108, "SER", 0.33),
        (0, 216, "Tgt", 0.29), (0, 216, "Union", 0.28), (0, 216, "SER", 0.30),
        (1, 0, "Src", 0.55), (1, 0, "Oracle", 0.06),
        (1, 108, "Tgt", 0.42), (1, 108, "Union", 0.37), (1, 108, "SER", 0.35),
        (1, 216, "Tgt", 0.30), (1, 216, "Union", 0.31), (1, 216, "SER", 0.27),
    ]
    return pd.DataFrame(rows, columns=["seed", "n_T", "method", err_col])


def _g(gains, seed, n_T, method, col):
    row = gains[(gains.seed == seed) & (gains.n_T == n_T) & (gains.method == method)]
    assert len(row) == 1
    return row[col].iloc[0]


def test_gains_by_hand():
    g = compute_gains(_results())
    assert _g(g, 0, 108, "SER", "g_tgt") == pytest.approx(0.40 - 0.33)
    # G_ref : min(Src 0,60 ; Tgt 0,40 ; Union 0,38) = 0,38.
    assert _g(g, 0, 108, "SER", "g_ref") == pytest.approx(0.38 - 0.33)
    assert _g(g, 0, 216, "SER", "g_tgt") == pytest.approx(-0.01)   # perd contre Tgt
    assert _g(g, 1, 216, "SER", "g_ref") == pytest.approx(0.30 - 0.27)
    assert _g(g, 0, 108, "Tgt", "g_tgt") == 0


def test_src_and_oracle_are_copied_on_every_budget():
    g = compute_gains(_results())
    assert set(g.loc[g.method == "Src", "n_T"]) == {108, 216}
    assert _g(g, 1, 216, "Src", "score") == pytest.approx(0.55)
    assert _g(g, 0, 216, "Oracle", "g_tgt") == pytest.approx(0.29 - 0.06)
    assert g["n_T"].notna().all() and (g["n_T"] > 0).all()


def test_empty_n_T_like_after_notebook_conversion():
    # Le notebook remplace n_T = 0 par NaN avant d'analyser : même résultat.
    r = _results()
    r["n_T"] = r["n_T"].replace(0, np.nan)
    pd.testing.assert_frame_equal(compute_gains(r), compute_gains(_results()), check_dtype=False)


def test_old_err_column_is_read_as_error():
    pd.testing.assert_frame_equal(compute_gains(_results("err")), compute_gains(_results()))


def test_refs_can_include_union_p():
    r = pd.concat([_results(), pd.DataFrame(
        [(0, 108, "Union-P", 0.31), (0, 216, "Union-P", 0.22),
         (1, 108, "Union-P", 0.30), (1, 216, "Union-P", 0.23)],
        columns=["seed", "n_T", "method", "error"])])
    g = compute_gains(r, refs=("Src", "Tgt", "Union", "Union-P"))
    assert _g(g, 0, 108, "SER", "g_ref") == pytest.approx(0.31 - 0.33)   # SER perd contre Union-P
    assert _g(g, 0, 108, "SER", "g_tgt") == pytest.approx(0.07)         # G_Tgt inchangé


def test_metric_ber():
    r = _results()
    r["ber"] = r["error"] + 0.05
    g = compute_gains(r, metric="ber")
    assert (g.metric == "ber").all()
    assert _g(g, 0, 108, "SER", "score") == pytest.approx(0.38)
    with pytest.raises(ValueError, match="f1"):
        compute_gains(r, metric="f1")


def test_missing_reference_gives_nan_with_warning():
    r = _results()
    r = r[~((r.seed == 1) & (r.n_T == 216) & (r.method == "Union"))]
    with pytest.warns(UserWarning, match="Union"):
        g = compute_gains(r)
    assert np.isnan(_g(g, 1, 216, "SER", "g_ref"))
    assert _g(g, 1, 216, "SER", "g_tgt") == pytest.approx(0.03)          # Tgt est là


def test_duplicates_are_rejected():
    with pytest.raises(ValueError, match="plusieurs"):
        compute_gains(pd.concat([_results(), _results().iloc[[2]]]))


def test_several_scenarios():
    a, b = _results(), _results()
    a["scenario"], b["scenario"] = "S1", "S2"
    b["error"] += 0.1
    g = compute_gains(pd.concat([a, b]))
    s2 = g[(g.scenario == "S2") & (g.seed == 0) & (g.n_T == 108) & (g.method == "SER")]
    assert s2["g_tgt"].iloc[0] == pytest.approx(0.07)        # même écart, pas de mélange
    assert len(g) == 2 * len(compute_gains(_results()))


# -------------------------------------------------------------------- synthèse
def test_summary_by_hand():
    s = summarize(compute_gains(_results()), n_boot=2000)
    ser = s[(s.method == "SER") & (s.n_T == 108)].iloc[0]
    assert ser["n_seeds"] == 2
    assert ser["g_tgt_mean"] == pytest.approx((0.07 + 0.07) / 2)
    assert ser["g_tgt_win_rate"] == 1.0
    ser216 = s[(s.method == "SER") & (s.n_T == 216)].iloc[0]
    assert ser216["g_tgt_win_rate"] == 0.5                     # perd graine 0, gagne graine 1
    tgt = s[(s.method == "Tgt") & (s.n_T == 108)].iloc[0]
    assert tgt["g_tgt_win_rate"] == 0.0                        # égalité : pas une victoire


def test_bootstrap_ci_on_20_seeds():
    rng = np.random.default_rng(2)
    g = 0.05 + rng.normal(0, 0.01, size=20)
    lo, hi = bootstrap_ci(g)
    assert lo < g.mean() < hi
    assert lo < 0.05 < hi
    assert hi - lo < 0.02
    assert np.isnan(bootstrap_ci([0.1])[0])                    # une seule graine : pas d'IC


def test_summary_is_reproducible():
    g = compute_gains(_results())
    pd.testing.assert_frame_equal(summarize(g), summarize(g))
