"""Tests de base des méthodes de Segev (SER, STRUT, SER*, STRUT*, MIX, Relab) sur données synthétiques."""
import numpy as np
import pytest

from transfer_forest.contracts import Context, check_proba, make_forest
from transfer_forest.methods import make_method
from transfer_forest.methods.segev import _Node, _clone, _ser, _strut

K = 3
NAMES = ["SER", "STRUT", "SER*", "STRUT*", "MIX", "Relab"]


def _data(n, shift, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n).astype(np.int64)
    X = rng.normal(size=(n, 4)) + y[:, None] * 2.0 + shift   # classes séparées, domaine décalé
    return X.astype(np.float64), y


@pytest.fixture(scope="module")
def ctx_and_test():
    Xs, ys = _data(1500, 0.0, 0)
    Xt, yt = _data(900, 1.5, 1)
    Xl, yl, Xte, yte = Xt[:150], yt[:150], Xt[150:], yt[150:]
    rf = make_forest(0, 20).fit(Xs, ys)
    return Context(Xs, ys, Xl, yl, Xte[:0], rf, K, 0, n_trees=20), Xte, yte


@pytest.mark.parametrize("name", NAMES)
def test_probas_valides(name, ctx_and_test):
    ctx, Xte, _ = ctx_and_test
    P = make_method(name).fit(ctx).predict_proba(Xte)
    check_proba(P, len(Xte), K)


@pytest.mark.parametrize("name", NAMES)
def test_foret_source_non_modifiee(name, ctx_and_test):
    ctx, Xte, _ = ctx_and_test
    before = ctx.rf_src.predict_proba(Xte)
    values = [e.tree_.value.copy() for e in ctx.rf_src.estimators_]
    make_method(name).fit(ctx)
    assert np.array_equal(before, ctx.rf_src.predict_proba(Xte))
    assert all(np.array_equal(v, e.tree_.value) for v, e in zip(values, ctx.rf_src.estimators_))


@pytest.mark.parametrize("name", NAMES)
def test_deterministe(name, ctx_and_test):
    ctx, Xte, _ = ctx_and_test
    a = make_method(name).fit(ctx).predict_proba(Xte)
    b = make_method(name).fit(ctx).predict_proba(Xte)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ["SER", "SER*"])
def test_ser_ameliore_sous_decalage(name, ctx_and_test):
    ctx, Xte, yte = ctx_and_test
    src_err = (ctx.rf_src.predict(Xte) != yte).mean()
    err = (make_method(name).fit(ctx).predict_proba(Xte).argmax(1) != yte).mean()
    assert err < src_err


def test_clone_seuils_float64_exacts(ctx_and_test):
    """Les seuils copiés sont ceux de scikit-learn, en float64 (pas arrondis en float32)."""
    ctx, _, _ = ctx_and_test
    est = ctx.rf_src.estimators_[0]
    root = _clone(est.tree_, est.classes_.astype(int), K)
    assert isinstance(root.threshold, np.float64)
    assert root.threshold == est.tree_.threshold[0]


def test_strut_etoile_egal_strut_si_proportions_egales():
    """Si P_S(Y) = P_T(Y), λ_k = 1 et STRUT* retrouve STRUT."""
    Xs, ys = _data(1500, 0.0, 0)
    Xl = np.vstack([Xs[ys == k][:60] for k in range(K)])
    yl = np.repeat(np.arange(K), 60).astype(np.int64)
    # source rééquilibrée : mêmes proportions que le budget cible
    keep = np.concatenate([np.flatnonzero(ys == k)[:400] for k in range(K)])
    Xs, ys = Xs[keep], ys[keep]
    rf = make_forest(0, 10).fit(Xs, ys)
    ctx = Context(Xs, ys, Xl + 1.0, yl, Xl[:0], rf, K, 0, n_trees=10)
    Xte = _data(300, 1.0, 5)[0]
    a = make_method("STRUT").fit(ctx).predict_proba(Xte)
    b = make_method("STRUT*").fit(ctx).predict_proba(Xte)
    assert np.allclose(a, b)


def test_ser_elagage_sur_egalite():
    """Erreurs égales (0 et 0) : le sous-arbre est élagué en une feuille (réduction de SER).

    Avec une inégalité stricte, rien ne serait jamais élagué après l'expansion.
    """
    root = _Node(np.array([.5, .5]), np.array([.5, .5]))
    root.feature, root.threshold = 0, np.float64(0.5)
    root.left = _Node(np.array([1., 0.]), np.array([1., 0.]))
    root.right = _Node(np.array([0., 1.]), np.array([0., 1.]))
    X = np.array([[0.2], [0.8]], dtype=np.float32)
    y = np.array([0, 0])
    _ser(root, X, y, np.arange(2), 2, 0, None)
    assert root.is_leaf and root.proba[0] == 1


def test_ser_etoile_protege_la_classe_rare():
    """SER* garde une feuille source de la classe protégée qu'aucun exemple cible n'atteint."""
    def tree():
        root = _Node(np.array([.5, .5]), np.array([.5, .5]))
        root.feature, root.threshold = 0, np.float64(0.5)
        root.left = _Node(np.array([1., 0.]), np.array([1., 0.]))
        root.right = _Node(np.array([0., 1.]), np.array([0., 1.]))
        return root
    X = np.array([[0.2], [0.3]], dtype=np.float32)    # tous à gauche : la feuille de la classe 1
    y = np.array([0, 0])                              # n'est atteinte par aucun exemple cible
    plain, protected = tree(), tree()
    _ser(plain, X, y, np.arange(2), 2, 0, None)
    _ser(protected, X, y, np.arange(2), 2, 0, 1)
    assert plain.is_leaf and not protected.is_leaf
    assert protected.right.proba[1] == 1               # la feuille protégée garde la source

    reached = tree()                                   # atteinte, la feuille n'est pas protégée
    _ser(reached, np.array([[0.2], [0.8]], dtype=np.float32), y, np.arange(2), 2, 0, 1)
    assert reached.is_leaf


def test_strut_ne_retient_que_les_maxima_locaux():
    """Une extrémité n'est candidate que si elle bat sa voisine.

    Ici le gain de divergence préférerait une coupure extrême (1,5 ou 5,5), qui imite la
    distribution source 50/50 sans rien séparer ; le seul maximum local du gain
    d'information est 3,5, la coupure parfaite.
    """
    half = np.array([.5, .5])
    root = _Node(half, half)
    root.feature, root.threshold = 0, np.float64(10.0)
    root.left, root.right = _Node(half, half), _Node(half, half)
    X = np.arange(1, 7, dtype=np.float32)[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    _strut(root, X, y, np.arange(6), 2, np.ones(2), root.src)
    assert root.threshold == 3.5
