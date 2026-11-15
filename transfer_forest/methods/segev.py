"""Les méthodes de transfert de Segev et al. (2017) et leurs variantes pour classes rares.

    SER      étendre puis élaguer : on greffe un arbre cible sur les feuilles impures,
             puis on élague de bas en haut
    STRUT    garder la structure, déplacer les seuils (maximise le gain de divergence)
    SER*     SER, mais une feuille source de la classe rare qu'aucun exemple cible n'atteint
             est conservée, et ses ancêtres ne sont pas élagués (Minvielle et al., 2019 ;
             règle du code des auteurs, reprise par ADAPT)
    STRUT*   STRUT, avec les distributions source des fils corrigées par
             λ_k = P_T(k) / P_S(k) (Minvielle et al., 2019, algorithme 5)
    MIX      moyenne des probabilités des forêts SER et STRUT (100 arbres)
    Relab    contrôle : toutes les coupures source gardées, feuilles réestimées sur la cible

Chaque méthode adapte une copie de chaque arbre de ctx.rf_src (jamais modifiée) avec
les exemples cibles étiquetés (Xl, yl). Les observations source ne sont pas relues :
ys ne sert qu'à estimer P_S(Y), pour STRUT*.

Choix d'implémentation que l'état de l'art laisse ouverts :
    - un nœud qu'aucun exemple cible n'atteint : SER le garde tel quel (feuille source) ;
      STRUT l'élague en une feuille qui reprend la distribution cible de son parent (le papier
      garde la décision source du nœud ; sans effet mesuré : ces nœuds sont presque jamais atteints) ;
    - une feuille atteinte par des exemples cibles prédit leurs proportions de classes ;
      une feuille sans exemple cible prédit la distribution de la source ;
    - élagage de SER quand LErr(v) <= STErr(v) (égalité comprise, algorithme 3 de l'état de l'art).
      Le pseudo-code de Segev et al. (2017, algorithme 1) écrit < ; or, après l'expansion, chaque
      feuille atteinte par des exemples cibles a 0 erreur sur eux, donc STErr(v) = 0 et la règle
      stricte n'élague jamais : SER ne ferait qu'étendre (et SER* serait identique à SER) ;
    - STRUT cherche le seuil parmi les maxima locaux du gain d'information cible ;
    - classe rare de SER* : ctx.rare_class, sinon la classe la moins fréquente de yl ;
    - λ_k de STRUT* : proportions de ys et de yl (STRUT : λ_k = 1, donc STRUT* = STRUT
      quand les proportions coïncident).
"""
import warnings

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from transfer_forest.contracts import Context, Method


class _Node:
    """Un nœud d'arbre modifiable (les arbres scikit-learn sont figés)."""
    __slots__ = ("feature", "threshold", "left", "right", "src", "proba")

    def __init__(self, src: np.ndarray, proba: np.ndarray):
        self.feature = self.threshold = self.left = self.right = None
        self.src = src        # distribution de classes de la SOURCE à ce nœud
        self.proba = proba    # distribution prédite si le nœud est (ou devient) une feuille

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    def make_leaf(self, proba: np.ndarray) -> None:
        self.feature = self.threshold = self.left = self.right = None
        self.proba = proba


def _clone(tree, classes, K: int, i: int = 0) -> _Node:
    """Recopie un arbre scikit-learn ; src = proportions des classes à chaque nœud."""
    v = np.zeros(K)
    v[classes] = tree.value[i].ravel()
    v /= v.sum()
    node = _Node(v, v)
    if tree.children_left[i] != _tree.TREE_LEAF:
        node.feature, node.threshold = int(tree.feature[i]), np.float64(tree.threshold[i])
        node.left = _clone(tree, classes, K, tree.children_left[i])
        node.right = _clone(tree, classes, K, tree.children_right[i])
    return node


# ---------------------------------------------------------------- SER

def _has_leaf_of(node: _Node, c: int) -> bool:
    """Le sous-arbre contient-il une feuille dont la classe majoritaire source est c ?"""
    if node.is_leaf:
        return node.src.argmax() == c
    return _has_leaf_of(node.left, c) or _has_leaf_of(node.right, c)


def _ser(node: _Node, X, y, idx, K: int, seed: int, protected: int | None) -> tuple[int, bool]:
    """Adapte le sous-arbre de `node` avec les exemples cibles `idx`.

    Renvoie le nombre d'erreurs du sous-arbre adapté sur ces exemples, et s'il
    contient une feuille protégée par SER* (ce qui interdit d'élaguer ses ancêtres).
    """
    n = len(idx)
    if n == 0:                                     # rien ne l'atteint : on garde la source
        return 0, protected is not None and _has_leaf_of(node, protected)
    counts = np.bincount(y[idx], minlength=K)
    leaf_err = n - counts.max()                    # erreurs si le nœud devenait une feuille
    if node.is_leaf:
        if leaf_err > 0:                           # expansion : arbre complet sur les cibles
            with warnings.catch_warnings():        # greffe sur quelques exemples : scikit-learn
                warnings.simplefilter("ignore")    # croirait à tort à une régression
                graft = DecisionTreeClassifier(random_state=seed).fit(X[idx], y[idx])
            sub = _clone(graft.tree_, graft.classes_.astype(int), K)
            node.feature, node.threshold = sub.feature, sub.threshold
            node.left, node.right, node.proba = sub.left, sub.right, sub.proba
        else:
            node.proba = counts / n                # feuille pure : réétiquetée si besoin
        return 0, False
    go = X[idx, node.feature] <= node.threshold
    err_l, prot_l = _ser(node.left, X, y, idx[go], K, seed, protected)
    err_r, prot_r = _ser(node.right, X, y, idx[~go], K, seed, protected)
    err, prot = err_l + err_r, prot_l or prot_r
    if leaf_err <= err and not prot:               # réduction (égalité comprise)
        node.make_leaf(counts / n)
        return leaf_err, False
    return err, prot


# ---------------------------------------------------------------- Relab

def _relab(node: _Node, X, y, idx, K: int) -> None:
    """Garde toutes les coupures ; chaque feuille atteinte prend les proportions cibles."""
    if len(idx) == 0:
        return                                     # feuille non atteinte : distribution source
    if node.is_leaf:
        node.proba = np.bincount(y[idx], minlength=K) / len(idx)
        return
    go = X[idx, node.feature] <= node.threshold
    _relab(node.left, X, y, idx[go], K)
    _relab(node.right, X, y, idx[~go], K)


# ---------------------------------------------------------------- STRUT

def _entropy(c: np.ndarray) -> np.ndarray:
    """Entropie (bits) de chaque ligne d'effectifs."""
    p = c / np.maximum(c.sum(axis=-1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=-1)


def _jsd(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Divergence de Jensen-Shannon entre chaque ligne de P et la distribution q."""
    Q = np.broadcast_to(q, P.shape)
    M = (P + Q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = lambda A: np.where(A > 0, A * np.log(A / M), 0.0).sum(axis=1)
        return (kl(P) + kl(Q)) / 2


def _correct(P: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """P*_k = λ_k P_k / Σ_j λ_j P_j (équation 10) ; P inchangée si la somme est nulle."""
    w = lam * P
    s = w.sum()
    return w / s if s > 0 else P


def _strut(node: _Node, X, y, idx, K: int, lam: np.ndarray, parent_proba: np.ndarray) -> None:
    """Recalcule les seuils de haut en bas ; les nœuds sans exemple cible sont élagués."""
    n = len(idx)
    if n == 0:
        node.make_leaf(parent_proba)
        return
    counts = np.bincount(y[idx], minlength=K)
    proba = counts / n
    if node.is_leaf:
        node.proba = proba
        return
    x = X[idx, node.feature]
    values = np.unique(x).astype(np.float64)       # seuils en float64, comme scikit-learn
    if len(values) < 2:                            # rien à séparer sur cette variable
        node.make_leaf(proba)
        return
    cand = (values[:-1] + values[1:]) / 2          # seuils candidats
    order = np.argsort(x, kind="stable")
    cum = np.cumsum(np.eye(K)[y[idx][order]], axis=0)
    cL = cum[np.searchsorted(x[order], cand, side="right") - 1]
    cR = counts - cL
    nL, nR = cL.sum(axis=1), cR.sum(axis=1)
    gain = _entropy(counts[None])[0] - nL / n * _entropy(cL) - nR / n * _entropy(cR)
    g = np.r_[-np.inf, gain, -np.inf]              # maxima locaux du gain d'information ;
    k = np.flatnonzero((g[1:-1] >= g[:-2]) & (g[1:-1] >= g[2:]))   # une extrémité doit battre sa voisine
    QL, QR = cL[k] / nL[k, None], cR[k] / nR[k, None]
    wL, wR = nL[k] / n, nR[k] / n
    PL, PR = _correct(node.left.src, lam), _correct(node.right.src, lam)
    dg = np.column_stack([1 - wL * _jsd(QL, PL) - wR * _jsd(QR, PR),    # (6) / (11)
                          1 - wL * _jsd(QL, PR) - wR * _jsd(QR, PL)]).ravel()  # fils échangés
    i = int(np.argmax(dg))
    node.threshold = np.float64(cand[k[i // 2]])
    if i % 2:
        node.left, node.right = node.right, node.left
    go = x <= node.threshold
    _strut(node.left, X, y, idx[go], K, lam, proba)
    _strut(node.right, X, y, idx[~go], K, lam, proba)


# ---------------------------------------------------------------- prédiction

def _compile(root: _Node):
    """Aplatit un arbre en tableaux : la prédiction se fait ensuite sans récursion."""
    feat, thr, left, right, proba = [], [], [], [], []

    def add(n: _Node) -> int:
        i = len(feat)
        feat.append(-1 if n.is_leaf else n.feature)
        thr.append(0.0 if n.is_leaf else n.threshold)
        left.append(-1)
        right.append(-1)
        proba.append(n.proba)
        if not n.is_leaf:
            left[i] = add(n.left)
            right[i] = add(n.right)
        return i

    add(root)
    return (np.array(feat), np.array(thr), np.array(left), np.array(right), np.array(proba))


def _route(tree, X: np.ndarray) -> np.ndarray:
    """Distribution de la feuille atteinte par chaque ligne de X."""
    feat, thr, left, right, proba = tree
    cur = np.zeros(len(X), dtype=np.intp)
    rows = np.arange(len(X))
    while True:
        f = feat[cur]
        inner = f >= 0
        if not inner.any():
            return proba[cur]
        go = X[rows, np.where(inner, f, 0)] <= thr[cur]
        cur = np.where(inner, np.where(go, left[cur], right[cur]), cur)


# ---------------------------------------------------------------- les méthodes

class _Segev(Method):
    """Adapte une copie de chaque arbre source. Une classe fille définit prepare et adapt."""
    uses_budget = True

    def prepare(self, ctx: Context) -> None:
        pass

    def adapt(self, root: _Node, X, y, idx, ctx: Context, seed: int) -> None:
        raise NotImplementedError

    def fit(self, ctx: Context) -> "_Segev":
        self._K = ctx.K
        X = np.asarray(ctx.Xl, dtype=np.float32)   # scikit-learn compare en float32
        y = np.asarray(ctx.yl)
        idx = np.arange(len(y))
        self.prepare(ctx)
        self._trees = []
        for i, est in enumerate(ctx.rf_src.estimators_):
            root = _clone(est.tree_, est.classes_.astype(int), ctx.K)
            self.adapt(root, X, y, idx, ctx, ctx.seed + i)
            self._trees.append(_compile(root))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.mean([_route(t, X) for t in self._trees], axis=0)


class SER(_Segev):
    name = "SER"
    protect = False

    def prepare(self, ctx):
        self._protected = None
        if self.protect:
            c = ctx.rare_class
            self._protected = int(np.argmin(np.bincount(ctx.yl, minlength=ctx.K))) if c is None else c

    def adapt(self, root, X, y, idx, ctx, seed):
        _ser(root, X, y, idx, ctx.K, seed, self._protected)


class SERStar(SER):
    name = "SER*"
    protect = True


class STRUT(_Segev):
    name = "STRUT"
    correct_priors = False

    def prepare(self, ctx):
        self._lam = np.ones(ctx.K)
        if self.correct_priors:
            p_t = np.bincount(ctx.yl, minlength=ctx.K) / len(ctx.yl)
            p_s = np.bincount(ctx.ys, minlength=ctx.K) / len(ctx.ys)
            self._lam = p_t / p_s

    def adapt(self, root, X, y, idx, ctx, seed):
        _strut(root, X, y, idx, ctx.K, self._lam, root.src)


class STRUTStar(STRUT):
    name = "STRUT*"
    correct_priors = True


class Relab(_Segev):
    name = "Relab"

    def adapt(self, root, X, y, idx, ctx, seed):
        _relab(root, X, y, idx, ctx.K)


class MIX(Method):
    """Union des forêts SER et STRUT : 2 x 50 arbres, probabilités moyennées."""
    name = "MIX"
    uses_budget = True

    def fit(self, ctx: Context) -> "MIX":
        self._forests = [SER().fit(ctx), STRUT().fit(ctx)]
        return self

    def predict_proba(self, X):
        return np.mean([f.predict_proba(X) for f in self._forests], axis=0)
