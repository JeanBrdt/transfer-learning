"""Les références sans transfert, que toute méthode de transfert doit battre.

    Src      forêt source appliquée telle quelle : mesure le décalage
    Tgt      cible seule : la référence principale
    Union    source + cible mélangées : le transfert « naïf »
    Union-P  idem, cibles surpondérées : masse totale cible = masse totale source
    Oracle   toute la réserve cible étiquetée : le plafond, pas un concurrent
"""
import numpy as np

from transfer_forest.contracts import Context, Method, full_proba, make_forest


class Src(Method):
    name = "Src"
    uses_budget = False

    def fit(self, ctx: Context) -> "Src":
        # aucun apprentissage : on reprend la forêt source partagée, sans la modifier
        self._K, self._rf = ctx.K, ctx.rf_src
        return self

    def predict_proba(self, X):
        return full_proba(self._rf, X, self._K)


class _Forest(Method):
    """Une forêt neuve apprise sur des données tirées du contexte.

    Chaque référence ne définit que training_data : (X, y, poids ou None).
    """

    def training_data(self, ctx: Context):
        raise NotImplementedError

    def fit(self, ctx: Context) -> "_Forest":
        X, y, w = self.training_data(ctx)
        self._K = ctx.K
        self._rf = make_forest(ctx.seed, ctx.n_trees).fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return full_proba(self._rf, X, self._K)


class Tgt(_Forest):
    name = "Tgt"
    uses_budget = True

    def training_data(self, ctx):
        return ctx.Xl, ctx.yl, None


class Union(_Forest):
    name = "Union"
    uses_budget = True

    def training_data(self, ctx):
        return np.vstack([ctx.Xs, ctx.Xl]), np.concatenate([ctx.ys, ctx.yl]), None


class UnionP(_Forest):
    name = "Union-P"
    uses_budget = True

    def training_data(self, ctx):
        n_S, n_T = len(ctx.ys), len(ctx.yl)
        w = np.concatenate([np.ones(n_S), np.full(n_T, n_S / n_T)])
        return np.vstack([ctx.Xs, ctx.Xl]), np.concatenate([ctx.ys, ctx.yl]), w


class Oracle(_Forest):
    name = "Oracle"
    uses_budget = False

    def training_data(self, ctx):
        return ctx.X_reserve, ctx.y_reserve, None
