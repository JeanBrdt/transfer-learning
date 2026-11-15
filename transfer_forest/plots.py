"""Les figures d'un scénario (protocole, section 6).

    plot_scores : note moyenne de chaque méthode selon n_T (± écart type sur les graines),
                  Src et Oracle en lignes horizontales ;
    plot_gains  : gains G_Tgt et G_ref selon n_T, avec leurs IC à 95 %.

Les figures sont construites sans pyplot : elles s'enregistrent partout, même sans écran.
"""
import pandas as pd
from matplotlib.figure import Figure

REFERENCES = ("Src", "Tgt", "Oracle")      # pas de courbe de gain pour elles


def plot_scores(results: pd.DataFrame, metric: str, path) -> None:
    fig = Figure(figsize=(9, 5.5))
    ax = fig.subplots()
    for name, r in results.groupby("method", sort=False):
        if r["n_T"].isna().all():                        # Src et Oracle : pas de budget
            ax.axhline(r[metric].mean(), color="k", ls="--" if name == "Src" else ":", label=name)
        else:
            curve = r.groupby("n_T")[metric]
            ax.errorbar(curve.mean().index, curve.mean(), yerr=curve.std(), marker="o", capsize=3,
                        label=name)
    ax.set(xscale="log", xlabel="$n_T$ : exemples cibles étiquetés",
           ylabel=f"{metric} sur le test cible", title="Note moyenne (± écart type sur les graines)")
    ax.legend(ncol=2, fontsize="small")
    fig.savefig(path, bbox_inches="tight")


def plot_gains(summary: pd.DataFrame, path) -> None:
    fig = Figure(figsize=(13, 5))
    axes = fig.subplots(1, 2, sharey=True)
    for ax, gain, title in zip(axes, ("g_tgt", "g_ref"), ("$G_{Tgt}$", "$G_{ref}$")):
        for name, s in summary.groupby("method", sort=False):
            if name in REFERENCES:
                continue
            mean = s[f"{gain}_mean"]
            ax.errorbar(s["n_T"], mean, yerr=[mean - s[f"{gain}_ci_low"], s[f"{gain}_ci_high"] - mean],
                        marker="o", capsize=3, label=name)
        ax.axhline(0, color="k", lw=1)
        ax.set(xscale="log", xlabel="$n_T$", title=f"Gain {title} (moyenne, IC à 95 %)")
    axes[0].set_ylabel("gain (> 0 : mieux que la référence)")
    axes[1].legend(fontsize="small")
    fig.savefig(path, bbox_inches="tight")
