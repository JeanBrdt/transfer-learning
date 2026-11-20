"""Scénario S6 : HHAR (UCI, id 344), reconnaissance d'activité avec des capteurs hétérogènes.

Données : accéléromètre et gyroscope de téléphones et de montres, 9 utilisateurs, 6 activités
(bike, sit, stairsdown, stairsup, stand, walk). Les appareils n'échantillonnent pas à la même
fréquence (50 à 200 Hz) et ne se portent pas au même endroit : le décalage entre domaines est réel.

Une forêt ne lit pas des mesures brutes une à une : on les résume en fenêtres.

    fenêtre   2 s, sans recouvrement, par (appareil, utilisateur), découpée sur Creation_Time ;
    variables 32 = 2 capteurs (acc, gyro) x 4 signaux (x, y, z, norme) x 4 résumés
              (moyenne, écart-type, min, max) ;
    étiquette l'activité majoritaire de la fenêtre. On écarte les fenêtres « null » (aucune
              activité), celles où l'activité change (moins de 90 % d'échantillons d'une même
              étiquette) et celles de moins de 40 échantillons par capteur.

Les `Samsung Galaxy S+` (samsungold) n'ont pas de gyroscope : ils sont écartés.

Source et cible : un type d'appareil contre l'autre, par défaut montres (source, 7 521 fenêtres)
vers téléphones (cible, 35 175 fenêtres). Le sens inverse est impossible avec le protocole : le
budget maximal (20 % de la source = 7 035) dépasserait la réserve de la cible (50 % de 7 521).
``groups_t`` de la cible donne l'utilisateur de chaque fenêtre.

Données brutes : https://archive.ics.uci.edu/dataset/344 (fichier « Activity recognition exp.zip »),
à décompresser dans data/hhar/. Le résumé en fenêtres (une fois pour toutes) est mis en cache
dans data/hhar/windows.csv.gz :

    python -m transfer_forest.data.s6_hhar        # construit le cache (quelques minutes)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_forest.contracts import Scenario, check_scenario

DATA_DIR = Path(__file__).parents[2] / "data" / "hhar"
CACHE = DATA_DIR / "windows.csv.gz"
PARTIALS = DATA_DIR / "partials"            # un résumé intermédiaire par fichier brut

FILES = {("phone", "acc"): "Phones_accelerometer.csv", ("phone", "gyro"): "Phones_gyroscope.csv",
         ("watch", "acc"): "Watch_accelerometer.csv", ("watch", "gyro"): "Watch_gyroscope.csv"}
ACTIVITIES = ("bike", "sit", "stairsdown", "stairsup", "stand", "walk")   # ordre alphabétique
NULL = "null"
WINDOW_S = 2.0
MIN_SAMPLES = 40                            # échantillons minimum par capteur dans une fenêtre
MIN_PURITY = 0.9                            # part minimale de l'étiquette majoritaire
SIGNALS = ("x", "y", "z", "m")              # m = norme du vecteur (x, y, z)
STATS = ("mean", "std", "min", "max")
KEY = ["device_type", "Model", "Device", "User", "win"]


def feature_names() -> tuple[str, ...]:
    return tuple(f"{sensor}_{sig}_{stat}" for sensor in ("acc", "gyro") for sig in SIGNALS for stat in STATS)


# --------------------------------------------------------------------------- résumé des fenêtres

def _partial(chunk: pd.DataFrame, device_type: str, window_ns: int) -> pd.DataFrame:
    """Sommes, minimums, maximums et comptes d'étiquettes par fenêtre, pour un morceau du fichier."""
    v = {s: chunk[s].to_numpy(np.float64) for s in "xyz"}
    v["m"] = np.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
    g = pd.DataFrame({"Model": chunk["Model"].astype(str), "Device": chunk["Device"].astype(str),
                      "User": chunk["User"].astype(str), "win": chunk["Creation_Time"] // window_ns,
                      "gt": chunk["gt"].astype(str), "n": 1})
    for s in SIGNALS:
        g[f"sum_{s}"], g[f"sq_{s}"], g[f"min_{s}"], g[f"max_{s}"] = v[s], v[s] ** 2, v[s], v[s]
    keys = ["Model", "Device", "User", "win"]
    agg = {"n": "sum"}
    for s in SIGNALS:
        agg |= {f"sum_{s}": "sum", f"sq_{s}": "sum", f"min_{s}": "min", f"max_{s}": "max"}
    stats = g.groupby(keys).agg(agg)
    labels = g.groupby(keys + ["gt"]).size().unstack("gt", fill_value=0)
    labels = labels.reindex(columns=list(ACTIVITIES) + [NULL], fill_value=0).add_prefix("gt_")
    # une étiquette hors liste serait ignorée en silence : on s'arrête plutôt
    unknown = set(g["gt"]) - set(ACTIVITIES) - {NULL}
    if unknown:
        raise ValueError(f"étiquettes inconnues dans HHAR : {sorted(unknown)}")
    out = stats.join(labels).reset_index()
    out.insert(0, "device_type", device_type)
    return out


def _merge(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Réunit les résumés de plusieurs morceaux (une fenêtre peut être coupée entre deux morceaux)."""
    df = pd.concat(parts, ignore_index=True)
    agg = {c: "sum" for c in df.columns if c.startswith(("n", "sum_", "sq_", "gt_")) and c not in KEY}
    agg |= {c: "min" for c in df.columns if c.startswith("min_")}
    agg |= {c: "max" for c in df.columns if c.startswith("max_")}
    return df.groupby(KEY, sort=False).agg(agg).reset_index()


def summarize_file(path: Path, device_type: str, window_s: float = WINDOW_S,
                   chunksize: int = 2_000_000) -> pd.DataFrame:
    """Résume un fichier brut (un capteur d'un type d'appareil) en une ligne par fenêtre.

    Lecture par morceaux : les fichiers font plus d'un Go.
    `keep_default_na=False` est indispensable : sinon pandas lit la chaîne « null » (colonne gt)
    comme une valeur manquante.
    """
    window_ns = int(window_s * 1e9)
    dtypes = {"Creation_Time": "int64", "x": "float64", "y": "float64", "z": "float64"}
    cols = ["Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"]
    parts = [_partial(ch, device_type, window_ns)
             for ch in pd.read_csv(path, usecols=cols, dtype=dtypes, keep_default_na=False,
                                   chunksize=chunksize)]
    return _merge(parts)


def finalize(summary: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Moyenne, écart-type, min et max de chaque signal ; garde n et les comptes d'étiquettes."""
    n = summary["n"].to_numpy(np.float64)
    out = summary[KEY].copy()
    out[f"n_{sensor}"] = summary["n"].astype(int)
    for s in SIGNALS:
        mean = summary[f"sum_{s}"] / n
        var = np.maximum(summary[f"sq_{s}"] / n - mean ** 2, 0.0)
        out[f"{sensor}_{s}_mean"] = mean
        out[f"{sensor}_{s}_std"] = np.sqrt(var)
        out[f"{sensor}_{s}_min"] = summary[f"min_{s}"]
        out[f"{sensor}_{s}_max"] = summary[f"max_{s}"]
    for a in list(ACTIVITIES) + [NULL]:
        out[f"gt_{a}_{sensor}"] = summary[f"gt_{a}"].astype(int)
    return out


def assemble(acc: pd.DataFrame, gyro: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par fenêtre valide, avec les variables des deux capteurs et l'étiquette."""
    w = finalize(acc, "acc").merge(finalize(gyro, "gyro"), on=KEY, how="inner")
    w = w[(w["n_acc"] >= MIN_SAMPLES) & (w["n_gyro"] >= MIN_SAMPLES)]
    counts = np.stack([w[f"gt_{a}_acc"] + w[f"gt_{a}_gyro"] for a in list(ACTIVITIES) + [NULL]], axis=1)
    best = counts.argmax(axis=1)
    purity = counts.max(axis=1) / counts.sum(axis=1)
    keep = (best < len(ACTIVITIES)) & (purity >= MIN_PURITY)       # ni null, ni changement d'activité
    w = w[keep].assign(label=np.array(ACTIVITIES)[best[keep]], purity=purity[keep])
    meta = KEY + ["label", "purity", "n_acc", "n_gyro"]
    return w[meta + list(feature_names())].sort_values(KEY).reset_index(drop=True)


def build_cache(raw_dir: Path | None = None, cache: Path = CACHE, window_s: float = WINDOW_S) -> pd.DataFrame:
    """Lit les quatre fichiers bruts, écrit le cache des fenêtres. Reprend là où il s'est arrêté."""
    raw_dir = Path(raw_dir) if raw_dir else DATA_DIR
    PARTIALS.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for (device_type, sensor), name in FILES.items():
        part = PARTIALS / f"{device_type}_{sensor}.pkl"
        if not part.exists():
            found = sorted(raw_dir.rglob(name))
            if not found:
                raise FileNotFoundError(
                    f"{name} introuvable sous {raw_dir}. Télécharger « Activity recognition exp.zip » sur "
                    f"https://archive.ics.uci.edu/dataset/344 et le décompresser dans {DATA_DIR}.")
            start = time.perf_counter()
            summarize_file(found[0], device_type, window_s).to_pickle(part)
            print(f"{name} : résumé en {time.perf_counter() - start:.0f} s", file=sys.stderr)
        summaries[(device_type, sensor)] = pd.read_pickle(part)
    windows = pd.concat([assemble(summaries[(t, "acc")], summaries[(t, "gyro")]) for t in ("phone", "watch")],
                        ignore_index=True)
    windows.to_csv(cache, index=False, float_format="%.6g")
    return windows


# --------------------------------------------------------------------------- le scénario

def load_windows() -> pd.DataFrame:
    if not CACHE.exists():
        build_cache()
    return pd.read_csv(CACHE)


def make_scenario(windows: pd.DataFrame, source: str = "watch", target: str = "phone",
                  name: str = "S6", family: str = "S6") -> Scenario:
    """Source = un type d'appareil (« phone » ou « watch »), cible = l'autre."""
    if source == target or {source, target} - {"phone", "watch"}:
        raise ValueError("source et cible doivent être « phone » et « watch », dans un ordre ou l'autre")
    names = list(feature_names())
    src, tgt = windows[windows["device_type"] == source], windows[windows["device_type"] == target]
    code = {a: k for k, a in enumerate(ACTIVITIES)}
    users = sorted(windows["User"].unique())
    scn = Scenario(name=name, family=family,
                   Xs=src[names].to_numpy(np.float64), ys=src["label"].map(code).to_numpy(np.int64),
                   Xt=tgt[names].to_numpy(np.float64), yt=tgt["label"].map(code).to_numpy(np.int64),
                   class_names=ACTIVITIES, feature_names=tuple(names), metric="error",
                   groups_t=tgt["User"].map({u: k for k, u in enumerate(users)}).to_numpy(np.int64),
                   params={"source": source, "target": target, "window_s": WINDOW_S,
                           "source_devices": sorted(src["Device"].unique()),
                           "target_devices": sorted(tgt["Device"].unique()), "users": users})
    check_scenario(scn)
    return scn


def load_s6() -> list[Scenario]:
    return [make_scenario(load_windows())]


if __name__ == "__main__":
    w = build_cache(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(f"{len(w)} fenêtres écrites dans {CACHE}")
