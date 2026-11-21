"""Scénario S5 : C-MAPSS (NASA), turboréacteurs simulés suivis jusqu'à la panne.

Une ligne = un cycle de vol d'un moteur, décrit par 3 réglages de vol et 21 capteurs.
Tâche : prédire une panne dans les 30 cycles (classe « panne » : durée de vie restante <= 30).

    source : FD001, 100 moteurs, un seul régime de vol (niveau de la mer), 20 631 cycles ;
    cible  : FD002, 260 moteurs, six régimes de vol, 53 759 cycles.

Le mode de panne est le même (usure du compresseur haute pression) ; ce qui change, ce sont
les conditions de vol, dont dépendent toutes les mesures. Un seul des six régimes de FD002
(15 % des cycles) est celui de la source. La part de cycles « panne » reste de 15 %.

Les cycles d'un même moteur se ressemblent : groups_t donne le moteur de chaque cycle
cible, pour que chaque moteur aille entier dans le test ou dans la réserve.

Données : fichiers train_FD001 et train_FD002 de l'archive CMAPSSData (NASA Prognostics
Data Repository, « Turbofan Engine Degradation Simulation »), recompressés dans data/cmapss/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_forest.contracts import Scenario, check_scenario

DATA_DIR = Path(__file__).parents[2] / "data" / "cmapss"
HORIZON = 30                                      # panne dans les 30 cycles
FEATURES = ("setting1", "setting2", "setting3") + tuple(f"sensor{i}" for i in range(1, 22))


def read(subset: str) -> pd.DataFrame:
    """Un fichier C-MAPSS, avec la durée de vie restante (rul) de chaque cycle."""
    df = pd.read_csv(DATA_DIR / f"train_{subset}.txt.gz", sep=r"\s+", header=None,
                     names=("unit", "cycle") + FEATURES)
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[list(FEATURES)].to_numpy(np.float64), (df["rul"] <= HORIZON).to_numpy(np.int64)


def load_s5() -> list[Scenario]:
    source, target = read("FD001"), read("FD002")
    (Xs, ys), (Xt, yt) = xy(source), xy(target)
    scn = Scenario(name="S5", family="S5", Xs=Xs, ys=ys, Xt=Xt, yt=yt,
                   class_names=("sain", "panne"), feature_names=FEATURES, metric="ber",
                   critical_class=1, groups_t=target["unit"].to_numpy(),
                   params={"horizon": HORIZON})
    check_scenario(scn)
    return [scn]
