"""Scénario S1 : Letter (UCI, id 59), coupé comme dans Segev et al. (2017).

20 000 lettres capitales décrites par 16 variables entières, 26 classes.
La coupe se fait sur x2bar (variance horizontale des pixels allumés) :
    source : x2bar <= médiane, lettres étroites  (10 822 exemples) ;
    cible  : x2bar >  médiane, lettres étalées   ( 9 178 exemples).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_forest.contracts import Scenario, check_scenario

DATA_FILE = Path(__file__).parents[2] / "data" / "letter_recognition.csv"
SPLIT_FEATURE = "x2bar"


def load_s1() -> list[Scenario]:
    df = pd.read_csv(DATA_FILE)
    class_names, y = np.unique(df.pop("lettr").to_numpy(str), return_inverse=True)
    X = df.to_numpy(np.float64)
    j = df.columns.get_loc(SPLIT_FEATURE)
    src = X[:, j] <= np.median(X[:, j])

    scn = Scenario(name="S1", family="S1",
                   Xs=X[src], ys=y[src].astype(np.int64),
                   Xt=X[~src], yt=y[~src].astype(np.int64),
                   class_names=tuple(class_names), feature_names=tuple(df.columns),
                   metric="error", params={"split_feature_idx": j})
    check_scenario(scn)
    return [scn]
