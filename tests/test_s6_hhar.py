"""Tests du scénario S6 (HHAR) sur de petits fichiers synthétiques : pas besoin des 3 Go."""
import numpy as np
import pandas as pd
import pytest

from transfer_forest.data import s6_hhar as s6
from transfer_forest.splits import budgets, split_target

WIN = int(s6.WINDOW_S * 1e9)
BASE = 500 * WIN                      # début d'une fenêtre : les fenêtres sont alignées sur WIN


def raw_file(path, sensor_offset, rng):
    """Un appareil, un utilisateur, 4 fenêtres de 2 s à 50 Hz (100 échantillons).

    fenêtre 0 : walk (valide) ; 1 : null ; 2 : moitié walk, moitié sit (impure) ;
    3 : walk mais 10 échantillons seulement (trop courte).
    """
    rows = []
    for w, (labels, n) in enumerate([(["walk"] * 100, 100), (["null"] * 100, 100),
                                     (["walk"] * 50 + ["sit"] * 50, 100), (["walk"] * 10, 10)]):
        for i in range(n):
            rows.append((BASE + w * WIN + i * int(0.02e9), rng.normal() + sensor_offset,
                         rng.normal(), rng.normal(), "a", "m1", "m1_1", labels[i]))
    df = pd.DataFrame(rows, columns=["Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"])
    df = df.sample(frac=1, random_state=0)                       # les vrais fichiers ne sont pas triés
    df.insert(0, "Index", range(len(df)))
    df.insert(1, "Arrival_Time", df["Creation_Time"] // 1_000_000)
    df.to_csv(path, index=False)
    return df


@pytest.fixture
def two_sensors(tmp_path):
    rng = np.random.default_rng(0)
    a, g = tmp_path / "acc.csv", tmp_path / "gyro.csv"
    return raw_file(a, 0.0, rng), raw_file(g, 5.0, rng), a, g


def test_null_is_a_label_not_a_missing_value(two_sensors):
    """pandas lit « null » comme valeur manquante : le résumé doit pourtant la compter."""
    _, _, a, _ = two_sensors
    s = s6.summarize_file(a, "phone")
    assert s["gt_null"].sum() == 100
    assert s["n"].sum() == 310


def test_chunks_do_not_change_the_result(two_sensors):
    """Une fenêtre coupée entre deux morceaux du fichier doit être recollée."""
    _, _, a, _ = two_sensors
    whole = s6.summarize_file(a, "phone", chunksize=10_000)
    cut = s6.summarize_file(a, "phone", chunksize=7)
    pd.testing.assert_frame_equal(whole.sort_values(s6.KEY).reset_index(drop=True),
                                  cut.sort_values(s6.KEY).reset_index(drop=True))


def test_features_match_numpy(two_sensors):
    raw_acc, _, a, _ = two_sensors
    f = s6.finalize(s6.summarize_file(a, "phone", chunksize=7), "acc").sort_values("win")
    first = raw_acc[raw_acc["Creation_Time"] // WIN == 500]
    row = f.iloc[0]
    assert row["acc_x_mean"] == pytest.approx(first["x"].mean())
    assert row["acc_x_std"] == pytest.approx(first["x"].std(ddof=0))
    assert row["acc_y_min"] == pytest.approx(first["y"].min())
    assert row["acc_z_max"] == pytest.approx(first["z"].max())
    m = np.sqrt(first["x"] ** 2 + first["y"] ** 2 + first["z"] ** 2)
    assert row["acc_m_mean"] == pytest.approx(m.mean())


def test_only_the_clean_window_is_kept(two_sensors):
    """Ni null, ni changement d'activité, ni fenêtre trop courte : reste la fenêtre 0."""
    _, _, a, g = two_sensors
    w = s6.assemble(s6.summarize_file(a, "phone"), s6.summarize_file(g, "phone"))
    assert len(w) == 1
    assert w["label"].iloc[0] == "walk" and w["win"].iloc[0] == 500
    assert w["purity"].iloc[0] == 1.0
    assert list(w.columns[-32:]) == list(s6.feature_names())
    assert w["gyro_x_mean"].iloc[0] > w["acc_x_mean"].iloc[0] + 3      # les deux capteurs ne sont pas mélangés


def test_feature_names():
    names = s6.feature_names()
    assert len(names) == len(set(names)) == 32
    assert names[0] == "acc_x_mean" and names[-1] == "gyro_m_max"


def fake_windows(n_phone, n_watch, rng):
    rows = []
    for device_type, n in (("phone", n_phone), ("watch", n_watch)):
        for k in range(n):
            rows.append({"device_type": device_type, "Model": "m", "Device": f"{device_type}_{k % 2}",
                         "User": "abcde"[k % 5], "win": k, "label": s6.ACTIVITIES[k % 6], "purity": 1.0,
                         "n_acc": 100, "n_gyro": 100, **dict(zip(s6.feature_names(), rng.normal(size=32)))})
    return pd.DataFrame(rows)


def test_scenario_is_valid_and_directed():
    w = fake_windows(300, 60, np.random.default_rng(0))
    scn = s6.make_scenario(w)                                  # montres -> téléphones par défaut
    assert (scn.n_S, len(scn.yt), scn.K, scn.d) == (60, 300, 6, 32)
    assert scn.class_names == s6.ACTIVITIES
    assert scn.groups_t.shape == scn.yt.shape and set(scn.groups_t) == set(range(5))
    assert scn.params["source"] == "watch" and scn.params["target_devices"] == ["phone_0", "phone_1"]
    back = s6.make_scenario(w, source="phone", target="watch")
    assert back.n_S == 300 and len(back.yt) == 60
    with pytest.raises(ValueError):
        s6.make_scenario(w, source="phone", target="phone")


def test_default_direction_fits_the_protocol():
    """Montres -> téléphones : le plus gros budget tient dans la réserve ; le sens inverse non."""
    w = fake_windows(3500, 750, np.random.default_rng(0))      # mêmes proportions que HHAR
    scn = s6.make_scenario(w)
    reserve, _ = split_target(scn)
    assert max(budgets(scn, len(reserve)).values()) <= len(reserve)
    back = s6.make_scenario(w, source="phone", target="watch")
    reserve, _ = split_target(back)
    with pytest.raises(ValueError):
        budgets(back, len(reserve))
