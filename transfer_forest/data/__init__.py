"""Annuaire des scénarios : le moteur ne les connaît que par leur nom.

Ajouter un scénario = ajouter son chargeur au dictionnaire ci-dessous.
"""
from transfer_forest.contracts import Scenario
from transfer_forest.data.s1_letter import load_s1
from transfer_forest.data.s6_hhar import load_s6

LOADERS = {"S1": load_s1, "S6": load_s6}


def load_scenarios(name: str) -> list[Scenario]:
    """Une liste de scénarios : certaines familles (S0, S4) en produisent plusieurs."""
    return LOADERS[name]()
