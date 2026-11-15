"""Annuaire des méthodes : le moteur ne les connaît que par leur nom officiel.

Ajouter une méthode = l'ajouter à la liste ci-dessous.
"""
from transfer_forest.contracts import Method
from transfer_forest.methods.references import Oracle, Src, Tgt, Union, UnionP
from transfer_forest.methods.segev import MIX, SER, STRUT, Relab, SERStar, STRUTStar

METHODS = {m.name: m for m in (Src, Tgt, Union, UnionP, Oracle,
                               Relab, SER, STRUT, MIX, SERStar, STRUTStar)}


def make_method(name: str) -> Method:
    """Une instance neuve à chaque appel : une méthode ne garde rien d'une expérience à l'autre."""
    return METHODS[name]()
