"""
n2bio.kinetics
==============
Biological kinetics sub-package for the N2Bio framework.

Exports
-------
KineticParameters    : Dataclass holding all kinetic rate constants.
ReservoirConditions  : Dataclass for physical/chemical reservoir state.
MonodKinetics        : Triple co-limitation Monod growth model.
NitrogenaseKinetics  : Nitrogenase rate with H₂ and aromatic inhibition.
HCDegradationModel   : Two-pool HC degradation and dark fermentation H₂.
"""

from .parameters import KineticParameters, ReservoirConditions
from .monod import MonodKinetics
from .nitrogenase import NitrogenaseKinetics
from .hc_degradation import HCDegradationModel

__all__ = [
    "KineticParameters",
    "ReservoirConditions",
    "MonodKinetics",
    "NitrogenaseKinetics",
    "HCDegradationModel",
]
