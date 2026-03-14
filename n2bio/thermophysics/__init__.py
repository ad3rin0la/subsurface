"""
n2bio.thermophysics
===================
Thermophysical property sub-package for the N2Bio framework.

Exports
-------
PengRobinsonEOS      : Pure-component PR EOS
MixturePREOS         : Mixture PR EOS (van der Waals mixing rules)
FrictionTheoryViscosity : f-theory viscosity for N₂, CO₂, H₂
GasMixtureViscosity  : Wilke mixing rule for gas mixture viscosity
N2Solubility         : N₂ Henry constant / dissolved concentration
CO2Solubility        : CO₂ Henry constant / dissolved concentration
H2Solubility         : H₂ Henry constant / dissolved concentration
dissolved_N2         : Convenience function
dissolved_CO2        : Convenience function
dissolved_H2         : Convenience function
"""

from .eos import PengRobinsonEOS, MixturePREOS
from .viscosity import FrictionTheoryViscosity, GasMixtureViscosity
from .solubility import (
    N2Solubility,
    CO2Solubility,
    H2Solubility,
    dissolved_N2,
    dissolved_CO2,
    dissolved_H2,
)

__all__ = [
    "PengRobinsonEOS",
    "MixturePREOS",
    "FrictionTheoryViscosity",
    "GasMixtureViscosity",
    "N2Solubility",
    "CO2Solubility",
    "H2Solubility",
    "dissolved_N2",
    "dissolved_CO2",
    "dissolved_H2",
]
