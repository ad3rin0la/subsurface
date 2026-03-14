"""
components.py
=============
Defines component data for the N2Bio framework: critical properties,
acentric factors, molecular weights, and volume shift parameters for
use with Peng-Robinson EOS.

Components modelled:
  N2, CO2, H2, NH3, H2O, HC_ali (pseudo), HC_arom (pseudo), NaCl, Temperature
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class Component:
    """
    Thermodynamic component descriptor.

    Attributes
    ----------
    name : str
        Component identifier string.
    Tc : float
        Critical temperature [K].
    Pc : float
        Critical pressure [Pa].
    omega : float
        Acentric factor [-].
    MW : float
        Molecular weight [g/mol].
    volume_shift : float
        Peneloux volume shift parameter [m³/mol].  Negative values correct
        PR EOS liquid volumes upward (gas volumes are less sensitive).
    formula : str, optional
        Chemical formula for display purposes.
    """
    name: str
    Tc: float           # K
    Pc: float           # Pa
    omega: float        # [-]
    MW: float           # g/mol
    volume_shift: float = 0.0   # m³/mol  (Peneloux correction)
    formula: str = ""


# ---------------------------------------------------------------------------
# Component library – critical properties from NIST / standard references
# ---------------------------------------------------------------------------

COMPONENTS: Dict[str, Component] = {
    # ---------- light gases ----------
    "N2": Component(
        name="N2",
        Tc=126.19,          # K   (NIST WebBook)
        Pc=33.96e5,         # Pa  (33.96 bar)
        omega=0.037,
        MW=28.014,
        volume_shift=-1.5e-6,    # m³/mol  ~−1.5 cm³/mol (Peneloux gas correction)
        formula="N₂",
    ),
    "CO2": Component(
        name="CO2",
        Tc=304.13,          # K
        Pc=73.77e5,         # Pa  (73.77 bar)
        omega=0.224,
        MW=44.010,
        volume_shift=-2.5e-6,    # m³/mol  ~−2.5 cm³/mol
        formula="CO₂",
    ),
    "H2": Component(
        name="H2",
        Tc=33.14,           # K
        Pc=12.96e5,         # Pa  (12.96 bar)
        omega=-0.216,
        MW=2.016,
        volume_shift=-1.0e-6,    # m³/mol  ~−1.0 cm³/mol
        formula="H₂",
    ),
    # ---------- nitrogen species ----------
    "NH3": Component(
        name="NH3",
        Tc=405.56,          # K
        Pc=113.53e5,        # Pa
        omega=0.252,
        MW=17.031,
        volume_shift=0.0,
        formula="NH₃",
    ),
    # ---------- water ----------
    "H2O": Component(
        name="H2O",
        Tc=647.096,         # K
        Pc=220.64e5,        # Pa  (220.64 bar)
        omega=0.345,
        MW=18.015,
        volume_shift=-1.0e-6,    # m³/mol  ~−1 cm³/mol
        formula="H₂O",
    ),
    # ---------- hydrocarbon pseudo-components ----------
    # HC_ali: aliphatic pool, fermentable (acetate equivalent, C2H4O2)
    "HC_ali": Component(
        name="HC_ali",
        Tc=590.0,           # K  (representative C4 alkane average)
        Pc=38.0e5,          # Pa
        omega=0.200,
        MW=60.052,          # g/mol  (acetic acid as acetate equiv.)
        volume_shift=0.0,
        formula="HC_ali",
    ),
    # HC_arom: aromatic pool, inhibitory (toluene representative)
    "HC_arom": Component(
        name="HC_arom",
        Tc=591.75,          # K  (toluene)
        Pc=41.06e5,         # Pa
        omega=0.264,
        MW=92.141,          # g/mol  (toluene)
        volume_shift=0.0,
        formula="HC_arom",
    ),
    # ---------- salinity pseudo-component ----------
    "NaCl": Component(
        name="NaCl",
        Tc=3400.0,          # K  (fictitious, purely for bookkeeping)
        Pc=1.0e5,
        omega=0.0,
        MW=58.443,
        volume_shift=0.0,
        formula="NaCl",
    ),
}


class ComponentSystem:
    """
    Tracks the full set of N2Bio components and their phase assignments.

    The 9-component system is:
        0  CO₂
        1  N₂
        2  H₂
        3  NH₄⁺ / NH₃  (total ammonium-N, speciated separately)
        4  HC_ali       (aliphatic HC pool)
        5  HC_arom      (aromatic HC pool)
        6  H₂O
        7  NaCl
        8  Temperature  (carried as a pseudo-component for transport bookkeeping)

    Phase assignments:
        - Gas  phase: CO₂, N₂, H₂  (can also dissolve in brine)
        - Aqueous phase: all components
        - Solid / immobile: HC_ali, HC_arom (treated as dissolved organic carbon
          pools in the aqueous phase for kinetics, but tracked separately)
    """

    COMPONENT_NAMES: List[str] = [
        "CO2", "N2", "H2", "NH3", "HC_ali", "HC_arom", "H2O", "NaCl", "Temperature"
    ]

    GAS_COMPONENTS: List[str] = ["CO2", "N2", "H2"]
    AQUEOUS_COMPONENTS: List[str] = ["CO2", "N2", "H2", "NH3", "HC_ali", "HC_arom",
                                      "H2O", "NaCl"]
    BIOLOGICAL_COMPONENTS: List[str] = ["N2", "NH3", "HC_ali", "HC_arom", "H2"]

    def __init__(self):
        self._components: Dict[str, Component] = {
            k: COMPONENTS[k] for k in self.COMPONENT_NAMES if k in COMPONENTS
        }
        # Temperature is a pseudo-component – use H2O entry as placeholder
        self._components["Temperature"] = Component(
            name="Temperature",
            Tc=0.0, Pc=0.0, omega=0.0, MW=0.0,
            formula="T"
        )

    def __getitem__(self, name: str) -> Component:
        return self._components[name]

    def __contains__(self, name: str) -> bool:
        return name in self._components

    @property
    def gas_components(self) -> List[Component]:
        """Returns Component objects for the gas-phase species."""
        return [COMPONENTS[n] for n in self.GAS_COMPONENTS]

    @property
    def aqueous_components(self) -> List[Component]:
        """Returns Component objects for dissolved species."""
        return [COMPONENTS[n] for n in self.AQUEOUS_COMPONENTS]

    def index(self, name: str) -> int:
        """Returns the index of a component in COMPONENT_NAMES list."""
        return self.COMPONENT_NAMES.index(name)

    def mw(self, name: str) -> float:
        """Molecular weight [g/mol] for the named component."""
        return self._components[name].MW

    def __repr__(self) -> str:
        return (f"ComponentSystem(n={len(self.COMPONENT_NAMES)}, "
                f"components={self.COMPONENT_NAMES})")
