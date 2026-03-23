"""
simulation/state.py
====================
Biological and geochemical state variables for the N2Bio simulation.

The BiologicalState dataclass holds the complete state of a single grid
cell (or a well-mixed batch reactor) at a given instant in time.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class BiologicalState:
    """
    Complete state vector for a single N2Bio grid cell.

    All concentrations are in mol/L.  Gas-phase partial pressures are in bar.
    Biomass is in g VSS/L.  Temperature in K.

    Attributes
    ----------
    N2_aq : float
        Dissolved N₂ concentration [mol/L].
    CO2_aq : float
        Dissolved CO₂ concentration [mol/L].
    H2_aq : float
        Dissolved H₂ concentration [mol/L].
    NH4_aq : float
        Ammonium concentration (total fixed-N as NH₄⁺+NH₃) [mol/L].
    HC_ali : float
        Aliphatic hydrocarbon pool concentration [mol/L].
    HC_arom : float
        Aromatic hydrocarbon pool concentration [mol/L].
    NaCl_aq : float
        Dissolved NaCl (molality approximation) [mol/L].
    X : float
        Biomass concentration [g VSS/L].
    P_N2 : float
        N₂ partial pressure in gas phase [bar].
    P_CO2 : float
        CO₂ partial pressure in gas phase [bar].
    P_H2 : float
        H₂ partial pressure in gas phase [bar].
    P_total : float
        Total reservoir pressure [bar].
    T : float
        Temperature [K].
    pH : float
        Aqueous phase pH.
    cum_H2 : float
        Cumulative H₂ produced [mol/L].  Tracked as an integrated ODE state.
    cum_NH4 : float
        Cumulative NH₄⁺ produced [mol/L].
    time : float
        Elapsed simulation time [h].
    """

    # Dissolved concentrations (mol/L)
    N2_aq:    float = 0.0
    CO2_aq:   float = 0.0
    H2_aq:    float = 0.0
    NH4_aq:   float = 0.0
    HC_ali:   float = 0.0
    HC_arom:  float = 0.0
    NaCl_aq:  float = 0.0

    # Biomass (g VSS/L)
    X: float = 0.0

    # Gas-phase partial pressures (bar)
    P_N2:    float = 0.0
    P_CO2:   float = 0.0
    P_H2:    float = 0.0
    P_total: float = 150.0

    # Conditions
    T:   float = 358.15   # K  (85°C default)
    pH:  float = 7.0

    # Cumulative production / tracking
    cum_H2:  float = 0.0
    cum_NH4: float = 0.0
    time:    float = 0.0   # hours

    # ---- Sulfur assimilation intermediates (mol/L) ----
    # Coupling 1: H₂S as direct OASTL substrate (bypasses APR/SiR chain)
    # Coupling 2: Acetate as acetyl-CoA donor to SAT (closes fermentation loop)
    # Coupling 3: Cysteine as structural requirement for nitrogenase Fe-S clusters
    H2S_aq:     float = 0.0     # total dissolved sulfide (H₂S + HS⁻) [mol/L]
    Acetate_aq: float = 0.0     # free acetate [mol/L]
    OAS_aq:     float = 0.0     # O-acetylserine (SAT product, OASTL substrate) [mol/L]
    Cys_aq:     float = 1.0e-5  # L-cysteine [mol/L] — small non-zero default avoids cold-start Cys starvation

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def f_arom(self) -> float:
        """
        Aromatic fraction of total HC pool [-].

        f_arom = HC_arom / (HC_ali + HC_arom)
        Returns 0 when both pools are zero.
        """
        total = self.HC_ali + self.HC_arom
        if total < 1.0e-12:
            return 0.0
        return self.HC_arom / total

    @property
    def f_ali(self) -> float:
        """Aliphatic fraction of total HC pool [-]."""
        return 1.0 - self.f_arom

    @property
    def HC_total(self) -> float:
        """Total HC pool concentration [mol/L]."""
        return self.HC_ali + self.HC_arom

    @property
    def P_H2_atm(self) -> float:
        """H₂ partial pressure in atmospheres [atm]."""
        return self.P_H2 / 1.01325   # bar → atm

    @property
    def P_N2_atm(self) -> float:
        """N₂ partial pressure in atmospheres [atm]."""
        return self.P_N2 / 1.01325

    @property
    def T_C(self) -> float:
        """Temperature in Celsius [°C]."""
        return self.T - 273.15

    @property
    def P_gas_total(self) -> float:
        """Total gas partial pressure (N₂ + CO₂ + H₂) [bar]."""
        return self.P_N2 + self.P_CO2 + self.P_H2

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def as_array(self) -> np.ndarray:
        """
        Return the ODE state vector as a numpy array.

        Order (11 elements):
            [0]  N2_aq     [mol/L]
            [1]  H2_aq     [mol/L]
            [2]  NH4_aq    [mol/L]
            [3]  HC_ali    [mol/L]
            [4]  HC_arom   [mol/L]
            [5]  X         [g VSS/L]
            [6]  cum_H2    [mol/L]
            [7]  H2S_aq    [mol/L]
            [8]  Acetate_aq[mol/L]
            [9]  OAS_aq    [mol/L]
            [10] Cys_aq    [mol/L]

        Returns
        -------
        np.ndarray  State vector of length 11.
        """
        return np.array([
            self.N2_aq,
            self.H2_aq,
            self.NH4_aq,
            self.HC_ali,
            self.HC_arom,
            self.X,
            self.cum_H2,
            self.H2S_aq,
            self.Acetate_aq,
            self.OAS_aq,
            self.Cys_aq,
        ])

    @classmethod
    def from_array(cls, y: np.ndarray, template: 'BiologicalState') -> 'BiologicalState':
        """
        Reconstruct a BiologicalState from an ODE solution array.

        Accepts both the legacy 7-element vector (indices 0–6) and the
        extended 11-element vector (indices 0–10).  Non-ODE fields
        (CO2_aq, NaCl_aq, pressures, T, pH) are copied from template.

        Parameters
        ----------
        y : np.ndarray
            State vector of length 7 or 11.
        template : BiologicalState
            Reference state for non-integrated variables.

        Returns
        -------
        BiologicalState
        """
        n = len(y)
        return cls(
            N2_aq      = max(float(y[0]), 0.0),
            CO2_aq     = template.CO2_aq,
            H2_aq      = max(float(y[1]), 0.0),
            NH4_aq     = max(float(y[2]), 0.0),
            HC_ali     = max(float(y[3]), 0.0),
            HC_arom    = max(float(y[4]), 0.0),
            NaCl_aq    = template.NaCl_aq,
            X          = max(float(y[5]), 0.0),
            P_N2       = template.P_N2,
            P_CO2      = template.P_CO2,
            P_H2       = max(float(y[1]) * 10.0, 0.0),
            P_total    = template.P_total,
            T          = template.T,
            pH         = template.pH,
            cum_H2     = float(y[6]),
            H2S_aq     = max(float(y[7]),  0.0) if n > 7  else template.H2S_aq,
            Acetate_aq = max(float(y[8]),  0.0) if n > 8  else template.Acetate_aq,
            OAS_aq     = max(float(y[9]),  0.0) if n > 9  else template.OAS_aq,
            Cys_aq     = max(float(y[10]), 0.0) if n > 10 else template.Cys_aq,
        )

    def copy(self) -> 'BiologicalState':
        """Return a shallow copy of this state."""
        import copy
        return copy.copy(self)

    def __repr__(self) -> str:
        return (
            f"BiologicalState(\n"
            f"  T={self.T_C:.1f}°C, P={self.P_total:.0f} bar, pH={self.pH:.2f}\n"
            f"  N2_aq={self.N2_aq*1e3:.4f} mmol/L, H2_aq={self.H2_aq*1e3:.4f} mmol/L\n"
            f"  NH4_aq={self.NH4_aq*1e3:.4f} mmol/L\n"
            f"  HC_ali={self.HC_ali*1e3:.4f} mmol/L, HC_arom={self.HC_arom*1e3:.4f} mmol/L\n"
            f"  f_arom={self.f_arom:.3f}, X={self.X*1e3:.2f} mg VSS/L\n"
            f"  H2S_aq={self.H2S_aq*1e6:.2f} μmol/L, Cys_aq={self.Cys_aq*1e6:.2f} μmol/L\n"
            f"  Acetate_aq={self.Acetate_aq*1e3:.4f} mmol/L, OAS_aq={self.OAS_aq*1e6:.2f} μmol/L\n"
            f"  cum_H2={self.cum_H2*1e3:.4f} mmol/L, time={self.time:.1f} h"
            f"\n)"
        )
