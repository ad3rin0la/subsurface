"""
kinetics/nitrogenase.py
=======================
Nitrogenase kinetics for biological N₂ fixation in the N2Bio framework.

The nitrogenase reaction (conventional stoichiometry):
    N₂ + 8 H⁺ + 8 e⁻ + 16 ATP  →  2 NH₃ + H₂ + 16 ADP + 16 Pᵢ

Key features modelled:
  - Monod-type saturation with dissolved N₂
  - H₂ product inhibition (competitive inhibition of nitrogenase activity)
  - Aromatic HC inhibition (HC_arom acts as a toxicant)
  - Cysteine co-limitation: Fe-S cluster assembly requires Cys; in sulfide-rich
    reservoirs H₂S → Cys via OASTL amplifies nitrogenase activity rather than
    acting as a ceiling.
  - Stoichiometric coupling: 1 mol H₂ + 2 mol NH₃ per mol N₂ fixed

References
----------
Burgess, B.K.; Lowe, D.J. (1996). Chem. Rev. 96, 2983.
Dixon, R.; Kahn, D. (2004). Nature Rev. Microbiol. 2, 621.
Zheng, L.; Dean, D.R. (1994). J. Biol. Chem. 269, 18723. (nifS/Fe-S assembly)
"""

import numpy as np
from .parameters import KineticParameters


class NitrogenaseKinetics:
    """
    Nitrogenase enzyme kinetics with dual inhibition.

    The rate expression accounts for:
    1. Dissolved N₂ availability (Michaelis-Menten saturation).
    2. H₂ product inhibition on nitrogenase (non-competitive).
    3. Aromatic HC inhibition (competitive / non-competitive mixed).

    Parameters
    ----------
    params : KineticParameters
        Kinetic parameter dataclass.
    """

    def __init__(self, params: KineticParameters):
        self.p = params

    # ------------------------------------------------------------------
    # Core nitrogenase rate
    # ------------------------------------------------------------------

    def rate(self, N2_aq: float, P_H2_atm: float, HC_arom_aq: float,
             Cys_aq: float = None) -> float:
        """
        Specific nitrogenase rate [mol N₂ fixed / g VSS / h].

        r_nit = r_nit_max
                * [N₂]_aq  / (K_N₂  + [N₂]_aq)           ← N₂ saturation
                * K_I_H₂   / (K_I_H₂  + P_H₂)             ← H₂ inhibition
                * K_I_arom / (K_I_arom + [HC_arom]_aq)     ← aromatic inhibition
                * [Cys]    / (K_Cys_nit + [Cys])           ← Fe-S cluster supply

        The Cys co-limitation term captures the requirement for continuous
        cysteine supply to the ISC machinery for Fe-S cluster assembly in
        both the Fe-protein and MoFe-protein of nitrogenase.  When H₂S is
        available from geological sources or SRB activity, the OASTL pathway
        maintains [Cys] ≫ K_Cys_nit and the term approaches 1 (no limitation).
        In sulfide-poor formations it is the operative ceiling on r_nit.

        Parameters
        ----------
        N2_aq : float
            Dissolved N₂ concentration [mol/L].
        P_H2_atm : float
            H₂ partial pressure [atm].
        HC_arom_aq : float
            Aqueous aromatic HC concentration [mol/L].
        Cys_aq : float, optional
            Aqueous cysteine concentration [mol/L].
            If None, cysteine co-limitation is not applied (backward-compatible).

        Returns
        -------
        float
            Specific nitrogenase rate [mol N₂ / g VSS / h].  Always ≥ 0.
        """
        N2_aq      = max(N2_aq,      0.0)
        P_H2_atm   = max(P_H2_atm,   0.0)
        HC_arom_aq = max(HC_arom_aq, 0.0)

        f_N2         = N2_aq / (self.p.K_N2 + N2_aq)
        f_H2_inhib   = self.p.K_I_H2   / (self.p.K_I_H2   + P_H2_atm)
        f_arom_inhib = self.p.K_I_arom / (self.p.K_I_arom + HC_arom_aq)

        f_Cys = 1.0
        if Cys_aq is not None:
            Cys_aq = max(Cys_aq, 0.0)
            K_Cys  = getattr(self.p, "K_Cys_nit", 2.0e-5)
            f_Cys  = Cys_aq / (K_Cys + Cys_aq)

        return self.p.r_nit_max * f_N2 * f_H2_inhib * f_arom_inhib * f_Cys

    # ------------------------------------------------------------------
    # Derived rates
    # ------------------------------------------------------------------

    def H2_production_rate(self, N2_aq: float, P_H2_atm: float,
                           HC_arom_aq: float, X_biomass: float,
                           Cys_aq: float = None) -> float:
        """
        Volumetric H₂ production rate from nitrogenase [mol H₂ / L / h].

        From the stoichiometry: 1 mol H₂ produced per mol N₂ fixed.

        r_H₂_nit = r_nit * X_biomass * 1  (mol H₂/mol N₂ = 1)

        Parameters
        ----------
        N2_aq : float         Dissolved N₂ [mol/L].
        P_H2_atm : float      H₂ partial pressure [atm].
        HC_arom_aq : float    Aromatic HC [mol/L].
        X_biomass : float     Biomass concentration [g VSS/L].

        Returns
        -------
        float  Volumetric H₂ production rate [mol H₂/L/h].
        """
        X_biomass = max(X_biomass, 0.0)
        r_nit = self.rate(N2_aq, P_H2_atm, HC_arom_aq, Cys_aq)
        return r_nit * X_biomass   # 1:1 H₂:N₂ stoichiometry

    def NH4_production_rate(self, N2_aq: float, P_H2_atm: float,
                            HC_arom_aq: float, X_biomass: float,
                            Cys_aq: float = None) -> float:
        """
        Volumetric NH₄⁺ production rate [mol NH₄⁺ / L / h].

        From the stoichiometry: 2 mol NH₃ per mol N₂ fixed.
        At typical reservoir pH (6-8) nearly all NH₃ is protonated to NH₄⁺.

        r_NH₄ = r_nit * X_biomass * 2.0

        Parameters
        ----------
        N2_aq : float         Dissolved N₂ [mol/L].
        P_H2_atm : float      H₂ partial pressure [atm].
        HC_arom_aq : float    Aromatic HC [mol/L].
        X_biomass : float     Biomass [g VSS/L].

        Returns
        -------
        float  Volumetric NH₄⁺ production rate [mol/L/h].
        """
        X_biomass = max(X_biomass, 0.0)
        r_nit = self.rate(N2_aq, P_H2_atm, HC_arom_aq, Cys_aq)
        return 2.0 * r_nit * X_biomass

    def N2_consumption_rate(self, N2_aq: float, P_H2_atm: float,
                            HC_arom_aq: float, X_biomass: float,
                            Cys_aq: float = None) -> float:
        """
        Volumetric N₂ consumption rate [mol N₂ / L / h].

        r_N₂_consumed = r_nit * X_biomass

        Parameters
        ----------
        N2_aq, P_H2_atm, HC_arom_aq : float
            See rate().
        X_biomass : float  Biomass [g VSS/L].

        Returns
        -------
        float  N₂ consumption rate [mol/L/h].
        """
        X_biomass = max(X_biomass, 0.0)
        return self.rate(N2_aq, P_H2_atm, HC_arom_aq, Cys_aq) * X_biomass

    # ------------------------------------------------------------------
    # Inhibition factor diagnostics
    # ------------------------------------------------------------------

    def H2_inhibition_factor(self, P_H2_atm: float) -> float:
        """
        H₂ inhibition factor for nitrogenase [-].

        f_H₂ = K_I_H₂ / (K_I_H₂ + P_H₂)

        Returns 1 when P_H₂ = 0, falls to 0.5 at P_H₂ = K_I_H₂.

        Parameters
        ----------
        P_H2_atm : float  H₂ partial pressure [atm].

        Returns
        -------
        float  Factor in (0, 1].
        """
        P_H2_atm = max(P_H2_atm, 0.0)
        return self.p.K_I_H2 / (self.p.K_I_H2 + P_H2_atm)

    def aromatic_inhibition_factor(self, HC_arom_aq: float) -> float:
        """
        Aromatic HC inhibition factor for nitrogenase [-].

        f_arom = K_I_arom / (K_I_arom + [HC_arom])

        Returns 1 when [HC_arom] = 0.

        Parameters
        ----------
        HC_arom_aq : float  Aqueous aromatic HC concentration [mol/L].

        Returns
        -------
        float  Factor in (0, 1].
        """
        HC_arom_aq = max(HC_arom_aq, 0.0)
        return self.p.K_I_arom / (self.p.K_I_arom + HC_arom_aq)

    def cys_factor(self, Cys_aq: float) -> float:
        """
        Cysteine co-limitation factor for nitrogenase [-].

        f_Cys = [Cys] / (K_Cys_nit + [Cys])

        Returns 0 when [Cys] → 0 (complete Fe-S cluster starvation),
        approaches 1 when [Cys] ≫ K_Cys_nit (no limitation).

        Parameters
        ----------
        Cys_aq : float  Cysteine concentration [mol/L].

        Returns
        -------
        float  Factor in [0, 1].
        """
        Cys_aq = max(Cys_aq, 0.0)
        K_Cys  = getattr(self.p, "K_Cys_nit", 2.0e-5)
        return Cys_aq / (K_Cys + Cys_aq)

    def N2_saturation_factor(self, N2_aq: float) -> float:
        """
        N₂ saturation factor for nitrogenase [-].

        f_N₂ = [N₂]_aq / (K_N₂ + [N₂]_aq)

        Parameters
        ----------
        N2_aq : float  Dissolved N₂ [mol/L].

        Returns
        -------
        float  Factor in [0, 1].
        """
        N2_aq = max(N2_aq, 0.0)
        return N2_aq / (self.p.K_N2 + N2_aq)

    def __repr__(self) -> str:
        return (f"NitrogenaseKinetics(r_nit_max={self.p.r_nit_max} mol N₂/g VSS/h, "
                f"K_N2={self.p.K_N2*1e6:.0f} μmol/L, "
                f"K_I_H2={self.p.K_I_H2} atm, "
                f"K_I_arom={self.p.K_I_arom*1e3:.2f} mmol/L)")
