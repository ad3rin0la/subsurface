"""
kinetics/monod.py
=================
Monod-type growth kinetics for diazotrophic microorganisms in the N2Bio
framework.

The complete growth rate includes triple co-limitation by:
    1. HC_ali  – the fermentable aliphatic hydrocarbon pool (electron donor)
    2. NH₄⁺   – fixed nitrogen for cell biosynthesis
    3. N₂(aq) – dissolved dinitrogen for nitrogenase activity

Reference
---------
Monod, J. (1942). Recherches sur la croissance des cultures bactériennes.
    Hermann, Paris.
Rittmann, B.E.; McCarty, P.L. (2001). Environmental Biotechnology: Principles
    and Applications. McGraw-Hill.
"""

import numpy as np
from .parameters import KineticParameters


class MonodKinetics:
    """
    Monod growth kinetics with triple substrate co-limitation.

    Parameters
    ----------
    params : KineticParameters
        Kinetic parameter set.
    """

    def __init__(self, params: KineticParameters):
        self.p = params

    # ------------------------------------------------------------------
    # Core growth rate
    # ------------------------------------------------------------------

    def growth_rate(self, HC_ali: float, NH4: float, N2_aq: float) -> float:
        """
        Complete Monod growth rate with triple co-limitation [h⁻¹].

        μ = μ_max * S_HC/(K_S + S_HC) * [NH₄⁺]/(K_N_fixed + [NH₄⁺])
                  * [N₂]_aq/(K_N₂ + [N₂]_aq)

        Parameters
        ----------
        HC_ali : float
            Aliphatic HC concentration [mol/L].  Non-negative.
        NH4 : float
            Ammonium concentration [mol/L].  Non-negative.
        N2_aq : float
            Dissolved N₂ concentration [mol/L].  Non-negative.

        Returns
        -------
        float
            Specific growth rate [h⁻¹].  Always ≥ 0.
        """
        # Clamp negative inputs (numerical safety)
        HC_ali = max(HC_ali, 0.0)
        NH4    = max(NH4, 0.0)
        N2_aq  = max(N2_aq, 0.0)

        f_HC      = HC_ali / (self.p.K_S         + HC_ali)
        f_N_fixed = NH4    / (self.p.K_N_fixed   + NH4)
        f_N2      = N2_aq  / (self.p.K_N2        + N2_aq)

        return self.p.mu_max * f_HC * f_N_fixed * f_N2

    # ------------------------------------------------------------------
    # Individual limitation factors (for diagnostics / plotting)
    # ------------------------------------------------------------------

    def HC_limitation(self, HC_ali: float) -> float:
        """
        Monod limitation factor for aliphatic HC (electron donor) [-].

        f_HC = [HC_ali] / (K_S + [HC_ali])

        Parameters
        ----------
        HC_ali : float  Aliphatic HC concentration [mol/L].

        Returns
        -------
        float  Limitation factor in [0, 1].
        """
        HC_ali = max(HC_ali, 0.0)
        return HC_ali / (self.p.K_S + HC_ali)

    def N_limitation(self, NH4: float) -> float:
        """
        Monod limitation factor for NH₄⁺ (fixed N, biosynthesis) [-].

        f_N_fixed = [NH₄⁺] / (K_N_fixed + [NH₄⁺])

        Parameters
        ----------
        NH4 : float  Ammonium concentration [mol/L].

        Returns
        -------
        float  Limitation factor in [0, 1].
        """
        NH4 = max(NH4, 0.0)
        return NH4 / (self.p.K_N_fixed + NH4)

    def N2_limitation(self, N2_aq: float) -> float:
        """
        Monod limitation factor for dissolved N₂ (nitrogen source) [-].

        f_N2 = [N₂]_aq / (K_N₂ + [N₂]_aq)

        Parameters
        ----------
        N2_aq : float  Dissolved N₂ [mol/L].

        Returns
        -------
        float  Limitation factor in [0, 1].
        """
        N2_aq = max(N2_aq, 0.0)
        return N2_aq / (self.p.K_N2 + N2_aq)

    # ------------------------------------------------------------------
    # Inhibition factors
    # ------------------------------------------------------------------

    def NH4_inhibition_factor(self, NH4: float) -> float:
        """
        NH₄⁺ catabolite repression (inhibition) factor for nitrogenase [-].

        When [NH₄⁺] >> K_I_NH4, nitrogenase expression is suppressed
        (cells prefer pre-fixed N over the energetically expensive
        nitrogenase pathway).

        f_inhib = K_I_NH4 / (K_I_NH4 + [NH₄⁺])

        Parameters
        ----------
        NH4 : float  Ammonium concentration [mol/L].

        Returns
        -------
        float  Inhibition factor in (0, 1].  1 = no inhibition.
        """
        NH4 = max(NH4, 0.0)
        return self.p.K_I_NH4 / (self.p.K_I_NH4 + NH4)

    # ------------------------------------------------------------------
    # Net biomass dynamics
    # ------------------------------------------------------------------

    def net_growth_rate(self, HC_ali: float, NH4: float, N2_aq: float) -> float:
        """
        Net specific growth rate accounting for endogenous decay [h⁻¹].

        μ_net = μ - b_decay

        Parameters
        ----------
        HC_ali, NH4, N2_aq : float  Concentrations [mol/L].

        Returns
        -------
        float  Net specific growth rate [h⁻¹].  Can be negative.
        """
        return self.growth_rate(HC_ali, NH4, N2_aq) - self.p.b_decay

    def biomass_rate(self, HC_ali: float, NH4: float, N2_aq: float,
                     X: float) -> float:
        """
        Biomass production rate dX/dt [g VSS/L/h].

        dX/dt = (μ - b_decay) * X

        Parameters
        ----------
        HC_ali, NH4, N2_aq : float  Concentrations [mol/L].
        X : float                    Biomass concentration [g VSS/L].

        Returns
        -------
        float  Rate [g VSS/L/h].
        """
        X = max(X, 0.0)
        return self.net_growth_rate(HC_ali, NH4, N2_aq) * X

    # ------------------------------------------------------------------
    # Convenience: maximum possible rates for normalisation
    # ------------------------------------------------------------------

    @property
    def mu_max(self) -> float:
        """Maximum specific growth rate [h⁻¹]."""
        return self.p.mu_max

    def __repr__(self) -> str:
        return (f"MonodKinetics(mu_max={self.p.mu_max} h⁻¹, "
                f"K_S={self.p.K_S*1e3:.3f} mmol/L, "
                f"K_N2={self.p.K_N2*1e6:.1f} μmol/L)")
