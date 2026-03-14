"""
kinetics/hc_degradation.py
===========================
Hydrocarbon degradation kinetics for the N2Bio two-pool HC model.

The HC pool is split into:
  HC_ali  – aliphatic (fermentable) fraction, degraded by Thermotoga-like
             dark fermentation at rate k_ali * X
  HC_arom – aromatic fraction, transformed more slowly; acts as a
             nitrogenase inhibitor as it accumulates

Dark fermentation stoichiometry (Thermotoga maritima-type):
  Glucose (C₆H₁₂O₆) + 2 H₂O  →  2 CH₃COO⁻ + 2 CO₂ + 4 H₂
  ~ 4 mol H₂ / mol hexose = ~0.35 mol H₂ / g COD  (Y_HC_H₂)

References
----------
Schut, G.J.; Adams, M.W.W. (2009). J. Bacteriol. 191, 4451.
Levin, D.B.; Pitt, L.; Love, M. (2004). Int. J. Hydrogen Energy 29, 173.
"""

import numpy as np
from .parameters import KineticParameters


class HCDegradationModel:
    """
    Two-pool hydrocarbon degradation model.

    Tracks aliphatic (fermentable) and aromatic (inhibitory) HC fractions
    under biological control.  Provides rates for both pools and total H₂
    production from fermentation.

    Parameters
    ----------
    params : KineticParameters
        Kinetic parameter dataclass.
    """

    def __init__(self, params: KineticParameters):
        self.p = params

    # ------------------------------------------------------------------
    # HC pool degradation rates
    # ------------------------------------------------------------------

    def HC_ali_rate(self, HC_ali: float, X: float) -> float:
        """
        Rate of change of aliphatic HC [mol/L/h].

        d[HC_ali]/dt = -k_ali * [HC_ali] * X

        First-order in substrate and biomass (bimolecular).  Represents
        enzymatic dark fermentation under electron-donor limitation.

        Parameters
        ----------
        HC_ali : float  Aliphatic HC concentration [mol/L].
        X : float       Biomass [g VSS/L].

        Returns
        -------
        float  Rate [mol/L/h].  Always ≤ 0.
        """
        HC_ali = max(HC_ali, 0.0)
        X = max(X, 0.0)
        return -self.p.k_ali * HC_ali * X

    def HC_arom_rate(self, HC_arom: float, X: float) -> float:
        """
        Rate of change of aromatic HC [mol/L/h].

        d[HC_arom]/dt = -k_arom * [HC_arom] * X

        Aromatics are degraded/transformed slowly; the rate constant is
        ~20× lower than for aliphatics (k_arom ≪ k_ali).

        Parameters
        ----------
        HC_arom : float  Aromatic HC concentration [mol/L].
        X : float        Biomass [g VSS/L].

        Returns
        -------
        float  Rate [mol/L/h].  Always ≤ 0.
        """
        HC_arom = max(HC_arom, 0.0)
        X = max(X, 0.0)
        return -self.p.k_arom * HC_arom * X

    # ------------------------------------------------------------------
    # H₂ production from dark fermentation
    # ------------------------------------------------------------------

    def fermentation_H2_rate(self, HC_ali: float, X: float) -> float:
        """
        H₂ production rate from dark fermentation of aliphatic HC [mol H₂/L/h].

        r_H₂_ferm = k_ali * [HC_ali] * X * Y_HC_H₂

        The yield Y_HC_H₂ [mol H₂ / g COD] converts mass COD consumed to
        moles of H₂.  Using molar mass of HC_ali (acetic-acid equivalent,
        MW ≈ 60 g/mol, COD factor ≈ 1.07 g COD/g) gives
        Y in mol H₂ / mol HC_ali ≈ 0.35 * 60 * 1.07 ≈ 22.5 mol H₂/mol.

        NOTE: HC_ali is in mol/L, X in g VSS/L.
              k_ali has units h⁻¹, Y_HC_H2 has units mol H₂/g COD.
              To obtain mol H₂/L/h we need to convert mol HC_ali → g COD.
              COD of acetic-acid equivalent = MW_HC_ali * COD_factor / 1000 g/mmol
              We use an effective yield Y_eff = Y_HC_H2 * MW_HC_ali * COD_factor [mol/mol]

        Parameters
        ----------
        HC_ali : float  Aliphatic HC [mol/L].
        X : float       Biomass [g VSS/L].

        Returns
        -------
        float  H₂ fermentation rate [mol H₂/L/h].
        """
        HC_ali = max(HC_ali, 0.0)
        X = max(X, 0.0)
        r_HC_consumed = self.p.k_ali * HC_ali * X   # mol HC_ali / L / h
        # Convert to g COD/L/h: MW_HC_ali ~ 60 g/mol, COD factor ~1.07
        MW_HC_ali = 60.052    # g/mol  (acetic acid equivalent)
        COD_factor = 1.0667   # g COD / g (= (2*32)/(2*12+4*1+2*16) for CH3COOH)
        g_COD_per_mol = MW_HC_ali * COD_factor / 1000.0   # g COD / mmol → g COD / mol × 1e-3
        # r_COD in g COD/L/h
        r_COD = r_HC_consumed * MW_HC_ali * COD_factor   # g COD/L/h
        # r_H2 in mol H2/L/h
        r_H2 = r_COD * self.p.Y_HC_H2
        return r_H2

    # ------------------------------------------------------------------
    # Pool composition diagnostics
    # ------------------------------------------------------------------

    def aromatic_fraction(self, HC_ali: float, HC_arom: float) -> float:
        """
        Aromatic mass fraction of total HC pool [-].

        f_arom = [HC_arom] / ([HC_ali] + [HC_arom])

        Returns initial f_arom0 when both pools are exhausted to avoid
        division by zero.

        Parameters
        ----------
        HC_ali : float   Aliphatic HC [mol/L].
        HC_arom : float  Aromatic HC [mol/L].

        Returns
        -------
        float  Aromatic fraction in [0, 1].
        """
        HC_ali  = max(HC_ali,  0.0)
        HC_arom = max(HC_arom, 0.0)
        total = HC_ali + HC_arom
        if total < 1.0e-12:
            return self.p.f_arom0   # default when exhausted
        return HC_arom / total

    def aliphatic_fraction(self, HC_ali: float, HC_arom: float) -> float:
        """
        Aliphatic mass fraction of total HC pool [-].

        f_ali = 1 - f_arom

        Parameters
        ----------
        HC_ali, HC_arom : float  HC concentrations [mol/L].

        Returns
        -------
        float  Aliphatic fraction in [0, 1].
        """
        return 1.0 - self.aromatic_fraction(HC_ali, HC_arom)

    # ------------------------------------------------------------------
    # Total H₂ production (combined pathways)
    # ------------------------------------------------------------------

    def total_H2_rate(self, HC_ali: float, HC_arom: float,
                      N2_aq: float, P_H2_atm: float,
                      X: float, nitrogenase_kinetics) -> float:
        """
        Total H₂ production rate from nitrogenase + dark fermentation [mol H₂/L/h].

        r_H₂_total = r_H₂_nit + r_H₂_ferm

        where:
          r_H₂_nit  = nitrogenase H₂ (1 mol H₂ per mol N₂ fixed)
          r_H₂_ferm = dark fermentation H₂

        Parameters
        ----------
        HC_ali : float          Aliphatic HC [mol/L].
        HC_arom : float         Aromatic HC [mol/L].
        N2_aq : float           Dissolved N₂ [mol/L].
        P_H2_atm : float        H₂ partial pressure [atm].
        X : float               Biomass [g VSS/L].
        nitrogenase_kinetics    NitrogenaseKinetics instance.

        Returns
        -------
        float  Total H₂ production rate [mol H₂/L/h].
        """
        r_nit_H2  = nitrogenase_kinetics.H2_production_rate(
            N2_aq, P_H2_atm, HC_arom, X
        )
        r_ferm_H2 = self.fermentation_H2_rate(HC_ali, X)
        return r_nit_H2 + r_ferm_H2

    # ------------------------------------------------------------------
    # Substrate consumed for biomass synthesis (electron donor)
    # ------------------------------------------------------------------

    def substrate_for_growth(self, mu: float, X: float) -> float:
        """
        HC_ali consumed for biomass synthesis (assimilative consumption)
        [mol/L/h].

        From stoichiometry: r_HC_assimil = (μ * X) / Y

        This is the portion of HC degradation going to cell material rather
        than H₂.  The total HC consumption is split between:
          - dissimilation → H₂/CO₂ (captured in HC_ali_rate + fermentation_H2_rate)
          - assimilation  → new biomass (captured here)

        In the two-pool model we lump both into k_ali for simplicity.

        Parameters
        ----------
        mu : float  Specific growth rate [h⁻¹].
        X : float   Biomass [g VSS/L].

        Returns
        -------
        float  Substrate consumption rate [mol HC_ali / L / h].
        """
        X = max(X, 0.0)
        MW_HC_ali = 60.052   # g/mol
        COD_factor = 1.0667  # g COD/g
        # g COD / L / h for growth
        g_COD_per_h = mu * X / self.p.Y
        # Convert to mol/L/h
        return g_COD_per_h / (MW_HC_ali * COD_factor)

    def __repr__(self) -> str:
        return (f"HCDegradationModel(k_ali={self.p.k_ali:.3e} h⁻¹, "
                f"k_arom={self.p.k_arom:.3e} h⁻¹, "
                f"f_arom0={self.p.f_arom0:.2f}, "
                f"Y_HC_H2={self.p.Y_HC_H2} mol H₂/g COD)")
