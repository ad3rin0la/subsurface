"""
kinetics/sulfur.py
==================
Sulfur assimilation kinetics for the N2Bio diazotrophic chassis.

Three missed couplings corrected here:

1. H₂S as direct OASTL substrate
   In an anaerobic reservoir with H₂S present (geological or SRB-derived),
   the diazotroph bypasses the energetically expensive SO₄²⁻ → APS → SO₃²⁻ →
   S²⁻ reductive chain and feeds dissolved sulfide directly into OASTL:

       OAS  +  HS⁻  →  Cys  +  acetate  +  H⁺

   pH-dependent speciation (H₂S ⇌ HS⁻, pKₐ ≈ 7.0 at 80 °C) controls the
   fraction available to OASTL.

2. Acetate as the acetyl-CoA donor to SAT
   SAT (serine acetyltransferase) catalyses:
       L-serine  +  acetyl-CoA  →  OAS  +  CoA

   Acetyl-CoA is in equilibrium with free acetate via acetyl-CoA synthetase:
       acetate  +  CoA  +  ATP  →  acetyl-CoA  +  AMP  +  PPᵢ

   Consequently OAS synthesis is co-limited by NH₄⁺ (gates serine via
   GS/GOGAT) AND by acetate availability, which tracks dark-fermentation
   throughput of HC_ali.  Cysteine product inhibition closes the feedback.

3. Cysteine structural requirement for nitrogenase
   Both the Fe-protein and the MoFe-protein of nitrogenase contain Fe-S
   clusters assembled via the ISC machinery using cysteine-derived sulfide.
   Cluster turnover is continuous, creating an ongoing Cys drain proportional
   to active nitrogenase content (tracked as a constant fraction of biomass X).

   The net effect: H₂S availability → Cys ↑ → nitrogenase active fraction ↑
   → r_nit ↑ → H₂ and NH₄⁺ ↑ → OAS synthesis ↑ → more Cys (positive loop).

References
----------
Hell, R. (1997) Planta 202, 138–148. (SAT/OASTL review)
Berkowitz, O. et al. (2002) J. Biol. Chem. 277, 27267. (Cys feedback on SAT)
Zheng, L.; Dean, D.R. (1994) J. Biol. Chem. 269, 18723. (nifS for Fe-S assembly)
Hesse, H.; Hoefgen, R. (2003) Trends Plant Sci. 8, 259. (N-S metabolic coupling)
"""

from __future__ import annotations
import numpy as np
from .parameters import KineticParameters


class SulfurAssimilationKinetics:
    """
    SAT / OASTL sulfur assimilation pathway for anaerobic diazotrophs.

    Models two enzymatic reactions that connect the sulfide pool to cysteine
    and thereby to nitrogenase Fe-S cluster maintenance:

    SAT:   L-serine + acetyl-CoA  →  OAS + CoA
           (acetate used as dissolved proxy for intracellular acetyl-CoA)

    OASTL: OAS + HS⁻  →  Cys + acetate + H⁺

    Parameters
    ----------
    params : KineticParameters
    """

    def __init__(self, params: KineticParameters):
        self.p = params

    # ------------------------------------------------------------------
    # H₂S / HS⁻ speciation
    # ------------------------------------------------------------------

    def hs_minus_fraction(self, pH: float) -> float:
        """
        Fraction of total dissolved sulfide present as HS⁻ at given pH.

        HS⁻ is the reactive species for OASTL (not H₂S).

        f_HS⁻ = 1 / (1 + 10^(pKa − pH))

        At pH < pKa  → mostly H₂S; at pH > pKa → mostly HS⁻.
        pKa ≈ 7.0 at 80 °C (slightly lower than 25 °C value of 7.05).

        Parameters
        ----------
        pH : float

        Returns
        -------
        float  Fraction in (0, 1).
        """
        return 1.0 / (1.0 + 10.0 ** (self.p.pKa_H2S - pH))

    def active_sulfide(self, H2S_aq: float, pH: float) -> float:
        """
        Effective OASTL-accessible sulfide concentration [mol/L].

        S_active = H₂S_total × f_HS⁻(pH)

        Parameters
        ----------
        H2S_aq : float  Total dissolved sulfide [mol/L].
        pH     : float  Aqueous pH.

        Returns
        -------
        float  [mol/L]
        """
        H2S_aq = max(H2S_aq, 0.0)
        return H2S_aq * self.hs_minus_fraction(pH)

    # ------------------------------------------------------------------
    # SAT rate
    # ------------------------------------------------------------------

    def sat_rate(self, NH4_aq: float, Acetate_aq: float,
                 Cys_aq: float, X: float) -> float:
        """
        Volumetric SAT rate [mol OAS / L / h].

        r_SAT = r_SAT_max
                × [NH₄⁺] / (K_OAS_SAT + [NH₄⁺])     ← N (serine) co-limitation
                × [acetate] / (K_Ac_SAT + [acetate])   ← acetyl-CoA co-limitation
                × K_I_Cys_SAT / (K_I_Cys_SAT + [Cys]) ← Cys allosteric feedback
                × X

        NH₄⁺ gates serine availability via GS/GOGAT (Glu → Ser is NH₄⁺-driven).
        Acetate is the dissolved proxy for intracellular acetyl-CoA via ACS.
        Cys product-inhibits SAT allosterically (well-characterised in bacteria).

        Parameters
        ----------
        NH4_aq     : float  Ammonium [mol/L].
        Acetate_aq : float  Free acetate [mol/L].
        Cys_aq     : float  Cysteine [mol/L].
        X          : float  Biomass [g VSS/L].

        Returns
        -------
        float  [mol OAS/L/h].  Always ≥ 0.
        """
        NH4_aq     = max(NH4_aq,     0.0)
        Acetate_aq = max(Acetate_aq, 0.0)
        Cys_aq     = max(Cys_aq,     0.0)
        X          = max(X,          0.0)

        f_N    = NH4_aq     / (self.p.K_N_fixed  + NH4_aq)
        f_Ac   = Acetate_aq / (self.p.K_Ac_SAT   + Acetate_aq)
        f_iCys = self.p.K_I_Cys_SAT / (self.p.K_I_Cys_SAT + Cys_aq)

        return self.p.r_SAT_max * f_N * f_Ac * f_iCys * X

    # ------------------------------------------------------------------
    # OASTL rate
    # ------------------------------------------------------------------

    def oastl_rate(self, OAS_aq: float, H2S_aq: float,
                   Cys_aq: float, X: float, pH: float) -> float:
        """
        Volumetric OASTL rate [mol Cys / L / h].

        r_OASTL = r_OASTL_max
                  × [OAS]     / (K_OAS_OASTL  + [OAS])     ← substrate
                  × S_active  / (K_H2S_OASTL  + S_active)  ← pH-corrected sulfide
                  × X

        No strong product inhibition reported for bacterial OASTL in anoxic
        conditions; Cys feedback is primarily via SAT (above).

        Parameters
        ----------
        OAS_aq  : float  O-acetylserine [mol/L].
        H2S_aq  : float  Total dissolved sulfide [mol/L].
        Cys_aq  : float  Cysteine [mol/L].
        X       : float  Biomass [g VSS/L].
        pH      : float  Aqueous pH (controls H₂S/HS⁻ speciation).

        Returns
        -------
        float  [mol Cys/L/h].  Always ≥ 0.
        """
        OAS_aq = max(OAS_aq, 0.0)
        X      = max(X,      0.0)

        S_act = self.active_sulfide(H2S_aq, pH)

        f_OAS = OAS_aq / (self.p.K_OAS_OASTL + OAS_aq)
        f_S   = S_act  / (self.p.K_H2S_OASTL + S_act)

        return self.p.r_OASTL_max * f_OAS * f_S * X

    # ------------------------------------------------------------------
    # Cys sinks
    # ------------------------------------------------------------------

    def cys_biomass_demand(self, mu: float, X: float) -> float:
        """
        Cys drain for structural biomass synthesis [mol Cys / L / h].

        Cys is the main organic sulfur source in cellular protein.
        Sulfur content of dry biomass: Y_S ≈ 0.010 g S / g VSS.
        All structural sulfur assumed to enter via cysteine (MW 121 g/mol;
        1 S per Cys, so mol Cys ≈ mol S).

        drain = (μ · X / Y) · (Y_S / MW_S)

        Parameters
        ----------
        mu : float  Specific growth rate [h⁻¹].
        X  : float  Biomass [g VSS/L].

        Returns
        -------
        float  [mol Cys/L/h].  Always ≥ 0.
        """
        X  = max(X,  0.0)
        mu = max(mu, 0.0)
        MW_S = 32.06   # g/mol
        return (mu * X / self.p.Y) * (self.p.Y_S / MW_S)

    def cys_fes_maintenance(self, X: float) -> float:
        """
        Cys drain for nitrogenase Fe-S cluster turnover [mol Cys / L / h].

        Fe-S clusters in the Fe-protein (4Fe-4S) and MoFe-protein (P-clusters
        + FeMo-cofactor) are assembled by the ISC machinery using
        cysteine-derived sulfide.  Cluster lifetime is finite; continuous
        turnover creates an ongoing Cys drain proportional to biomass.

        drain = r_FeS_maintenance · X

        Parameters
        ----------
        X : float  Biomass [g VSS/L].

        Returns
        -------
        float  [mol Cys/L/h].  Always ≥ 0.
        """
        return self.p.r_FeS_maintenance * max(X, 0.0)

    # ------------------------------------------------------------------
    # H₂S source / sink summary
    # ------------------------------------------------------------------

    def dH2S(self, OAS_aq: float, H2S_aq: float,
             Cys_aq: float, X: float, pH: float) -> float:
        """
        Net rate of change of dissolved H₂S [mol/L/h].

        dH₂S/dt = H2S_source              ← geological / SRB boundary
                − r_OASTL                  ← consumed by OASTL (1:1 H₂S:Cys)
                − k_strip_H2S · H₂S_aq   ← gas-phase stripping

        Gas-phase stripping is significant: kH(H₂S) ≈ 100 L·bar/mol at 80 °C.
        Low dissolved H₂S maintains the driving force from geological sources.

        Parameters
        ----------
        OAS_aq, H2S_aq, Cys_aq, X, pH : see oastl_rate().

        Returns
        -------
        float  [mol/L/h].
        """
        r_oastl  = self.oastl_rate(OAS_aq, H2S_aq, Cys_aq, X, pH)
        r_strip  = self.p.k_strip_H2S * max(H2S_aq, 0.0)
        return self.p.H2S_source - r_oastl - r_strip

    def __repr__(self) -> str:
        return (
            f"SulfurAssimilationKinetics("
            f"r_SAT_max={self.p.r_SAT_max:.2e}, "
            f"r_OASTL_max={self.p.r_OASTL_max:.2e}, "
            f"K_Cys_nit={self.p.K_Cys_nit*1e6:.1f} μmol/L, "
            f"pKa_H2S={self.p.pKa_H2S})"
        )
