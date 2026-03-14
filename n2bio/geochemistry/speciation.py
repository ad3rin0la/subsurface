"""
geochemistry/speciation.py
==========================
Geochemical speciation of the NH₃/NH₄⁺ acid-base pair in the N2Bio
framework.

The equilibrium:
    NH₃(aq) + H⁺  ⇌  NH₄⁺      Ka = [NH₄⁺] / ([NH₃][H⁺])

pKa at 25°C = 9.25.  Temperature dependence is modelled via Van't Hoff.

References
----------
Bates, R.G.; Pinching, G.D. (1949). J. Res. NBS 42, 419.
Stumm, W.; Morgan, J.J. (1996). Aquatic Chemistry, 3rd ed. Wiley.
Millero, F.J. (1995). Geochim. Cosmochim. Acta 59, 661.
"""

import numpy as np
from typing import Tuple


class NH3Speciation:
    """
    NH₃/NH₄⁺ acid-base speciation as a function of temperature and pH.

    The pKa of the ammonium equilibrium is temperature-corrected using
    the Van't Hoff equation, giving a slight decrease in pKa with
    increasing temperature (less basic at high T).

    Notes
    -----
    At pH 7 and 25°C, >99.9% of total ammonium-N is NH₄⁺.
    At pH 7 and 85°C, ~99.7% is still NH₄⁺ (pKa decreases slightly).
    At pH 9 and 25°C, the split is ~64% NH₄⁺ / 36% NH₃.

    Class Attributes
    ----------------
    pKa_25C : float
        pKa of NH₄⁺ at 25°C (298.15 K).
    dH_deprotonation : float
        Standard enthalpy of NH₄⁺ deprotonation [J/mol].
        Used for Van't Hoff temperature correction.
    """

    pKa_25C: float = 9.25
    dH_deprotonation: float = 52210.0   # J/mol  (Bates & Pinching 1949)
    _R: float = 8.314                   # J/(mol·K)

    @classmethod
    def pKa(cls, T_K: float) -> float:
        """
        Temperature-corrected pKa of NH₄⁺ [dimensionless].

        Uses the Van't Hoff integrated form:
            pKa(T) = pKa(T_ref) + (ΔH/R/ln10) * (1/T - 1/T_ref)

        At 358.15 K (85°C): pKa ≈ 8.85

        Parameters
        ----------
        T_K : float
            Temperature [K].

        Returns
        -------
        float
            pKa at temperature T_K.
        """
        T_ref = 298.15   # K
        correction = (cls.dH_deprotonation / (cls._R * np.log(10.0))) * (
            1.0 / T_K - 1.0 / T_ref
        )
        return cls.pKa_25C + correction

    @classmethod
    def Ka(cls, T_K: float) -> float:
        """
        Acid dissociation constant Ka at temperature T_K [-].

        Ka = 10^(-pKa(T))

        Parameters
        ----------
        T_K : float  Temperature [K].

        Returns
        -------
        float  Ka.
        """
        return 10.0 ** (-cls.pKa(T_K))

    @classmethod
    def NH4_fraction(cls, T_K: float, pH: float) -> float:
        """
        Fraction of total ammonia-nitrogen present as NH₄⁺ [-].

        f_NH₄ = 1 / (1 + 10^(pH - pKa))

        At pH < pKa, f_NH₄ → 1 (mostly protonated).
        At pH > pKa, f_NH₄ → 0 (mostly free NH₃).

        Parameters
        ----------
        T_K : float  Temperature [K].
        pH : float   Solution pH.

        Returns
        -------
        float  NH₄⁺ fraction in [0, 1].
        """
        pKa = cls.pKa(T_K)
        return 1.0 / (1.0 + 10.0 ** (pH - pKa))

    @classmethod
    def NH3_fraction(cls, T_K: float, pH: float) -> float:
        """
        Fraction of total ammonia-nitrogen present as NH₃(aq) [-].

        f_NH₃ = 1 - f_NH₄ = 10^(pH-pKa) / (1 + 10^(pH-pKa))

        Parameters
        ----------
        T_K : float  Temperature [K].
        pH : float   Solution pH.

        Returns
        -------
        float  NH₃ fraction in [0, 1].
        """
        return 1.0 - cls.NH4_fraction(T_K, pH)

    @classmethod
    def speciate(cls, total_ammonium_N: float,
                 T_K: float, pH: float) -> Tuple[float, float]:
        """
        Speciate total dissolved ammonium-N into NH₄⁺ and NH₃ [mol/L].

        Parameters
        ----------
        total_ammonium_N : float
            Total dissolved ammonium-N concentration [mol/L].
            (= [NH₄⁺] + [NH₃])
        T_K : float
            Temperature [K].
        pH : float
            Solution pH.

        Returns
        -------
        (NH4_conc, NH3_conc) : Tuple[float, float]
            Concentrations [mol/L] of NH₄⁺ and NH₃ respectively.
        """
        total_ammonium_N = max(total_ammonium_N, 0.0)
        f_NH4 = cls.NH4_fraction(T_K, pH)
        NH4_conc = total_ammonium_N * f_NH4
        NH3_conc = total_ammonium_N * (1.0 - f_NH4)
        return NH4_conc, NH3_conc

    @classmethod
    def NH3_from_NH4(cls, NH4_conc: float,
                     T_K: float, pH: float) -> float:
        """
        Free NH₃ concentration given [NH₄⁺] [mol/L].

        [NH₃] = [NH₄⁺] * 10^(pH - pKa)

        Parameters
        ----------
        NH4_conc : float  NH₄⁺ concentration [mol/L].
        T_K : float       Temperature [K].
        pH : float        Solution pH.

        Returns
        -------
        float  [NH₃] [mol/L].
        """
        pKa = cls.pKa(T_K)
        return NH4_conc * 10.0 ** (pH - pKa)

    @classmethod
    def total_from_NH4(cls, NH4_conc: float, T_K: float, pH: float) -> float:
        """
        Total ammonium-N from [NH₄⁺] alone [mol/L].

        C_total = [NH₄⁺] / f_NH₄

        Parameters
        ----------
        NH4_conc : float  NH₄⁺ concentration [mol/L].
        T_K : float       Temperature [K].
        pH : float        Solution pH.

        Returns
        -------
        float  Total ammonium-N [mol/L].
        """
        f = cls.NH4_fraction(T_K, pH)
        if f < 1.0e-15:
            return 0.0
        return NH4_conc / f

    @classmethod
    def pKa_table(cls, T_range_C: np.ndarray) -> np.ndarray:
        """
        Compute pKa over a range of temperatures.

        Parameters
        ----------
        T_range_C : np.ndarray
            Temperature array [°C].

        Returns
        -------
        np.ndarray
            pKa values at each temperature.
        """
        return np.array([cls.pKa(T + 273.15) for T in T_range_C])
