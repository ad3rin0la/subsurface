"""
thermophysics/solubility.py
============================
Gas solubility in water / NaCl brine using Henry's law.

References
----------
Fernandez-Prini, R.; Alvarez, J.L.; Harvey, A.H. (2003).
    J. Phys. Chem. Ref. Data 32, 903-916.
Ziabakhsh-Ganji, Z.; Kooi, H. (2012). Geophys. Res. Lett. 39, L01405.
Battino, R. (1982). Solubility Data Series vol 7 (N₂/O₂).
Setschenow, J.Z. (1889). Z. Phys. Chem. 4, 117.

Units:
    T        – Kelvin
    P_gas    – bar
    m_NaCl   – mol/kg (molality)
    kH       – bar (mole-fraction-basis Henry constant)
    [gas]_aq – mol/L dissolved concentration
"""

import numpy as np


# ---------------------------------------------------------------------------
# Water saturation pressure (Antoine approximation)
# ---------------------------------------------------------------------------

def _water_sat_pressure(T: float) -> float:
    """
    Saturation pressure of water from Antoine equation [bar].

    Valid range: 275 – 650 K (approximate).

    Parameters
    ----------
    T : float  Temperature [K].

    Returns
    -------
    float  Saturation pressure [bar].
    """
    # Antoine constants for water (T in K, result in kPa)
    # log10(P/kPa) = A - B/(C + T)
    # Wagner & Pruss (2002) IAPWS values simplified to Antoine form
    A, B, C = 16.3872, 3885.70, -42.98
    P_kPa = np.exp(A - B / (T + C))   # kPa  (note: uses natural log form)
    return P_kPa / 100.0               # → bar


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _GasSolubility:
    """
    Base class for Fernandez-Prini Henry's law gas solubility.

    Subclasses supply the FP2003 parameters (eta, c_list, sigma) and the
    gas name.

    The Henry constant is on the mole-fraction (x) basis [bar], returned
    by henry_constant_pure_water.  It is converted to a molality-basis
    [mol/kg/bar] and then to a volumetric basis [mol/L/bar] for use in
    the dissolved_concentration method.
    """

    # To be set by subclass
    _name: str = ""
    _eta: float = 0.0
    _c: list = []
    _sigma: float = 0.355
    _Tc_water: float = 647.096   # K  (IAPWS 1997)

    # Setschenow parameters (defaults; subclass may override)
    _lambda_gas_Na: float = 0.0
    _zeta_gas_Na_Cl: float = 0.0

    def henry_constant_pure_water(self, T: float) -> float:
        """
        Henry constant for gas in pure water on mole-fraction basis [bar].

        Fernandez-Prini (2003) correlation:
            ln(kH / p_sat*) = η/T* + Σᵢ cᵢ*(1-T*)^(i·σ)

        where T* = T / Tc_water.

        Parameters
        ----------
        T : float  Temperature [K].

        Returns
        -------
        float  Henry constant kH [bar], mole-fraction basis.
        """
        T_star = T / self._Tc_water
        p_sat = _water_sat_pressure(T)   # bar

        ln_sum = sum(
            self._c[i] * (1.0 - T_star) ** ((i + 1) * self._sigma)
            for i in range(len(self._c))
        )
        ln_kH_over_psat = self._eta / T_star + ln_sum
        kH_bar = np.exp(ln_kH_over_psat) * p_sat   # bar (mole-fraction basis)
        return kH_bar

    def henry_constant_molar(self, T: float) -> float:
        """
        Henry constant on a volumetric (molar) basis [L·bar/mol].

        kH_molar = kH_xfrac / (c_water)
        where c_water ≈ 55.5 mol/L (density of water / MW_water).

        Equivalently, kH_molar = kH_xfrac * 0.018015 / rho_water_kg_per_L
        where rho_water ≈ 1.0 kg/L at reservoir conditions (approximate).

        Parameters
        ----------
        T : float  Temperature [K].

        Returns
        -------
        float  kH [L·bar/mol].
        """
        kH_xfrac = self.henry_constant_pure_water(T)
        # molar concentration of pure water: ~1000 g/L / 18.015 g/mol
        c_water_mol_per_L = 1000.0 / 18.015   # ≈ 55.51 mol/L
        return kH_xfrac / c_water_mol_per_L   # L·bar/mol

    def henry_constant_brine(self, T: float, m_NaCl: float) -> float:
        """
        Henry constant in NaCl brine (Setschenow salting-out) [L·bar/mol].

        ln(kH_brine / kH_water) = (λ_gas-Na + ζ_gas-Na-Cl * m_Cl) * m_Na

        For NaCl: m_Na = m_Cl = m_NaCl.

        Parameters
        ----------
        T : float      Temperature [K].
        m_NaCl : float Molality of NaCl [mol/kg].

        Returns
        -------
        float  Henry constant in brine [L·bar/mol].
        """
        kH_w = self.henry_constant_molar(T)
        m_Na = m_NaCl
        m_Cl = m_NaCl
        ln_correction = (self._lambda_gas_Na + self._zeta_gas_Na_Cl * m_Cl) * m_Na
        return kH_w * np.exp(ln_correction)

    def dissolved_concentration(self, T: float, P_gas_bar: float,
                                m_NaCl: float = 0.0) -> float:
        """
        Dissolved gas concentration via Henry's law [mol/L].

        [gas]_aq = P_gas / kH_brine

        Parameters
        ----------
        T : float           Temperature [K].
        P_gas_bar : float   Partial pressure of gas [bar].
        m_NaCl : float      NaCl molality [mol/kg]. Default 0 (pure water).

        Returns
        -------
        float  Dissolved concentration [mol/L].
        """
        kH = self.henry_constant_brine(T, m_NaCl)
        if kH <= 0.0:
            return 0.0
        return P_gas_bar / kH   # (bar) / (L·bar/mol) = mol/L

    def solubility_bar_per_mol_L(self, T: float, m_NaCl: float = 0.0) -> float:
        """
        Return Henry constant for direct use: bar / (mol/L) = L·bar/mol.

        Convenience alias for henry_constant_brine.
        """
        return self.henry_constant_brine(T, m_NaCl)


# ---------------------------------------------------------------------------
# N₂ Solubility
# ---------------------------------------------------------------------------

class N2Solubility(_GasSolubility):
    """
    N₂ solubility in water / NaCl brine.

    Fernandez-Prini (2003) parameters for N₂:
        η = -9.67578
        c₁ = 4.72162, c₂ = 11.70585, c₃ = -12.78108
        σ = 0.355

    Setschenow (Battino 1982):
        λ_N₂-Na  = 0.1015
        ζ_N₂-NaCl = -0.0009
    """

    _name = "N2"
    _eta = -9.67578
    _c = [4.72162, 11.70585, -12.78108]
    _sigma = 0.355
    _lambda_gas_Na = 0.1015
    _zeta_gas_Na_Cl = -0.0009


# ---------------------------------------------------------------------------
# CO₂ Solubility
# ---------------------------------------------------------------------------

class CO2Solubility(_GasSolubility):
    """
    CO₂ solubility in water / NaCl brine.

    Fernandez-Prini (2003) parameters for CO₂:
        η = -8.55445
        c₁ = 4.01195, c₂ = 9.52345, c₃ = -9.31346
        σ = 0.355

    Setschenow parameters from Duan & Sun (2003):
        λ_CO₂-Na  ≈ 0.1070
        ζ_CO₂-NaCl ≈ -0.0020
    """

    _name = "CO2"
    _eta = -8.55445
    _c = [4.01195, 9.52345, -9.31346]
    _sigma = 0.355
    _lambda_gas_Na = 0.1070
    _zeta_gas_Na_Cl = -0.0020


# ---------------------------------------------------------------------------
# H₂ Solubility
# ---------------------------------------------------------------------------

class H2Solubility(_GasSolubility):
    """
    H₂ solubility in water / NaCl brine.

    Fernandez-Prini (2003) parameters for H₂:
        η = -4.73284
        c₁ = 6.08954, c₂ = 6.06861, c₃ = -6.62223
        σ = 0.355

    Setschenow parameters (approximate, from Wiebe & Gaddy 1934):
        λ_H₂-Na  ≈ 0.0920
        ζ_H₂-NaCl ≈ -0.0008
    """

    _name = "H2"
    _eta = -4.73284
    _c = [6.08954, 6.06861, -6.62223]
    _sigma = 0.355
    _lambda_gas_Na = 0.0920
    _zeta_gas_Na_Cl = -0.0008


# ---------------------------------------------------------------------------
# Convenience module-level functions (backward-compatible API)
# ---------------------------------------------------------------------------

_n2_sol = N2Solubility()
_co2_sol = CO2Solubility()
_h2_sol = H2Solubility()


def henry_constant_N2_pure_water(T: float) -> float:
    """Henry constant for N₂ in pure water [L·bar/mol]."""
    return _n2_sol.henry_constant_molar(T)


def henry_constant_N2_brine(T: float, m_NaCl: float) -> float:
    """Henry constant for N₂ in NaCl brine [L·bar/mol]."""
    return _n2_sol.henry_constant_brine(T, m_NaCl)


def dissolved_N2(T: float, P_N2_bar: float, m_NaCl: float = 0.0) -> float:
    """
    Dissolved N₂ concentration [mol/L].

    Parameters
    ----------
    T : float           Temperature [K].
    P_N2_bar : float    N₂ partial pressure [bar].
    m_NaCl : float      NaCl molality [mol/kg].

    Returns
    -------
    float  [N₂]_aq [mol/L].
    """
    return _n2_sol.dissolved_concentration(T, P_N2_bar, m_NaCl)


def dissolved_CO2(T: float, P_CO2_bar: float, m_NaCl: float = 0.0) -> float:
    """Dissolved CO₂ concentration [mol/L]."""
    return _co2_sol.dissolved_concentration(T, P_CO2_bar, m_NaCl)


def dissolved_H2(T: float, P_H2_bar: float, m_NaCl: float = 0.0) -> float:
    """Dissolved H₂ concentration [mol/L]."""
    return _h2_sol.dissolved_concentration(T, P_H2_bar, m_NaCl)
