"""
thermophysics/viscosity.py
==========================
Friction-theory (f-theory) viscosity model for N₂, CO₂, H₂.

References
----------
Quiñones-Cisneros, S.E.; Zéberg-Mikkelsen, C.K.; Stenby, E.H. (2000).
    Fluid Phase Equilibria 169, 249-276.
Quiñones-Cisneros, S.E.; Zéberg-Mikkelsen, C.K.; Stenby, E.H. (2001).
    Fluid Phase Equilibria 178, 1-16.
Chapman, S.; Cowling, T.G. (1970). The Mathematical Theory of Non-Uniform
    Gases. Cambridge University Press.

Units throughout:
    T  in K
    P  in Pa  (internally converted as needed)
    η  in μPa·s
"""

import numpy as np
from typing import Tuple


# Gas constant
R = 8.314  # J/(mol·K)


class FrictionTheoryViscosity:
    """
    Viscosity of a pure gas component via the friction theory model.

    The total viscosity is split into:
        η = η₀(T) + η_friction(T, P_r, P_a)

    where
        η_friction = κ_r(T)*P_r + κ_a(T)*P_a + κ_rr(T)*P_r²

    and P_r, P_a are the repulsive and attractive pressure contributions
    from the PR EOS at the given T, P.

    Parameters
    ----------
    component_name : str
        One of 'N2', 'CO2', 'H2'.
    eos : PengRobinsonEOS
        PR EOS instance for the same component.
    """

    # ------------------------------------------------------------------
    # Friction coefficient tables fitted to Quiñones-Cisneros et al.
    # For each component: (A_r, B_r, A_a, B_a, A_rr)
    # κ_r(T) = A_r + B_r*(Tc/T)        [Pa⁻¹·μPa·s]
    # κ_a(T) = A_a + B_a*(Tc/T)        [Pa⁻¹·μPa·s]
    # κ_rr(T) = A_rr                   [Pa⁻²·μPa·s]  (small correction)
    # ------------------------------------------------------------------
    _COEFF: dict = {
        # N₂: Tc = 126.19 K
        "N2": {
            "A_r":   1.73e-8,    # Pa⁻¹·μPa·s
            "B_r":   1.95e-9,    # Pa⁻¹·μPa·s
            "A_a":  -8.50e-9,    # Pa⁻¹·μPa·s
            "B_a":  -4.20e-9,    # Pa⁻¹·μPa·s
            "A_rr":  2.50e-18,   # Pa⁻²·μPa·s
        },
        # CO₂: Tc = 304.13 K
        "CO2": {
            "A_r":   2.19e-8,
            "B_r":   3.50e-9,
            "A_a":  -1.10e-8,
            "B_a":  -6.00e-9,
            "A_rr":  3.20e-18,
        },
        # H₂: Tc = 33.14 K
        "H2": {
            "A_r":   0.85e-8,
            "B_r":   0.60e-9,
            "A_a":  -3.00e-9,
            "B_a":  -1.50e-9,
            "A_rr":  0.80e-18,
        },
    }

    # Lennard-Jones parameters for Chapman-Enskog dilute gas viscosity
    # (sigma in Angstrom, eps/k in K)
    _LJ: dict = {
        "N2":  {"sigma": 3.798, "eps_k": 71.4,  "Tc": 126.19},
        "CO2": {"sigma": 3.941, "eps_k": 195.2, "Tc": 304.13},
        "H2":  {"sigma": 2.827, "eps_k": 59.7,  "Tc": 33.14},
    }

    def __init__(self, component_name: str, eos):
        if component_name not in self._COEFF:
            raise ValueError(
                f"Friction theory viscosity not parameterised for '{component_name}'. "
                f"Available: {list(self._COEFF.keys())}"
            )
        self.name = component_name
        self.eos = eos
        self._c = self._COEFF[component_name]
        self._lj = self._LJ[component_name]
        self.Tc = self._lj["Tc"]

    # ------------------------------------------------------------------
    # Dilute gas viscosity via Chapman-Enskog + power-law fit
    # ------------------------------------------------------------------

    @staticmethod
    def _omega_viscosity(T_star: float) -> float:
        """
        Neufeld collision integral Ω^(2,2)* for viscosity calculation.

        Neufeld, P.D.; Janzen, A.R.; Aziz, R.A. (1972). J. Chem. Phys. 57, 1100.

        Parameters
        ----------
        T_star : float
            Reduced temperature T* = kT/ε.

        Returns
        -------
        float
            Collision integral Ω^(2,2)* [-].
        """
        A, B, C, D, E, F, G, H = (1.16145, 0.14874, 0.52487, 0.7732,
                                   2.16178, 2.43787, 0.0, 0.0)
        return A / T_star**B + C / np.exp(D * T_star) + E / np.exp(F * T_star)

    def dilute_viscosity(self, T: float) -> float:
        """
        Dilute gas viscosity η₀(T) via Chapman-Enskog theory [μPa·s].

        η₀ = 2.6693e-6 * √(M*T) / (σ² * Ω^(2,2)*)

        where σ is the Lennard-Jones diameter [Å] and M is molar mass [g/mol].
        The pre-factor gives η in Pa·s; multiply by 1e6 for μPa·s.

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        float
            Dilute gas viscosity [μPa·s].
        """
        M = self.eos.MW          # g/mol
        sigma = self._lj["sigma"]  # Å
        eps_k = self._lj["eps_k"]  # K  (ε/k_B)

        T_star = T / eps_k
        Omega = self._omega_viscosity(T_star)

        # Chapman-Enskog: η₀ in Pa·s
        eta0_Pa_s = 2.6693e-6 * np.sqrt(M * T) / (sigma ** 2 * Omega)
        return eta0_Pa_s * 1e6   # convert to μPa·s

    # ------------------------------------------------------------------
    # Friction (residual) viscosity coefficients
    # ------------------------------------------------------------------

    def friction_coefficients(self, T: float) -> Tuple[float, float, float]:
        """
        Temperature-dependent friction coefficients.

        κ_r(T)  = A_r + B_r*(Tc/T)   [Pa⁻¹·μPa·s]
        κ_a(T)  = A_a + B_a*(Tc/T)   [Pa⁻¹·μPa·s]
        κ_rr(T) = A_rr                [Pa⁻²·μPa·s]

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        (kappa_r, kappa_a, kappa_rr) : Tuple[float, float, float]
        """
        c = self._c
        ratio = self.Tc / T
        kappa_r  = c["A_r"]  + c["B_r"]  * ratio
        kappa_a  = c["A_a"]  + c["B_a"]  * ratio
        kappa_rr = c["A_rr"]
        return kappa_r, kappa_a, kappa_rr

    # ------------------------------------------------------------------
    # Main viscosity calculation
    # ------------------------------------------------------------------

    def viscosity(self, T: float, P: float, phase: str = 'gas') -> float:
        """
        Total dynamic viscosity via friction theory [μPa·s].

        Procedure:
        1. Get molar volume V from PR EOS.
        2. Decompose into P_r and P_a.
        3. Apply friction model: η = η₀ + κ_r*P_r + κ_a*P_a + κ_rr*P_r²

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        phase : str
            'gas' or 'liquid'.

        Returns
        -------
        float
            Viscosity [μPa·s].
        """
        # Dilute gas contribution
        eta0 = self.dilute_viscosity(T)

        # Pressure decomposition using raw (un-shifted) volume
        P_r, P_a = self.eos.pressure_contributions_at_TP(T, P, phase)

        # Friction coefficients at T
        kappa_r, kappa_a, kappa_rr = self.friction_coefficients(T)

        # Friction (residual) contribution
        eta_friction = kappa_r * P_r + kappa_a * P_a + kappa_rr * P_r ** 2

        eta_total = eta0 + eta_friction
        # Viscosity must be positive; clamp to dilute value at worst
        return max(eta_total, 0.5 * eta0)

    def viscosity_Pa_s(self, T: float, P: float, phase: str = 'gas') -> float:
        """
        Convenience wrapper returning viscosity in Pa·s.

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        phase : str

        Returns
        -------
        float
            Viscosity [Pa·s].
        """
        return self.viscosity(T, P, phase) * 1e-6


class GasMixtureViscosity:
    """
    Viscosity of a N₂-CO₂-H₂ gas mixture via Wilke mixing rule.

    Wilke, C.R. (1950). J. Chem. Phys. 18, 517.

    Parameters
    ----------
    component_names : list of str
        e.g. ['N2', 'CO2', 'H2']
    eos_list : list of PengRobinsonEOS
        One EOS per component, same ordering.
    """

    def __init__(self, component_names: list, eos_list: list):
        self.names = component_names
        self._visc_models = [
            FrictionTheoryViscosity(n, eos)
            for n, eos in zip(component_names, eos_list)
        ]
        self._MW = np.array([eos.MW for eos in eos_list])

    def viscosity(self, T: float, P: float,
                  mole_fractions: np.ndarray,
                  phase: str = 'gas') -> float:
        """
        Mixture viscosity via Wilke mixing rule [μPa·s].

        η_mix = Σᵢ xᵢ ηᵢ / Σⱼ xⱼ Φᵢⱼ
        Φᵢⱼ = [1 + (ηᵢ/ηⱼ)^(1/2) * (Mⱼ/Mᵢ)^(1/4)]² / √(8*(1+Mᵢ/Mⱼ))

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        mole_fractions : np.ndarray
        phase : str

        Returns
        -------
        float
            Mixture viscosity [μPa·s].
        """
        n = len(self.names)
        x = np.asarray(mole_fractions, dtype=float)
        eta = np.array([m.viscosity(T, P, phase) for m in self._visc_models])
        MW = self._MW

        # Wilke interaction coefficients Φᵢⱼ
        Phi = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                num = (1.0 + np.sqrt(eta[i] / eta[j]) * (MW[j] / MW[i]) ** 0.25) ** 2
                denom = np.sqrt(8.0 * (1.0 + MW[i] / MW[j]))
                Phi[i, j] = num / denom

        eta_mix = 0.0
        for i in range(n):
            denom_i = sum(x[j] * Phi[i, j] for j in range(n))
            eta_mix += x[i] * eta[i] / denom_i

        return eta_mix
