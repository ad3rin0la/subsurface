"""
thermophysics/eos.py
====================
Peng-Robinson Equation of State (PR EOS) implementation for N2Bio.

References
----------
Peng, D.-Y.; Robinson, D.B. (1976). Ind. Eng. Chem. Fundam. 15, 59-64.
Peneloux, A. et al. (1982). Fluid Phase Equilibria 8, 7-23.  (volume shift)

All pressures in Pa, temperatures in K, volumes in m³/mol internally.
"""

import numpy as np
from typing import Optional, List, Tuple

# Gas constant
R = 8.314  # J/(mol·K)


class PengRobinsonEOS:
    """
    Peng-Robinson EOS for a pure component.

    Parameters
    ----------
    component : Component
        Component dataclass containing Tc, Pc, omega, MW, volume_shift.
    """

    def __init__(self, component):
        self.comp = component
        self.Tc = component.Tc        # K
        self.Pc = component.Pc        # Pa
        self.omega = component.omega  # acentric factor
        self.MW = component.MW        # g/mol
        self.c = component.volume_shift   # m³/mol  (Peneloux shift)

        # Pre-compute temperature-independent part of a and b
        self._kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        self._a0 = 0.45724 * R ** 2 * self.Tc ** 2 / self.Pc   # J²·s²/kg/mol  = Pa·m⁶/mol²
        self._b = 0.07780 * R * self.Tc / self.Pc               # m³/mol

    # ------------------------------------------------------------------
    # Core EOS parameters
    # ------------------------------------------------------------------

    def alpha(self, T: float) -> float:
        """
        Temperature-dependent alpha function.

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        float
            α(T) [-].
        """
        Tr = T / self.Tc
        return (1.0 + self._kappa * (1.0 - np.sqrt(Tr))) ** 2

    def a_param(self, T: float) -> float:
        """
        Temperature-dependent attractive parameter a(T) [Pa·m⁶/mol²].

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        float
            a(T) [Pa·m⁶/mol²].
        """
        return self._a0 * self.alpha(T)

    @property
    def b_param(self) -> float:
        """
        Repulsive co-volume b [m³/mol].  Temperature-independent.
        """
        return self._b

    # ------------------------------------------------------------------
    # Dimensionless A, B parameters (needed for cubic in Z)
    # ------------------------------------------------------------------

    def _AB(self, T: float, P: float) -> Tuple[float, float]:
        """
        Compute dimensionless A and B.

        A = a(T)*P / (R*T)²
        B = b*P / (R*T)

        Parameters
        ----------
        T : float  – temperature [K]
        P : float  – pressure [Pa]

        Returns
        -------
        (A, B) : Tuple[float, float]
        """
        a = self.a_param(T)
        b = self._b
        RT = R * T
        A = a * P / RT ** 2
        B = b * P / RT
        return A, B

    # ------------------------------------------------------------------
    # Cubic Z-factor solve
    # ------------------------------------------------------------------

    def _solve_cubic_Z(self, T: float, P: float) -> np.ndarray:
        """
        Solve the PR EOS cubic in Z:
            Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]

        Returns
        -------
        np.ndarray
            Array of real, positive roots.
        """
        A, B = self._AB(T, P)
        # Coefficients for Z³ + c2*Z² + c1*Z + c0 = 0
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B ** 2 - 2.0 * B
        c0 = -(A * B - B ** 2 - B ** 3)
        roots = np.roots([1.0, c2, c1, c0])
        # Keep real, positive roots
        real_roots = np.array([r.real for r in roots if abs(r.imag) < 1e-8 and r.real > B])
        return np.sort(real_roots)

    def Z_factor(self, T: float, P: float, phase: str = 'gas') -> float:
        """
        Compressibility factor Z = PV/(RT).

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        phase : str – 'gas' (largest root) or 'liquid' (smallest root)

        Returns
        -------
        float
            Compressibility factor Z [-].
        """
        roots = self._solve_cubic_Z(T, P)
        if len(roots) == 0:
            raise ValueError(f"No real positive Z roots at T={T} K, P={P/1e5:.2f} bar")
        if phase == 'gas':
            return float(roots[-1])   # largest root → gas
        else:
            return float(roots[0])    # smallest root → liquid

    # ------------------------------------------------------------------
    # Molar volume and density
    # ------------------------------------------------------------------

    def volume(self, T: float, P: float, phase: str = 'gas') -> float:
        """
        Molar volume with optional Peneloux volume shift [m³/mol].

        V_shifted = Z*R*T/P + c
        (c is typically slightly negative for gases, giving a slightly smaller
        corrected volume; the convention here matches Peneloux 1982 where
        the correction is added to the raw PR volume)

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        phase : str – 'gas' or 'liquid'

        Returns
        -------
        float
            Molar volume [m³/mol].
        """
        Z = self.Z_factor(T, P, phase)
        V_raw = Z * R * T / P
        return V_raw + self.c   # Peneloux correction (c is negative for denser correction)

    def volume_raw(self, T: float, P: float, phase: str = 'gas') -> float:
        """
        Raw (uncorrected) molar volume Z*R*T/P [m³/mol].

        Used internally for pressure decomposition (P_r + P_a).

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        phase : str – 'gas' or 'liquid'

        Returns
        -------
        float
            Raw molar volume [m³/mol].
        """
        Z = self.Z_factor(T, P, phase)
        return Z * R * T / P

    def density(self, T: float, P: float, phase: str = 'gas') -> float:
        """
        Mass density [kg/m³].

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        phase : str – 'gas' or 'liquid'

        Returns
        -------
        float
            Density [kg/m³].
        """
        V = self.volume(T, P, phase)
        # MW in g/mol → kg/mol by dividing by 1000
        return (self.MW * 1e-3) / V

    # ------------------------------------------------------------------
    # Repulsive and attractive pressure contributions (for friction theory)
    # ------------------------------------------------------------------

    def pressure_contributions(self, T: float, V: float) -> Tuple[float, float]:
        """
        Decompose total pressure into repulsive (P_r) and attractive (P_a).

        The input V should be the raw (un-shifted) molar volume from Z*R*T/P.
        When called with a shifted volume (from self.volume()), the result
        will not exactly match P; use volume_raw() for thermodynamically
        consistent decomposition.

        P_r = R*T / (V - b)
        P_a = -a(T) / [V*(V+b) + b*(V-b)]
        P   = P_r + P_a

        Parameters
        ----------
        T : float – temperature [K]
        V : float – raw molar volume [m³/mol]  (Z*R*T/P, without Peneloux shift)

        Returns
        -------
        (P_r, P_a) : Tuple[float, float]  both in Pa
        """
        a = self.a_param(T)
        b = self._b
        P_r = R * T / (V - b)
        P_a = -a / (V * (V + b) + b * (V - b))
        return P_r, P_a

    def pressure_contributions_at_TP(self, T: float, P: float,
                                      phase: str = 'gas') -> Tuple[float, float]:
        """
        Pressure decomposition at given T, P using the raw (unshifted) volume.

        Convenience wrapper that internally calls volume_raw().

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        phase : str

        Returns
        -------
        (P_r, P_a) : Tuple[float, float]  both in Pa, sum = P
        """
        V_raw = self.volume_raw(T, P, phase)
        return self.pressure_contributions(T, V_raw)

    def pressure(self, T: float, V: float) -> float:
        """
        Compute pressure from T and raw molar volume V [Pa].

        V should be the raw (un-shifted) volume.

        Parameters
        ----------
        T : float – temperature [K]
        V : float – raw molar volume [m³/mol]

        Returns
        -------
        float
            Pressure [Pa].
        """
        P_r, P_a = self.pressure_contributions(T, V)
        return P_r + P_a

    # ------------------------------------------------------------------
    # Fugacity coefficient
    # ------------------------------------------------------------------

    def fugacity_coefficient(self, T: float, P: float, phase: str = 'gas') -> float:
        """
        Fugacity coefficient ln(φ) from PR EOS departure functions.

        ln(φ) = (Z-1) - ln(Z-B) - A/(2√2·B) * ln[(Z+(1+√2)B)/(Z+(1-√2)B)]

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        phase : str – 'gas' or 'liquid'

        Returns
        -------
        float
            Fugacity coefficient φ [-].
        """
        Z = self.Z_factor(T, P, phase)
        A, B = self._AB(T, P)
        sqrt2 = np.sqrt(2.0)
        ln_phi = (Z - 1.0) - np.log(Z - B) - (A / (2.0 * sqrt2 * B)) * np.log(
            (Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B)
        )
        return np.exp(ln_phi)

    # ------------------------------------------------------------------
    # Departure enthalpy
    # ------------------------------------------------------------------

    def departure_enthalpy(self, T: float, P: float, phase: str = 'gas') -> float:
        """
        Residual (departure) enthalpy H_dep = H - H_ig [J/mol].

        H_dep = RT(Z-1) + [T*da/dT - a] / (2√2·b) * ln[(Z+(1+√2)B)/(Z+(1-√2)B)]

        where da/dT is computed analytically from the alpha function.

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        phase : str – 'gas' or 'liquid'

        Returns
        -------
        float
            Departure enthalpy [J/mol].
        """
        Z = self.Z_factor(T, P, phase)
        A, B = self._AB(T, P)
        sqrt2 = np.sqrt(2.0)

        # Analytical da/dT
        Tr = T / self.Tc
        sqrt_Tr = np.sqrt(Tr)
        alpha_val = self.alpha(T)
        # d(alpha)/dT = -kappa/Tc * sqrt(1/Tr) * (1 + kappa*(1-sqrt(Tr)))
        #             = -kappa * (1 + kappa*(1-sqrt(Tr))) / (Tc * sqrt(Tr))
        d_alpha_dT = -self._kappa * (1.0 + self._kappa * (1.0 - sqrt_Tr)) / (self.Tc * sqrt_Tr)
        da_dT = self._a0 * d_alpha_dT   # Pa·m⁶/(mol²·K)

        b = self._b
        log_term = np.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))

        H_dep = (R * T * (Z - 1.0)
                 + (T * da_dT - self.a_param(T)) / (2.0 * sqrt2 * b) * log_term)
        return H_dep

    # ------------------------------------------------------------------
    # Mixture PR EOS helper methods (van der Waals mixing rules)
    # ------------------------------------------------------------------

    @staticmethod
    def mixture_a(T: float, compositions: np.ndarray,
                  components: list,
                  kij: Optional[np.ndarray] = None) -> float:
        """
        Mixture attractive parameter using van der Waals mixing rules.

        a_mix = Σᵢ Σⱼ xᵢ xⱼ √(aᵢ aⱼ) (1 - kᵢⱼ)

        Parameters
        ----------
        T : float
            Temperature [K].
        compositions : np.ndarray
            Mole fractions [n_comp].
        components : list of Component
            Component objects (must have same ordering as compositions).
        kij : np.ndarray, optional
            Binary interaction parameters [n_comp × n_comp].  If None, all
            kij = 0.

        Returns
        -------
        float
            Mixture a(T) [Pa·m⁶/mol²].
        """
        n = len(components)
        if kij is None:
            kij = np.zeros((n, n))
        a_i = np.array([
            PengRobinsonEOS(comp).a_param(T) for comp in components
        ])
        a_mix = 0.0
        for i in range(n):
            for j in range(n):
                a_ij = np.sqrt(a_i[i] * a_i[j]) * (1.0 - kij[i, j])
                a_mix += compositions[i] * compositions[j] * a_ij
        return a_mix

    @staticmethod
    def mixture_b(compositions: np.ndarray, components: list) -> float:
        """
        Mixture co-volume using linear mixing rule.

        b_mix = Σᵢ xᵢ bᵢ

        Parameters
        ----------
        compositions : np.ndarray
            Mole fractions.
        components : list of Component

        Returns
        -------
        float
            Mixture b [m³/mol].
        """
        b_i = np.array([PengRobinsonEOS(comp).b_param for comp in components])
        return float(np.dot(compositions, b_i))


class MixturePREOS:
    """
    Peng-Robinson EOS for N₂-CO₂-H₂ gas mixtures.

    Uses van der Waals mixing rules with optional binary interaction
    parameters (kij).  Provides Z-factor, molar volume, density and
    fugacity coefficients for the gas mixture.

    Parameters
    ----------
    components : list of Component
        Gas-phase components.
    kij : np.ndarray, optional
        Binary interaction matrix [n × n].  Defaults to zero matrix.
    """

    def __init__(self, components: list, kij: Optional[np.ndarray] = None):
        self.components = components
        self.n = len(components)
        self.kij = kij if kij is not None else np.zeros((self.n, self.n))
        self._eos_list = [PengRobinsonEOS(c) for c in components]

    def _mixture_AB(self, T: float, P: float,
                    compositions: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute mixture A, B and the underlying a_mix, b_mix.

        Returns
        -------
        (A, B, a_mix, b_mix)
        """
        a_mix = PengRobinsonEOS.mixture_a(T, compositions, self.components, self.kij)
        b_mix = PengRobinsonEOS.mixture_b(compositions, self.components)
        RT = R * T
        A = a_mix * P / RT ** 2
        B = b_mix * P / RT
        return A, B, a_mix, b_mix

    def _solve_cubic_Z(self, T: float, P: float,
                       compositions: np.ndarray) -> np.ndarray:
        """Solve mixture PR cubic for real positive Z roots."""
        A, B, _, _ = self._mixture_AB(T, P, compositions)
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B ** 2 - 2.0 * B
        c0 = -(A * B - B ** 2 - B ** 3)
        roots = np.roots([1.0, c2, c1, c0])
        real_roots = np.array([r.real for r in roots if abs(r.imag) < 1e-8 and r.real > B])
        return np.sort(real_roots)

    def Z_factor(self, T: float, P: float, compositions: np.ndarray,
                 phase: str = 'gas') -> float:
        """
        Mixture compressibility factor.

        Parameters
        ----------
        T : float – temperature [K]
        P : float – pressure [Pa]
        compositions : np.ndarray – mole fractions (must sum to 1)
        phase : str – 'gas' or 'liquid'

        Returns
        -------
        float
        """
        roots = self._solve_cubic_Z(T, P, compositions)
        if len(roots) == 0:
            raise ValueError(f"No real Z roots for mixture at T={T} K, P={P/1e5:.2f} bar")
        return float(roots[-1]) if phase == 'gas' else float(roots[0])

    def volume(self, T: float, P: float, compositions: np.ndarray,
               phase: str = 'gas') -> float:
        """
        Mixture molar volume [m³/mol].
        """
        Z = self.Z_factor(T, P, compositions, phase)
        return Z * R * T / P

    def density(self, T: float, P: float, compositions: np.ndarray,
                phase: str = 'gas') -> float:
        """
        Mixture mass density [kg/m³].

        Uses mole-fraction-weighted average molecular weight.
        """
        MW_mix = float(np.dot(compositions, [c.MW for c in self.components]))  # g/mol
        V = self.volume(T, P, compositions, phase)
        return (MW_mix * 1e-3) / V   # kg/m³

    def fugacity_coefficients(self, T: float, P: float,
                               compositions: np.ndarray,
                               phase: str = 'gas') -> np.ndarray:
        """
        Component fugacity coefficients in the mixture.

        ln(φᵢ) = bᵢ/b_mix*(Z-1) - ln(Z-B)
                 - A/(2√2·B) * (2·Σⱼ xⱼ√(aᵢaⱼ)(1-kᵢⱼ)/a_mix - bᵢ/b_mix)
                 * ln[(Z+(1+√2)B)/(Z+(1-√2)B)]

        Parameters
        ----------
        T, P, compositions, phase : as above

        Returns
        -------
        np.ndarray
            Array of φᵢ for each component.
        """
        A, B, a_mix, b_mix = self._mixture_AB(T, P, compositions)
        Z = self.Z_factor(T, P, compositions, phase)
        sqrt2 = np.sqrt(2.0)
        log_term = np.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))

        a_i = np.array([eos.a_param(T) for eos in self._eos_list])
        b_i = np.array([eos.b_param for eos in self._eos_list])

        phi = np.zeros(self.n)
        for i in range(self.n):
            # Sum over j: 2 Σⱼ xⱼ √(aᵢaⱼ)(1-kᵢⱼ) / a_mix
            cross_sum = 0.0
            for j in range(self.n):
                cross_sum += compositions[j] * np.sqrt(a_i[i] * a_i[j]) * (
                    1.0 - self.kij[i, j])
            cross_sum *= 2.0 / a_mix

            ln_phi_i = (b_i[i] / b_mix * (Z - 1.0)
                        - np.log(Z - B)
                        - A / (2.0 * sqrt2 * B) * (cross_sum - b_i[i] / b_mix) * log_term)
            phi[i] = np.exp(ln_phi_i)

        return phi
