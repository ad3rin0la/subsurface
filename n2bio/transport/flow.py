"""
transport/flow.py
=================
1D two-phase (gas-liquid) flow on an IFDM grid using Darcy's law.

Tracks a gas phase (N₂/H₂ mixture) and a liquid brine phase.
Relative permeabilities follow the Brooks-Corey / Corey model.

References
----------
Brooks, R.H.; Corey, A.T. (1964). Hydrology Paper No. 3. Colorado State Univ.
Darcy, H. (1856). Les fontaines publiques de la ville de Dijon. Dalmont, Paris.
"""

import numpy as np
from typing import Optional
from .grid import IFDMGrid1D
from ..kinetics.parameters import KineticParameters, ReservoirConditions


class MultiphaseFlow1D:
    """
    1D two-phase Darcy flow on an IFDM grid.

    Tracks gas-phase saturation S_g and liquid (brine) saturation S_l = 1 - S_g.
    Relative permeabilities use the Corey model.

    Parameters
    ----------
    grid : IFDMGrid1D
        Spatial discretisation.
    params : KineticParameters
        Kinetic parameters (for biological source terms).
    conditions : ReservoirConditions
        Reservoir physical conditions (permeability, porosity).
    S_gr : float
        Residual gas saturation [-].  Default 0.05.
    S_lr : float
        Residual liquid saturation [-].  Default 0.20.
    """

    def __init__(self,
                 grid: IFDMGrid1D,
                 params: KineticParameters,
                 conditions: ReservoirConditions,
                 S_gr: float = 0.05,
                 S_lr: float = 0.20):
        self.grid  = grid
        self.p     = params
        self.cond  = conditions
        self.n     = grid.n_cells
        self.S_gr  = S_gr
        self.S_lr  = S_lr

    # ------------------------------------------------------------------
    # Relative permeability models (Brooks-Corey / Corey)
    # ------------------------------------------------------------------

    def relative_permeability_gas(self, S_g: np.ndarray) -> np.ndarray:
        """
        Gas-phase relative permeability (Corey model) [-].

        k_rg = S_e² * (1 - (1 - S_e)²)   (simplified Corey for gas)

        where S_e = (S_g - S_gr) / (1 - S_gr - S_lr)

        Parameters
        ----------
        S_g : np.ndarray  Gas saturation per cell [-].

        Returns
        -------
        np.ndarray  k_rg per cell [-].
        """
        S_g = np.asarray(S_g, dtype=float)
        S_e = (S_g - self.S_gr) / (1.0 - self.S_gr - self.S_lr)
        S_e = np.clip(S_e, 0.0, 1.0)
        return S_e ** 2 * (1.0 - (1.0 - S_e) ** 2)

    def relative_permeability_liquid(self, S_l: np.ndarray) -> np.ndarray:
        """
        Liquid-phase relative permeability (Corey model) [-].

        k_rl = S_e⁴   (Corey with λ = 2)

        where S_e = (S_l - S_lr) / (1 - S_lr)

        Parameters
        ----------
        S_l : np.ndarray  Liquid saturation per cell [-].

        Returns
        -------
        np.ndarray  k_rl per cell [-].
        """
        S_l = np.asarray(S_l, dtype=float)
        S_e = (S_l - self.S_lr) / (1.0 - self.S_lr)
        S_e = np.clip(S_e, 0.0, 1.0)
        return S_e ** 4

    def capillary_pressure(self, S_l: np.ndarray,
                           entry_pressure: float = 1.0e4) -> np.ndarray:
        """
        Capillary pressure P_c = P_g - P_l [Pa] (Brooks-Corey model).

        P_c = P_e * S_e^(-1/λ)

        with λ = 2 (Corey), P_e = entry pressure.

        Parameters
        ----------
        S_l : np.ndarray     Liquid saturation [-].
        entry_pressure : float  Entry pressure [Pa].

        Returns
        -------
        np.ndarray  Capillary pressure per cell [Pa].
        """
        S_l = np.asarray(S_l, dtype=float)
        S_e = (S_l - self.S_lr) / (1.0 - self.S_lr)
        S_e = np.clip(S_e, 1.0e-6, 1.0)
        lam = 2.0   # Corey pore-size index
        return entry_pressure * S_e ** (-1.0 / lam)

    # ------------------------------------------------------------------
    # Darcy flux
    # ------------------------------------------------------------------

    def darcy_flux(self, P: np.ndarray, S_g: np.ndarray,
                   rho: np.ndarray, mu: np.ndarray,
                   phase: str = 'gas') -> np.ndarray:
        """
        Darcy flux at each cell interface [m³/s].

        Q_ij = T_ij * k_r * ΔP_ij

        Uses upstream-weighted relative permeability (upwind scheme).

        Parameters
        ----------
        P : np.ndarray    Phase pressure per cell [Pa].  Length n_cells.
        S_g : np.ndarray  Gas saturation per cell [-].   Length n_cells.
        rho : np.ndarray  Phase density per cell [kg/m³].
        mu : np.ndarray   Phase viscosity per cell [Pa·s].
        phase : str       'gas' or 'liquid'.

        Returns
        -------
        np.ndarray  Volumetric flux per connection [m³/s].
                    Positive = flow from cell i to cell j (left to right).
        """
        P   = np.asarray(P,   dtype=float)
        S_g = np.asarray(S_g, dtype=float)
        mu  = np.asarray(mu,  dtype=float)
        k   = self.cond.permeability   # m²
        n   = self.n
        Q   = np.zeros(n - 1)

        for idx, (i, j) in enumerate(self.grid.connections):
            dP  = P[j] - P[i]
            d   = self.grid.d_ij[idx]
            A   = self.grid.interface_areas[idx]
            mu_avg = 0.5 * (mu[i] + mu[j])

            # Upwind saturations for relative permeability
            if dP >= 0.0:
                # flow from i to j → use upwind cell i
                S_g_up  = S_g[i]
                S_l_up  = 1.0 - S_g[i]
            else:
                S_g_up  = S_g[j]
                S_l_up  = 1.0 - S_g[j]

            if phase == 'gas':
                kr = float(self.relative_permeability_gas(np.array([S_g_up]))[0])
            else:
                kr = float(self.relative_permeability_liquid(np.array([S_l_up]))[0])

            T_ij = A * k * kr / (mu_avg * d)   # m³/(Pa·s)
            Q[idx] = T_ij * dP                  # m³/s

        return Q

    # ------------------------------------------------------------------
    # Mass flux (kg/s per interface)
    # ------------------------------------------------------------------

    def mass_flux(self, P: np.ndarray, S_g: np.ndarray,
                  rho: np.ndarray, mu: np.ndarray,
                  phase: str = 'gas') -> np.ndarray:
        """
        Mass flux at each interface [kg/s].

        F_ij = Q_ij * ρ_ij  (ρ averaged at interface)

        Parameters
        ----------
        P, S_g, rho, mu, phase : as for darcy_flux.

        Returns
        -------
        np.ndarray  Mass flux [kg/s].
        """
        Q   = self.darcy_flux(P, S_g, rho, mu, phase)
        rho = np.asarray(rho, dtype=float)
        rho_ij = 0.5 * (rho[:-1] + rho[1:])
        return Q * rho_ij

    # ------------------------------------------------------------------
    # Saturation update (explicit)
    # ------------------------------------------------------------------

    def saturation_update_explicit(self, S_g: np.ndarray,
                                   Q_g: np.ndarray,
                                   Q_l: np.ndarray,
                                   source_g: np.ndarray,
                                   dt: float) -> np.ndarray:
        """
        Explicit (forward Euler) update of gas saturation.

        φ * V * dS_g/dt = -(flux_out - flux_in)_gas + Q_source_g

        Parameters
        ----------
        S_g : np.ndarray    Current gas saturation [-].
        Q_g : np.ndarray    Gas Darcy flux at each connection [m³/s].
        Q_l : np.ndarray    Liquid Darcy flux (not needed for saturation).
        source_g : np.ndarray  Gas source per cell [m³/s].
        dt : float          Time step [s].

        Returns
        -------
        np.ndarray  Updated gas saturation [-].
        """
        phi = self.cond.porosity
        V   = self.grid.volumes

        dS_g = np.zeros(self.n)
        for idx, (i, j) in enumerate(self.grid.connections):
            # Flux from i → j: positive Q_g means gas leaves cell i, enters j
            dS_g[i] -= Q_g[idx] / (phi * V[i])
            dS_g[j] += Q_g[idx] / (phi * V[j])

        # Source / sink terms
        dS_g += source_g / (phi * V)

        S_g_new = S_g + dt * dS_g
        return np.clip(S_g_new, 0.0, 1.0)

    # ------------------------------------------------------------------
    # CFL time-step criterion
    # ------------------------------------------------------------------

    def CFL_timestep(self, Q_g: np.ndarray, S_g: np.ndarray,
                     CFL_max: float = 0.5) -> float:
        """
        Estimate stable explicit time step from CFL condition [s].

        Δt_CFL = CFL_max * min_i(φ * V_i / |Q_i_net|)

        Parameters
        ----------
        Q_g : np.ndarray  Gas flux at interfaces [m³/s].
        S_g : np.ndarray  Gas saturation (unused here, for signature).
        CFL_max : float   Maximum CFL number (default 0.5).

        Returns
        -------
        float  Suggested time step [s].
        """
        phi = self.cond.porosity
        pore_vols = self.grid.volumes * phi

        # Net flux magnitude per cell
        net_flux = np.zeros(self.n)
        for idx, (i, j) in enumerate(self.grid.connections):
            net_flux[i] += abs(Q_g[idx])
            net_flux[j] += abs(Q_g[idx])

        # Avoid division by zero
        net_flux = np.maximum(net_flux, 1.0e-30)
        dt_cells = pore_vols / net_flux
        return float(CFL_max * np.min(dt_cells))

    def __repr__(self) -> str:
        return (f"MultiphaseFlow1D(n_cells={self.n}, "
                f"k={self.cond.permeability*1e15:.1f} mD, "
                f"phi={self.cond.porosity:.3f}, "
                f"S_gr={self.S_gr}, S_lr={self.S_lr})")
