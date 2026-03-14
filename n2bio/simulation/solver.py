"""
simulation/solver.py
====================
ODE solver for the N2Bio batch-reactor (well-mixed single-cell) system.

The stiff ODE system integrates 7 state variables simultaneously:
    y[0]  N2_aq    – dissolved N₂ [mol/L]
    y[1]  H2_aq    – dissolved H₂ [mol/L]
    y[2]  NH4_aq   – ammonium [mol/L]
    y[3]  HC_ali   – aliphatic HC pool [mol/L]
    y[4]  HC_arom  – aromatic HC pool [mol/L]
    y[5]  X        – biomass [g VSS/L]
    y[6]  cum_H2   – cumulative H₂ produced [mol/L]

Uses scipy.integrate.solve_ivp with the BDF method (robust for stiff
biological systems with multiple timescales).
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, List

from ..kinetics.parameters import KineticParameters, ReservoirConditions
from ..kinetics.monod import MonodKinetics
from ..kinetics.nitrogenase import NitrogenaseKinetics
from ..kinetics.hc_degradation import HCDegradationModel
from ..geochemistry.speciation import NH3Speciation
from .state import BiologicalState
from .lifecycle import LifecycleClassifier, LifecyclePhase


class N2BioBatchSolver:
    """
    Batch (well-mixed) reactor solver for the N2Bio ODE system.

    The reactor represents a single reservoir grid cell with no spatial
    gradients.  Gas-phase partial pressures (N₂, H₂) are treated as
    quasi-static boundary conditions updated from the initial state; the
    dissolved concentrations evolve according to the biological ODEs.

    Parameters
    ----------
    params : KineticParameters
        Kinetic parameter dataclass.
    conditions : ReservoirConditions
        Reservoir physical and geochemical conditions.
    initial_state : BiologicalState
        Initial values of all state variables.
    """

    def __init__(self,
                 params: KineticParameters,
                 conditions: ReservoirConditions,
                 initial_state: BiologicalState):
        self.p = params
        self.cond = conditions
        self.s0 = initial_state

        # Kinetic sub-models
        self.monod       = MonodKinetics(params)
        self.nitrogenase = NitrogenaseKinetics(params)
        self.hc_model    = HCDegradationModel(params)
        self.nh3_spec    = NH3Speciation()
        self.lifecycle   = LifecycleClassifier(params)

        # Cache initial values for lifecycle classification
        self._HC_ali_initial = initial_state.HC_ali
        self._X_max_seen     = initial_state.X

        # H₂ stripping / degassing rate constant [h⁻¹]
        # Represents equilibration between dissolved and gas phase
        self._k_strip_H2 = 0.1   # h⁻¹

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def odes(self, t: float, y: np.ndarray) -> list:
        """
        Right-hand side of the N2Bio ODE system.

        State vector y:
            [0] N2_aq  [mol/L]
            [1] H2_aq  [mol/L]
            [2] NH4_aq [mol/L]
            [3] HC_ali [mol/L]
            [4] HC_arom[mol/L]
            [5] X      [g VSS/L]
            [6] cum_H2 [mol/L] (monotone integral)

        Parameters
        ----------
        t : float         Current time [h].
        y : np.ndarray    Current state vector.

        Returns
        -------
        list
            Time derivatives dy/dt.
        """
        # Unpack and clamp non-negative
        N2_aq   = max(y[0], 0.0)
        H2_aq   = max(y[1], 0.0)
        NH4_aq  = max(y[2], 0.0)
        HC_ali  = max(y[3], 0.0)
        HC_arom = max(y[4], 0.0)
        X       = max(y[5], 0.0)
        cum_H2  = y[6]

        # ------ H₂ partial pressure estimate [atm] ------
        # Dissolved H₂ in equilibrium with gas phase.
        # Use a simple proportionality: P_H2 ~ H2_aq * kH_H2_eff
        # kH for H₂ at 85°C, 0.5 mol/kg NaCl ≈ 15 L·bar/mol (rough)
        kH_H2_eff = 15.0   # L·bar/mol
        P_H2_bar = H2_aq * kH_H2_eff
        P_H2_atm = P_H2_bar / 1.01325   # bar → atm

        # ------ Monod growth rate ------
        mu = self.monod.growth_rate(HC_ali, NH4_aq, N2_aq)

        # ------ Nitrogenase: H₂ + NH₄⁺ production ------
        r_nit_H2 = self.nitrogenase.H2_production_rate(
            N2_aq, P_H2_atm, HC_arom, X
        )
        r_NH4_prod = self.nitrogenase.NH4_production_rate(
            N2_aq, P_H2_atm, HC_arom, X
        )

        # N₂ consumed (1 mol per mol NH₄⁺ pair, i.e. r_NH4/2)
        r_N2_consumed = r_NH4_prod / 2.0   # mol/L/h

        # ------ HC degradation ------
        dHC_ali  = self.hc_model.HC_ali_rate(HC_ali, X)
        dHC_arom = self.hc_model.HC_arom_rate(HC_arom, X)
        r_ferm_H2 = self.hc_model.fermentation_H2_rate(HC_ali, X)

        # ------ Total H₂ production ------
        r_H2_total = r_nit_H2 + r_ferm_H2

        # ------ N₂: consumed by nitrogenase ------
        dN2 = -r_N2_consumed

        # ------ H₂: produced, minus gas-phase stripping ------
        # Equilibrium dissolved concentration at current gas pressure
        P_N2_bar    = self.s0.P_N2    # treat as boundary (slowly depleting)
        kH_H2_eq    = 15.0            # L·bar/mol (same as above)
        H2_eq       = P_H2_bar / kH_H2_eq   # current equilibrium value
        dH2 = r_H2_total - self._k_strip_H2 * H2_aq

        # ------ NH₄⁺: produced by nitrogenase, consumed for biosynthesis ------
        # N demand for biomass growth: mol N / L / h
        # = (mu * X / Y) * Y_N / MW_N
        # Y  = g VSS / g COD
        # Y_N = g N / g VSS
        # MW_N = 14 g/mol N
        N_demand = (mu * X / self.p.Y) * (self.p.Y_N / 14.0)   # mol N/L/h
        dNH4 = r_NH4_prod - N_demand

        # ------ Biomass ------
        dX = mu * X - self.p.b_decay * X

        # ------ Cumulative H₂ ------
        dcum_H2 = r_H2_total

        return [dN2, dH2, dNH4, dHC_ali, dHC_arom, dX, dcum_H2]

    # ------------------------------------------------------------------
    # Solver entry points
    # ------------------------------------------------------------------

    def solve(self, t_end_hours: float,
              n_points: int = 500) -> object:
        """
        Solve the ODE system over [0, t_end_hours].

        Uses scipy's BDF integrator (good for stiff biological ODEs).

        Parameters
        ----------
        t_end_hours : float
            Simulation end time [h].
        n_points : int
            Number of output time points.

        Returns
        -------
        scipy.integrate.OdeSolution
            Solution object with attributes .t (time) and .y (state).
        """
        s = self.s0
        y0 = [s.N2_aq, s.H2_aq, s.NH4_aq, s.HC_ali, s.HC_arom, s.X, s.cum_H2]
        t_span = (0.0, t_end_hours)
        t_eval = np.linspace(0.0, t_end_hours, n_points)

        sol = solve_ivp(
            self.odes,
            t_span,
            y0,
            method='BDF',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-10,
            dense_output=True,
        )

        if not sol.success:
            import warnings
            warnings.warn(
                f"ODE solver did not converge: {sol.message}", RuntimeWarning
            )

        return sol

    def solve_with_lifecycle(self, t_end_hours: float,
                              n_points: int = 500) -> Tuple[object, List[LifecyclePhase]]:
        """
        Solve the ODE system and annotate each time step with a lifecycle phase.

        Parameters
        ----------
        t_end_hours : float
            Simulation end time [h].
        n_points : int
            Number of output time points.

        Returns
        -------
        (sol, phases) : Tuple
            sol    – scipy OdeSolution
            phases – list of LifecyclePhase, one per time point
        """
        sol = self.solve(t_end_hours, n_points)

        HC_ali_initial = self.s0.HC_ali
        # Determine peak biomass from solution
        X_trajectory = np.maximum(sol.y[5], 0.0)
        X_max = float(np.max(X_trajectory)) if len(X_trajectory) > 0 else self.s0.X

        phases: List[LifecyclePhase] = []
        for i in range(len(sol.t)):
            state = BiologicalState(
                N2_aq   = max(float(sol.y[0][i]), 0.0),
                CO2_aq  = self.s0.CO2_aq,
                H2_aq   = max(float(sol.y[1][i]), 0.0),
                NH4_aq  = max(float(sol.y[2][i]), 0.0),
                HC_ali  = max(float(sol.y[3][i]), 0.0),
                HC_arom = max(float(sol.y[4][i]), 0.0),
                NaCl_aq = self.s0.NaCl_aq,
                X       = max(float(sol.y[5][i]), 0.0),
                P_N2    = self.s0.P_N2,
                P_CO2   = self.s0.P_CO2,
                P_H2    = max(float(sol.y[1][i]) * 15.0, 0.0),
                P_total = self.cond.pressure,
                T       = self.cond.temperature,
                pH      = self.cond.pH,
                cum_H2  = float(sol.y[6][i]),
                time    = float(sol.t[i]),
            )
            phase = self.lifecycle.classify(state, X_max, HC_ali_initial)
            phases.append(phase)

        return sol, phases

    def state_at_time(self, sol: object, t: float) -> BiologicalState:
        """
        Reconstruct a BiologicalState at an arbitrary time using the dense solution.

        Parameters
        ----------
        sol : scipy OdeSolution (must have dense_output=True)
        t : float  Time [h].

        Returns
        -------
        BiologicalState
        """
        y = sol.sol(t)
        return BiologicalState.from_array(y, self.s0)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def reaction_rates_at(self, state: BiologicalState) -> dict:
        """
        Compute all reaction rates at a given state.

        Parameters
        ----------
        state : BiologicalState

        Returns
        -------
        dict  Mapping rate name → value with units annotated in key.
        """
        HC_ali  = max(state.HC_ali,  0.0)
        HC_arom = max(state.HC_arom, 0.0)
        N2_aq   = max(state.N2_aq,   0.0)
        NH4_aq  = max(state.NH4_aq,  0.0)
        H2_aq   = max(state.H2_aq,   0.0)
        X       = max(state.X,       0.0)

        kH_H2_eff = 15.0
        P_H2_bar = H2_aq * kH_H2_eff
        P_H2_atm = P_H2_bar / 1.01325

        mu = self.monod.growth_rate(HC_ali, NH4_aq, N2_aq)
        r_nit = self.nitrogenase.rate(N2_aq, P_H2_atm, HC_arom)
        r_nit_H2  = r_nit * X
        r_nit_NH4 = 2.0 * r_nit * X
        r_ferm_H2 = self.hc_model.fermentation_H2_rate(HC_ali, X)

        return {
            "mu [h-1]":                  mu,
            "r_nit [mol N2/gVSS/h]":    r_nit,
            "r_nit_H2 [mol H2/L/h]":    r_nit_H2,
            "r_nit_NH4 [mol NH4/L/h]":  r_nit_NH4,
            "r_ferm_H2 [mol H2/L/h]":   r_ferm_H2,
            "r_H2_total [mol H2/L/h]":  r_nit_H2 + r_ferm_H2,
            "dX/dt [gVSS/L/h]":         (mu - self.p.b_decay) * X,
            "dHC_ali/dt [mol/L/h]":     self.hc_model.HC_ali_rate(HC_ali, X),
            "P_H2 [atm]":               P_H2_atm,
            "f_arom [-]":               state.f_arom,
            "H2_inhib_factor [-]":      self.nitrogenase.H2_inhibition_factor(P_H2_atm),
            "arom_inhib_factor [-]":    self.nitrogenase.aromatic_inhibition_factor(HC_arom),
        }
