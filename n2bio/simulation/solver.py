"""
simulation/solver.py
====================
ODE solver for the N2Bio batch-reactor (well-mixed single-cell) system.

The stiff ODE system integrates 11 state variables simultaneously:
    y[0]  N2_aq      – dissolved N₂ [mol/L]
    y[1]  H2_aq      – dissolved H₂ [mol/L]
    y[2]  NH4_aq     – ammonium [mol/L]
    y[3]  HC_ali     – aliphatic HC pool [mol/L]
    y[4]  HC_arom    – aromatic HC pool [mol/L]
    y[5]  X          – biomass [g VSS/L]
    y[6]  cum_H2     – cumulative H₂ produced [mol/L]
    y[7]  H2S_aq     – dissolved sulfide [mol/L]
    y[8]  Acetate_aq – free acetate [mol/L]
    y[9]  OAS_aq     – O-acetylserine [mol/L]
    y[10] Cys_aq     – L-cysteine [mol/L]

Variables y[7]–y[10] implement the three missed N–S couplings:
  • H₂S as direct OASTL substrate (sulfide bypass of APR/SiR)
  • Acetate as acetyl-CoA donor to SAT (closes fermentation loop)
  • Cysteine as structural requirement for nitrogenase Fe-S clusters

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
from ..kinetics.sulfur import SulfurAssimilationKinetics
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
        self.sulfur      = SulfurAssimilationKinetics(params)
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
        Right-hand side of the N2Bio ODE system (11 state variables).

        State vector y:
            [0]  N2_aq      [mol/L]
            [1]  H2_aq      [mol/L]
            [2]  NH4_aq     [mol/L]
            [3]  HC_ali     [mol/L]
            [4]  HC_arom    [mol/L]
            [5]  X          [g VSS/L]
            [6]  cum_H2     [mol/L]
            [7]  H2S_aq     [mol/L]
            [8]  Acetate_aq [mol/L]
            [9]  OAS_aq     [mol/L]
            [10] Cys_aq     [mol/L]

        Parameters
        ----------
        t : float         Current time [h].
        y : np.ndarray    Current state vector (length 11).

        Returns
        -------
        list  Time derivatives dy/dt (length 11).
        """
        # ------ Unpack and clamp non-negative ------
        N2_aq      = max(y[0],  0.0)
        H2_aq      = max(y[1],  0.0)
        NH4_aq     = max(y[2],  0.0)
        HC_ali     = max(y[3],  0.0)
        HC_arom    = max(y[4],  0.0)
        X          = max(y[5],  0.0)
        cum_H2     = y[6]
        H2S_aq     = max(y[7],  0.0)
        Acetate_aq = max(y[8],  0.0)
        OAS_aq     = max(y[9],  0.0)
        Cys_aq     = max(y[10], 0.0)

        # ------ H₂ partial pressure estimate [atm] ------
        # Bulk Henry's law equilibrium with dissolved H₂.
        kH_H2_eff   = 15.0               # L·bar/mol at 85°C
        P_H2_bar    = H2_aq * kH_H2_eff
        P_H2_atm    = P_H2_bar / 1.01325

        # Apply buoyancy segregation correction.  Biogenic H₂ (M = 2 g/mol)
        # migrates buoyantly to the structural high of the anticline while the
        # biofilm colonises the deeper pore space near the injector.  The local
        # P(H₂) at the biofilm is therefore f_buoyancy_seg × bulk P(H₂).
        # f_buoyancy_seg = 1.0 recovers the well-mixed baseline.
        P_H2_atm_local = P_H2_atm * self.p.f_buoyancy_seg

        # ------ Monod growth rate ------
        mu = self.monod.growth_rate(HC_ali, NH4_aq, N2_aq)

        # ------ Sulfur assimilation rates ------
        pH = self.cond.pH

        r_SAT   = self.sulfur.sat_rate(NH4_aq, Acetate_aq, Cys_aq, X)
        r_OASTL = self.sulfur.oastl_rate(OAS_aq, H2S_aq, Cys_aq, X, pH)

        # Cys sinks: structural biomass + nitrogenase Fe-S maintenance
        r_Cys_biomass = self.sulfur.cys_biomass_demand(mu, X)
        r_Cys_FeS     = self.sulfur.cys_fes_maintenance(X)

        # ------ Nitrogenase: Cys co-limited ------
        r_nit_H2 = self.nitrogenase.H2_production_rate(
            N2_aq, P_H2_atm_local, HC_arom, X, Cys_aq
        )
        r_NH4_prod = self.nitrogenase.NH4_production_rate(
            N2_aq, P_H2_atm_local, HC_arom, X, Cys_aq
        )
        r_N2_consumed = r_NH4_prod / 2.0

        # ------ HC degradation + fermentation ------
        dHC_ali   = self.hc_model.HC_ali_rate(HC_ali, X)
        dHC_arom  = self.hc_model.HC_arom_rate(HC_arom, X)
        r_ferm_H2 = self.hc_model.fermentation_H2_rate(HC_ali, X)

        # Acetate coproduct of dark fermentation (same COD basis as H₂)
        # r_ferm_acetate [mol/L/h] = r_COD [g COD/L/h] × Y_acetate [mol/g COD]
        MW_HC_ali  = 60.052   # g/mol  (acetic-acid equivalent)
        COD_factor = 1.0667   # g COD/g
        r_COD      = self.p.k_ali * HC_ali * X * MW_HC_ali * COD_factor
        r_ferm_acetate = r_COD * self.p.Y_acetate

        # ------ Total H₂ ------
        r_H2_total = r_nit_H2 + r_ferm_H2

        # ------ ODEs ------

        # N₂: consumed by nitrogenase
        dN2 = -r_N2_consumed

        # H₂: produced, minus gas-phase stripping
        dH2 = r_H2_total - self._k_strip_H2 * H2_aq

        # NH₄⁺: produced by nitrogenase, consumed for biosynthesis
        N_demand = (mu * X / self.p.Y) * (self.p.Y_N / 14.0)
        dNH4 = r_NH4_prod - N_demand

        # HC pools
        # (dHC_ali and dHC_arom already computed above)

        # Biomass
        dX = mu * X - self.p.b_decay * X

        # Cumulative H₂ (monotone integral)
        dcum_H2 = r_H2_total

        # ---- Sulfur-pathway ODEs ----

        # H₂S: geological/SRB source − OASTL consumption − stripping
        dH2S = self.p.H2S_source \
               - r_OASTL \
               - self.p.k_strip_H2S * H2S_aq

        # Acetate: fermentation source + OASTL byproduct − SAT consumption
        # Net SAT/OASTL cycle: acetate consumed by SAT, returned by OASTL (1:1);
        # net acetate from cycle ≈ 0 at steady state; fermentation is the
        # only net source.
        dAcetate = r_ferm_acetate + r_OASTL - r_SAT

        # OAS: produced by SAT, consumed by OASTL
        dOAS = r_SAT - r_OASTL

        # Cys: produced by OASTL, consumed for structural biomass + Fe-S turnover
        dCys = r_OASTL - r_Cys_biomass - r_Cys_FeS

        return [dN2, dH2, dNH4, dHC_ali, dHC_arom, dX, dcum_H2,
                dH2S, dAcetate, dOAS, dCys]

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
        y0 = [s.N2_aq, s.H2_aq, s.NH4_aq, s.HC_ali, s.HC_arom, s.X, s.cum_H2,
              s.H2S_aq, s.Acetate_aq, s.OAS_aq, s.Cys_aq]
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
            ny = sol.y.shape[0]
            state = BiologicalState(
                N2_aq      = max(float(sol.y[0][i]), 0.0),
                CO2_aq     = self.s0.CO2_aq,
                H2_aq      = max(float(sol.y[1][i]), 0.0),
                NH4_aq     = max(float(sol.y[2][i]), 0.0),
                HC_ali     = max(float(sol.y[3][i]), 0.0),
                HC_arom    = max(float(sol.y[4][i]), 0.0),
                NaCl_aq    = self.s0.NaCl_aq,
                X          = max(float(sol.y[5][i]), 0.0),
                P_N2       = self.s0.P_N2,
                P_CO2      = self.s0.P_CO2,
                P_H2       = max(float(sol.y[1][i]) * 15.0, 0.0),
                P_total    = self.cond.pressure,
                T          = self.cond.temperature,
                pH         = self.cond.pH,
                cum_H2     = float(sol.y[6][i]),
                time       = float(sol.t[i]),
                H2S_aq     = max(float(sol.y[7][i]),  0.0) if ny > 7  else 0.0,
                Acetate_aq = max(float(sol.y[8][i]),  0.0) if ny > 8  else 0.0,
                OAS_aq     = max(float(sol.y[9][i]),  0.0) if ny > 9  else 0.0,
                Cys_aq     = max(float(sol.y[10][i]), 0.0) if ny > 10 else 1.0e-5,
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

        kH_H2_eff      = 15.0
        P_H2_bar       = H2_aq * kH_H2_eff
        P_H2_atm       = P_H2_bar / 1.01325
        P_H2_atm_local = P_H2_atm * self.p.f_buoyancy_seg

        Cys_aq  = max(state.Cys_aq,     0.0)
        H2S_aq  = max(state.H2S_aq,     0.0)
        OAS_aq  = max(state.OAS_aq,     0.0)
        Ac_aq   = max(state.Acetate_aq, 0.0)
        pH      = state.pH

        mu = self.monod.growth_rate(HC_ali, NH4_aq, N2_aq)
        r_nit    = self.nitrogenase.rate(N2_aq, P_H2_atm_local, HC_arom, Cys_aq)
        r_nit_H2  = r_nit * X
        r_nit_NH4 = 2.0 * r_nit * X
        r_ferm_H2 = self.hc_model.fermentation_H2_rate(HC_ali, X)

        r_SAT   = self.sulfur.sat_rate(NH4_aq, Ac_aq, Cys_aq, X)
        r_OASTL = self.sulfur.oastl_rate(OAS_aq, H2S_aq, Cys_aq, X, pH)

        return {
            "mu [h-1]":                    mu,
            "r_nit [mol N2/gVSS/h]":      r_nit,
            "r_nit_H2 [mol H2/L/h]":      r_nit_H2,
            "r_nit_NH4 [mol NH4/L/h]":    r_nit_NH4,
            "r_ferm_H2 [mol H2/L/h]":     r_ferm_H2,
            "r_H2_total [mol H2/L/h]":    r_nit_H2 + r_ferm_H2,
            "dX/dt [gVSS/L/h]":           (mu - self.p.b_decay) * X,
            "dHC_ali/dt [mol/L/h]":       self.hc_model.HC_ali_rate(HC_ali, X),
            "P_H2_bulk [atm]":             P_H2_atm,
            "P_H2_local [atm]":            P_H2_atm_local,
            "f_buoyancy_seg [-]":          self.p.f_buoyancy_seg,
            "f_arom [-]":                  state.f_arom,
            "H2_inhib_factor [-]":         self.nitrogenase.H2_inhibition_factor(P_H2_atm_local),
            "arom_inhib_factor [-]":       self.nitrogenase.aromatic_inhibition_factor(HC_arom),
            "Cys_factor [-]":              self.nitrogenase.cys_factor(Cys_aq),
            "r_SAT [mol OAS/L/h]":        r_SAT,
            "r_OASTL [mol Cys/L/h]":      r_OASTL,
            "Cys_aq [umol/L]":            Cys_aq * 1e6,
            "H2S_aq [umol/L]":            H2S_aq * 1e6,
        }
