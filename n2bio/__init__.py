"""
n2bio – TOUGHREACT-N2Bio Biogeochemical Modelling Framework
===========================================================

A scientific Python framework for simulating subsurface biological
hydrogen production via nitrogen fixation and hydrocarbon degradation
under multiphase flow conditions.

Key physical processes modelled
--------------------------------
- Peng-Robinson EOS for N₂, CO₂, H₂ thermophysics
- Friction-theory viscosity (Quiñones-Cisneros model)
- Henry's law gas solubility in NaCl brine (Fernandez-Prini 2003)
- Setschenow salting-out correction
- Monod-type diazotrophic growth with triple co-limitation
- Nitrogenase N₂ fixation with H₂ and aromatic inhibition
- Two-pool HC model (aliphatic fermentable + aromatic inhibitory)
- NH₃/NH₄⁺ acid-base speciation
- 1D reactive transport via IFDM
- Biological lifecycle phases (Phase 0–4)

Quick Start
-----------
>>> from n2bio import N2BioSimulation, KineticParameters
>>> params = KineticParameters()
>>> sim = N2BioSimulation.default_batch()
>>> sol, phases = sim.run(t_end_hours=3000)

Module structure
----------------
n2bio.components        – Component definitions and ComponentSystem
n2bio.thermophysics     – EOS, viscosity, solubility
n2bio.kinetics          – Monod, nitrogenase, HC degradation, parameters
n2bio.geochemistry      – NH₃/NH₄⁺ speciation
n2bio.transport         – IFDM grid, multiphase Darcy flow
n2bio.simulation        – BiologicalState, lifecycle, ODE solver
n2bio.verification      – Verification test suites (stages 1, 3, 4)
"""

__version__ = "1.0.0"
__author__  = "N2Bio Development Team"

# ---- Core component system ----
from .components import Component, COMPONENTS, ComponentSystem

# ---- Kinetic parameters ----
from .kinetics.parameters import KineticParameters, ReservoirConditions

# ---- Simulation classes ----
from .simulation.state import BiologicalState
from .simulation.lifecycle import LifecyclePhase, LifecycleClassifier
from .simulation.solver import N2BioBatchSolver

# ---- Convenience high-level class ----
from .thermophysics.eos import PengRobinsonEOS, MixturePREOS
from .thermophysics.solubility import N2Solubility, CO2Solubility, H2Solubility
from .geochemistry.speciation import NH3Speciation
from .transport.grid import IFDMGrid1D
from .transport.flow import MultiphaseFlow1D


class N2BioSimulation:
    """
    High-level interface to the N2Bio batch-reactor simulation.

    Provides convenient setup and execution methods wrapping the
    N2BioBatchSolver with sensible defaults.

    Parameters
    ----------
    params : KineticParameters
        Biological kinetics.
    conditions : ReservoirConditions
        Physical/chemical reservoir conditions.
    initial_state : BiologicalState
        Initial state vector.
    """

    def __init__(self,
                 params: KineticParameters,
                 conditions: ReservoirConditions,
                 initial_state: BiologicalState):
        self.params     = params
        self.conditions = conditions
        self.initial    = initial_state
        self._solver    = N2BioBatchSolver(params, conditions, initial_state)

    @classmethod
    def default_batch(cls,
                      T_C: float = 85.0,
                      P_bar: float = 150.0,
                      salinity: float = 0.5,
                      HC_total: float = 5.0e-3) -> 'N2BioSimulation':
        """
        Create an N2BioSimulation with default reservoir conditions.

        Parameters
        ----------
        T_C : float        Temperature [°C].
        P_bar : float      Total pressure [bar].
        salinity : float   NaCl molality [mol/kg].
        HC_total : float   Total initial HC concentration [mol/L].

        Returns
        -------
        N2BioSimulation
        """
        params = KineticParameters()
        cond   = ReservoirConditions(
            temperature=T_C + 273.15,
            pressure=P_bar,
            salinity_NaCl=salinity,
        )

        HC_ali_0  = HC_total * (1.0 - params.f_arom0)
        HC_arom_0 = HC_total * params.f_arom0

        initial = BiologicalState(
            N2_aq   = params.K_N2 * 10.0,
            CO2_aq  = 0.005,
            H2_aq   = 0.0,
            NH4_aq  = 1.0e-7,      # near-zero → triggers Phase 0 lag
            HC_ali  = HC_ali_0,
            HC_arom = HC_arom_0,
            NaCl_aq = salinity,
            X       = 0.005,       # g VSS/L inoculum
            P_N2    = 100.0,
            P_CO2   = 5.0,
            P_H2    = 0.01,
            P_total = P_bar,
            T       = T_C + 273.15,
            pH      = cond.pH,
        )

        return cls(params, cond, initial)

    def run(self, t_end_hours: float = 3000.0,
            n_points: int = 500):
        """
        Run the batch simulation.

        Parameters
        ----------
        t_end_hours : float
            Simulation end time [h].
        n_points : int
            Number of output time points.

        Returns
        -------
        (sol, phases)
            scipy OdeSolution and list of LifecyclePhase.
        """
        return self._solver.solve_with_lifecycle(t_end_hours, n_points)

    def __repr__(self) -> str:
        return (f"N2BioSimulation(T={self.conditions.temperature_C:.0f}°C, "
                f"P={self.conditions.pressure:.0f} bar, "
                f"mu_max={self.params.mu_max} h⁻¹)")


__all__ = [
    # Main entry point
    "N2BioSimulation",
    # Data classes
    "Component",
    "COMPONENTS",
    "ComponentSystem",
    "KineticParameters",
    "ReservoirConditions",
    "BiologicalState",
    "LifecyclePhase",
    "LifecycleClassifier",
    # Solvers / models
    "N2BioBatchSolver",
    "PengRobinsonEOS",
    "MixturePREOS",
    "N2Solubility",
    "CO2Solubility",
    "H2Solubility",
    "NH3Speciation",
    "IFDMGrid1D",
    "MultiphaseFlow1D",
]
