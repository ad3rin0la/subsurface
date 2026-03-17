"""
n2bio.fba – Matrix Manifold Flux Balance Analysis
==================================================

Constraint-based stoichiometric modelling of the N2Bio four-guild
consortium on three Riemannian manifolds:

* **Grassmann Gr(k, n)** — null-space geometry; geodesic distance between
  guilds measures metabolic complementarity.
* **Stiefel V_{n,k}** — orthonormal null-space bases; Riemannian gradient
  descent stays in null(S) exactly.
* **SPD Sym+(n)** — flux covariance manifold; Fisher-Rao metric for
  principled cross-guild comparison.

Guilds
------
0  Diazotroph       N₂-fixation + dark fermentation  (Thermotoga chassis)
1  Hydrogenotroph   H₂ + CO₂ → CH₄
2  SRB              H₂/HC + SO₄ → H₂S
3  AromaticDegrader HC_arom detoxification

Quick start
-----------
>>> from n2bio.fba import run_community_fba, lifecycle_scan, CommunityState
>>> import numpy as np
>>> state = CommunityState(np.array([0.55, 0.20, 0.15, 0.10]),
...                        NH4_conc=2.0, SO4_conc=1.0,
...                        H2_partial_pressure=0.20, HC_arom_conc=0.25,
...                        lifecycle_phase=2)
>>> result = run_community_fba(state)

Bridge from kinetic simulation
-------------------------------
>>> from n2bio import N2BioSimulation
>>> from n2bio.fba import run_community_fba
>>> from n2bio.fba.bridge import state_to_community
>>> sim = N2BioSimulation.default_batch()
>>> sol, phases = sim.run(t_end_hours=3000)
>>> # grab Phase-2 snapshot (index where phase == 2 first occurs):
>>> idx = next(i for i, p in enumerate(phases) if int(p) == 2)
>>> from n2bio.simulation.state import BiologicalState
>>> bio = BiologicalState.from_array(sol.y[:, idx], sim.conditions)
>>> community_state = state_to_community(bio, lifecycle_phase=2)
>>> result = run_community_fba(community_state)
"""

from .manifold_fba import (
    # Data structures
    GuildModel,
    FBAResult,
    CommunityState,
    CommunityFBAResult,
    LifecycleScan,
    # Guild builders
    build_all_guilds,
    build_diazotroph_model,
    build_hydrogenotroph_model,
    build_srb_model,
    build_aromatic_degrader_model,
    # Geometry
    compute_null_basis,
    grassmann_distance,
    grassmann_log,
    grassmann_exp,
    grassmann_geodesic,
    spd_distance,
    spd_mean,
    # Solvers
    run_standard_fba,
    run_manifold_fba,
    run_community_fba,
    lifecycle_scan,
    # Reporting
    print_guild_null_spaces,
    print_community_report,
    print_lifecycle_summary,
    # Constants
    EXCHANGE_METABOLITES,
    GUILD_NAMES,
    N_EXCH,
)

from .bridge import state_to_community

__all__ = [
    # Data structures
    "GuildModel",
    "FBAResult",
    "CommunityState",
    "CommunityFBAResult",
    "LifecycleScan",
    # Guild builders
    "build_all_guilds",
    "build_diazotroph_model",
    "build_hydrogenotroph_model",
    "build_srb_model",
    "build_aromatic_degrader_model",
    # Geometry
    "compute_null_basis",
    "grassmann_distance",
    "grassmann_log",
    "grassmann_exp",
    "grassmann_geodesic",
    "spd_distance",
    "spd_mean",
    # Solvers
    "run_standard_fba",
    "run_manifold_fba",
    "run_community_fba",
    "lifecycle_scan",
    # Reporting
    "print_guild_null_spaces",
    "print_community_report",
    "print_lifecycle_summary",
    # Constants
    "EXCHANGE_METABOLITES",
    "GUILD_NAMES",
    "N_EXCH",
    # Bridge
    "state_to_community",
]
