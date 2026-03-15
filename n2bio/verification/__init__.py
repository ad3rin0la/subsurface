"""
n2bio.verification
==================
Verification test modules for the N2Bio framework.

Modules
-------
stage1_thermo   : PR EOS density, viscosity, and N2 solubility verification.
stage2_flow     : IFDM grid geometry, Darcy flow, and N2 front propagation.
stage3_kinetics : Monod, nitrogenase, and HC degradation kinetics verification.
stage4_hc       : Full batch reactor lifecycle and H2 yield verification.
"""

from .stage1_thermo import (
    verify_N2_density,
    verify_N2_solubility,
    run_all_stage1_verifications,
)
from .stage2_flow import (
    verify_grid_geometry,
    verify_relative_permeability,
    verify_darcy_flux,
    verify_mass_balance,
    verify_N2_front_propagation,
    run_all_stage2_verifications,
)
from .stage3_kinetics import (
    verify_monod_kinetics,
    verify_nitrogenase_kinetics,
    verify_HC_degradation_kinetics,
    run_all_stage3_verifications,
)
from .stage4_hc import (
    run_batch_reactor_benchmark,
    verify_lifecycle_phases,
    run_all_stage4_verifications,
)

__all__ = [
    "verify_N2_density",
    "verify_N2_solubility",
    "run_all_stage1_verifications",
    "verify_grid_geometry",
    "verify_relative_permeability",
    "verify_darcy_flux",
    "verify_mass_balance",
    "verify_N2_front_propagation",
    "run_all_stage2_verifications",
    "verify_monod_kinetics",
    "verify_nitrogenase_kinetics",
    "verify_HC_degradation_kinetics",
    "run_all_stage3_verifications",
    "run_batch_reactor_benchmark",
    "verify_lifecycle_phases",
    "run_all_stage4_verifications",
]
