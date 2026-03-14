"""
n2bio.verification
==================
Verification test modules for the N2Bio framework.

Modules
-------
stage1_thermo   : PR EOS density, viscosity, and N₂ solubility verification.
stage3_kinetics : Monod, nitrogenase, and HC degradation kinetics verification.
stage4_hc       : Full batch reactor lifecycle and H₂ yield verification.
"""

from .stage1_thermo import (
    verify_N2_density,
    verify_N2_solubility,
    run_all_stage1_verifications,
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
    "verify_monod_kinetics",
    "verify_nitrogenase_kinetics",
    "verify_HC_degradation_kinetics",
    "run_all_stage3_verifications",
    "run_batch_reactor_benchmark",
    "verify_lifecycle_phases",
    "run_all_stage4_verifications",
]
