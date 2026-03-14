"""
n2bio.simulation
================
Simulation sub-package for the N2Bio framework.

Exports
-------
BiologicalState      : State vector dataclass for a single grid cell.
LifecyclePhase       : IntEnum of biological lifecycle phases (0-4).
LifecycleClassifier  : Rule-based lifecycle phase classifier.
N2BioBatchSolver     : ODE solver for well-mixed batch reactor.
"""

from .state import BiologicalState
from .lifecycle import LifecyclePhase, LifecycleClassifier
from .solver import N2BioBatchSolver

__all__ = [
    "BiologicalState",
    "LifecyclePhase",
    "LifecycleClassifier",
    "N2BioBatchSolver",
]
