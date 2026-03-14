"""
simulation/lifecycle.py
=======================
Biological lifecycle phase classification for N2Bio simulations.

The five phases describe the temporal evolution of a diazotrophic
microbial community in a subsurface reservoir undergoing N₂ injection:

  Phase 0 – Inoculation Lag
      [NH₄⁺] ≪ K_N_fixed.  Cells have no fixed N for biosynthesis.
      μ ≈ 0, H₂ ≈ 0.  N₂ must be fixed before growth can begin.

  Phase 1 – N₂ Fixation Bootstrap
      [NH₄⁺] rising from nitrogenase activity.
      r_HC increasing as biomass grows.  f_arom still low.
      Both growth pathways ramping up.

  Phase 2 – Peak Production
      X near maximum.  Both H₂ production pathways operating at or
      near peak.  P_H₂ approaching K_I_H₂ threshold.

  Phase 3 – HC Depletion / Aromatic Inhibition
      HC_ali nearly exhausted, f_arom rising.
      r_nit declining from aromatic inhibition.
      H₂ production starting to decline.

  Phase 4 – Biological Senescence
      X declining, electron donors exhausted.
      H₂ → 0.  Community approaching extinction or dormancy.
"""

from enum import IntEnum
import numpy as np
from .state import BiologicalState
from ..kinetics.parameters import KineticParameters


class LifecyclePhase(IntEnum):
    """
    Enumeration of N2Bio biological lifecycle phases.

    Phases are ordered chronologically (0 → 4) following the progression
    from inoculation through peak production to senescence.
    """

    INOCULATION_LAG        = 0   # Phase 0
    N_FIXATION_BOOTSTRAP   = 1   # Phase 1
    PEAK_PRODUCTION        = 2   # Phase 2
    HC_DEPLETION           = 3   # Phase 3
    BIOLOGICAL_SENESCENCE  = 4   # Phase 4

    @property
    def description(self) -> str:
        """Human-readable description of the phase."""
        _desc = {
            0: "Inoculation lag: NH₄⁺ << K_N_fixed, μ ≈ 0, H₂ ≈ 0",
            1: "N₂ fixation bootstrap: NH₄⁺ rising, r_HC increasing, f_arom low",
            2: "Peak production: X maximum, both H₂ pathways active",
            3: "HC depletion / aromatic inhibition: HC_ali exhausted, f_arom rising",
            4: "Biological senescence: X declining, H₂ → 0, donors exhausted",
        }
        return _desc[int(self)]


class LifecycleClassifier:
    """
    Classifies the current biological state into a LifecyclePhase.

    Classification is based on threshold rules applied to state variables:
    biomass concentration relative to its historical maximum, NH₄⁺
    relative to K_N_fixed, HC_ali depletion fraction, and f_arom.

    Parameters
    ----------
    params : KineticParameters
        Kinetic parameters (used for threshold comparisons).
    thresholds : dict, optional
        Override default classification thresholds.
    """

    _DEFAULT_THRESHOLDS = {
        # Phase 0: NH₄⁺ < factor * K_N_fixed
        'NH4_lag_factor':      0.05,
        # Phase 1: NH₄⁺ < factor * K_N_fixed (and not yet Phase 0)
        'NH4_bootstrap_factor': 0.5,
        # Phase 2: X > fraction * X_max
        'X_max_fraction':       0.9,
        # Phase 3/4: HC_ali < fraction * HC_ali_initial
        'HC_ali_depleted':      0.10,
        # Phase 3: f_arom > threshold → aromatic inhibition kicking in
        'f_arom_inhibition':    0.30,
        # Phase 4: X < fraction * X_max  (AND HC depleted)
        'X_senescence':         0.10,
    }

    def __init__(self, params: KineticParameters,
                 thresholds: dict = None):
        self.p = params
        self.thresholds = dict(self._DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    def classify(self, state: BiologicalState,
                 X_max: float,
                 HC_ali_initial: float) -> LifecyclePhase:
        """
        Classify a BiologicalState into a LifecyclePhase.

        Rules are evaluated in priority order:
        1. Phase 0 (nitrogen starvation lag) if NH₄⁺ very low.
        2. Phase 4 (senescence) if X collapsed AND HC exhausted.
        3. Phase 3 (HC depletion) if HC_ali < 10% initial or f_arom high.
        4. Phase 2 (peak production) if X near X_max.
        5. Phase 1 (bootstrap) otherwise.

        Parameters
        ----------
        state : BiologicalState
            Current simulation state.
        X_max : float
            Historical peak biomass [g VSS/L].
        HC_ali_initial : float
            Initial aliphatic HC concentration [mol/L].

        Returns
        -------
        LifecyclePhase
        """
        t = self.thresholds

        # ------ Phase 0: Inoculation lag (N-starved) ------
        if state.NH4_aq < t['NH4_lag_factor'] * self.p.K_N_fixed:
            return LifecyclePhase.INOCULATION_LAG

        # ------ Phase 4: Biological senescence ------
        HC_depleted = (
            HC_ali_initial > 0 and
            state.HC_ali < t['HC_ali_depleted'] * HC_ali_initial
        )
        X_collapsed = (
            X_max > 0 and
            state.X < t['X_senescence'] * X_max
        )
        if HC_depleted and X_collapsed:
            return LifecyclePhase.BIOLOGICAL_SENESCENCE

        # ------ Phase 3: HC depletion / aromatic inhibition ------
        f_arom_high = state.f_arom > t['f_arom_inhibition']
        if HC_depleted or f_arom_high:
            return LifecyclePhase.HC_DEPLETION

        # ------ Phase 2: Peak production ------
        if X_max > 0 and state.X > t['X_max_fraction'] * X_max:
            return LifecyclePhase.PEAK_PRODUCTION

        # ------ Phase 1: N₂ fixation bootstrap ------
        return LifecyclePhase.N_FIXATION_BOOTSTRAP

    def classify_trajectory(self, states: list,
                             HC_ali_initial: float) -> list:
        """
        Classify an entire trajectory of states.

        Parameters
        ----------
        states : list of BiologicalState
        HC_ali_initial : float

        Returns
        -------
        list of LifecyclePhase
        """
        if not states:
            return []
        X_max = max(s.X for s in states)
        return [self.classify(s, X_max, HC_ali_initial) for s in states]

    def phase_transitions(self, time_array: np.ndarray,
                          phases: list) -> list:
        """
        Identify phase transition times.

        Parameters
        ----------
        time_array : np.ndarray  Time [h].
        phases : list of LifecyclePhase

        Returns
        -------
        list of (time, LifecyclePhase) tuples at each transition.
        """
        if len(phases) == 0:
            return []
        transitions = [(time_array[0], phases[0])]
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                transitions.append((time_array[i], phases[i]))
        return transitions

    def summary(self, time_array: np.ndarray, phases: list) -> str:
        """
        Generate a human-readable phase transition summary.

        Parameters
        ----------
        time_array : np.ndarray  Time [h].
        phases : list of LifecyclePhase

        Returns
        -------
        str
        """
        transitions = self.phase_transitions(time_array, phases)
        lines = ["Biological Lifecycle Phase Transitions", "=" * 45]
        for t, ph in transitions:
            lines.append(f"  t = {t:8.1f} h  →  Phase {int(ph)}: {ph.name}")
            lines.append(f"               ({ph.description})")
        return "\n".join(lines)
