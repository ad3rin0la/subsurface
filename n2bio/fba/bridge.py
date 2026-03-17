"""
fba/bridge.py
=============
Bridge between the N2Bio kinetic simulation state and the manifold FBA
community state.

Maps a ``BiologicalState`` (ODE-derived snapshot) to a ``CommunityState``
suitable for calling ``run_community_fba`` or ``lifecycle_scan``.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .manifold_fba import CommunityState

_BAR_TO_ATM = 1.0 / 1.01325   # 1 bar = 0.9869 atm


def state_to_community(
    bio,                                        # BiologicalState (avoid circular import)
    lifecycle_phase: int = 0,
    SO4_conc_mM: float = 2.0,
    biomass_fractions: Optional[np.ndarray] = None,
) -> CommunityState:
    """
    Convert a :class:`~n2bio.simulation.state.BiologicalState` to a
    :class:`~n2bio.fba.manifold_fba.CommunityState`.

    Parameters
    ----------
    bio : BiologicalState
        Current ODE snapshot from the N2Bio batch or reservoir solver.
    lifecycle_phase : int
        Current lifecycle phase (0–4), typically from
        :class:`~n2bio.simulation.lifecycle.LifecycleClassifier`.
    SO4_conc_mM : float
        Sulfate concentration [mmol/L].  Not tracked in the single-guild
        kinetic model; caller must supply from geochemical context.
    biomass_fractions : array_like of shape (4,), optional
        Consortium guild biomass fractions
        [Diazotroph, Hydrogenotroph, SRB, AromaticDegrader].
        Defaults to the Phase-2 canonical split ``[0.55, 0.20, 0.15, 0.10]``
        if *None*.

    Returns
    -------
    CommunityState

    Unit conversions
    ----------------
    * ``NH4_aq``  mol/L   →  ``NH4_conc``          mmol/L  (×1000)
    * ``P_H2``    bar     →  ``H2_partial_pressure`` atm    (÷1.01325)
    * ``HC_arom`` mol/L   →  ``HC_arom_conc``       mmol/L  (×1000)
    """
    if biomass_fractions is None:
        # Default: Phase-2 distribution from lifecycle_scan
        biomass_fractions = np.array([0.55, 0.20, 0.15, 0.10])

    return CommunityState(
        biomass_fractions=np.asarray(biomass_fractions, dtype=float),
        NH4_conc=bio.NH4_aq * 1000.0,
        SO4_conc=float(SO4_conc_mM),
        H2_partial_pressure=bio.P_H2 * _BAR_TO_ATM,
        HC_arom_conc=bio.HC_arom * 1000.0,
        lifecycle_phase=lifecycle_phase,
    )
