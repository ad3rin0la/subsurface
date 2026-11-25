"""Tests for hard constraints and aligned geometry in the reduced FBA model."""

import numpy as np

from n2bio.fba import (
    CommunityState,
    build_diazotroph_model,
    run_community_fba,
    run_manifold_fba,
)


def test_manifold_solver_preserves_balance_bounds_and_exchange_limit():
    guild = build_diazotroph_model()
    result = run_manifold_fba(
        guild,
        exchange_bounds={"N2_aq": (-0.4, 0.0)},
    )

    residual = guild.S[:guild.n_internal, :] @ result.fluxes
    nitrogen_flux = guild.S_exchange[0, :] @ result.fluxes
    assert result.status == "manifold_optimal"
    assert np.max(np.abs(residual)) < 1e-8
    assert np.all(result.fluxes >= guild.lb - 1e-8)
    assert np.all(result.fluxes <= guild.ub + 1e-8)
    assert -0.4 - 1e-8 <= nitrogen_flux <= 1e-8


def test_community_geometry_is_finite_in_shared_exchange_basis():
    state = CommunityState(
        biomass_fractions=np.array([0.55, 0.20, 0.15, 0.10]),
        NH4_conc=2.0,
        SO4_conc=1.0,
        H2_partial_pressure=0.20,
        HC_arom_conc=0.25,
    )
    result = run_community_fba(state)

    assert result.grassmann_distances.shape == (4, 4)
    assert np.all(np.isfinite(result.grassmann_distances))
    assert np.allclose(result.grassmann_distances, result.grassmann_distances.T)
    assert np.allclose(np.diag(result.grassmann_distances), 0.0)
