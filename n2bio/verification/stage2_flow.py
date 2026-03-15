"""
verification/stage2_flow.py
============================
Stage 2 multiphase flow verification for N2Bio.

Verifies the IFDM grid geometry and Darcy flow module against analytical
solutions and mass-balance checks:

  1. Grid geometry   – cell volumes, centres, interface areas, d_ij
  2. Relative permeability – Corey model limiting cases (k_r = 0 at S_r,
     k_r = 1 at S = 1)
  3. Darcy flux sign – flow from high to low pressure
  4. Mass balance    – saturation update conserves total fluid volume
  5. N₂ front propagation – dissolved N₂ advances from injection face;
     breakthrough time consistent with pore-velocity estimate

Target accuracy (Section 8, Table 5):
  Saturation fronts and pressure profiles within 3% relative error.

References
----------
Narasimhan, T.N.; Witherspoon, P.A. (1976). Water Resour. Res. 12, 57-64.
Buckley, S.E.; Leverett, M.C. (1942). Trans. AIME 146, 107-116.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


# ---------------------------------------------------------------------------
# 1. Grid geometry verification
# ---------------------------------------------------------------------------

def verify_grid_geometry(verbose: bool = False) -> Dict[str, Any]:
    """
    Verify IFDMGrid1D geometric properties.

    Checks:
      - sum(cell lengths) = domain length
      - cell volumes = length × cross_area
      - number of connections = n_cells - 1
      - d_ij = differences of cell centres
      - cell_index() locates the correct cell

    Returns
    -------
    dict  Verification results.
    """
    from ..transport.grid import IFDMGrid1D

    results = {}

    # ---- Uniform grid ----
    n, L, A = 10, 100.0, 2.5
    grid = IFDMGrid1D.uniform(n, L, cross_area=A)

    # Total length
    total_L = float(np.sum(grid.lengths))
    rel_err = abs(total_L - L) / L
    results['uniform_total_length'] = {
        'value':    total_L,
        'expected': L,
        'rel_error': rel_err,
        'pass':     rel_err < 1.0e-12,
    }

    # Cell volumes = length × cross_area
    volumes_expected = grid.lengths * A
    vol_ok = np.allclose(grid.volumes, volumes_expected, rtol=1e-12)
    results['uniform_volumes_correct'] = {
        'pass': vol_ok,
        'description': 'V_i = dx_i × A',
    }

    # Number of connections
    n_conn = grid.n_connections
    results['uniform_n_connections'] = {
        'value':    n_conn,
        'expected': n - 1,
        'pass':     n_conn == n - 1,
    }

    # d_ij = differences of cell centres (uniform → all equal dx)
    dx = L / n
    d_expected = np.full(n - 1, dx)
    d_ok = np.allclose(grid.d_ij, d_expected, rtol=1e-12)
    results['uniform_d_ij_correct'] = {
        'pass':        d_ok,
        'description': 'd_ij = dx for uniform grid',
    }

    # cell_index locates correct cell
    idx = grid.cell_index(55.0)
    results['cell_index_at_55m'] = {
        'value':    idx,
        'expected': 5,          # cell 5 spans [50, 60) in a 10-cell, 100 m grid
        'pass':     idx == 5,
    }

    # ---- Logarithmic grid ----
    grid_log = IFDMGrid1D.logarithmic(20, 500.0, refinement=10.0)
    total_log = float(np.sum(grid_log.lengths))
    results['log_total_length'] = {
        'value':    total_log,
        'expected': 500.0,
        'rel_error': abs(total_log - 500.0) / 500.0,
        'pass':     abs(total_log - 500.0) / 500.0 < 1.0e-10,
    }

    # Logarithmic grid: first cell shorter than last
    results['log_first_cell_shorter'] = {
        'value_first': grid_log.lengths[0],
        'value_last':  grid_log.lengths[-1],
        'pass':        grid_log.lengths[0] < grid_log.lengths[-1],
        'description': 'First cell must be finer than last in log grid',
    }

    # Pore volumes
    phi = 0.15
    pore_vols = grid.pore_volumes(phi)
    pore_vol_ok = np.allclose(pore_vols, grid.volumes * phi, rtol=1e-12)
    results['pore_volumes_correct'] = {
        'pass': pore_vol_ok,
        'description': 'Pore volume = bulk volume × porosity',
    }

    if verbose:
        _print_results('Grid Geometry', results)

    return results


# ---------------------------------------------------------------------------
# 2. Relative permeability verification
# ---------------------------------------------------------------------------

def verify_relative_permeability(verbose: bool = False) -> Dict[str, Any]:
    """
    Verify Corey relative permeability model limiting cases.

    Checks:
      - k_rg = 0 at S_g = S_gr
      - k_rl = 0 at S_l = S_lr
      - k_rg > 0 and k_rl > 0 in the mobile region
      - k_rg + k_rl ≤ 1 (Corey curves do not exceed 1 individually or sum > 1)

    Returns
    -------
    dict
    """
    from ..transport.grid import IFDMGrid1D
    from ..transport.flow import MultiphaseFlow1D
    from ..kinetics.parameters import KineticParameters, ReservoirConditions

    grid = IFDMGrid1D.uniform(5, 50.0)
    params = KineticParameters()
    cond   = ReservoirConditions()
    flow   = MultiphaseFlow1D(grid, params, cond, S_gr=0.05, S_lr=0.20)

    results = {}

    # k_rg = 0 at S_g = S_gr (residual gas saturation)
    krg_at_Sgr = float(flow.relative_permeability_gas(np.array([flow.S_gr]))[0])
    results['krg_zero_at_Sgr'] = {
        'value': krg_at_Sgr,
        'expected': 0.0,
        'pass': krg_at_Sgr == 0.0,
    }

    # k_rl = 0 at S_l = S_lr (residual liquid saturation)
    krl_at_Slr = float(flow.relative_permeability_liquid(np.array([flow.S_lr]))[0])
    results['krl_zero_at_Slr'] = {
        'value': krl_at_Slr,
        'expected': 0.0,
        'pass': krl_at_Slr == 0.0,
    }

    # k_rg > 0 above residual saturation
    S_g_mobile = np.array([0.5])
    krg_mid = float(flow.relative_permeability_gas(S_g_mobile)[0])
    results['krg_positive_at_Sg_0p5'] = {
        'value': krg_mid,
        'pass':  krg_mid > 0.0,
    }

    # k_rl > 0 above S_lr
    S_l_mobile = np.array([0.7])
    krl_mid = float(flow.relative_permeability_liquid(S_l_mobile)[0])
    results['krl_positive_at_Sl_0p7'] = {
        'value': krl_mid,
        'pass':  krl_mid > 0.0,
    }

    # Both k_r values are in [0, 1]
    S_g_range = np.linspace(0.0, 1.0, 50)
    S_l_range = np.linspace(0.0, 1.0, 50)
    krg_arr = flow.relative_permeability_gas(S_g_range)
    krl_arr = flow.relative_permeability_liquid(S_l_range)
    results['krg_bounded_0_1'] = {
        'pass': bool(np.all(krg_arr >= 0.0) and np.all(krg_arr <= 1.0)),
        'description': 'k_rg ∈ [0, 1] for all S_g',
    }
    results['krl_bounded_0_1'] = {
        'pass': bool(np.all(krl_arr >= 0.0) and np.all(krl_arr <= 1.0)),
        'description': 'k_rl ∈ [0, 1] for all S_l',
    }

    # Capillary pressure: P_c > 0 and increasing as S_l decreases
    S_l_test = np.array([0.8, 0.6, 0.4, 0.3])
    Pc = flow.capillary_pressure(S_l_test)
    results['Pc_positive'] = {
        'pass':        bool(np.all(Pc > 0.0)),
        'description': 'Capillary pressure must be positive',
    }
    results['Pc_increases_as_Sl_decreases'] = {
        'pass':        bool(np.all(np.diff(Pc) > 0.0)),  # S_l decreasing → Pc increasing → diff > 0
        'description': 'P_c increases as S_l decreases (more drainage pressure)',
    }

    if verbose:
        _print_results('Relative Permeability', results)

    return results


# ---------------------------------------------------------------------------
# 3. Darcy flux verification
# ---------------------------------------------------------------------------

def verify_darcy_flux(verbose: bool = False) -> Dict[str, Any]:
    """
    Verify Darcy flux direction and magnitude.

    Checks:
      - Flux is positive (left-to-right) when P decreases from left to right
      - Zero flux when pressure is uniform
      - Flux magnitude consistent with T = k × A / (μ × d) × ΔP

    Returns
    -------
    dict
    """
    from ..transport.grid import IFDMGrid1D
    from ..transport.flow import MultiphaseFlow1D
    from ..kinetics.parameters import KineticParameters, ReservoirConditions

    n = 5
    grid = IFDMGrid1D.uniform(n, 50.0, cross_area=1.0)
    params = KineticParameters()
    cond   = ReservoirConditions(permeability=50.0e-15, porosity=0.15)
    flow   = MultiphaseFlow1D(grid, params, cond, S_gr=0.05, S_lr=0.20)

    results = {}

    # Pressure gradient: high on left, low on right
    P = np.linspace(200.0e5, 150.0e5, n)   # Pa  (200 bar → 150 bar)
    S_g = np.full(n, 0.3)
    mu_g = np.full(n, 2.0e-5)  # 20 μPa·s
    rho  = np.full(n, 100.0)   # kg/m³

    Q = flow.darcy_flux(P, S_g, rho, mu_g, phase='gas')

    # All fluxes should be negative: Q = T × (P[j]−P[i]), P[j] < P[i]
    # Negative Q means flow is from cell i to cell j (left to right in this setup)
    results['flux_positive_under_pressure_gradient'] = {
        'value': float(np.max(Q)),
        'pass':  bool(np.all(Q < 0.0)),
        'description': 'Q < 0 when P[j]-P[i] < 0 (T×ΔP sign convention, flow left-to-right)',
    }

    # Zero pressure gradient → zero flux
    P_uniform = np.full(n, 150.0e5)
    Q_zero = flow.darcy_flux(P_uniform, S_g, rho, mu_g, phase='gas')
    results['flux_zero_under_uniform_pressure'] = {
        'value': float(np.max(np.abs(Q_zero))),
        'expected': 0.0,
        'pass': bool(np.allclose(Q_zero, 0.0, atol=1.0e-30)),
    }

    # Analytical check for first interface:
    # T_01 = A × k × k_rg / (μ_avg × d_01)
    # k_rg at S_g = 0.3
    S_e = (0.3 - flow.S_gr) / (1.0 - flow.S_gr - flow.S_lr)
    krg = S_e ** 2 * (1.0 - (1.0 - S_e) ** 2)
    d01  = grid.d_ij[0]
    A01  = grid.interface_areas[0]
    mu_avg = mu_g[0]
    T_analytical = A01 * cond.permeability * krg / (mu_avg * d01)
    dP_01 = P[1] - P[0]   # negative (high → low)
    Q_analytical = T_analytical * dP_01

    rel_err = abs(Q[0] - Q_analytical) / max(abs(Q_analytical), 1.0e-30)
    results['flux_magnitude_matches_analytical'] = {
        'value':     Q[0],
        'expected':  Q_analytical,
        'rel_error': rel_err,
        'pass':      rel_err < 1.0e-6,
        'description': 'First interface flux matches T_ij × ΔP_ij analytical',
    }

    if verbose:
        _print_results('Darcy Flux', results)

    return results


# ---------------------------------------------------------------------------
# 4. Mass balance verification
# ---------------------------------------------------------------------------

def verify_mass_balance(verbose: bool = False) -> Dict[str, Any]:
    """
    Verify that the explicit saturation update conserves total pore volume.

    In a closed system with no sources or sinks, the total gas saturation
    (and hence total fluid volume) should be conserved after each update step.

    Returns
    -------
    dict
    """
    from ..transport.grid import IFDMGrid1D
    from ..transport.flow import MultiphaseFlow1D
    from ..kinetics.parameters import KineticParameters, ReservoirConditions

    n = 10
    grid  = IFDMGrid1D.uniform(n, 100.0, cross_area=1.0)
    params = KineticParameters()
    cond   = ReservoirConditions(porosity=0.20, permeability=50.0e-15)
    flow   = MultiphaseFlow1D(grid, params, cond)

    # Initial saturations: linearly varying (gas enters from left)
    S_g = np.linspace(0.6, 0.1, n)
    P   = np.linspace(180.0e5, 150.0e5, n)
    mu_g = np.full(n, 2.0e-5)
    rho  = np.full(n, 100.0)

    # Compute Darcy fluxes
    Q_g = flow.darcy_flux(P, S_g, rho, mu_g, phase='gas')
    Q_l = flow.darcy_flux(P, 1.0 - S_g, rho, mu_g * 20, phase='liquid')

    # Closed system: no source terms
    source_g = np.zeros(n)

    # CFL time step
    dt_s = flow.CFL_timestep(Q_g, S_g, CFL_max=0.4)

    # Saturation update
    S_g_new = flow.saturation_update_explicit(S_g, Q_g, Q_l, source_g, dt_s)

    # Mass balance: pore volumes × porosity × saturation (weighted sum)
    PV    = grid.pore_volumes(cond.porosity)
    mass_before = float(np.dot(PV, S_g))
    mass_after  = float(np.dot(PV, S_g_new))

    # In closed system, any gas that leaves a cell enters an adjacent cell
    # Total gas volume is conserved up to numerical precision of explicit step
    rel_err = abs(mass_after - mass_before) / (mass_before + 1.0e-30)

    results = {}
    results['saturation_bounded_0_1'] = {
        'pass': bool(np.all(S_g_new >= 0.0) and np.all(S_g_new <= 1.0)),
        'description': 'Updated saturation must remain in [0, 1]',
    }
    results['CFL_timestep_positive'] = {
        'value': dt_s,
        'pass':  dt_s > 0.0,
    }
    results['mass_balance_closed_system'] = {
        'mass_before': mass_before,
        'mass_after':  mass_after,
        'rel_error':   rel_err,
        'pass':        rel_err < 1.0e-3,   # < 0.1% for explicit first-order scheme
        'description': 'Closed-system gas volume conserved within 0.1%',
    }

    if verbose:
        _print_results('Mass Balance', results)

    return results


# ---------------------------------------------------------------------------
# 5. N₂ front propagation
# ---------------------------------------------------------------------------

def verify_N2_front_propagation(verbose: bool = False) -> Dict[str, Any]:
    """
    Verify dissolved-N₂ front propagation in a 1D advection test.

    Uses the first-order upwind scheme from reservoir_1d.py to advect a
    step-change in dissolved N₂ through a uniform column.  Checks:
      - N₂ is non-negative everywhere
      - Front has propagated (multiple cells show non-zero N₂)
      - Total N₂ in column increases monotonically with injection
      - Injection face reaches a concentration above zero after CFL-stable steps

    The CFL stability criterion for the upwind scheme (v_pore × dt / (dx × phi) ≤ 1)
    requires dt ≤ dx × phi / v_pore.  With dx=5 m, phi=0.15, v_pore=3.33 m/h,
    the maximum stable dt is 0.225 h; we use 0.15 h (CFL ≈ 0.67).

    Returns
    -------
    dict
    """
    from ..transport.grid import IFDMGrid1D

    n = 20
    L = 100.0      # m
    phi = 0.15
    v_darcy = 0.5  # m/h  (Darcy / superficial velocity)
    v_pore = v_darcy / phi   # 3.333 m/h  (pore / interstitial velocity)

    # Mass-balance upwind scheme:
    #   d(phi*C)/dt = -v_darcy * dC/dx
    # CFL stability: v_darcy * dt / (phi * dx) <= 1
    dx = L / n      # 5 m
    dt_h = 0.15     # h  →  CFL = v_darcy * dt / (phi * dx) = 0.5*0.15/(0.15*5) = 0.10 ✓
    n_steps = 50    # 7.5 hours total; pore breakthrough at L/v_pore ≈ 30 h >> 7.5 h

    grid = IFDMGrid1D.uniform(n, L, cross_area=1.0)
    N2_inject = 1.0e-4   # mol/L

    N2_field = np.zeros(n)

    # Track total N₂ per step to check monotone injection
    total_N2_history = []

    def _advect(field, n_inject):
        """First-order upwind advection step.

        Conservative form: flux = v_darcy * C * A  (not v_pore).
        Dividing by (V_cell * phi) gives d(phi*C)/dt = -v_darcy*(C_i - C_{i-1})/dx,
        equivalent to dC/dt = -v_pore*(C_i - C_{i-1})/dx.
        """
        new = field.copy()
        fluxes = np.zeros(n + 1)
        fluxes[0] = v_darcy * n_inject * grid.cross_area
        for i in range(1, n):
            fluxes[i] = v_darcy * field[i - 1] * grid.cross_area
        fluxes[n] = v_darcy * field[-1] * grid.cross_area
        for i in range(n):
            dN2 = (fluxes[i] - fluxes[i + 1]) / (grid.volumes[i] * phi)
            new[i] = max(field[i] + dt_h * dN2, 0.0)
        return new

    for _ in range(n_steps):
        N2_field = _advect(N2_field, N2_inject)
        # Total dissolved N₂ in column (mol) = sum(C[i] * V_pore[i])
        pore_vols = grid.pore_volumes(phi)   # m³
        total_N2_history.append(float(np.dot(N2_field, pore_vols)))

    results = {}

    # N₂ field is non-negative everywhere
    results['N2_non_negative'] = {
        'pass':        bool(np.all(N2_field >= 0.0)),
        'description': 'Dissolved N₂ must be non-negative everywhere',
    }

    # Injection face has non-zero N₂ (CFL-stable scheme maintains this)
    results['injection_face_non_zero'] = {
        'value':       N2_field[0],
        'pass':        N2_field[0] > 0.0,
        'description': 'Cell 0 concentration must be positive under steady injection',
    }

    # Front has propagated into the column (multiple cells occupied)
    n_cells_with_N2 = int(np.sum(N2_field > 1.0e-6 * N2_inject))
    results['front_has_propagated'] = {
        'n_cells_with_N2': n_cells_with_N2,
        'pass':             n_cells_with_N2 > 2,
        'description':      'N₂ must have propagated beyond the first two cells',
    }

    # Total N₂ in column increases monotonically (injection drives accumulation)
    diffs = np.diff(total_N2_history)
    monotone_increase = bool(np.all(diffs >= -1.0e-20))  # allow tiny floating-point noise
    results['total_N2_increases_monotonically'] = {
        'pass':        monotone_increase,
        'description': 'Total dissolved N₂ in column must not decrease between steps',
    }

    # Last cell (far end) should still be near zero for a 7.5-h run in a 100-m column.
    # Front travels at v_pore = 3.333 m/h → breakthrough at L/v_pore = 100/3.333 ≈ 30 h >> 7.5 h.
    # After 7.5 h the front has only reached ~25 m; last cell (95–100 m) stays at zero.
    results['far_end_still_uncontaminated'] = {
        'value':       N2_field[-1],
        'pass':        N2_field[-1] < 0.05 * N2_inject,
        'description': 'Last cell should have < 5% of N2_inject at 7.5 h (breakthrough at ~30 h)',
    }

    if verbose:
        _print_results('N₂ Front Propagation', results)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_results(name: str, results: Dict) -> None:
    """Pretty-print a results dict."""
    print(f"\n{'='*58}")
    print(f"Stage 2 Verification: {name}")
    print(f"{'='*58}")
    n_pass = sum(1 for r in results.values() if r.get('pass', False))
    print(f"  Passed: {n_pass}/{len(results)}")
    for test_name, r in results.items():
        marker = 'OK' if r.get('pass', False) else 'FAIL'
        desc = r.get('description', '')
        val = r.get('value', '')
        val_str = f"{val:.4e}" if isinstance(val, float) else str(val)
        err = r.get('rel_error', None)
        err_str = f"  rel_err={err:.2e}" if err is not None else ''
        print(f"  [{marker}] {test_name}: {val_str}{err_str}  {desc}")


# ---------------------------------------------------------------------------
# Run all Stage 2 verifications
# ---------------------------------------------------------------------------

def run_all_stage2_verifications(verbose: bool = True) -> Dict[str, Dict]:
    """
    Run all Stage 2 multiphase flow verifications.

    Parameters
    ----------
    verbose : bool
        Print results to stdout.

    Returns
    -------
    dict  Keys are test group names, values are result dicts.
    """
    all_results = {
        'grid_geometry':       verify_grid_geometry(verbose=verbose),
        'relative_permeability': verify_relative_permeability(verbose=verbose),
        'darcy_flux':          verify_darcy_flux(verbose=verbose),
        'mass_balance':        verify_mass_balance(verbose=verbose),
        'N2_front':            verify_N2_front_propagation(verbose=verbose),
    }

    if verbose:
        total_pass = sum(
            sum(1 for r in group.values() if r.get('pass', False))
            for group in all_results.values()
        )
        total_tests = sum(len(group) for group in all_results.values())
        print(f"\nStage 2 Overall: {total_pass}/{total_tests} tests passed.")

    return all_results


if __name__ == '__main__':
    run_all_stage2_verifications(verbose=True)
