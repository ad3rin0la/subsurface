"""
N2Bio 1D Reservoir Column Example
==================================
Simulates a 1D column of subsurface reservoir rock under N₂ injection
from one end.  Models spatial transport of dissolved N₂ and the coupled
biological reactions (N₂ fixation, HC degradation, H₂ production) in
each grid cell.

Demonstrates:
  - IFDMGrid1D discretisation
  - Operator-splitting: transport step + reaction step per dt
  - Spatial front propagation of dissolved N₂
  - Biological activity concentrated near N₂ injection point

Run from /Users/derinfasipe/subsurface/:
    python examples/reservoir_1d.py

Or with plots:
    python examples/reservoir_1d.py --plot
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2bio.kinetics.parameters import KineticParameters, ReservoirConditions
from n2bio.simulation.state import BiologicalState
from n2bio.simulation.solver import N2BioBatchSolver
from n2bio.simulation.lifecycle import LifecyclePhase
from n2bio.transport.grid import IFDMGrid1D
from n2bio.thermophysics.solubility import N2Solubility
from n2bio.thermophysics.viscosity import FrictionTheoryViscosity
from n2bio.thermophysics.eos import PengRobinsonEOS
from n2bio.components import COMPONENTS


def build_initial_column(grid: IFDMGrid1D,
                          params: KineticParameters,
                          cond: ReservoirConditions) -> list:
    """
    Build a list of BiologicalState objects, one per grid cell.

    Cell 0 (injection point) has elevated dissolved N₂.
    All cells have the same HC and inoculum conditions.

    Parameters
    ----------
    grid   : IFDMGrid1D
    params : KineticParameters
    cond   : ReservoirConditions

    Returns
    -------
    list of BiologicalState  (length = n_cells)
    """
    n2sol = N2Solubility()
    N2_boundary = n2sol.dissolved_concentration(
        cond.temperature, 100.0, cond.salinity_NaCl
    )

    HC_total  = 5.0e-3
    HC_ali_0  = HC_total * (1.0 - params.f_arom0)
    HC_arom_0 = HC_total * params.f_arom0

    states = []
    for i in range(grid.n_cells):
        # Dissolved N₂ decays linearly from injection face
        frac  = 1.0 - float(i) / grid.n_cells
        N2_aq = N2_boundary * frac * 0.8 + N2_boundary * 0.2

        state = BiologicalState(
            N2_aq   = N2_aq,
            CO2_aq  = 0.005,
            H2_aq   = 0.0,
            NH4_aq  = 1.0e-7,
            HC_ali  = HC_ali_0,
            HC_arom = HC_arom_0,
            NaCl_aq = cond.salinity_NaCl,
            X       = 0.005,            # uniform inoculum
            P_N2    = 100.0 * frac + 50.0 * (1.0 - frac),
            P_CO2   = 5.0,
            P_H2    = 0.0,
            P_total = cond.pressure,
            T       = cond.temperature,
            pH      = cond.pH,
        )
        states.append(state)

    return states


def advect_N2(N2_field: np.ndarray, grid: IFDMGrid1D,
              darcy_velocity: float, dt_h: float,
              porosity: float, N2_inject: float) -> np.ndarray:
    """
    First-order upwind advection of dissolved N₂.

    Simulates pore-water flow carrying dissolved N₂ from the injection
    face (cell 0) through the column.

    Parameters
    ----------
    N2_field : np.ndarray   Dissolved N₂ per cell [mol/L].
    grid : IFDMGrid1D
    darcy_velocity : float  Darcy velocity [m/h].
    dt_h : float            Time step [h].
    porosity : float
    N2_inject : float       N₂ concentration at injection face [mol/L].

    Returns
    -------
    np.ndarray  Updated N₂ field.
    """
    n = grid.n_cells
    v_pore = darcy_velocity / porosity    # pore velocity [m/h]
    N2_new = N2_field.copy()

    # Upwind fluxes (positive flow direction: cell 0 → cell n-1)
    fluxes = np.zeros(n + 1)
    # Injection face
    fluxes[0] = v_pore * N2_inject * grid.cross_area   # mol/h
    # Interior faces
    for i in range(1, n):
        # Upwind: use concentration from cell i-1 (upstream)
        fluxes[i] = v_pore * N2_field[i - 1] * grid.cross_area
    # Outflow face (free outflow)
    fluxes[n] = v_pore * N2_field[-1] * grid.cross_area

    # Update
    for i in range(n):
        dN2 = (fluxes[i] - fluxes[i + 1]) / (grid.volumes[i] * porosity)  # mol/L/h
        N2_new[i] = max(N2_field[i] + dt_h * dN2, 0.0)

    return N2_new


def run_1d_column(n_cells: int = 20,
                   domain_length: float = 100.0,
                   t_end_days: float = 30.0,
                   dt_h: float = 6.0,
                   darcy_velocity: float = 0.5,
                   verbose: bool = True):
    """
    Run the 1D column simulation.

    Uses operator splitting:
      1. Transport step  (advect dissolved N₂)
      2. Reaction step   (batch ODE per cell)

    Parameters
    ----------
    n_cells : int         Number of grid cells.
    domain_length : float Column length [m].
    t_end_days : float    Simulation end time [days].
    dt_h : float          Operator-splitting time step [h].
    darcy_velocity : float Darcy velocity for N₂-saturated brine [m/h].
    verbose : bool

    Returns
    -------
    dict  Results including spatial profiles at selected times.
    """
    params = KineticParameters()
    cond   = ReservoirConditions()
    grid   = IFDMGrid1D.uniform(n_cells, domain_length, cross_area=1.0)

    # Injection boundary condition: dissolved N₂ in equilibrium at P_N2=100 bar
    n2sol = N2Solubility()
    N2_inject = n2sol.dissolved_concentration(
        cond.temperature, 100.0, cond.salinity_NaCl
    )

    # Build initial states per cell
    cell_states = build_initial_column(grid, params, cond)

    # Store spatial profiles at selected snapshots
    t_snapshots = np.linspace(0, t_end_days * 24.0, 7)  # hours
    snapshot_idx = 0
    snapshots = {}

    t_h = 0.0
    t_end_h = t_end_days * 24.0
    n_steps = int(t_end_h / dt_h)

    if verbose:
        print(f"\nN2Bio 1D Column Simulation")
        print(f"{'='*55}")
        print(f"  Domain      : {domain_length:.0f} m, {n_cells} cells")
        print(f"  Temperature : {cond.temperature_C:.0f} °C")
        print(f"  Pressure    : {cond.pressure:.0f} bar")
        print(f"  Darcy vel.  : {darcy_velocity:.2f} m/h")
        print(f"  dt          : {dt_h:.1f} h")
        print(f"  t_end       : {t_end_days:.0f} days ({t_end_h:.0f} h)")
        print(f"  [N₂]_inject : {N2_inject*1e6:.1f} μmol/L")
        print()

    # Extract N₂ field for advection
    N2_field = np.array([s.N2_aq for s in cell_states])

    for step in range(n_steps):
        t_h = step * dt_h

        # 1. Transport: advect dissolved N₂
        N2_field = advect_N2(
            N2_field, grid, darcy_velocity, dt_h,
            cond.porosity, N2_inject
        )

        # 2. Reaction: integrate each cell for dt_h hours
        for i in range(n_cells):
            # Update N₂ in state from transport
            cs = cell_states[i]
            cs.N2_aq = N2_field[i]

            # Run ODE for this cell
            solver = N2BioBatchSolver(params, cond, cs)
            sol_cell = solver.solve(t_end_hours=dt_h, n_points=10)

            # Update state from end of ODE integration
            y_end = sol_cell.y[:, -1]
            cell_states[i] = BiologicalState(
                N2_aq   = max(y_end[0], 0.0),
                CO2_aq  = cs.CO2_aq,
                H2_aq   = max(y_end[1], 0.0),
                NH4_aq  = max(y_end[2], 0.0),
                HC_ali  = max(y_end[3], 0.0),
                HC_arom = max(y_end[4], 0.0),
                NaCl_aq = cs.NaCl_aq,
                X       = max(y_end[5], 0.0),
                P_N2    = cs.P_N2,
                P_CO2   = cs.P_CO2,
                P_H2    = max(y_end[1] * 15.0, 0.0),
                P_total = cs.P_total,
                T       = cs.T,
                pH      = cs.pH,
                cum_H2  = y_end[6],
                time    = t_h + dt_h,
            )
            # Keep N₂ field in sync with ODE consumption
            N2_field[i] = max(y_end[0], 0.0)

        # Snapshot check
        t_current = (step + 1) * dt_h
        if (snapshot_idx < len(t_snapshots) and
                t_current >= t_snapshots[snapshot_idx]):
            snapshots[t_current] = {
                'x':        grid.centers.copy(),
                'N2_aq':    np.array([s.N2_aq for s in cell_states]),
                'X':        np.array([s.X for s in cell_states]),
                'H2_aq':    np.array([s.H2_aq for s in cell_states]),
                'NH4_aq':   np.array([s.NH4_aq for s in cell_states]),
                'HC_ali':   np.array([s.HC_ali for s in cell_states]),
                'cum_H2':   np.array([s.cum_H2 for s in cell_states]),
                'f_arom':   np.array([s.f_arom for s in cell_states]),
            }
            if verbose:
                max_X   = max(s.X for s in cell_states)
                max_H2  = max(s.cum_H2 for s in cell_states)
                print(f"  t={t_current:7.1f} h  "
                      f"X_max={max_X*1000:.3f} mg/L  "
                      f"cum_H2_max={max_H2*1e6:.3f} μmol/L")
            snapshot_idx += 1

    return {
        'grid':         grid,
        'params':       params,
        'cond':         cond,
        'final_states': cell_states,
        'snapshots':    snapshots,
    }


def plot_1d_results(results: dict) -> None:
    """Plot spatial profiles from 1D column simulation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plots.")
        return

    grid      = results['grid']
    snapshots = results['snapshots']
    x         = grid.centers

    if not snapshots:
        print("No snapshots to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("N2Bio 1D Column – Spatial Profiles", fontsize=13)

    times    = sorted(snapshots.keys())
    cmap     = plt.cm.viridis
    colors   = [cmap(i / max(1, len(times) - 1)) for i in range(len(times))]
    var_info = [
        ('N2_aq',  '[N₂]_aq [μmol/L]',  1e6),
        ('X',      'Biomass [mg/L]',      1e3),
        ('H2_aq',  '[H₂]_aq [μmol/L]',  1e6),
        ('NH4_aq', '[NH₄⁺] [μmol/L]',   1e6),
        ('HC_ali', 'HC_ali [mmol/L]',    1e3),
        ('f_arom', 'f_arom [-]',         1.0),
    ]
    for ax, (var, ylabel, scale) in zip(axes.flat, var_info):
        for t, col in zip(times, colors):
            snap = snapshots[t]
            ax.plot(snap['x'], snap[var] * scale, color=col,
                    label=f"t={t/24:.1f}d")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split('[')[0].strip())
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(7, len(times)),
               fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("reservoir_1d_results.png", dpi=150, bbox_inches='tight')
    print("  Plot saved to reservoir_1d_results.png")
    plt.show()


if __name__ == '__main__':
    plot_flag = '--plot' in sys.argv

    results = run_1d_column(
        n_cells=10,
        domain_length=100.0,
        t_end_days=10.0,
        dt_h=12.0,
        darcy_velocity=0.5,
        verbose=True,
    )

    print("\nFinal spatial profile summary:")
    grid = results['grid']
    for i, s in enumerate(results['final_states']):
        print(f"  Cell {i:2d} (x={grid.centers[i]:.1f} m): "
              f"X={s.X*1000:.3f} mg/L  "
              f"cum_H2={s.cum_H2*1e6:.4f} μmol/L  "
              f"f_arom={s.f_arom:.3f}")

    if plot_flag:
        plot_1d_results(results)
