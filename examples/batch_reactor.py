"""
N2Bio Batch Reactor Example
============================
Simulates a well-mixed batch reactor representing a subsurface depleted
reservoir cell undergoing N₂ injection, nitrogen fixation, and
hydrocarbon degradation.

Demonstrates:
  - Phase 0 inoculation lag (N-limited)
  - Phase 1 N₂ fixation bootstrap
  - Phase 2 peak H₂ production
  - Phase 3 aromatic inhibition onset
  - Phase 4 biological senescence

Run from /Users/derinfasipe/subsurface/:
    python examples/batch_reactor.py

Or with matplotlib plots:
    python examples/batch_reactor.py --plot
"""

import numpy as np
import sys
import os

# Allow running as a script without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2bio.kinetics.parameters import KineticParameters, ReservoirConditions
from n2bio.simulation.state import BiologicalState
from n2bio.simulation.solver import N2BioBatchSolver
from n2bio.simulation.lifecycle import LifecyclePhase


def run_standard_case(verbose: bool = True):
    """
    Run the standard batch reactor case with default reservoir conditions.

    Parameters
    ----------
    verbose : bool  Print output to stdout.

    Returns
    -------
    (sol, phases) : OdeSolution, list of LifecyclePhase
    """
    params = KineticParameters()
    cond   = ReservoirConditions()

    HC_total  = 5.0e-3    # mol/L total HC
    HC_ali_0  = HC_total * (1.0 - params.f_arom0)
    HC_arom_0 = HC_total * params.f_arom0

    initial = BiologicalState(
        N2_aq   = params.K_N2 * 10.0,  # 10 × K_N2 ~ dissolved N₂ at 150 bar
        CO2_aq  = 0.005,
        H2_aq   = 0.0,
        NH4_aq  = 1.0e-7,              # near-zero → triggers Phase 0 lag
        HC_ali  = HC_ali_0,
        HC_arom = HC_arom_0,
        NaCl_aq = cond.salinity_NaCl,
        X       = 0.005,               # g VSS/L inoculum
        P_N2    = 100.0,               # bar
        P_CO2   = 5.0,
        P_H2    = 0.01,
        P_total = cond.pressure,
        T       = cond.temperature,
        pH      = cond.pH,
    )

    if verbose:
        print("N2Bio Batch Reactor Simulation")
        print("=" * 55)
        print(f"  Temperature    : {cond.temperature_C:.0f} °C")
        print(f"  Pressure       : {cond.pressure:.0f} bar")
        print(f"  Salinity (NaCl): {cond.salinity_NaCl:.2f} mol/kg")
        print(f"  pH             : {cond.pH:.1f}")
        print()
        print(f"  Initial HC_ali  : {initial.HC_ali*1000:.3f} mmol/L")
        print(f"  Initial HC_arom : {initial.HC_arom*1000:.3f} mmol/L")
        print(f"  Initial f_arom  : {params.f_arom0:.2f}")
        print(f"  Initial Biomass : {initial.X*1000:.1f} mg VSS/L")
        print(f"  Initial N₂_aq   : {initial.N2_aq*1e6:.1f} μmol/L")
        print()

    solver = N2BioBatchSolver(params, cond, initial)
    sol, phases = solver.solve_with_lifecycle(t_end_hours=3000.0, n_points=600)

    if verbose:
        # Print phase transitions
        print("Biological Lifecycle Phases:")
        current_phase = phases[0]
        print(f"  t = {sol.t[0]:8.1f} h  →  Phase {int(current_phase)}: "
              f"{current_phase.name}")
        for i in range(1, len(sol.t)):
            if phases[i] != current_phase:
                current_phase = phases[i]
                print(f"  t = {sol.t[i]:8.1f} h  →  Phase {int(current_phase)}: "
                      f"{current_phase.name}")
        print()

        # Summary statistics
        print("Summary Statistics:")
        print(f"  Peak biomass          : {max(sol.y[5])*1000:.3f} mg VSS/L")
        print(f"  Cumulative H₂ (final) : {sol.y[6,-1]*1e6:.4f} μmol/L")
        print(f"  Final NH₄⁺            : {sol.y[2,-1]*1e6:.4f} μmol/L")
        print(f"  Final HC_ali          : {max(sol.y[3,-1],0)*1e9:.3f} nmol/L")
        total_HC_final = max(sol.y[3,-1],0) + max(sol.y[4,-1],0)
        f_arom_final   = (max(sol.y[4,-1],0) / total_HC_final
                          if total_HC_final > 1e-15 else params.f_arom0)
        print(f"  Final f_arom          : {f_arom_final:.4f}")
        print(f"  Solver success        : {sol.success}")
        print()

        # Reaction rates at t=0
        rates = solver.reaction_rates_at(initial)
        print("Reaction Rates at t=0:")
        for k, v in rates.items():
            print(f"  {k:<40s}: {v:.4e}")

    return sol, phases


def run_parametric_salinity(verbose: bool = True):
    """
    Run batch reactor at three salinity levels to show Setschenow effect.

    Returns
    -------
    list of (salinity, sol, phases)
    """
    from n2bio.thermophysics.solubility import N2Solubility
    n2sol = N2Solubility()

    salinities = [0.0, 0.5, 2.0]   # mol/kg NaCl
    results = []

    if verbose:
        print("\nParametric Salinity Study")
        print("=" * 55)

    for m_NaCl in salinities:
        params = KineticParameters()
        cond   = ReservoirConditions(salinity_NaCl=m_NaCl)

        # Adjust initial N₂ for salinity (Setschenow effect)
        N2_aq_0 = n2sol.dissolved_concentration(cond.temperature, 100.0, m_NaCl)

        HC_total = 5.0e-3
        initial = BiologicalState(
            N2_aq   = N2_aq_0,
            CO2_aq  = 0.005,
            H2_aq   = 0.0,
            NH4_aq  = 1.0e-7,
            HC_ali  = HC_total * (1 - params.f_arom0),
            HC_arom = HC_total * params.f_arom0,
            NaCl_aq = m_NaCl,
            X       = 0.005,
            P_N2    = 100.0, P_CO2=5.0, P_H2=0.01,
            P_total = cond.pressure,
            T       = cond.temperature,
            pH      = cond.pH,
        )

        solver = N2BioBatchSolver(params, cond, initial)
        sol, phases = solver.solve_with_lifecycle(3000.0, 300)
        results.append((m_NaCl, sol, phases))

        if verbose:
            kH = n2sol.henry_constant_brine(cond.temperature, m_NaCl)
            print(f"  Salinity={m_NaCl:.1f} mol/kg: "
                  f"[N2]_0={N2_aq_0*1e6:.1f} μmol/L, "
                  f"kH={kH:.2f} L·bar/mol, "
                  f"cum_H2={sol.y[6,-1]*1e6:.2f} μmol/L")

    return results


def plot_results(sol, phases, title: str = "N2Bio Batch Reactor"):
    """
    Generate multi-panel plot of simulation results.

    Requires matplotlib.

    Parameters
    ----------
    sol    : scipy OdeSolution
    phases : list of LifecyclePhase
    title  : str
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available – skipping plots.")
        return

    t = sol.t
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Phase colour map
    phase_colors = {0: '#d0d0d0', 1: '#a8d5a2', 2: '#4caf50', 3: '#ff9800', 4: '#f44336'}
    phase_int = np.array([int(p) for p in phases])

    def shade_phases(ax):
        for ph_val, col in phase_colors.items():
            mask = phase_int == ph_val
            if not np.any(mask):
                continue
            idxs = np.where(mask)[0]
            # Find contiguous segments
            breaks = np.where(np.diff(idxs) > 1)[0] + 1
            segs = np.split(idxs, breaks)
            for seg in segs:
                ax.axvspan(t[seg[0]], t[seg[-1]], alpha=0.15, color=col, lw=0)

    # 1. Biomass
    ax = axes[0, 0]
    ax.semilogy(t, np.maximum(sol.y[5], 1e-15) * 1000, 'b-', lw=2)
    shade_phases(ax)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Biomass [mg VSS/L]")
    ax.set_title("Biomass (X)")
    ax.grid(True, alpha=0.3)

    # 2. HC pools
    ax = axes[0, 1]
    ax.semilogy(t, np.maximum(sol.y[3], 1e-15) * 1e3, 'g-', lw=2, label='HC_ali')
    ax.semilogy(t, np.maximum(sol.y[4], 1e-15) * 1e3, 'r-', lw=2, label='HC_arom')
    shade_phases(ax)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Concentration [mmol/L]")
    ax.set_title("Hydrocarbon Pools")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. f_arom
    ax = axes[1, 0]
    HC_ali_arr  = np.maximum(sol.y[3], 0)
    HC_arom_arr = np.maximum(sol.y[4], 0)
    total_HC    = HC_ali_arr + HC_arom_arr
    f_arom      = np.where(total_HC > 1e-12, HC_arom_arr / total_HC, 0.15)
    ax.plot(t, f_arom, 'r-', lw=2)
    shade_phases(ax)
    ax.axhline(0.30, ls='--', color='orange', label='Inhibition threshold (0.3)')
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("f_arom [-]")
    ax.set_title("Aromatic Fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Dissolved N₂ and NH₄⁺
    ax = axes[1, 1]
    ax.semilogy(t, np.maximum(sol.y[0], 1e-15) * 1e6, 'b-', lw=2, label='[N₂]_aq')
    ax.semilogy(t, np.maximum(sol.y[2], 1e-15) * 1e6, 'm-', lw=2, label='[NH₄⁺]')
    shade_phases(ax)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Concentration [μmol/L]")
    ax.set_title("N₂ and NH₄⁺")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Dissolved H₂
    ax = axes[2, 0]
    ax.semilogy(t, np.maximum(sol.y[1], 1e-15) * 1e6, 'c-', lw=2)
    shade_phases(ax)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("[H₂]_aq [μmol/L]")
    ax.set_title("Dissolved H₂")
    ax.grid(True, alpha=0.3)

    # 6. Cumulative H₂
    ax = axes[2, 1]
    ax.plot(t, sol.y[6] * 1e6, 'c-', lw=2)
    shade_phases(ax)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Cumulative H₂ [μmol/L]")
    ax.set_title("Cumulative H₂ Production")
    ax.grid(True, alpha=0.3)

    # Legend for phases
    legend_handles = [
        mpatches.Patch(color=phase_colors[ph],
                       label=f"Phase {ph}: {LifecyclePhase(ph).name}")
        for ph in sorted(phase_colors.keys())
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("batch_reactor_results.png", dpi=150, bbox_inches='tight')
    print("  Plot saved to batch_reactor_results.png")
    plt.show()


if __name__ == '__main__':
    plot_flag = '--plot' in sys.argv

    print("\n" + "=" * 60)
    print("N2Bio Batch Reactor – Standard Case")
    print("=" * 60)
    sol, phases = run_standard_case(verbose=True)

    print("\n" + "=" * 60)
    print("Parametric Salinity Study")
    print("=" * 60)
    run_parametric_salinity(verbose=True)

    if plot_flag:
        plot_results(sol, phases, "N2Bio Batch Reactor – 85°C, 150 bar, 0.5 mol/kg NaCl")
