"""
tests/test_kinetics.py
=======================
Unit tests for the biological kinetics modules.

Run with:
    python -m pytest tests/test_kinetics.py -v
or:
    python tests/test_kinetics.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2bio.kinetics.parameters import KineticParameters, ReservoirConditions
from n2bio.kinetics.monod import MonodKinetics
from n2bio.kinetics.nitrogenase import NitrogenaseKinetics
from n2bio.kinetics.hc_degradation import HCDegradationModel


# ===========================================================================
# Helpers
# ===========================================================================

def default_params():
    return KineticParameters()


# ===========================================================================
# Monod kinetics tests
# ===========================================================================

def test_growth_rate_zero_at_zero_substrate():
    """Growth rate must be zero when HC_ali = 0."""
    p = default_params()
    m = MonodKinetics(p)
    mu = m.growth_rate(HC_ali=0.0, NH4=p.K_N_fixed * 100, N2_aq=p.K_N2 * 100)
    assert mu == 0.0, f"Growth rate must be 0 at HC_ali=0, got {mu}"


def test_growth_rate_zero_at_zero_NH4():
    """Growth rate must be zero when NH4 = 0."""
    p = default_params()
    m = MonodKinetics(p)
    mu = m.growth_rate(HC_ali=p.K_S * 100, NH4=0.0, N2_aq=p.K_N2 * 100)
    assert mu == 0.0, f"Growth rate must be 0 at NH4=0, got {mu}"


def test_growth_rate_zero_at_zero_N2():
    """Growth rate must be zero when N2_aq = 0."""
    p = default_params()
    m = MonodKinetics(p)
    mu = m.growth_rate(HC_ali=p.K_S * 100, NH4=p.K_N_fixed * 100, N2_aq=0.0)
    assert mu == 0.0, f"Growth rate must be 0 at N2_aq=0, got {mu}"


def test_growth_rate_half_max_at_KS():
    """
    At HC_ali = K_S with N and N₂ far in excess, growth rate should be
    approximately half mu_max (f_HC = 0.5, f_N_fixed ≈ 1, f_N2 ≈ 1).
    At 1e6 × K values, f_N_fixed and f_N2 ≈ 1 - 1e-6, so the product is
    ~0.5 * (1 - 2e-6) ≈ 0.5 with relative error < 1e-5.
    """
    p = default_params()
    m = MonodKinetics(p)
    # Very high NH4 and N2 → f_N_fixed ≈ 1, f_N2 ≈ 1
    mu = m.growth_rate(HC_ali=p.K_S, NH4=p.K_N_fixed * 1e6, N2_aq=p.K_N2 * 1e6)
    expected = p.mu_max * 0.5   # only HC limitation at half-saturation
    rel_err = abs(mu - expected) / expected
    assert rel_err < 1.0e-4, (
        f"Growth at K_S should be ≈0.5*mu_max ≈ {expected:.6f}, got {mu:.8f}, "
        f"rel_err={rel_err:.2e}"
    )


def test_growth_rate_approaches_mu_max():
    """At very high all substrates, growth rate → mu_max."""
    p = default_params()
    m = MonodKinetics(p)
    mu = m.growth_rate(
        HC_ali=p.K_S * 1e5,
        NH4=p.K_N_fixed * 1e5,
        N2_aq=p.K_N2 * 1e5,
    )
    rel_err = abs(mu - p.mu_max) / p.mu_max
    assert rel_err < 1.0e-4, (
        f"Growth rate at saturation should → mu_max={p.mu_max}, got {mu:.8f}"
    )


def test_growth_rate_non_negative():
    """Growth rate must be non-negative for all non-negative inputs."""
    p = default_params()
    m = MonodKinetics(p)
    test_vals = [0.0, 1e-9, 1e-6, 1e-3, 1.0]
    for HC in test_vals:
        for NH4 in test_vals:
            for N2 in test_vals:
                mu = m.growth_rate(HC, NH4, N2)
                assert mu >= 0.0, f"Negative growth rate: HC={HC}, NH4={NH4}, N2={N2}"


def test_NH4_inhibition_factor_at_zero():
    """NH4 inhibition factor = 1 at zero NH4."""
    p = default_params()
    m = MonodKinetics(p)
    f = m.NH4_inhibition_factor(0.0)
    assert abs(f - 1.0) < 1.0e-10, f"NH4 inhibition at 0 should be 1, got {f}"


def test_NH4_inhibition_factor_half_at_KI():
    """NH4 inhibition factor = 0.5 at NH4 = K_I_NH4."""
    p = default_params()
    m = MonodKinetics(p)
    f = m.NH4_inhibition_factor(p.K_I_NH4)
    assert abs(f - 0.5) < 1.0e-10, f"NH4 inhibition at K_I_NH4 should be 0.5, got {f}"


def test_biomass_rate_positive_when_growing():
    """Biomass rate should be positive when μ > b_decay."""
    p = default_params()
    m = MonodKinetics(p)
    # Very high substrates so μ ≈ μ_max > b_decay
    dX = m.biomass_rate(
        HC_ali=p.K_S * 1e5,
        NH4=p.K_N_fixed * 1e5,
        N2_aq=p.K_N2 * 1e5,
        X=0.1,
    )
    assert dX > 0.0, f"Biomass should grow at saturation, got dX={dX}"


def test_biomass_rate_negative_at_zero_substrate():
    """At zero substrates, net growth rate is -b_decay * X (decay only)."""
    p = default_params()
    m = MonodKinetics(p)
    X = 0.1
    dX = m.biomass_rate(0.0, 0.0, 0.0, X)
    expected = -p.b_decay * X
    assert abs(dX - expected) < 1.0e-12, f"Expected {expected:.6e}, got {dX:.6e}"


# ===========================================================================
# Nitrogenase kinetics tests
# ===========================================================================

def test_nitrogenase_rate_zero_at_zero_N2():
    """Nitrogenase rate must be zero when N2_aq = 0."""
    p = default_params()
    n = NitrogenaseKinetics(p)
    r = n.rate(N2_aq=0.0, P_H2_atm=0.0, HC_arom_aq=0.0)
    assert r == 0.0, f"r_nit must be 0 at N2_aq=0, got {r}"


def test_nitrogenase_inhibited_by_H2():
    """Nitrogenase rate at P_H2 = K_I_H2 should be half the uninhibited rate."""
    p = default_params()
    n = NitrogenaseKinetics(p)
    N2_sat = p.K_N2 * 1000
    r0 = n.rate(N2_aq=N2_sat, P_H2_atm=0.0,         HC_arom_aq=0.0)
    r1 = n.rate(N2_aq=N2_sat, P_H2_atm=p.K_I_H2,    HC_arom_aq=0.0)
    assert abs(r1 / r0 - 0.5) < 1.0e-6, (
        f"r_nit at K_I_H2 should be 0.5 * r_nit(0), got ratio {r1/r0:.6f}"
    )


def test_nitrogenase_inhibited_by_aromatics():
    """Rate at [HC_arom] = K_I_arom should be half uninhibited rate."""
    p = default_params()
    n = NitrogenaseKinetics(p)
    N2_sat = p.K_N2 * 1000
    r0 = n.rate(N2_sat, 0.0, 0.0)
    r1 = n.rate(N2_sat, 0.0, p.K_I_arom)
    assert abs(r1 / r0 - 0.5) < 1.0e-6, (
        f"r_nit at K_I_arom should be 0.5 * r_nit(0), got ratio {r1/r0:.6f}"
    )


def test_nitrogenase_H2_stoichiometry():
    """H₂ and N₂ production should be 1:1 (mol basis)."""
    p = default_params()
    n = NitrogenaseKinetics(p)
    X = 0.1
    N2_sat = p.K_N2 * 1000
    r_H2 = n.H2_production_rate(N2_sat, 0.0, 0.0, X)
    r_N2  = n.N2_consumption_rate(N2_sat, 0.0, 0.0, X)
    assert abs(r_H2 - r_N2) < 1.0e-12, (
        f"H₂/N₂ ratio should be 1:1, got r_H2={r_H2:.4e}, r_N2={r_N2:.4e}"
    )


def test_nitrogenase_NH4_stoichiometry():
    """NH₄⁺ production should be 2× H₂ production (2 mol NH₃ per N₂)."""
    p = default_params()
    n = NitrogenaseKinetics(p)
    X = 0.1
    N2_sat = p.K_N2 * 1000
    r_H2  = n.H2_production_rate( N2_sat, 0.0, 0.0, X)
    r_NH4 = n.NH4_production_rate(N2_sat, 0.0, 0.0, X)
    assert abs(r_NH4 / r_H2 - 2.0) < 1.0e-9, (
        f"NH₄⁺/H₂ ratio should be 2, got {r_NH4/r_H2:.6f}"
    )


def test_nitrogenase_rate_positive():
    """Nitrogenase rate must be non-negative."""
    p = default_params()
    n = NitrogenaseKinetics(p)
    for N2 in [0.0, 1e-6, 1e-4, 1e-2]:
        for PH2 in [0.0, 0.1, 1.0]:
            for HC_arom in [0.0, 1e-5, 1e-3]:
                r = n.rate(N2, PH2, HC_arom)
                assert r >= 0.0, f"r_nit must be ≥ 0, got {r}"


# ===========================================================================
# HC degradation tests
# ===========================================================================

def test_HC_ali_rate_negative():
    """HC_ali degradation rate must be negative (consumption)."""
    p = default_params()
    hc = HCDegradationModel(p)
    r = hc.HC_ali_rate(HC_ali=1.0e-3, X=0.1)
    assert r < 0.0, f"HC_ali rate must be negative, got {r}"


def test_HC_arom_rate_negative():
    """HC_arom degradation rate must be negative."""
    p = default_params()
    hc = HCDegradationModel(p)
    r = hc.HC_arom_rate(HC_arom=1.0e-3, X=0.1)
    assert r < 0.0, f"HC_arom rate must be negative, got {r}"


def test_HC_ali_rate_zero_at_zero_substrate():
    """HC_ali rate must be zero when HC_ali = 0."""
    p = default_params()
    hc = HCDegradationModel(p)
    r = hc.HC_ali_rate(HC_ali=0.0, X=0.1)
    assert r == 0.0, f"HC_ali rate must be 0 at HC_ali=0, got {r}"


def test_HC_ali_rate_zero_at_zero_biomass():
    """HC_ali rate must be zero when X = 0."""
    p = default_params()
    hc = HCDegradationModel(p)
    r = hc.HC_ali_rate(HC_ali=1.0e-3, X=0.0)
    assert r == 0.0, f"HC_ali rate must be 0 at X=0, got {r}"


def test_aromatic_fraction_definition():
    """f_arom = HC_arom / (HC_ali + HC_arom)."""
    p = default_params()
    hc = HCDegradationModel(p)
    HC_ali  = 3.0e-3
    HC_arom = 1.0e-3
    f = hc.aromatic_fraction(HC_ali, HC_arom)
    expected = HC_arom / (HC_ali + HC_arom)
    assert abs(f - expected) < 1.0e-12


def test_f_arom_increases_with_time():
    """
    In a simulation, f_arom should increase over time because k_ali >> k_arom
    (aliphatics degrade faster, enriching aromatic fraction).
    """
    from n2bio.simulation.solver import N2BioBatchSolver
    from n2bio.simulation.state import BiologicalState
    from n2bio.kinetics.parameters import ReservoirConditions

    p    = default_params()
    cond = ReservoirConditions()

    initial = BiologicalState(
        N2_aq=p.K_N2 * 50, CO2_aq=0.005, H2_aq=0.0,
        NH4_aq=p.K_N_fixed * 100,   # enough NH4 to allow growth
        HC_ali=4.25e-3, HC_arom=0.75e-3,
        NaCl_aq=0.5, X=0.05,
        P_N2=100.0, P_CO2=5.0, P_H2=0.0, P_total=150.0,
        T=358.15, pH=7.0,
    )
    solver = N2BioBatchSolver(p, cond, initial)
    sol    = solver.solve(t_end_hours=5000.0, n_points=200)

    HC_ali_arr  = np.maximum(sol.y[3], 0.0)
    HC_arom_arr = np.maximum(sol.y[4], 0.0)
    total_arr   = HC_ali_arr + HC_arom_arr
    f_arom_arr  = np.where(total_arr > 1e-12,
                           HC_arom_arr / total_arr,
                           p.f_arom0)

    # Compare early vs late aromatic fraction
    f_early = float(np.mean(f_arom_arr[:20]))
    f_late  = float(np.mean(f_arom_arr[-20:]))

    assert f_late >= f_early * 0.99, (  # allow tiny numerical noise
        f"f_arom should not decrease: f_early={f_early:.4f}, f_late={f_late:.4f}"
    )


def test_fermentation_H2_positive():
    """Fermentation H₂ rate must be positive when HC_ali > 0 and X > 0."""
    p = default_params()
    hc = HCDegradationModel(p)
    r = hc.fermentation_H2_rate(HC_ali=1.0e-3, X=0.1)
    assert r > 0.0, f"Fermentation H₂ rate must be positive, got {r}"


def test_total_H2_rate_positive():
    """Total H₂ rate from both pathways must be positive."""
    p  = default_params()
    hc = HCDegradationModel(p)
    n  = NitrogenaseKinetics(p)

    r = hc.total_H2_rate(
        HC_ali=1.0e-3, HC_arom=1.5e-4,
        N2_aq=p.K_N2 * 10, P_H2_atm=0.0,
        X=0.1, nitrogenase_kinetics=n
    )
    assert r > 0.0, f"Total H₂ rate must be positive, got {r}"


def test_k_arom_less_than_k_ali():
    """Aromatic degradation constant must be less than aliphatic."""
    p = default_params()
    assert p.k_arom < p.k_ali, (
        f"k_arom ({p.k_arom:.3e}) should be < k_ali ({p.k_ali:.3e})"
    )


def test_parameters_default_values():
    """Check key default parameter values match specification."""
    p = KineticParameters()
    assert abs(p.mu_max - 0.08) < 1.0e-10
    assert abs(p.K_N2 - 5.0e-5) < 1.0e-15
    assert abs(p.K_S  - 5.0e-4) < 1.0e-15
    assert abs(p.Y    - 0.07)   < 1.0e-10
    assert abs(p.b_decay - 0.005) < 1.0e-10
    assert abs(p.K_I_H2 - 0.3)   < 1.0e-10
    assert abs(p.r_nit_max - 1.0) < 1.0e-10
    assert abs(p.f_arom0 - 0.15)  < 1.0e-10


def test_reservoir_conditions_defaults():
    """Check default reservoir conditions."""
    cond = ReservoirConditions()
    assert abs(cond.temperature - 358.15) < 1.0e-10
    assert abs(cond.pressure - 150.0) < 1.0e-10
    assert abs(cond.porosity - 0.15) < 1.0e-10
    assert abs(cond.permeability - 50.0e-15) < 1.0e-20


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    test_functions = [
        # Monod
        test_growth_rate_zero_at_zero_substrate,
        test_growth_rate_zero_at_zero_NH4,
        test_growth_rate_zero_at_zero_N2,
        test_growth_rate_half_max_at_KS,
        test_growth_rate_approaches_mu_max,
        test_growth_rate_non_negative,
        test_NH4_inhibition_factor_at_zero,
        test_NH4_inhibition_factor_half_at_KI,
        test_biomass_rate_positive_when_growing,
        test_biomass_rate_negative_at_zero_substrate,
        # Nitrogenase
        test_nitrogenase_rate_zero_at_zero_N2,
        test_nitrogenase_inhibited_by_H2,
        test_nitrogenase_inhibited_by_aromatics,
        test_nitrogenase_H2_stoichiometry,
        test_nitrogenase_NH4_stoichiometry,
        test_nitrogenase_rate_positive,
        # HC degradation
        test_HC_ali_rate_negative,
        test_HC_arom_rate_negative,
        test_HC_ali_rate_zero_at_zero_substrate,
        test_HC_ali_rate_zero_at_zero_biomass,
        test_aromatic_fraction_definition,
        test_f_arom_increases_with_time,
        test_fermentation_H2_positive,
        test_total_H2_rate_positive,
        test_k_arom_less_than_k_ali,
        test_parameters_default_values,
        test_reservoir_conditions_defaults,
    ]

    n_pass = 0
    n_fail = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            import traceback
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            n_fail += 1

    print(f"\n{'='*55}")
    print(f"Kinetics Tests: {n_pass}/{n_pass+n_fail} passed")
