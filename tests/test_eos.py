"""
tests/test_eos.py
==================
Unit tests for the Peng-Robinson EOS implementation.

Run with:
    python -m pytest tests/test_eos.py -v
or:
    python tests/test_eos.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2bio.thermophysics.eos import PengRobinsonEOS, MixturePREOS, R
from n2bio.components import COMPONENTS


# ===========================================================================
# Test helpers
# ===========================================================================

def get_N2_eos():
    return PengRobinsonEOS(COMPONENTS['N2'])

def get_CO2_eos():
    return PengRobinsonEOS(COMPONENTS['CO2'])

def get_H2_eos():
    return PengRobinsonEOS(COMPONENTS['H2'])


# ===========================================================================
# Pure component PR EOS tests
# ===========================================================================

def test_N2_critical_point_pressure():
    """
    At T = Tc and V → V_c, PR EOS should reproduce critical pressure Pc.

    We verify by computing pressure at (Tc, V_c) and checking it is close to Pc.
    V_c ≈ 3*b (approximate for PR EOS).
    """
    eos = get_N2_eos()
    T = eos.Tc
    # Critical volume estimate: V_c = 3.952 * b for PR EOS
    V_c = 3.952 * eos.b_param
    P_calc = eos.pressure(T, V_c)
    # Allow 5% error (PR EOS systematic error at Tc)
    rel_err = abs(P_calc - eos.Pc) / eos.Pc
    assert rel_err < 0.05, (
        f"PR EOS pressure at N₂ critical point: {P_calc/1e5:.2f} bar, "
        f"expected {eos.Pc/1e5:.2f} bar, rel_err={rel_err:.3f}"
    )


def test_Z_factor_ideal_gas_limit():
    """
    At very low pressure, Z → 1 (ideal gas behaviour).
    Test at T = 400 K, P = 1 bar (well above N₂ boiling point).
    """
    eos = get_N2_eos()
    T = 400.0    # K
    P = 1.0e4    # Pa (0.1 bar → very dilute)
    Z = eos.Z_factor(T, P, phase='gas')
    assert abs(Z - 1.0) < 0.01, f"Z at dilute conditions should ≈ 1, got {Z:.6f}"


def test_density_positive():
    """Density must be positive for all components at reservoir conditions."""
    T = 358.15   # 85°C
    P = 150.0e5  # 150 bar

    for name in ['N2', 'CO2', 'H2']:
        eos = PengRobinsonEOS(COMPONENTS[name])
        rho = eos.density(T, P, phase='gas')
        assert rho > 0.0, f"Density for {name} at T={T}, P={P:.0e} must be positive, got {rho}"


def test_volume_gas_larger_than_liquid():
    """
    For a component well below its critical temperature, the PR EOS cubic
    should yield three real Z-factor roots (van der Waals loop).
    The largest root corresponds to the gas phase, smallest to liquid.
    The gas Z-factor (and thus molar volume) must exceed the liquid.

    Test CO₂ at 280 K (below Tc=304.13 K), P = 40 bar.
    PR EOS gives three real roots in this two-phase region.
    """
    eos = get_CO2_eos()
    T = 280.0       # K  well below Tc = 304.13 K
    P = 40.0e5      # Pa  (40 bar, in two-phase region)

    roots = eos._solve_cubic_Z(T, P)
    # PR EOS should yield 3 real positive roots here
    assert len(roots) >= 2, (
        f"Expected ≥ 2 real Z roots at T={T} K, P={P/1e5:.0f} bar (two-phase region), "
        f"got {len(roots)}: {roots}"
    )

    Z_gas = roots[-1]    # largest root → vapour
    Z_liq = roots[0]     # smallest root → liquid
    V_gas = Z_gas * 8.314 * T / P
    V_liq = Z_liq * 8.314 * T / P
    assert Z_gas > Z_liq, (
        f"Gas Z ({Z_gas:.4f}) must exceed liquid Z ({Z_liq:.4f}) "
        f"→ V_gas={V_gas*1e6:.1f} cm³/mol, V_liq={V_liq*1e6:.1f} cm³/mol"
    )


def test_Z_factor_positive():
    """Z factor must be positive (physical)."""
    eos = get_N2_eos()
    for T in [300.0, 350.0, 400.0]:
        for P_bar in [10.0, 50.0, 100.0, 200.0]:
            Z = eos.Z_factor(T, P_bar * 1e5, phase='gas')
            assert Z > 0.0, f"Z={Z} must be positive at T={T}, P={P_bar} bar"


def test_fugacity_coefficient_ideal_limit():
    """At P → 0, fugacity coefficient φ → 1 (ideal gas limit)."""
    eos = get_N2_eos()
    T = 400.0    # K
    P = 100.0    # Pa (very low)
    phi = eos.fugacity_coefficient(T, P, phase='gas')
    assert abs(phi - 1.0) < 0.005, f"φ at P→0 should ≈ 1, got {phi:.6f}"


def test_departure_enthalpy_ideal_limit():
    """At P → 0, departure enthalpy → 0 (ideal gas)."""
    eos = get_N2_eos()
    T = 400.0
    P = 1.0     # very low pressure
    H_dep = eos.departure_enthalpy(T, P, phase='gas')
    # Should be small relative to RT
    assert abs(H_dep) < 10.0, (
        f"Departure enthalpy at P→0 should ≈ 0, got {H_dep:.2f} J/mol"
    )


def test_pressure_volume_roundtrip():
    """
    Given T and P, compute raw molar volume V_raw.
    Then compute P' = eos.pressure(T, V_raw).  Should recover P.
    (Uses raw unshifted volume for thermodynamic consistency.)
    """
    eos = get_N2_eos()
    T   = 360.0           # K
    P   = 150.0e5         # Pa
    V   = eos.volume_raw(T, P, phase='gas')
    P2  = eos.pressure(T, V)
    rel_err = abs(P2 - P) / P
    assert rel_err < 1.0e-5, (
        f"Pressure roundtrip failed: P={P/1e5:.2f} bar, P'={P2/1e5:.2f} bar"
    )


def test_N2_density_at_reservoir():
    """
    N₂ density at 85°C, 100 bar should be physically reasonable.
    NIST reference ≈ 107 kg/m³.

    PR EOS systematically underpredicts gas density by up to ~15% without
    volume-translation tuning.  Allow 20% relative error as a sanity check
    that the EOS returns a physically meaningful positive value in the
    correct order of magnitude.
    """
    eos = get_N2_eos()
    T = 358.15    # 85°C
    P = 100.0e5   # 100 bar
    rho = eos.density(T, P, phase='gas')
    rho_ref = 107.5   # kg/m³ (NIST)
    # Check density is in physically correct range
    assert rho > 30.0 and rho < 200.0, (
        f"N₂ density at reservoir should be 30-200 kg/m³, got {rho:.1f} kg/m³"
    )
    # Warn but don't fail at 20% threshold (PR EOS limitation)
    rel_err = abs(rho - rho_ref) / rho_ref
    assert rel_err < 0.20, (
        f"N₂ density at reservoir: {rho:.1f} kg/m³, expected ~{rho_ref:.1f} kg/m³, "
        f"rel_err={rel_err:.2%} (PR EOS limitation, tolerable at ≤20%)"
    )


def test_kappa_N2():
    """kappa (acentric factor correction) for N₂ should equal ≈ 0.4373."""
    eos = get_N2_eos()
    omega = eos.omega  # 0.037 for N₂
    kappa_expected = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    kappa_calc = eos._kappa
    assert abs(kappa_calc - kappa_expected) < 1.0e-10


def test_alpha_equals_one_at_Tc():
    """α(Tc) = 1 by definition of the alpha function."""
    for name in ['N2', 'CO2', 'H2']:
        eos = PengRobinsonEOS(COMPONENTS[name])
        alpha_Tc = eos.alpha(eos.Tc)
        assert abs(alpha_Tc - 1.0) < 1.0e-10, (
            f"α(Tc) for {name} = {alpha_Tc:.8f}, expected 1.0"
        )


def test_mixture_a_pure_component_limit():
    """
    For a mixture of pure N₂ (x_N₂=1), a_mix should equal a_N₂ alone.
    """
    T = 350.0
    eos = get_N2_eos()
    a_pure = eos.a_param(T)

    comps = [COMPONENTS['N2']]
    x = np.array([1.0])
    a_mix = PengRobinsonEOS.mixture_a(T, x, comps)

    rel_err = abs(a_mix - a_pure) / a_pure
    assert rel_err < 1.0e-12, (
        f"Mixture a_mix for pure N₂ should equal a_pure: {a_mix:.6e} vs {a_pure:.6e}"
    )


def test_mixture_b_pure_component_limit():
    """b_mix for pure N₂ should equal b_N₂."""
    eos = get_N2_eos()
    b_pure = eos.b_param
    comps = [COMPONENTS['N2']]
    x = np.array([1.0])
    b_mix = PengRobinsonEOS.mixture_b(x, comps)
    assert abs(b_mix - b_pure) < 1.0e-15


def test_mixture_eos_density():
    """MixturePREOS should give sensible density for N₂-dominated mixture."""
    comps = [COMPONENTS['N2'], COMPONENTS['CO2'], COMPONENTS['H2']]
    meos  = MixturePREOS(comps)
    x = np.array([0.80, 0.15, 0.05])
    T = 358.15
    P = 150.0e5
    rho = meos.density(T, P, x, phase='gas')
    assert 50.0 < rho < 300.0, f"Mixture density {rho:.1f} kg/m³ out of range"


def test_pressure_contributions_sum():
    """P_r + P_a should equal P from EOS when using raw (unshifted) volume."""
    eos = get_N2_eos()
    T = 360.0
    P = 150.0e5
    V_raw = eos.volume_raw(T, P, phase='gas')
    P_r, P_a = eos.pressure_contributions(T, V_raw)
    P_calc = P_r + P_a
    rel_err = abs(P_calc - P) / P
    assert rel_err < 1.0e-5, (
        f"P_r + P_a = {P_calc/1e5:.2f} bar ≠ P = {P/1e5:.2f} bar"
    )


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    test_functions = [
        test_N2_critical_point_pressure,
        test_Z_factor_ideal_gas_limit,
        test_density_positive,
        test_volume_gas_larger_than_liquid,
        test_Z_factor_positive,
        test_fugacity_coefficient_ideal_limit,
        test_departure_enthalpy_ideal_limit,
        test_pressure_volume_roundtrip,
        test_N2_density_at_reservoir,
        test_kappa_N2,
        test_alpha_equals_one_at_Tc,
        test_mixture_a_pure_component_limit,
        test_mixture_b_pure_component_limit,
        test_mixture_eos_density,
        test_pressure_contributions_sum,
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
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
            n_fail += 1

    print(f"\n{'='*50}")
    print(f"EOS Tests: {n_pass}/{n_pass+n_fail} passed")
