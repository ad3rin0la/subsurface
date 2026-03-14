"""
verification/stage3_kinetics.py
================================
Stage 3 kinetics verification tests for N2Bio.

Verifies that the Monod, nitrogenase, and HC degradation kinetic
expressions give physically correct results at limiting conditions:
  - Zero rates at zero substrate
  - Half-maximum rate at K_S, K_N2, K_N_fixed
  - Correct inhibition behaviour
  - Biomass ODE self-consistency
"""

import numpy as np
from typing import Dict, Any, List


def verify_monod_kinetics(verbose: bool = False) -> Dict[str, Any]:
    """
    Verify Monod growth kinetics satisfy theoretical limiting cases.

    Returns
    -------
    dict  Keys: test_name → {value, expected, pass, rel_error}.
    """
    from ..kinetics.parameters import KineticParameters
    from ..kinetics.monod import MonodKinetics

    p = KineticParameters()
    m = MonodKinetics(p)

    results = {}

    # ---- Zero substrate → zero growth ----
    mu_zero = m.growth_rate(HC_ali=0.0, NH4=p.K_N_fixed * 10, N2_aq=p.K_N2 * 10)
    results['zero_HC_zero_growth'] = {
        'value':    mu_zero,
        'expected': 0.0,
        'pass':     mu_zero == 0.0,
    }

    # ---- Half-max at K_S ----
    mu_half_S = m.growth_rate(
        HC_ali=p.K_S, NH4=p.K_N_fixed * 1000, N2_aq=p.K_N2 * 1000
    )
    expected_half = p.mu_max * 0.5
    rel_err = abs(mu_half_S - expected_half) / expected_half
    results['half_max_at_KS'] = {
        'value':     mu_half_S,
        'expected':  expected_half,
        'rel_error': rel_err,
        'pass':      rel_err < 1.0e-6,
    }

    # ---- Half-max at K_N_fixed ----
    mu_half_N = m.growth_rate(
        HC_ali=p.K_S * 1000, NH4=p.K_N_fixed, N2_aq=p.K_N2 * 1000
    )
    expected_half_N = p.mu_max * 0.5
    rel_err_N = abs(mu_half_N - expected_half_N) / expected_half_N
    results['half_max_at_K_N_fixed'] = {
        'value':     mu_half_N,
        'expected':  expected_half_N,
        'rel_error': rel_err_N,
        'pass':      rel_err_N < 1.0e-6,
    }

    # ---- Half-max at K_N2 ----
    mu_half_N2 = m.growth_rate(
        HC_ali=p.K_S * 1000, NH4=p.K_N_fixed * 1000, N2_aq=p.K_N2
    )
    expected_half_N2 = p.mu_max * 0.5
    rel_err_N2 = abs(mu_half_N2 - expected_half_N2) / expected_half_N2
    results['half_max_at_KN2'] = {
        'value':     mu_half_N2,
        'expected':  expected_half_N2,
        'rel_error': rel_err_N2,
        'pass':      rel_err_N2 < 1.0e-6,
    }

    # ---- mu_max approached at saturation ----
    mu_sat = m.growth_rate(
        HC_ali=p.K_S * 1e5, NH4=p.K_N_fixed * 1e5, N2_aq=p.K_N2 * 1e5
    )
    rel_sat = abs(mu_sat - p.mu_max) / p.mu_max
    results['saturation_approaches_mu_max'] = {
        'value':     mu_sat,
        'expected':  p.mu_max,
        'rel_error': rel_sat,
        'pass':      rel_sat < 1.0e-4,
    }

    # ---- NH4 inhibition factor ----
    f_inhib_zero = m.NH4_inhibition_factor(0.0)
    results['NH4_inhibition_zero_NH4'] = {
        'value':    f_inhib_zero,
        'expected': 1.0,
        'pass':     abs(f_inhib_zero - 1.0) < 1.0e-10,
    }
    f_inhib_half = m.NH4_inhibition_factor(p.K_I_NH4)
    results['NH4_inhibition_half_at_KI'] = {
        'value':    f_inhib_half,
        'expected': 0.5,
        'pass':     abs(f_inhib_half - 0.5) < 1.0e-10,
    }

    if verbose:
        _print_results('Monod Kinetics', results)

    return results


def verify_nitrogenase_kinetics(verbose: bool = False) -> Dict[str, Any]:
    """
    Verify nitrogenase rate expressions.

    Returns
    -------
    dict
    """
    from ..kinetics.parameters import KineticParameters
    from ..kinetics.nitrogenase import NitrogenaseKinetics

    p = KineticParameters()
    n = NitrogenaseKinetics(p)

    results = {}

    # ---- Zero N₂ → zero rate ----
    r_zero = n.rate(N2_aq=0.0, P_H2_atm=0.0, HC_arom_aq=0.0)
    results['zero_N2_zero_rate'] = {
        'value': r_zero, 'expected': 0.0, 'pass': r_zero == 0.0
    }

    # ---- H₂ inhibition: rate at K_I_H2 should be 0.5 * rate at P_H2=0 ----
    r_no_H2   = n.rate(N2_aq=p.K_N2 * 1000, P_H2_atm=0.0, HC_arom_aq=0.0)
    r_half_H2 = n.rate(N2_aq=p.K_N2 * 1000, P_H2_atm=p.K_I_H2, HC_arom_aq=0.0)
    rel_err_H2 = abs(r_half_H2 / r_no_H2 - 0.5)
    results['H2_inhibition_half_at_KI_H2'] = {
        'value': r_half_H2 / r_no_H2, 'expected': 0.5,
        'rel_error': rel_err_H2,
        'pass': rel_err_H2 < 1.0e-6,
    }

    # ---- Aromatic inhibition ----
    r_no_arom   = n.rate(N2_aq=p.K_N2 * 1000, P_H2_atm=0.0, HC_arom_aq=0.0)
    r_half_arom = n.rate(N2_aq=p.K_N2 * 1000, P_H2_atm=0.0,
                         HC_arom_aq=p.K_I_arom)
    rel_err_arom = abs(r_half_arom / r_no_arom - 0.5)
    results['aromatic_inhibition_half_at_KI_arom'] = {
        'value': r_half_arom / r_no_arom, 'expected': 0.5,
        'rel_error': rel_err_arom,
        'pass': rel_err_arom < 1.0e-6,
    }

    # ---- NH4 stoichiometry: 2x H2 production ----
    X = 0.1  # g VSS/L
    r_H2  = n.H2_production_rate( p.K_N2 * 10, 0.0, 0.0, X)
    r_NH4 = n.NH4_production_rate(p.K_N2 * 10, 0.0, 0.0, X)
    stoich_ratio = r_NH4 / r_H2 if r_H2 > 0 else 0
    results['NH4_H2_stoichiometry_2_to_1'] = {
        'value': stoich_ratio, 'expected': 2.0,
        'rel_error': abs(stoich_ratio - 2.0),
        'pass': abs(stoich_ratio - 2.0) < 1.0e-9,
    }

    if verbose:
        _print_results('Nitrogenase Kinetics', results)

    return results


def verify_HC_degradation_kinetics(verbose: bool = False) -> Dict[str, Any]:
    """
    Verify HC degradation kinetic expressions.

    Returns
    -------
    dict
    """
    from ..kinetics.parameters import KineticParameters
    from ..kinetics.hc_degradation import HCDegradationModel

    p = KineticParameters()
    hc = HCDegradationModel(p)

    results = {}

    # ---- HC_ali rate is negative (consumption) ----
    r_ali = hc.HC_ali_rate(HC_ali=1.0e-3, X=0.1)
    results['HC_ali_rate_negative'] = {
        'value': r_ali, 'pass': r_ali < 0.0,
        'description': 'HC_ali rate should be negative (consumption)'
    }

    # ---- HC_arom rate is negative ----
    r_arom = hc.HC_arom_rate(HC_arom=1.0e-3, X=0.1)
    results['HC_arom_rate_negative'] = {
        'value': r_arom, 'pass': r_arom < 0.0,
        'description': 'HC_arom rate should be negative (consumption)'
    }

    # ---- f_arom when only aromatics remain ----
    f = hc.aromatic_fraction(HC_ali=0.0, HC_arom=1.0e-3)
    results['f_arom_all_aromatic'] = {
        'value': f, 'expected': 1.0, 'pass': abs(f - 1.0) < 1.0e-10
    }

    # ---- f_arom = 0 when only aliphatics ----
    f2 = hc.aromatic_fraction(HC_ali=1.0e-3, HC_arom=0.0)
    results['f_arom_all_aliphatic'] = {
        'value': f2, 'expected': 0.0, 'pass': f2 == 0.0
    }

    # ---- H2 fermentation rate positive ----
    r_H2 = hc.fermentation_H2_rate(HC_ali=1.0e-3, X=0.1)
    results['fermentation_H2_positive'] = {
        'value': r_H2, 'pass': r_H2 > 0.0,
    }

    # ---- k_arom < k_ali (aromatics degrade slower) ----
    results['k_arom_lt_k_ali'] = {
        'value':    p.k_arom / p.k_ali,
        'expected': '< 1',
        'pass':     p.k_arom < p.k_ali,
    }

    if verbose:
        _print_results('HC Degradation Kinetics', results)

    return results


def _print_results(name: str, results: Dict) -> None:
    """Pretty-print a results dict."""
    print(f"\n{'='*55}")
    print(f"Stage 3 Verification: {name}")
    print(f"{'='*55}")
    n_pass = sum(1 for r in results.values() if r.get('pass', False))
    print(f"  Passed: {n_pass}/{len(results)}")
    for test_name, r in results.items():
        marker = 'OK' if r.get('pass', False) else 'FAIL'
        val_str = f"{r.get('value', '?'):.6g}" if isinstance(r.get('value'), float) else str(r.get('value', '?'))
        exp_str = f"exp={r.get('expected', '?')}" if 'expected' in r else ''
        err_str = f"rel_err={r.get('rel_error', 0):.2e}" if 'rel_error' in r else ''
        print(f"  [{marker}] {test_name}: {val_str}  {exp_str}  {err_str}")


def run_all_stage3_verifications(verbose: bool = True) -> Dict[str, Dict]:
    """
    Run all Stage 3 kinetics verifications.

    Parameters
    ----------
    verbose : bool

    Returns
    -------
    dict  Keys are test group names.
    """
    all_results = {
        'monod':        verify_monod_kinetics(verbose),
        'nitrogenase':  verify_nitrogenase_kinetics(verbose),
        'hc_degrad':    verify_HC_degradation_kinetics(verbose),
    }

    if verbose:
        total_pass  = sum(
            sum(1 for r in group.values() if r.get('pass', False))
            for group in all_results.values()
        )
        total_tests = sum(len(group) for group in all_results.values())
        print(f"\nStage 3 Overall: {total_pass}/{total_tests} tests passed.")

    return all_results


if __name__ == '__main__':
    run_all_stage3_verifications(verbose=True)
