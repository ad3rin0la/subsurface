"""
examples/field_data_io.py
=========================
Demonstrates the n2bio.io module: loading field data from CSV, LAS,
JSON, and plain dicts, then feeding the result into N2BioBatchSolver.

Run from the repo root::

    python examples/field_data_io.py

No external dependencies beyond n2bio itself (and numpy/scipy).
The script generates its own synthetic test files in /tmp.
"""

import io
import json
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2bio.io import ConditionsBuilder, load_field_data, convert_record


# ===========================================================================
# 1.  Build from a plain Python dict (industry-standard field names)
# ===========================================================================

def demo_from_dict():
    print("\n" + "=" * 55)
    print("1. Building ReservoirConditions from a dict")
    print("=" * 55)

    cond = ConditionsBuilder.from_dict(
        {
            "BHT":          87.3,    # bottom-hole temperature in °C
            "pressure_psi": 2176,    # ~150 bar
            "phi":          0.14,    # porosity as fraction
            "perm":         48.0,    # mD
            "TDS":          18_500,  # ppm NaCl  (~0.316 mol/kg)
            "ph":           6.9,
        },
        warn_missing=True,
        warn_unusual=False,
    )
    print(cond.summary())
    return cond


# ===========================================================================
# 2.  Build from a CSV file (synthetic core-analysis table)
# ===========================================================================

def demo_from_csv():
    print("\n" + "=" * 55)
    print("2. Building ReservoirConditions from a CSV file")
    print("=" * 55)

    csv_content = """\
Depth_m,Temperature_C,Pressure_bar,Porosity,Permeability,Salinity_ppm,pH
2080,83.1,148.0,0.132,38.0,20000,7.1
2100,85.4,150.5,0.141,46.0,18800,7.0
2120,86.1,151.8,0.148,52.0,18200,6.9
2140,87.3,153.0,0.155,61.0,17500,6.8
2160,88.0,154.2,0.149,55.0,17900,6.9
2200,91.0,158.0,0.121,29.0,22000,7.2
"""
    # Write to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        cond = ConditionsBuilder.from_file(
            tmp_path,
            depth_interval=(2100.0, 2160.0),   # restrict to pay zone
            aggregation="median",
        )
        print(f"Source: {tmp_path}")
        print(f"Depth interval: 2100–2160 m  (aggregation: median)")
        print(cond.summary())
    finally:
        os.unlink(tmp_path)

    return cond


# ===========================================================================
# 3.  Build from a LAS 2.0 file (synthetic wireline log)
# ===========================================================================

def demo_from_las():
    print("\n" + "=" * 55)
    print("3. Building ReservoirConditions from a LAS 2.0 file")
    print("=" * 55)

    las_content = """\
~VERSION INFORMATION
 VERS.                    2.0  : CWLS LOG ASCII STANDARD -VERSION 2.0
 WRAP.                     NO  : ONE LINE PER DEPTH STEP
~WELL INFORMATION
 STRT.M                2000.0  : START DEPTH
 STOP.M                2300.0  : STOP DEPTH
 STEP.M                   1.0  : STEP
 NULL.                 -9999.25: NULL VALUE
 WELL.                   A-01  : WELL NAME
 BHT .DEGC               87.0  : BOTTOM HOLE TEMPERATURE
 RES_PRES.BAR           151.5  : RESERVOIR PRESSURE
~CURVE INFORMATION
 DEPT.M                        : DEPTH
 PHIE.V/V                      : EFFECTIVE POROSITY
 PERM.MD                       : PERMEABILITY
 TDS .PPM                      : TOTAL DISSOLVED SOLIDS
 PH  .                         : pH
~A  DEPTH   PHIE   PERM    TDS     PH
2100.0  0.132  38.0  20000.0  7.1
2110.0  0.141  46.0  18800.0  7.0
2120.0  0.148  52.0  18200.0  6.9
2130.0  0.155  61.0  17500.0  6.8
2140.0  0.149  55.0  17900.0  6.9
2150.0  0.138  44.0  19100.0  7.0
2160.0  0.129  35.0  20500.0  7.1
2200.0  0.091  12.0  26000.0  7.3
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".las", delete=False, encoding="utf-8"
    ) as f:
        f.write(las_content)
        tmp_path = f.name

    try:
        cond = ConditionsBuilder.from_file(
            tmp_path,
            depth_interval=(2100.0, 2160.0),
            aggregation="median",
        )
        print(f"Source: {tmp_path}")
        print(cond.summary())
    finally:
        os.unlink(tmp_path)

    return cond


# ===========================================================================
# 4.  Build from a JSON field card
# ===========================================================================

def demo_from_json():
    print("\n" + "=" * 55)
    print("4. Building ReservoirConditions from a JSON file")
    print("=" * 55)

    card = {
        "well": "A-01",
        "operator": "Anabaena Energy",
        "reservoir": {
            "temperature":   87.3,     # °C
            "pressure_bar":  152.0,
            "porosity":      0.148,
            "permeability":  52.0,     # mD
            "salinity_ppm":  18_200,
            "pH":            6.9,
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(card, f, indent=2)
        tmp_path = f.name

    try:
        cond = ConditionsBuilder.from_file(tmp_path)
        print(f"Source: {tmp_path}")
        print(cond.summary())
    finally:
        os.unlink(tmp_path)

    return cond


# ===========================================================================
# 5.  inspect_file — debug a file without building conditions
# ===========================================================================

def demo_inspect():
    print("\n" + "=" * 55)
    print("5. Inspecting a CSV file before building")
    print("=" * 55)

    csv_content = """\
Depth_m,BHT,Pressure_psi,phi,perm,TDS
2100,87.3,2205,14.8,52.0,18200
2120,88.1,2220,15.5,61.0,17500
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        info = ConditionsBuilder.inspect_file(tmp_path)
        print(f"Raw parsed fields:   {list(info['raw'].keys())}")
        print(f"Converted values:")
        for k, v in info["converted"].items():
            print(f"  {k:20s} = {v:.6g}")
        if info["missing"]:
            print(f"Missing RC fields:   {info['missing']}")
        if info["warnings"]:
            for w in info["warnings"]:
                print(f"  [WARN] {w}")
    finally:
        os.unlink(tmp_path)


# ===========================================================================
# 6.  Full simulation using parsed conditions
# ===========================================================================

def demo_full_simulation():
    print("\n" + "=" * 55)
    print("6. Running a batch simulation with field-derived conditions")
    print("=" * 55)

    # Parse from dict — no temp file needed
    cond = ConditionsBuilder.from_dict(
        {
            "temperature_C":  87.3,
            "pressure_bar":   152.0,
            "porosity":       0.148,
            "perm":           52.0,    # mD
            "salinity_ppm":   18_200,
            "pH":             6.9,
        },
        warn_missing=False,
        warn_unusual=False,
    )
    print("Conditions derived from field data:")
    print(cond.summary())

    # Run simulation
    from n2bio.kinetics.parameters import KineticParameters
    from n2bio.simulation.state import BiologicalState
    from n2bio.simulation.solver import N2BioBatchSolver

    params = KineticParameters()
    HC_total = 5.0e-3
    initial = BiologicalState(
        N2_aq   = params.K_N2 * 10.0,
        CO2_aq  = 0.005,
        H2_aq   = 0.0,
        NH4_aq  = 1.0e-7,
        HC_ali  = HC_total * (1.0 - params.f_arom0),
        HC_arom = HC_total * params.f_arom0,
        NaCl_aq = cond.salinity_NaCl,
        X       = 0.005,
        P_N2    = 100.0,
        P_CO2   = 5.0,
        P_H2    = 0.01,
        P_total = cond.pressure,
        T       = cond.temperature,
        pH      = cond.pH,
    )

    solver = N2BioBatchSolver(params, cond, initial)
    sol, phases = solver.solve_with_lifecycle(t_end_hours=3000, n_points=200)

    print(f"\nSimulation complete:")
    print(f"  Final cumulative H₂ : {sol.y[6, -1]*1e6:.2f} μmol/L")

    phase_changes = []
    for i, ph in enumerate(phases):
        if i == 0 or phases[i] != phases[i-1]:
            phase_changes.append(f"t={sol.t[i]:.0f} h  Phase {int(ph)}: {ph.name}")
    for line in phase_changes:
        print(f"  {line}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    demo_from_dict()
    demo_from_csv()
    demo_from_las()
    demo_from_json()
    demo_inspect()
    demo_full_simulation()
    print("\nAll demos complete.")
