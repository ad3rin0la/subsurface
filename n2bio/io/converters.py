"""
io/converters.py
================
Unit conversion utilities for field data destined for ``ReservoirConditions``.

All functions are pure scalar → scalar transforms.  The ``convert_record``
helper applies the full battery to a ``FieldRecord`` returned by the parsers
and resolves competing representations (e.g. temperature given in both °C
and °F) into a single canonical value in the N2Bio internal unit system.

N2Bio internal units (matching ``ReservoirConditions``)
-------------------------------------------------------
temperature  → Kelvin  [K]
pressure     → bar
porosity     → dimensionless  [-]   (fraction, not percent)
permeability → m²
salinity     → mol/kg NaCl (molality)
pH           → dimensionless  [-]
"""

from __future__ import annotations

from typing import Optional
from .parsers import FieldRecord


# ---------------------------------------------------------------------------
# Temperature converters
# ---------------------------------------------------------------------------

def celsius_to_kelvin(T_C: float) -> float:
    """°C → K."""
    return T_C + 273.15


def fahrenheit_to_kelvin(T_F: float) -> float:
    """°F → K."""
    return (T_F - 32.0) * 5.0 / 9.0 + 273.15


def kelvin_to_kelvin(T_K: float) -> float:
    """No-op; accepts K, returns K."""
    return float(T_K)


# ---------------------------------------------------------------------------
# Pressure converters
# ---------------------------------------------------------------------------

def bar_to_bar(P: float) -> float:
    return float(P)

def psi_to_bar(P_psi: float) -> float:
    """psi → bar  (1 psi = 0.0689476 bar)."""
    return P_psi * 0.0689476

def mpa_to_bar(P_MPa: float) -> float:
    """MPa → bar  (1 MPa = 10 bar)."""
    return P_MPa * 10.0

def kpa_to_bar(P_kPa: float) -> float:
    """kPa → bar  (1 kPa = 0.01 bar)."""
    return P_kPa * 0.01

def pa_to_bar(P_Pa: float) -> float:
    """Pa → bar  (1 Pa = 1e-5 bar)."""
    return P_Pa * 1.0e-5

def atm_to_bar(P_atm: float) -> float:
    """atm → bar  (1 atm = 1.01325 bar)."""
    return P_atm * 1.01325


# ---------------------------------------------------------------------------
# Permeability converters
# ---------------------------------------------------------------------------

def md_to_m2(k_mD: float) -> float:
    """milliDarcy → m²  (1 mD = 9.869233e-16 m²)."""
    return k_mD * 9.869233e-16

def darcy_to_m2(k_D: float) -> float:
    """Darcy → m²  (1 D = 9.869233e-13 m²)."""
    return k_D * 9.869233e-13

def m2_to_m2(k: float) -> float:
    return float(k)


# ---------------------------------------------------------------------------
# Salinity converters
# ---------------------------------------------------------------------------

_MW_NACL = 58.44   # g/mol

def ppm_to_mol_kg(ppm: float) -> float:
    """
    Convert NaCl concentration from ppm (mg/kg) to molality (mol/kg water).

    ppm is treated as mg NaCl per kg solution.  For dilute brines the
    approximation mol/kg ≈ (ppm/1e6) / (MW_NaCl / 1000) is used.

    Parameters
    ----------
    ppm : float  NaCl concentration in mg/kg (ppm by mass).

    Returns
    -------
    float  Molality [mol/kg water].
    """
    # ppm → g NaCl / kg solution → g NaCl / kg water (dilute approximation)
    g_per_kg = ppm / 1000.0          # mg/kg → g/kg
    # For dilute solutions: g_NaCl/kg_water ≈ g_NaCl/kg_solution
    return g_per_kg / _MW_NACL       # mol/kg

def g_L_to_mol_kg(g_L: float) -> float:
    """
    g/L NaCl → mol/kg water.

    Assumes solution density ≈ 1 kg/L (valid for dilute brines).
    """
    return g_L / _MW_NACL

def mol_kg_to_mol_kg(m: float) -> float:
    return float(m)

def percent_to_fraction(v: float) -> float:
    """Porosity given as percent (e.g. 14.0) → fraction (0.14)."""
    return v / 100.0


# ---------------------------------------------------------------------------
# Porosity normalisation
# ---------------------------------------------------------------------------

def normalise_porosity(phi: float) -> float:
    """
    Ensure porosity is in (0, 1).

    If ``phi > 1``, it is assumed to be a percentage and divided by 100.
    Values outside the physical range (0, 1) after normalisation raise
    ValueError.

    Parameters
    ----------
    phi : float  Raw porosity value.

    Returns
    -------
    float  Porosity fraction in (0, 1).
    """
    if phi > 1.0:
        phi = phi / 100.0
    if not (0.0 < phi < 1.0):
        raise ValueError(
            f"Porosity {phi:.4f} is outside the physical range (0, 1).  "
            "Check your input units."
        )
    return phi


# ---------------------------------------------------------------------------
# Master record converter
# ---------------------------------------------------------------------------

def convert_record(record: FieldRecord) -> dict:
    """
    Convert a raw ``FieldRecord`` (from any parser) to N2Bio internal units.

    Handles competing representations by applying a priority hierarchy:
    - Temperature: K > °C > °F
    - Pressure:    bar > MPa > kPa > psi > Pa
    - Permeability: m² > mD > D
    - Salinity:    mol/kg > ppm

    Parameters
    ----------
    record : FieldRecord
        Output of any ``parse_*`` function in ``parsers.py``.

    Returns
    -------
    dict  Keys are N2Bio ``ReservoirConditions`` field names:
          ``temperature``, ``pressure``, ``porosity``, ``permeability``,
          ``salinity_NaCl``, ``pH``.
          Only keys for which data was found are included.

    Raises
    ------
    ValueError
        If a value is present but fails physical sanity checks.
    """
    out: dict = {}

    # ---- Temperature (priority: K > C > F) ----
    if "temperature_K" in record:
        out["temperature"] = kelvin_to_kelvin(record["temperature_K"])
    elif "temperature_C" in record:
        out["temperature"] = celsius_to_kelvin(record["temperature_C"])
    elif "temperature_F" in record:
        out["temperature"] = fahrenheit_to_kelvin(record["temperature_F"])

    # ---- Pressure (priority: bar > MPa > kPa > psi > Pa) ----
    if "pressure_bar" in record:
        out["pressure"] = bar_to_bar(record["pressure_bar"])
    elif "pressure_MPa" in record:
        out["pressure"] = mpa_to_bar(record["pressure_MPa"])
    elif "pressure_kPa" in record:
        out["pressure"] = kpa_to_bar(record["pressure_kPa"])
    elif "pressure_psi" in record:
        out["pressure"] = psi_to_bar(record["pressure_psi"])
    elif "pressure_Pa" in record:
        out["pressure"] = pa_to_bar(record["pressure_Pa"])

    # ---- Porosity ----
    if "porosity" in record:
        out["porosity"] = normalise_porosity(record["porosity"])

    # ---- Permeability (priority: m² > mD > D) ----
    if "permeability_m2" in record:
        out["permeability"] = m2_to_m2(record["permeability_m2"])
    elif "permeability_mD" in record:
        out["permeability"] = md_to_m2(record["permeability_mD"])
    elif "permeability_D" in record:
        out["permeability"] = darcy_to_m2(record["permeability_D"])

    # ---- Salinity (priority: mol/kg > ppm) ----
    if "salinity_mol_kg" in record:
        out["salinity_NaCl"] = mol_kg_to_mol_kg(record["salinity_mol_kg"])
    elif "salinity_ppm" in record:
        out["salinity_NaCl"] = ppm_to_mol_kg(record["salinity_ppm"])

    # ---- pH ----
    if "pH" in record:
        pH = float(record["pH"])
        if not (0.0 <= pH <= 14.0):
            raise ValueError(f"pH value {pH} is outside [0, 14].")
        out["pH"] = pH

    return out


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------

def check_physical_ranges(converted: dict) -> list:
    """
    Return a list of warning strings for values that are physically unusual
    but not outright invalid for a petroleum reservoir.

    Parameters
    ----------
    converted : dict
        Output of :func:`convert_record`.

    Returns
    -------
    list of str  Empty if everything looks normal.
    """
    warnings: list = []

    T = converted.get("temperature")
    if T is not None:
        T_C = T - 273.15
        if T_C < 40:
            warnings.append(f"Temperature {T_C:.1f} °C is low for a petroleum reservoir (typical > 40 °C).")
        if T_C > 200:
            warnings.append(f"Temperature {T_C:.1f} °C is very high — check units.")

    P = converted.get("pressure")
    if P is not None:
        if P < 10:
            warnings.append(f"Pressure {P:.1f} bar is low for a subsurface reservoir.")
        if P > 1000:
            warnings.append(f"Pressure {P:.1f} bar is extreme — check units (expected bar).")

    k = converted.get("permeability")
    if k is not None:
        k_mD = k / 9.869233e-16
        if k_mD < 0.001:
            warnings.append(f"Permeability {k_mD:.4f} mD is very tight (< 0.001 mD).")
        if k_mD > 10_000:
            warnings.append(f"Permeability {k_mD:.0f} mD is very high — check units.")

    sal = converted.get("salinity_NaCl")
    if sal is not None:
        if sal > 6.0:
            warnings.append(f"Salinity {sal:.2f} mol/kg NaCl is above halite saturation (~6 mol/kg).")

    return warnings
