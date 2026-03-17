"""
io/parsers.py
=============
Parsers for common field data file formats used in petroleum engineering.

Each parser reads a file (or file-like object) and returns a standardised
``FieldRecord`` dict that the unit converters and ``ConditionsBuilder`` can
consume without caring about the original source format.

Supported formats
-----------------
CSV / Excel well-log tables
    Tabular data with a header row.  One row per depth sample or per well.
    Column names are matched case-insensitively against a built-in alias
    table so common petroleum-industry naming conventions work out of the box.

LAS 2.0 (Log ASCII Standard)
    Standard wireline log format.  The ``~W`` (well information) and ``~C``
    (curve) sections are parsed.  Scalar reservoir properties are inferred
    by averaging or taking the median of depth-dependent log curves over a
    user-specified depth interval.

JSON / YAML field data cards
    Simple flat key-value documents (or nested under a ``"reservoir"`` key).
    Keys are matched against the same alias table.

References
----------
Canadian Well Logging Society (1992). LAS 2.0 specification.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FieldRecord = Dict[str, Any]   # raw key-value pairs from a parsed file
PathLike = Union[str, Path, IO]


# ---------------------------------------------------------------------------
# Column / key alias table
# ---------------------------------------------------------------------------
# Maps every recognised synonym to the canonical N2Bio field name.
# All matching is done after lower-casing and stripping whitespace/underscores.

_ALIASES: Dict[str, str] = {
    # Temperature
    "temperature":          "temperature_C",
    "temp":                 "temperature_C",
    "reservoir_temperature":"temperature_C",
    "res_temp":             "temperature_C",
    "bottomhole_temperature":"temperature_C",
    "bht":                  "temperature_C",
    "temperature_c":        "temperature_C",
    "temperature_k":        "temperature_K",   # explicit Kelvin variant
    "temperature_f":        "temperature_F",   # explicit Fahrenheit variant

    # Pressure
    "pressure":             "pressure_bar",
    "reservoir_pressure":   "pressure_bar",
    "res_pressure":         "pressure_bar",
    "initial_reservoir_pressure": "pressure_bar",
    "pres":                 "pressure_bar",
    "pressure_bar":         "pressure_bar",
    "pressure_psi":         "pressure_psi",
    "pressure_mpa":         "pressure_MPa",
    "pressure_kpa":         "pressure_kPa",
    "pressure_pa":          "pressure_Pa",
    "pore_pressure":        "pressure_bar",

    # Porosity
    "porosity":             "porosity",
    "phi":                  "porosity",
    "phie":                 "porosity",        # effective porosity from logs
    "poro":                 "porosity",
    "total_porosity":       "porosity",
    "effective_porosity":   "porosity",
    "core_porosity":        "porosity",

    # Permeability
    "permeability":         "permeability_mD",
    "perm":                 "permeability_mD",
    "k":                    "permeability_mD",
    "kh":                   "permeability_mD",  # horizontal permeability
    "kair":                 "permeability_mD",
    "core_permeability":    "permeability_mD",
    "permeability_md":      "permeability_mD",
    "permeability_m2":      "permeability_m2",  # SI variant
    "permeability_d":       "permeability_D",   # Darcy variant

    # Salinity / NaCl
    "salinity":             "salinity_ppm",
    "nacl":                 "salinity_ppm",
    "nacl_concentration":   "salinity_ppm",
    "total_dissolved_solids":"salinity_ppm",
    "tds":                  "salinity_ppm",
    "salinity_ppm":         "salinity_ppm",
    "salinity_mol_kg":      "salinity_mol_kg",  # direct molality variant
    "salinity_molality":    "salinity_mol_kg",
    "chloride":             "salinity_ppm",     # approximation: Cl⁻ ≈ NaCl

    # pH
    "ph":                   "pH",

    # Depth (for LAS log interval selection)
    "depth":                "depth_m",
    "md":                   "depth_m",         # measured depth
    "tvd":                  "depth_m",         # true vertical depth
    "depth_m":              "depth_m",
    "depth_ft":             "depth_ft",
}


def _normalise_key(raw: str) -> str:
    """Lower-case, strip, collapse whitespace and non-alphanumerics to '_'."""
    s = raw.strip().lower()
    s = re.sub(r"[\s\-/\\]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _resolve_alias(raw: str) -> Optional[str]:
    """Return the canonical N2Bio field name for *raw*, or None if unknown."""
    return _ALIASES.get(_normalise_key(raw))


# ---------------------------------------------------------------------------
# CSV / Excel parser
# ---------------------------------------------------------------------------

def parse_csv(
    source: PathLike,
    delimiter: str = ",",
    depth_interval: Optional[Tuple[float, float]] = None,
    aggregation: str = "median",
) -> FieldRecord:
    """
    Parse a CSV (or TSV) file containing well-log or core-analysis data.

    Parameters
    ----------
    source : str | Path | file-like
        Path to a ``.csv`` or ``.tsv`` file, or an open file-like object.
    delimiter : str
        Column delimiter.  Defaults to ``","``; use ``"\\t"`` for TSV.
    depth_interval : (float, float) | None
        If the table contains a depth column, restrict aggregation to rows
        where depth falls within ``[depth_min, depth_max]`` (metres).
        Ignored when no depth column is present.
    aggregation : {"median", "mean", "min", "max"}
        How to collapse multiple rows into a single scalar.  Defaults to
        ``"median"``.

    Returns
    -------
    FieldRecord
        Dict mapping canonical N2Bio field names to raw numeric values
        (units are preserved as-is; conversion happens in ``converters.py``).

    Raises
    ------
    ValueError
        If no recognised columns are found in the header row.

    Examples
    --------
    >>> record = parse_csv("well_log.csv", depth_interval=(2100, 2250))
    >>> record
    {'temperature_C': 87.3, 'pressure_bar': 152.1, 'porosity': 0.14, ...}
    """
    # ---- open source ----
    if hasattr(source, "read"):
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8-sig")
        lines = io.StringIO(text)
    else:
        path = Path(source)
        with path.open(newline="", encoding="utf-8-sig") as fh:
            lines = io.StringIO(fh.read())

    reader = csv.DictReader(lines, delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError("CSV file appears to be empty or has no header row.")

    # ---- map header columns to canonical names ----
    col_map: Dict[str, str] = {}   # original col name → canonical name
    for col in reader.fieldnames:
        canonical = _resolve_alias(col)
        if canonical is not None:
            col_map[col] = canonical

    if not col_map:
        raise ValueError(
            f"No recognised columns found in CSV header: {list(reader.fieldnames)}\n"
            "Check the alias table in n2bio/io/parsers.py for supported names."
        )

    # ---- collect rows into per-canonical-field lists ----
    accum: Dict[str, List[float]] = {v: [] for v in col_map.values()}
    depth_col_orig: Optional[str] = None

    # find original column name for depth filtering
    for orig, can in col_map.items():
        if can in ("depth_m", "depth_ft"):
            depth_col_orig = orig
            break

    for row in reader:
        # depth filtering
        if depth_interval is not None and depth_col_orig is not None:
            try:
                d = float(row[depth_col_orig])
                if not (depth_interval[0] <= d <= depth_interval[1]):
                    continue
            except (ValueError, TypeError):
                continue

        for orig_col, canonical in col_map.items():
            raw_val = row.get(orig_col, "").strip()
            if raw_val in ("", "NA", "N/A", "NaN", "nan", "--", "null"):
                continue
            try:
                accum[canonical].append(float(raw_val))
            except ValueError:
                pass

    # ---- aggregate ----
    record: FieldRecord = {}
    agg_fn = {
        "median": np.median,
        "mean":   np.mean,
        "min":    np.min,
        "max":    np.max,
    }.get(aggregation, np.median)

    for canonical, values in accum.items():
        if values:
            record[canonical] = float(agg_fn(values))

    return record


def parse_excel(
    source: PathLike,
    sheet: Union[str, int] = 0,
    depth_interval: Optional[Tuple[float, float]] = None,
    aggregation: str = "median",
) -> FieldRecord:
    """
    Parse an Excel (.xlsx / .xls) file using openpyxl (or xlrd for .xls).

    Parameters
    ----------
    source : str | Path | file-like
        Path to the workbook.
    sheet : str | int
        Sheet name or zero-based index.
    depth_interval : (float, float) | None
        Depth filter applied to a recognised depth column (metres).
    aggregation : str
        Aggregation method: ``"median"``, ``"mean"``, ``"min"``, ``"max"``.

    Returns
    -------
    FieldRecord
    """
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required for Excel parsing: pip install openpyxl"
        ) from exc

    path = Path(source)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    if isinstance(sheet, int):
        ws = wb.worksheets[sheet]
    else:
        ws = wb[sheet]

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"Sheet '{sheet}' in {path} is empty.")

    header = [str(c) if c is not None else "" for c in rows[0]]
    col_map: Dict[int, str] = {}
    for idx, col in enumerate(header):
        canonical = _resolve_alias(col)
        if canonical is not None:
            col_map[idx] = canonical

    if not col_map:
        raise ValueError(
            f"No recognised columns in Excel header: {header}"
        )

    # Write to a temporary CSV buffer and reuse the CSV parser logic
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in rows[1:]:
        writer.writerow([("" if v is None else v) for v in row])
    buf.seek(0)

    wb.close()
    return parse_csv(buf, depth_interval=depth_interval, aggregation=aggregation)


# ---------------------------------------------------------------------------
# LAS 2.0 parser
# ---------------------------------------------------------------------------

def parse_las(
    source: PathLike,
    depth_interval: Optional[Tuple[float, float]] = None,
    aggregation: str = "median",
) -> FieldRecord:
    """
    Parse a LAS 2.0 wireline log file.

    Scalar values from the ``~W`` (Well Information) section are read
    directly.  Depth-curve values from the ``~A`` (ASCII data) section
    are aggregated over ``depth_interval`` if provided.

    Parameters
    ----------
    source : str | Path | file-like
        Path to a ``.las`` file or an open file-like object.
    depth_interval : (float, float) | None
        Restrict curve aggregation to this depth range [m].
    aggregation : str
        Aggregation method: ``"median"``, ``"mean"``, ``"min"``, ``"max"``.

    Returns
    -------
    FieldRecord

    Notes
    -----
    Only LAS 2.0 is supported.  LAS 3.0 files may parse partially.
    """
    if hasattr(source, "read"):
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8-sig", errors="replace")
    else:
        path = Path(source)
        with path.open(encoding="utf-8-sig", errors="replace") as fh:
            text = fh.read()

    lines = text.splitlines()
    record: FieldRecord = {}

    # ---- LAS section scanner ----
    current_section: str = ""
    curve_names: List[str] = []
    null_value: float = -9999.25
    data_rows: List[List[float]] = []
    wrap_mode: bool = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Section header
        if stripped.startswith("~"):
            current_section = stripped[1:2].upper()
            continue

        if current_section == "W":
            # Well information: "MNEM.UNIT  VALUE : DESCRIPTION"
            m = re.match(r"([A-Z0-9_]+)\s*\.(\S*)\s+(.*?)\s*:", stripped, re.IGNORECASE)
            if m:
                mnem = m.group(1).strip()
                value_str = m.group(3).strip().split()[0] if m.group(3).strip() else ""
                canonical = _resolve_alias(mnem)
                if canonical and value_str:
                    try:
                        record[canonical] = float(value_str)
                    except ValueError:
                        pass
            if re.search(r"WRAP\s*\..*:\s*YES", stripped, re.IGNORECASE):
                wrap_mode = True

        elif current_section == "C":
            # Curve information: collect curve names in order
            m = re.match(r"([A-Z0-9_]+)\s*\.", stripped, re.IGNORECASE)
            if m:
                curve_names.append(m.group(1).strip().upper())

        elif current_section == "A":
            # ASCII data block
            nums = []
            for tok in stripped.split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    pass
            if nums:
                data_rows.append(nums)

    # ---- aggregate curve data ----
    if curve_names and data_rows:
        try:
            arr = np.array(data_rows)
        except ValueError:
            # Ragged rows — pad or skip
            max_cols = max(len(r) for r in data_rows)
            arr = np.array([r + [np.nan] * (max_cols - len(r)) for r in data_rows])

        # Replace null values
        arr[arr == null_value] = np.nan

        # Depth column (always first in LAS 2.0)
        depth_col = arr[:, 0] if arr.ndim == 2 and arr.shape[1] > 0 else None

        mask = np.ones(arr.shape[0], dtype=bool)
        if depth_interval is not None and depth_col is not None:
            mask = (depth_col >= depth_interval[0]) & (depth_col <= depth_interval[1])

        agg_fn = {
            "median": np.nanmedian,
            "mean":   np.nanmean,
            "min":    np.nanmin,
            "max":    np.nanmax,
        }.get(aggregation, np.nanmedian)

        for i, curve in enumerate(curve_names):
            if i >= arr.shape[1]:
                break
            canonical = _resolve_alias(curve)
            if canonical and canonical not in record:
                vals = arr[mask, i]
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0:
                    record[canonical] = float(agg_fn(valid))

    return record


# ---------------------------------------------------------------------------
# JSON / YAML parser
# ---------------------------------------------------------------------------

def parse_json(source: PathLike) -> FieldRecord:
    """
    Parse a flat or nested JSON field data card.

    The JSON may be flat::

        {"temperature": 87, "pressure": 152, "porosity": 0.14}

    or nested under a ``"reservoir"`` key::

        {"well": "A-01", "reservoir": {"temperature": 87, ...}}

    Parameters
    ----------
    source : str | Path | file-like

    Returns
    -------
    FieldRecord
    """
    if hasattr(source, "read"):
        data = json.load(source)
    else:
        with open(source, encoding="utf-8") as fh:
            data = json.load(fh)

    # Flatten nested "reservoir" or "conditions" sub-dict
    if isinstance(data, dict):
        for key in ("reservoir", "conditions", "reservoir_conditions"):
            if key in data and isinstance(data[key], dict):
                data = data[key]
                break

    record: FieldRecord = {}
    for raw_key, value in data.items():
        canonical = _resolve_alias(raw_key)
        if canonical and isinstance(value, (int, float)):
            record[canonical] = float(value)

    return record


def parse_yaml(source: PathLike) -> FieldRecord:
    """
    Parse a flat or nested YAML field data card.

    Requires PyYAML (``pip install pyyaml``).

    Parameters
    ----------
    source : str | Path | file-like

    Returns
    -------
    FieldRecord
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML parsing: pip install pyyaml"
        ) from exc

    if hasattr(source, "read"):
        data = yaml.safe_load(source)
    else:
        with open(source, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

    # Flatten nested dicts the same way as JSON
    if isinstance(data, dict):
        for key in ("reservoir", "conditions", "reservoir_conditions"):
            if key in data and isinstance(data[key], dict):
                data = data[key]
                break

    record: FieldRecord = {}
    for raw_key, value in data.items():
        canonical = _resolve_alias(raw_key)
        if canonical and isinstance(value, (int, float)):
            record[canonical] = float(value)

    return record


# ---------------------------------------------------------------------------
# Auto-dispatch loader
# ---------------------------------------------------------------------------

def load_field_data(
    source: PathLike,
    *,
    delimiter: str = ",",
    sheet: Union[str, int] = 0,
    depth_interval: Optional[Tuple[float, float]] = None,
    aggregation: str = "median",
) -> FieldRecord:
    """
    Load field data from any supported file format, auto-detected by extension.

    Supported extensions
    --------------------
    ``.csv``, ``.tsv``, ``.txt``  →  :func:`parse_csv`
    ``.xlsx``, ``.xls``           →  :func:`parse_excel`
    ``.las``                      →  :func:`parse_las`
    ``.json``                     →  :func:`parse_json`
    ``.yaml``, ``.yml``           →  :func:`parse_yaml`

    Parameters
    ----------
    source : str | Path | file-like
        File path or open file-like object.  For file-like objects without a
        name attribute, the format defaults to CSV.
    delimiter : str
        Column delimiter for CSV/TSV files.
    sheet : str | int
        Sheet selection for Excel files.
    depth_interval : (float, float) | None
        Depth filter for CSV, Excel, and LAS files [metres].
    aggregation : str
        Aggregation method for tabular / log data: ``"median"`` (default),
        ``"mean"``, ``"min"``, or ``"max"``.

    Returns
    -------
    FieldRecord
        Standardised dict of canonical field names → raw numeric values.
    """
    # Determine file extension
    if hasattr(source, "name"):
        ext = Path(source.name).suffix.lower()
    elif isinstance(source, (str, Path)):
        ext = Path(source).suffix.lower()
    else:
        ext = ".csv"   # fallback for unnamed file-like objects

    if ext in (".csv", ".tsv", ".txt"):
        delim = "\t" if ext == ".tsv" else delimiter
        return parse_csv(source, delimiter=delim,
                         depth_interval=depth_interval, aggregation=aggregation)
    elif ext in (".xlsx", ".xls"):
        return parse_excel(source, sheet=sheet,
                           depth_interval=depth_interval, aggregation=aggregation)
    elif ext == ".las":
        return parse_las(source, depth_interval=depth_interval, aggregation=aggregation)
    elif ext == ".json":
        return parse_json(source)
    elif ext in (".yaml", ".yml"):
        return parse_yaml(source)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'.  "
            "Supported: .csv, .tsv, .xlsx, .xls, .las, .json, .yaml"
        )
