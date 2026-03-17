"""
n2bio.io
========
Field data I/O sub-package for the N2Bio framework.

Provides parsers for common petroleum-engineering file formats and a
high-level builder that converts field measurements directly into a
``ReservoirConditions`` object ready for simulation.

Supported file formats
----------------------
CSV / TSV          tabular well-log or core-analysis data
Excel (.xlsx/.xls) same tabular structure, sheet-selectable
LAS 2.0            wireline log ASCII standard
JSON               flat or nested key-value field cards
YAML               flat or nested key-value field cards

Quick start
-----------
::

    from n2bio.io import ConditionsBuilder

    # Auto-detect format from file extension
    cond = ConditionsBuilder.from_file(
        "core_analysis.csv",
        depth_interval=(2100.0, 2250.0),
    )

    # Inspect what a file contains before building
    info = ConditionsBuilder.inspect_file("well_A01.las")
    print(info["converted"])
    print(info["missing"])

    # Build from a plain dict using industry-standard field names
    cond = ConditionsBuilder.from_dict({
        "BHT": 87.3,          # °C
        "pressure_psi": 2200,
        "phi": 0.14,
        "perm": 45.0,         # mD
        "TDS": 18500,         # ppm NaCl
    })

Exports
-------
ConditionsBuilder    High-level builder (parse → convert → ReservoirConditions)
load_field_data      Auto-dispatching file loader (returns raw FieldRecord)
parse_csv            CSV / TSV parser
parse_excel          Excel parser (requires openpyxl)
parse_las            LAS 2.0 wireline log parser
parse_json           JSON field card parser
parse_yaml           YAML field card parser (requires pyyaml)
convert_record       Unit converter: FieldRecord → N2Bio internal units
"""

from .builder import ConditionsBuilder
from .parsers import (
    load_field_data,
    parse_csv,
    parse_excel,
    parse_las,
    parse_json,
    parse_yaml,
)
from .converters import convert_record

__all__ = [
    "ConditionsBuilder",
    "load_field_data",
    "parse_csv",
    "parse_excel",
    "parse_las",
    "parse_json",
    "parse_yaml",
    "convert_record",
]
