"""
io/builder.py
=============
High-level entry point for creating a ``ReservoirConditions`` object
directly from field data files.

``ConditionsBuilder`` chains the three steps — parse, convert, build —
into a single clean API.  It is the only class most users need to import
from the ``n2bio.io`` sub-package.

Usage
-----
::

    from n2bio.io import ConditionsBuilder

    # From a CSV well-log
    cond = ConditionsBuilder.from_file(
        "core_analysis.csv",
        depth_interval=(2100.0, 2250.0),
    )

    # From a LAS wireline log
    cond = ConditionsBuilder.from_file(
        "well_A01.las",
        depth_interval=(2150.0, 2200.0),
        aggregation="mean",
    )

    # From a JSON field card
    cond = ConditionsBuilder.from_file("reservoir_summary.json")

    # Override any parsed field
    cond = ConditionsBuilder.from_file(
        "core_analysis.csv",
        overrides={"pH": 6.8, "pressure": 180.0},
    )

    # Fill gaps with explicit defaults
    cond = ConditionsBuilder.from_file(
        "partial_data.csv",
        defaults={"salinity_NaCl": 0.3},
    )

    print(cond.summary())
    sim = N2BioSimulation(params, cond, initial_state)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .parsers import FieldRecord, PathLike, load_field_data
from .converters import convert_record, check_physical_ranges


# Lazy import to avoid circular dependency at module level
def _get_reservoir_conditions():
    from ..kinetics.parameters import ReservoirConditions
    return ReservoirConditions


class ConditionsBuilder:
    """
    Build a :class:`~n2bio.kinetics.parameters.ReservoirConditions` from
    field data files or raw dicts.

    The class is intentionally stateless — all methods are class methods
    or static methods so there is no object lifecycle to manage.
    """

    # Default fallback values (same as ReservoirConditions defaults)
    _DEFAULTS: Dict[str, float] = {
        "temperature":   358.15,   # K  (85 °C)
        "pressure":      150.0,    # bar
        "pH":            7.0,
        "salinity_NaCl": 0.5,      # mol/kg
        "porosity":      0.15,
        "permeability":  50.0e-15, # m²  (50 mD)
    }

    @classmethod
    def from_file(
        cls,
        source: PathLike,
        *,
        delimiter: str = ",",
        sheet: Union[str, int] = 0,
        depth_interval: Optional[Tuple[float, float]] = None,
        aggregation: str = "median",
        overrides: Optional[Dict[str, float]] = None,
        defaults: Optional[Dict[str, float]] = None,
        warn_missing: bool = True,
        warn_unusual: bool = True,
    ):
        """
        Parse *source*, convert units, apply overrides/defaults, and return
        a fully validated ``ReservoirConditions`` object.

        Parameters
        ----------
        source : str | Path | file-like
            Field data file.  Format is auto-detected from extension.
        delimiter : str
            Column delimiter for CSV/TSV files (default ``","``).
        sheet : str | int
            Sheet name or index for Excel files (default ``0``).
        depth_interval : (float, float) | None
            Restrict aggregation of tabular / log data to this depth range
            in metres, e.g. ``(2100.0, 2250.0)``.
        aggregation : {"median", "mean", "min", "max"}
            How to collapse multiple depth samples into a single scalar
            (default ``"median"``).
        overrides : dict | None
            Key-value pairs in N2Bio internal units that take priority over
            parsed values.  Keys must match ``ReservoirConditions`` field
            names: ``temperature`` [K], ``pressure`` [bar], ``porosity`` [-],
            ``permeability`` [m²], ``salinity_NaCl`` [mol/kg], ``pH`` [-].
        defaults : dict | None
            Fallback values for fields not found in the file.  Same unit
            conventions as *overrides*.  If a field is neither in the file
            nor in *defaults*, the ``ReservoirConditions`` dataclass default
            is used.
        warn_missing : bool
            Emit a ``UserWarning`` for each ``ReservoirConditions`` field that
            was not found in *source* and is falling back to a default.
        warn_unusual : bool
            Emit a ``UserWarning`` for physically unusual but valid values
            (e.g. very high or very low temperature).

        Returns
        -------
        ReservoirConditions

        Raises
        ------
        ValueError
            If a parsed or converted value fails the ``ReservoirConditions``
            validation checks (e.g. porosity outside (0, 1)).
        """
        # 1. Parse raw field record
        record: FieldRecord = load_field_data(
            source,
            delimiter=delimiter,
            sheet=sheet,
            depth_interval=depth_interval,
            aggregation=aggregation,
        )

        return cls.from_record(
            record,
            overrides=overrides,
            defaults=defaults,
            warn_missing=warn_missing,
            warn_unusual=warn_unusual,
        )

    @classmethod
    def from_record(
        cls,
        record: FieldRecord,
        *,
        overrides: Optional[Dict[str, float]] = None,
        defaults: Optional[Dict[str, float]] = None,
        warn_missing: bool = True,
        warn_unusual: bool = True,
    ):
        """
        Build ``ReservoirConditions`` from an already-parsed ``FieldRecord``.

        This is useful when you have obtained a ``FieldRecord`` from one of
        the low-level ``parse_*`` functions and want to apply unit conversion
        and construct the conditions object separately.

        Parameters
        ----------
        record : FieldRecord
            Raw output of any ``parse_*`` function.
        overrides, defaults, warn_missing, warn_unusual
            Same as :meth:`from_file`.

        Returns
        -------
        ReservoirConditions
        """
        # 2. Convert units to N2Bio internal system
        converted: dict = convert_record(record)

        # 3. Apply caller overrides (highest priority)
        if overrides:
            converted.update(overrides)

        # 4. Fill missing fields from caller-supplied defaults, then built-in defaults
        all_defaults = dict(cls._DEFAULTS)
        if defaults:
            all_defaults.update(defaults)

        _RC_FIELDS = ("temperature", "pressure", "pH",
                      "salinity_NaCl", "porosity", "permeability")

        missing: list = []
        for field in _RC_FIELDS:
            if field not in converted:
                if field in all_defaults:
                    converted[field] = all_defaults[field]
                    missing.append(field)

        # 5. Warn about missing fields
        if warn_missing and missing:
            warnings.warn(
                f"The following ReservoirConditions fields were not found in the "
                f"field data and are using default values: {missing}.\n"
                "Supply them in your file or pass them via the 'defaults' argument.",
                UserWarning,
                stacklevel=3,
            )

        # 6. Warn about unusual values
        if warn_unusual:
            issues = check_physical_ranges(converted)
            for msg in issues:
                warnings.warn(msg, UserWarning, stacklevel=3)

        # 7. Construct and return ReservoirConditions (validation in __post_init__)
        ReservoirConditions = _get_reservoir_conditions()
        return ReservoirConditions(
            temperature   = converted["temperature"],
            pressure      = converted["pressure"],
            pH            = converted["pH"],
            salinity_NaCl = converted["salinity_NaCl"],
            porosity      = converted["porosity"],
            permeability  = converted["permeability"],
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        overrides: Optional[Dict[str, float]] = None,
        defaults: Optional[Dict[str, float]] = None,
        warn_missing: bool = True,
        warn_unusual: bool = True,
    ):
        """
        Build ``ReservoirConditions`` from a plain Python dict.

        The dict keys are matched against the same alias table as file
        parsers, so common industry names work directly::

            cond = ConditionsBuilder.from_dict({
                "BHT":         87.3,   # °C
                "pressure_psi": 2200,
                "phi":          0.14,
                "perm":         45.0,  # mD
                "TDS":          18500, # ppm
            })

        Parameters
        ----------
        data : dict
            Arbitrary key-value pairs with recognised field names.
        overrides, defaults, warn_missing, warn_unusual
            Same as :meth:`from_file`.

        Returns
        -------
        ReservoirConditions
        """
        from .parsers import _resolve_alias
        record: FieldRecord = {}
        for raw_key, value in data.items():
            canonical = _resolve_alias(raw_key)
            if canonical and isinstance(value, (int, float)):
                record[canonical] = float(value)

        return cls.from_record(
            record,
            overrides=overrides,
            defaults=defaults,
            warn_missing=warn_missing,
            warn_unusual=warn_unusual,
        )

    @classmethod
    def inspect_file(cls, source: PathLike, **kwargs) -> dict:
        """
        Parse *source* and report what fields were found, their raw values,
        and the converted N2Bio values — without constructing a
        ``ReservoirConditions`` object.

        Useful for debugging field data files before a full run.

        Parameters
        ----------
        source : str | Path
            Field data file.
        **kwargs
            Passed to :func:`~n2bio.io.parsers.load_field_data`.

        Returns
        -------
        dict with keys:
            ``"raw"``       – the ``FieldRecord`` from the parser
            ``"converted"`` – N2Bio internal-unit values
            ``"warnings"``  – list of physical range warnings
            ``"missing"``   – list of RC fields not found in the file
        """
        record = load_field_data(source, **kwargs)
        converted = convert_record(record)
        issues = check_physical_ranges(converted)

        _RC_FIELDS = ("temperature", "pressure", "pH",
                      "salinity_NaCl", "porosity", "permeability")
        missing = [f for f in _RC_FIELDS if f not in converted]

        return {
            "raw":       record,
            "converted": converted,
            "warnings":  issues,
            "missing":   missing,
        }
