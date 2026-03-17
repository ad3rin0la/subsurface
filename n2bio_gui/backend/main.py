"""
N2Bio GUI Backend — FastAPI server
===================================
Wraps the n2bio Python package and exposes REST endpoints for the
single-page dashboard.

Endpoints
---------
POST /api/run              – Run a single batch simulation
POST /api/sweep            – Parameter sweep over one variable
POST /api/compare          – Run multiple named scenarios and compare
GET  /api/defaults         – Return default KineticParameters + ReservoirConditions
POST /api/upload-conditions– Parse an uploaded field-data file → ReservoirConditions
POST /api/geo/parse        – Parse an uploaded geo/well-log file → depth profiles
"""

from __future__ import annotations

import csv as _csv_module
import io as _io
import sys
import os
import re as _re_module
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from n2bio.kinetics.parameters import KineticParameters, ReservoirConditions
from n2bio.simulation.state import BiologicalState
from n2bio.simulation.solver import N2BioBatchSolver
from n2bio.simulation.lifecycle import LifecyclePhase

# ---------------------------------------------------------------------------
app = FastAPI(title="N2Bio GUI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class KineticParamsIn(BaseModel):
    mu_max: float = 0.08
    K_N2: float = 5.0e-5
    K_S: float = 5.0e-4
    Y: float = 0.07
    b_decay: float = 0.005
    K_I_H2: float = 0.3
    K_I_NH4: float = 0.1
    r_nit_max: float = 1.0
    k_ali: float = 4.167e-4
    k_arom: float = 2.083e-5
    K_I_arom: float = 5.0e-4
    f_arom0: float = 0.15
    Y_HC_H2: float = 0.35
    K_N_fixed: float = 5.0e-6
    Y_N: float = 0.14


class ReservoirCondIn(BaseModel):
    temperature_C: float = 85.0
    pressure: float = 150.0
    pH: float = 7.0
    salinity_NaCl: float = 0.5
    porosity: float = 0.15
    permeability_mD: float = 50.0


class InitialStateIn(BaseModel):
    N2_aq: float = 5.0e-4
    CO2_aq: float = 0.005
    H2_aq: float = 0.0
    NH4_aq: float = 1.0e-7
    HC_ali: float = 4.25e-3
    HC_arom: float = 0.75e-3
    NaCl_aq: float = 0.5
    X: float = 0.005
    P_N2: float = 100.0
    P_CO2: float = 5.0
    P_H2: float = 0.01
    P_total: float = 150.0


class RunRequest(BaseModel):
    params: KineticParamsIn = Field(default_factory=KineticParamsIn)
    conditions: ReservoirCondIn = Field(default_factory=ReservoirCondIn)
    initial: InitialStateIn = Field(default_factory=InitialStateIn)
    t_end_hours: float = 3000.0
    n_points: int = 400
    label: Optional[str] = None


class SweepRequest(BaseModel):
    base_params: KineticParamsIn = Field(default_factory=KineticParamsIn)
    base_conditions: ReservoirCondIn = Field(default_factory=ReservoirCondIn)
    base_initial: InitialStateIn = Field(default_factory=InitialStateIn)
    sweep_param: str  # e.g. "mu_max", "K_I_H2", "f_arom0", "temperature_C"
    sweep_values: List[float]
    t_end_hours: float = 3000.0
    n_points: int = 300


class CompareRequest(BaseModel):
    scenarios: List[RunRequest]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_LABELS = {
    0: "Inoculation Lag",
    1: "N₂ Fixation Bootstrap",
    2: "Peak Production",
    3: "HC Depletion",
    4: "Biological Senescence",
}

PHASE_COLORS = {
    0: "#4a5568",
    1: "#2b6cb0",
    2: "#276749",
    3: "#c05621",
    4: "#702459",
}


def _build_kinetic_params(p: KineticParamsIn) -> KineticParameters:
    kp = KineticParameters()
    for field in p.model_fields:
        setattr(kp, field, getattr(p, field))
    return kp


def _build_reservoir_cond(c: ReservoirCondIn) -> ReservoirConditions:
    return ReservoirConditions(
        temperature=c.temperature_C + 273.15,
        pressure=c.pressure,
        pH=c.pH,
        salinity_NaCl=c.salinity_NaCl,
        porosity=c.porosity,
        permeability=c.permeability_mD * 1e-15,
    )


def _build_initial_state(s: InitialStateIn, cond: ReservoirConditions) -> BiologicalState:
    return BiologicalState(
        N2_aq=s.N2_aq,
        CO2_aq=s.CO2_aq,
        H2_aq=s.H2_aq,
        NH4_aq=s.NH4_aq,
        HC_ali=s.HC_ali,
        HC_arom=s.HC_arom,
        NaCl_aq=s.NaCl_aq,
        X=s.X,
        P_N2=s.P_N2,
        P_CO2=s.P_CO2,
        P_H2=s.P_H2,
        P_total=s.P_total,
        T=cond.temperature,
        pH=cond.pH,
    )


def _run_one(req: RunRequest) -> Dict[str, Any]:
    kp = _build_kinetic_params(req.params)
    rc = _build_reservoir_cond(req.conditions)
    s0 = _build_initial_state(req.initial, rc)

    solver = N2BioBatchSolver(kp, rc, s0)
    sol, phases = solver.solve_with_lifecycle(
        t_end_hours=req.t_end_hours,
        n_points=req.n_points,
    )

    t = sol.t.tolist()
    y = sol.y

    # y indices: 0=N2_aq, 1=H2_aq, 2=NH4_aq, 3=HC_ali, 4=HC_arom, 5=X, 6=cum_H2
    f_arom = np.where(
        (y[3] + y[4]) > 1e-12,
        y[4] / (y[3] + y[4]),
        req.params.f_arom0,
    ).tolist()

    phase_int = [int(p) for p in phases]

    # Compute rates at each time point for diagnostic panel
    rates_mu = []
    for i in range(len(t)):
        state = BiologicalState(
            N2_aq=float(y[0, i]),
            H2_aq=float(y[1, i]),
            NH4_aq=float(y[2, i]),
            HC_ali=float(y[3, i]),
            HC_arom=float(y[4, i]),
            X=float(y[5, i]),
            P_N2=s0.P_N2,
            P_CO2=s0.P_CO2,
            P_H2=float(y[1, i]) * 15.0,
            P_total=s0.P_total,
            T=s0.T,
            pH=s0.pH,
        )
        rates = solver.compute_rates(state)
        rates_mu.append(rates.get("mu [h-1]", 0.0))

    # Summary metrics
    cum_H2_final = float(y[6, -1])
    X_max = float(np.max(y[5]))
    peak_H2_rate = float(np.max(np.gradient(y[6], sol.t)))
    NH4_final = float(y[2, -1])

    # Phase span segments for timeline
    phase_spans = []
    if phase_int:
        cur_phase = phase_int[0]
        cur_start = t[0]
        for i in range(1, len(phase_int)):
            if phase_int[i] != cur_phase:
                phase_spans.append({
                    "phase": cur_phase,
                    "label": PHASE_LABELS[cur_phase],
                    "color": PHASE_COLORS[cur_phase],
                    "t_start": cur_start,
                    "t_end": t[i],
                })
                cur_phase = phase_int[i]
                cur_start = t[i]
        phase_spans.append({
            "phase": cur_phase,
            "label": PHASE_LABELS[cur_phase],
            "color": PHASE_COLORS[cur_phase],
            "t_start": cur_start,
            "t_end": t[-1],
        })

    return {
        "label": req.label or "Simulation",
        "t": t,
        "N2_aq": y[0].tolist(),
        "H2_aq": y[1].tolist(),
        "NH4_aq": y[2].tolist(),
        "HC_ali": y[3].tolist(),
        "HC_arom": y[4].tolist(),
        "X": y[5].tolist(),
        "cum_H2": y[6].tolist(),
        "f_arom": f_arom,
        "mu": rates_mu,
        "phases": phase_int,
        "phase_spans": phase_spans,
        "metrics": {
            "cum_H2_mmol_L": cum_H2_final * 1000,
            "X_max_mg_L": X_max * 1000,
            "peak_H2_rate_umol_L_h": peak_H2_rate * 1e6,
            "NH4_final_mmol_L": NH4_final * 1000,
            "t_end_h": req.t_end_hours,
            "max_phase": max(phase_int) if phase_int else 0,
        },
    }


# ---------------------------------------------------------------------------
# Routes — simulation
# ---------------------------------------------------------------------------

@app.get("/api/defaults")
def get_defaults():
    kp = KineticParameters()
    rc = ReservoirConditions()
    return {
        "params": {f: getattr(kp, f) for f in KineticParamsIn.model_fields},
        "conditions": {
            "temperature_C": rc.temperature_C,
            "pressure": rc.pressure,
            "pH": rc.pH,
            "salinity_NaCl": rc.salinity_NaCl,
            "porosity": rc.porosity,
            "permeability_mD": rc.permeability * 1e15,
        },
        "initial": {f: getattr(InitialStateIn(), f) for f in InitialStateIn.model_fields},
        "param_meta": _PARAM_META,
    }


@app.post("/api/run")
def run_simulation(req: RunRequest):
    try:
        return _run_one(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


@app.post("/api/sweep")
def run_sweep(req: SweepRequest):
    results = []
    for val in req.sweep_values:
        p_dict = req.base_params.model_dump()
        c_dict = req.base_conditions.model_dump()
        i_dict = req.base_initial.model_dump()

        if req.sweep_param in p_dict:
            p_dict[req.sweep_param] = val
        elif req.sweep_param in c_dict:
            c_dict[req.sweep_param] = val
        elif req.sweep_param in i_dict:
            i_dict[req.sweep_param] = val
        else:
            raise HTTPException(status_code=400, detail=f"Unknown sweep param: {req.sweep_param}")

        run_req = RunRequest(
            params=KineticParamsIn(**p_dict),
            conditions=ReservoirCondIn(**c_dict),
            initial=InitialStateIn(**i_dict),
            t_end_hours=req.t_end_hours,
            n_points=req.n_points,
            label=f"{req.sweep_param}={val:.3g}",
        )
        try:
            results.append(_run_one(run_req))
        except Exception as e:
            results.append({"label": f"{req.sweep_param}={val:.3g}", "error": str(e)})

    return {
        "sweep_param": req.sweep_param,
        "sweep_values": req.sweep_values,
        "results": results,
    }


@app.post("/api/compare")
def compare_scenarios(req: CompareRequest):
    results = []
    for i, scenario in enumerate(req.scenarios):
        if not scenario.label:
            scenario.label = f"Scenario {i+1}"
        try:
            results.append(_run_one(scenario))
        except Exception as e:
            results.append({"label": scenario.label, "error": str(e)})
    return {"scenarios": results}


# ---------------------------------------------------------------------------
# Routes — field-data file upload
# ---------------------------------------------------------------------------

# Canonical field names for geo depth-profile data
_GEO_FIELD_ALIASES: Dict[str, str] = {
    # Depth
    "depth": "depth", "depth_m": "depth", "md": "depth", "tvd": "depth", "dept": "depth",
    # Porosity
    "porosity": "porosity", "phi": "porosity", "phie": "porosity", "poro": "porosity",
    "total_porosity": "porosity", "effective_porosity": "porosity", "core_porosity": "porosity",
    # Permeability [mD]
    "permeability": "permeability", "perm": "permeability", "k": "permeability",
    "kh": "permeability", "kair": "permeability", "core_permeability": "permeability",
    "permeability_md": "permeability",
    # Water saturation
    "sw": "sw", "water_saturation": "sw", "s_water": "sw", "swi": "sw", "sw_log": "sw",
    # HC fractions [mol/L]
    "hc_ali": "hc_ali", "hc_aliphatic": "hc_ali", "aliphatic": "hc_ali", "c_ali": "hc_ali",
    "hc_arom": "hc_arom", "hc_aromatic": "hc_arom", "aromatic": "hc_arom", "c_arom": "hc_arom",
    # Scalar reservoir conditions (also parseable from geo files)
    "temperature": "temperature", "temp": "temperature", "bht": "temperature",
    "temperature_c": "temperature",
    "pressure": "pressure", "pressure_bar": "pressure", "res_pressure": "pressure",
}

_GEO_KEY_FIELDS = ("depth", "porosity", "permeability", "sw", "hc_ali", "hc_arom",
                   "temperature", "pressure")


def _geo_norm_key(s: str) -> str:
    s = s.strip().lower()
    s = _re_module.sub(r"[\s\-/\\]+", "_", s)
    s = _re_module.sub(r"[^a-z0-9_]", "", s)
    s = _re_module.sub(r"_+", "_", s).strip("_")
    return s


def _geo_resolve(col: str) -> Optional[str]:
    return _GEO_FIELD_ALIASES.get(_geo_norm_key(col))


@app.post("/api/upload-conditions")
async def upload_conditions(
    file: UploadFile = File(...),
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    aggregation: str = "median",
):
    """
    Parse an uploaded field-data file (CSV, LAS, JSON, Excel, YAML) using
    n2bio.io and return the extracted reservoir conditions in GUI units.

    Returns
    -------
    {
        "conditions": { temperature_C, pressure, pH, salinity_NaCl,
                        porosity, permeability_mD },
        "warnings":   [ ... ],
        "missing":    [ ... ]
    }
    """
    import warnings as _warnings_mod
    from n2bio.io import ConditionsBuilder

    filename = file.filename or "data"
    contents = await file.read()

    # Wrap bytes in a named BytesIO so load_field_data can detect the extension
    buf = _io.BytesIO(contents)
    buf.name = filename  # type: ignore[attr-defined]

    depth_interval = (
        (depth_min, depth_max)
        if depth_min is not None and depth_max is not None
        else None
    )

    caught_warnings: List[str] = []
    try:
        with _warnings_mod.catch_warnings(record=True) as w:
            _warnings_mod.simplefilter("always")
            cond = ConditionsBuilder.from_file(
                buf,
                depth_interval=depth_interval,
                aggregation=aggregation,
                warn_missing=True,
                warn_unusual=True,
            )
            caught_warnings = [str(warning.message) for warning in w]
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse '{filename}': {e}",
        )

    # Re-inspect to get the list of missing fields (without building again)
    buf2 = _io.BytesIO(contents)
    buf2.name = filename  # type: ignore[attr-defined]
    try:
        info = ConditionsBuilder.inspect_file(buf2)
        missing = info.get("missing", [])
    except Exception:
        missing = []

    return {
        "conditions": {
            "temperature_C":  round(cond.temperature - 273.15, 2),
            "pressure":       round(cond.pressure, 2),
            "pH":             round(cond.pH, 2),
            "salinity_NaCl":  round(cond.salinity_NaCl, 4),
            "porosity":       round(cond.porosity, 4),
            "permeability_mD": round(cond.permeability * 1e15, 2),
        },
        "warnings": caught_warnings,
        "missing":  missing,
    }


@app.post("/api/geo/parse")
async def geo_parse(file: UploadFile = File(...)):
    """
    Parse an uploaded well-log / geo-data file and return depth-indexed
    column arrays for the Geological Data dashboard view.

    Supported formats: .csv, .tsv, .txt, .las, .json

    Returns
    -------
    {
        "filename":  str,
        "format":    "csv" | "las" | "json",
        "n_rows":    int,
        "columns":   [str, ...],
        "mapping":   { canonical_name: original_column_name },
        "data":      { depth, porosity, permeability, sw, hc_ali, hc_arom, ... },
        "raw_data":  { col_name: [float | null, ...] }
    }
    """
    import json as _json

    filename = file.filename or "data"
    ext = Path(filename).suffix.lower()
    contents = await file.read()

    try:
        text = contents.decode("utf-8-sig", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode file: {exc}")

    columns: List[str] = []
    raw_data: Dict[str, List] = {}
    fmt = ext.lstrip(".")

    # ---- CSV / TSV ----
    if ext in (".csv", ".tsv", ".txt"):
        delim = "\t" if ext == ".tsv" else ","
        reader = _csv_module.DictReader(_io.StringIO(text), delimiter=delim)
        if not reader.fieldnames:
            raise HTTPException(
                status_code=422, detail="CSV appears empty or has no header row."
            )
        columns = list(reader.fieldnames)
        raw_data = {c: [] for c in columns}
        for row in reader:
            for col in columns:
                v = (row.get(col) or "").strip()
                try:
                    raw_data[col].append(float(v))
                except (ValueError, TypeError):
                    raw_data[col].append(None)
        fmt = "csv"

    # ---- LAS 2.0 ----
    elif ext == ".las":
        current_section = ""
        curve_names: List[str] = []
        null_value = -9999.25
        data_rows: List[List[float]] = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("~"):
                current_section = stripped[1:2].upper()
                continue
            if current_section == "C":
                m = _re_module.match(r"([A-Z0-9_]+)\s*\.", stripped, _re_module.IGNORECASE)
                if m:
                    curve_names.append(m.group(1).strip().upper())
            elif current_section == "A":
                nums = []
                for tok in stripped.split():
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        pass
                if nums:
                    data_rows.append(nums)

        if not curve_names or not data_rows:
            raise HTTPException(
                status_code=422,
                detail="Could not find curve definitions (~C) or data (~A) in LAS file.",
            )

        max_cols = max(len(r) for r in data_rows)
        arr = np.array(
            [r + [np.nan] * (max_cols - len(r)) for r in data_rows],
            dtype=float,
        )
        arr[arr == null_value] = np.nan

        columns = curve_names[: arr.shape[1]]
        for i, c in enumerate(columns):
            if i < arr.shape[1]:
                raw_data[c] = [
                    None if np.isnan(v) else float(v) for v in arr[:, i]
                ]
        fmt = "las"

    # ---- JSON ----
    elif ext == ".json":
        try:
            jdata = _json.loads(text)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}")

        if isinstance(jdata, dict):
            for key in ("data", "logs", "curves", "records"):
                if key in jdata and isinstance(jdata[key], dict):
                    jdata = jdata[key]
                    break

        for k, v in (jdata.items() if isinstance(jdata, dict) else {}.items()):
            if isinstance(v, list):
                columns.append(k)
                raw_data[k] = v
        fmt = "json"

    else:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported extension '{ext}' for geo parsing. "
                "Supported: .csv, .tsv, .las, .json"
            ),
        )

    n_rows = max((len(v) for v in raw_data.values()), default=0)

    # Build canonical → original column mapping
    mapping: Dict[str, str] = {}
    for col in columns:
        canonical = _geo_resolve(col)
        if canonical and canonical not in mapping:
            mapping[canonical] = col

    def get_mapped(canon: str) -> List:
        orig = mapping.get(canon)
        return raw_data.get(orig, []) if orig else []

    data = {k: get_mapped(k) for k in _GEO_KEY_FIELDS if get_mapped(k)}
    clean_mapping = {k: v for k, v in mapping.items() if k in _GEO_KEY_FIELDS}

    return {
        "filename": filename,
        "format":   fmt,
        "n_rows":   n_rows,
        "columns":  columns,
        "mapping":  clean_mapping,
        "data":     data,
        "raw_data": raw_data,
    }


# ---------------------------------------------------------------------------
# Parameter metadata (for GUI sliders/inputs)
# ---------------------------------------------------------------------------

_PARAM_META = {
    # Growth kinetics
    "mu_max":     {"label": "μ_max", "unit": "h⁻¹", "min": 0.001, "max": 0.5,   "step": 0.001, "scale": "linear", "group": "Growth"},
    "K_N2":       {"label": "K_N₂",  "unit": "mol/L","min": 1e-6,  "max": 1e-3,  "step": None,  "scale": "log",    "group": "Growth"},
    "K_S":        {"label": "K_S",   "unit": "mol/L","min": 1e-5,  "max": 1e-2,  "step": None,  "scale": "log",    "group": "Growth"},
    "Y":          {"label": "Y",     "unit": "g VSS/g COD","min": 0.01,"max": 0.5,"step": 0.01, "scale": "linear", "group": "Growth"},
    "b_decay":    {"label": "b_decay","unit":"h⁻¹",  "min": 0.0001,"max": 0.05,  "step": 0.0001,"scale":"linear",  "group": "Growth"},
    # Nitrogenase
    "K_I_H2":     {"label": "K_I,H₂","unit": "atm",  "min": 0.01,  "max": 2.0,   "step": 0.01,  "scale": "linear", "group": "Nitrogenase"},
    "K_I_NH4":    {"label": "K_I,NH₄","unit":"mol/L", "min": 0.001, "max": 1.0,   "step": 0.001, "scale": "linear", "group": "Nitrogenase"},
    "r_nit_max":  {"label": "r_nit,max","unit":"mol N₂/g VSS/h","min":0.01,"max":5.0,"step":0.01,"scale":"linear","group":"Nitrogenase"},
    # HC degradation
    "k_ali":      {"label": "k_ali", "unit": "h⁻¹",  "min": 1e-5,  "max": 1e-2,  "step": None,  "scale": "log",    "group": "HC Degradation"},
    "k_arom":     {"label": "k_arom","unit": "h⁻¹",  "min": 1e-6,  "max": 1e-3,  "step": None,  "scale": "log",    "group": "HC Degradation"},
    "K_I_arom":   {"label": "K_I,arom","unit":"mol/L","min": 1e-5,  "max": 1e-2,  "step": None,  "scale": "log",    "group": "HC Degradation"},
    "f_arom0":    {"label": "f_arom₀","unit": "-",    "min": 0.0,   "max": 0.8,   "step": 0.01,  "scale": "linear", "group": "HC Degradation"},
    "Y_HC_H2":    {"label": "Y_HC→H₂","unit":"mol H₂/g COD","min":0.05,"max":0.8,"step":0.01,"scale":"linear","group":"HC Degradation"},
    # NH4 biosynthesis
    "K_N_fixed":  {"label": "K_N,fixed","unit":"mol/L","min":1e-7, "max":1e-4,   "step": None,  "scale": "log",    "group": "Nitrogen"},
    "Y_N":        {"label": "Y_N",  "unit": "g N/g VSS","min":0.05,"max":0.3,   "step": 0.005, "scale": "linear", "group": "Nitrogen"},
}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, reload=True)
