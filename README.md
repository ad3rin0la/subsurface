# TOUGHREACT-N2Bio

**A Framework for Simulating Subsurface Biological Hydrogen Production**
via Nitrogen Fixation and Hydrocarbon Degradation Under Multiphase Flow Conditions

Extending CO2Bio (Shabani & Vilcáez, 2019) to N₂-Based Biotic Reservoir Systems

---

## Overview

N2Bio is a scientific Python framework for simulating coupled multiphase flow, geochemistry, and biological kinetics in depleted petroleum reservoirs under N₂ injection. The target application is *in-situ* biological H₂ production by a thermophilic, diazotrophic chassis organism (*Thermotoga* sp. expressing the FS406-22 *nif* cluster).

The framework is modelled on the TOUGHREACT-CO2Bio module (Shabani & Vilcáez, 2019) and extends it in three critical ways:

| Feature | CO2Bio | N2Bio v2 |
|---|---|---|
| Injected gas | CO₂ | **N₂** (direct nitrogenase substrate) |
| H₂ directionality | consumed intermediate | **obligate product** (two sources) |
| Hydrocarbon role | passive electron donor | **active**: HC_ali powers the system; HC_arom inhibits nitrogenase |
| N limitation | not modelled | **central feature**: N₂ injection relieves reservoir N-limitation |

The residual hydrocarbon pool in a depleted reservoir acts as the electron donor that powers nitrogen fixation, while nitrogen fixation unlocks that same HC pool as a growth substrate — a mutually enabling positive feedback unique to this system.

---

## Installation

```bash
git clone https://github.com/anabaena/subsurface.git
cd subsurface
pip install -e .
```

**Requirements:** Python ≥ 3.8, numpy ≥ 1.24, scipy ≥ 1.10, matplotlib ≥ 3.7, cvxpy ≥ 1.4

---

## Quick Start

```python
from n2bio import N2BioSimulation

# Default batch reactor: 85°C, 150 bar, 0.5 mol/kg NaCl
sim = N2BioSimulation.default_batch()
sol, phases = sim.run(t_end_hours=3000)

# Phase transitions
for i, ph in enumerate(phases):
    if i == 0 or phases[i] != phases[i-1]:
        print(f"t={sol.t[i]:.0f} h  Phase {int(ph)}: {ph.name}")

# Cumulative H₂ produced
print(f"Final cumulative H₂: {sol.y[6,-1]*1e6:.2f} μmol/L")
```

Custom conditions:

```python
from n2bio import N2BioSimulation, KineticParameters, ReservoirConditions, BiologicalState

params = KineticParameters(f_arom0=0.25, k_ali=8e-4)  # high aromatics, fast ali degradation
sim = N2BioSimulation.default_batch(T_C=90.0, P_bar=200.0, HC_total=8.0e-3)
sol, phases = sim.run(t_end_hours=5000)
```

---

## Module Structure

```
n2bio/
├── components.py           Component definitions (N₂, CO₂, H₂, NH₃, HC_ali, HC_arom, …)
├── thermophysics/
│   ├── eos.py              Peng-Robinson EOS (pure + mixture), fugacity, departure enthalpy
│   ├── solubility.py       Henry's law gas solubility in NaCl brine (Fernandez-Prini 2003)
│   └── viscosity.py        Friction-theory viscosity (Quiñones-Cisneros model) + Wilke mixing
├── kinetics/
│   ├── parameters.py       KineticParameters and ReservoirConditions dataclasses
│   ├── monod.py            Triple co-limitation Monod growth (HC_ali × NH₄⁺ × N₂)
│   ├── nitrogenase.py      Nitrogenase rate with H₂ and aromatic inhibition
│   └── hc_degradation.py   Two-pool HC model (aliphatic fermentation + aromatic inhibition)
├── geochemistry/
│   └── speciation.py       NH₃/NH₄⁺ acid-base speciation (Van't Hoff temperature correction)
├── transport/
│   ├── grid.py             1D IFDM grid (Narasimhan & Witherspoon 1976)
│   └── flow.py             Multiphase Darcy flow, Brooks-Corey relative permeability
├── simulation/
│   ├── state.py            BiologicalState dataclass (full ODE state vector)
│   ├── lifecycle.py        Lifecycle phase classifier (Phase 0–4)
│   └── solver.py           N2BioBatchSolver — stiff BDF ODE integration
├── fba/
│   ├── manifold_fba.py     Matrix manifold FBA core (Grassmann · Stiefel · SPD)
│   └── bridge.py           BiologicalState → CommunityState unit-conversion bridge
└── verification/
    ├── stage1_thermo.py    PR EOS density, viscosity, and N₂ solubility benchmarks
    ├── stage2_flow.py      Multiphase flow and IFDM grid verification
    ├── stage3_kinetics.py  Monod, nitrogenase, HC degradation kinetic checks
    └── stage4_hc.py        Full batch lifecycle and H₂ yield quantitative verification
```

---

## Biological Lifecycle Phases

The coupled N₂-fixation / HC-fermentation system passes through five phases in a depleted reservoir:

| Phase | Name | Dominant process | Key state |
|---|---|---|---|
| 0 | Inoculation lag | Cell attachment; no fixed N | [NH₄⁺] ≪ K_N_fixed; μ ≈ 0 |
| 1 | N₂ fixation bootstrap | Nitrogenase active; [NH₄⁺] accumulates | r_HC increasing; f_arom low |
| 2 | Peak production | Max biomass; both H₂ pathways at capacity | P_H₂ near K_I_H₂ threshold |
| 3 | HC depletion / aromatic inhibition | HC_ali exhausted; f_arom rising | r_nit declining |
| 4 | Biological senescence | Electron donors exhausted; X decaying | H₂ → 0 |

---

## Component System (9 components)

| Component | Phase(s) | Role |
|---|---|---|
| N₂ | Gas + aqueous | **Primary** injected substrate; nitrogenase feedstock |
| CO₂ | Gas + aqueous | Present as EOR gas / metabolite |
| H₂ | Gas + aqueous | Obligate product of nitrogenase; also from dark fermentation |
| NH₃/NH₄⁺ | Aqueous | Co-product of fixation; biosynthesis substrate |
| HC_ali | Aqueous + adsorbed | Aliphatic pool; electron donor; fermented by Thermotoga |
| HC_arom | Aqueous + adsorbed | Aromatic pool; inhibits nitrogenase at elevated concentrations |
| H₂O | Liquid | Formation brine solvent |
| NaCl | Aqueous/solid | Salinity; Setschenow salting-out of gases |

Primary variable vector: `[P, X_NaCl, Z_N₂, Z_CO₂, Z_H₂, Z_NH₃, C_HC_ali, C_HC_arom, T]`

---

## Matrix Manifold FBA (`n2bio.fba`)

The `fba` subpackage adds constraint-based stoichiometric modelling of the full
four-guild consortium on three Riemannian manifolds, complementing the kinetic
ODE model with steady-state flux analysis.

### Geometric enhancements over standard LP-FBA

| Manifold | Role |
|---|---|
| **Grassmann Gr(k, n)** | Each guild's null space is a point on Gr(k, n). Geodesic distance between guilds quantifies metabolic complementarity. The lifecycle trajectory is a geodesic path across phases. |
| **Stiefel V_{n,k}** | Orthonormal null-space bases live on V_{n,k}. Riemannian gradient descent stays in null(S) exactly — no LP re-projection needed. |
| **SPD Sym⁺(n)** | Flux covariance lives on Sym⁺(n). Fisher-Rao metric enables principled cross-guild comparison and anisotropic sampling. |

### Guilds and exchange metabolites

| Guild | Chassis | Primary reaction |
|---|---|---|
| 0 Diazotroph | *Thermotoga* | N₂ fixation + dark fermentation |
| 1 Hydrogenotroph | Methanogen | H₂ + CO₂ → CH₄ |
| 2 SRB | Sulfate reducer | H₂/HC + SO₄²⁻ → H₂S |
| 3 Aromatic degrader | HC degrader | HC_arom detoxification |

Exchange pool (9 shared metabolites): N₂_aq, NH₄, H₂_aq, CO₂_aq, HC_ali, SO₄, H₂S, HC_arom, CH₄.

### Quick start — standalone FBA

```python
import numpy as np
from n2bio.fba import CommunityState, run_community_fba, lifecycle_scan

# Single phase (Phase 2 — peak production)
state = CommunityState(
    biomass_fractions=np.array([0.55, 0.20, 0.15, 0.10]),
    NH4_conc=2.0, SO4_conc=1.0,
    H2_partial_pressure=0.20, HC_arom_conc=0.25,
    lifecycle_phase=2,
)
result = run_community_fba(state, use_manifold=True)

from n2bio.fba import print_community_report
print_community_report(result, state)

# Full lifecycle scan (Phases 0 → 4)
scan = lifecycle_scan(n_phases=5, use_manifold=True)
from n2bio.fba import print_lifecycle_summary
print_lifecycle_summary(scan)
```

### Bridge from kinetic simulation

```python
from n2bio import N2BioSimulation
from n2bio.fba import run_community_fba
from n2bio.fba.bridge import state_to_community

sim = N2BioSimulation.default_batch()
sol, phases = sim.run(t_end_hours=3000)

# Find first Phase-2 snapshot and run FBA on it
idx = next(i for i, p in enumerate(phases) if int(p) == 2)
from n2bio.simulation.state import BiologicalState
bio = BiologicalState.from_array(sol.y[:, idx], sim.conditions)
community_state = state_to_community(bio, lifecycle_phase=2, SO4_conc_mM=1.0)
result = run_community_fba(community_state)
```

### N-S coupling in the FBA layer

`_exchange_bounds_from_state` encodes the same N–S competition that drives the
kinetic model:

```
# Diazotroph: nitrogenase flux capped by H₂ and HC_arom inhibition
nit_max = 10 × K_I_H₂/(K_I_H₂ + P_H₂) × K_I_arom/(K_I_arom + [HC_arom])

# SRB: dissimilatory SO₄ pool reduced as NH₄ drives OAS synthesis
SO₄_available = SO₄_total × (1 − 0.65 × NH₄/(K_OAS + NH₄))
```

---

## Key Kinetic Equations

**Complete growth rate (triple co-limitation):**

```
μ = μ_max × [HC_ali]/(K_S + [HC_ali]) × [NH₄⁺]/(K_N_fixed + [NH₄⁺]) × [N₂]_aq/(K_N₂ + [N₂]_aq)
```

**Nitrogenase rate with dual inhibition:**

```
r_nit = r_nit_max × [N₂]_aq/(K_N₂ + [N₂]_aq) × K_I_H₂/(K_I_H₂ + P_H₂) × K_I_arom/(K_I_arom + [HC_arom]_aq)
```

**HC pool evolution:**

```
d[HC_ali]/dt  = −k_ali  × [HC_ali]  × X    (fermentative consumption)
d[HC_arom]/dt = −k_arom × [HC_arom] × X    (slow thermolytic transformation)
f_arom        = [HC_arom] / ([HC_ali] + [HC_arom])   (computed dynamically per cell)
```

**Total H₂ production (two sources):**

```
r_H₂_total = r_nit × X  +  k_ali × [HC_ali] × X × Y_HC→H₂
              ↑ nitrogenase      ↑ dark fermentation (Thauer limit)
```

---

## Running Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Individual test suites
python tests/test_eos.py
python tests/test_kinetics.py
python tests/test_lifecycle.py

# Verification stages
python -m n2bio.verification.stage1_thermo
python -m n2bio.verification.stage2_flow
python -m n2bio.verification.stage3_kinetics
python -m n2bio.verification.stage4_hc
```

---

## Examples

```bash
# Batch reactor (well-mixed single cell)
python examples/batch_reactor.py

# With plots (requires matplotlib)
python examples/batch_reactor.py --plot

# 1D reservoir column (operator-split transport + reaction)
python examples/reservoir_1d.py --plot
```

---

## Verification Strategy (5 Stages)

| Stage | Approach | Target accuracy |
|---|---|---|
| 1 | PR EOS vs NIST REFPROP; N₂ solubility vs Battino (1982) | < 2% density; < 5% viscosity |
| 2 | IFDM flow vs analytical solution; mass balance closure | Saturation fronts within 3% |
| 3 | Monod / nitrogenase / HC kinetics at theoretical limits | Machine precision for analytical tests |
| 4 | Batch reactor H₂ yield vs Thermotoga CSTR data | < 15% cumulative H₂ error |
| 5 | Field-scale mass balance and phase-lifecycle ordering | < 0.01% mass balance error |

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `mu_max` | 0.08 h⁻¹ | Max diazotrophic growth rate |
| `K_N2` | 0.05 mmol/L | N₂ half-saturation constant |
| `K_S` | 0.5 mmol/L | HC_ali half-saturation constant |
| `K_I_H2` | 0.3 atm | H₂ inhibition on nitrogenase |
| `K_I_arom` | 0.5 mmol/L | Aromatic HC inhibition on nitrogenase |
| `k_ali` | 0.01 d⁻¹ | Aliphatic HC first-order degradation rate |
| `k_arom` | 0.0005 d⁻¹ | Aromatic HC transformation rate |
| `f_arom0` | 0.15 | Initial aromatic fraction of HC pool |
| `Y_HC_H2` | 0.35 mol H₂/g COD | Thermotoga dark fermentation H₂ yield |

---

## References

**Primary framework:**
- Shabani, B. & Vilcáez, J. (2019). TOUGHREACT-CO2Bio. *J. Nat. Gas Sci. Eng.* 63, 85–94.

**HC degradation and N fixation:**
- Knowles, R., Wishart, C. & Sargeant, E.I. (1977). *Environment Research* 13(3), 422–432.
- Foght, J. (2010). In *Handbook of Hydrocarbon and Lipid Microbiology*, pp. 1661–1668. Springer.
- Pradhan, N. et al. (2021). *Bioresource Technology* 332, 125127.


**Thermophysical:**
- Peng, D.-Y. & Robinson, D.B. (1976). *Ind. Eng. Chem. Fundam.* 15, 59–64.
- Fernandez-Prini, R. et al. (2003). *J. Phys. Chem. Ref. Data* 32, 903–916.
- Quiñones-Cisneros, S.E. et al. (2000, 2001). *Fluid Phase Equilibria* 169, 249–276; 178, 1–16.

---

*ANABAENA — TOUGHREACT-N2Bio Framework v2.0 — Confidential*
