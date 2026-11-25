"""SBML-backed multi-organism flux balance analysis.

This module keeps organism-specific intracellular networks separate while
placing mapped extracellular metabolites in one shared community compartment.
The resulting COBRA model is a single optimization problem, so secretion,
uptake, competition, and cross-feeding obey shared-pool mass balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import re

import numpy as np


FluxBounds = Tuple[float, float]


DEFAULT_EXTRACELLULAR_COMPARTMENTS = frozenset(
    {"e", "e0", "extra", "extracellular", "external"}
)

# Common BiGG-style aliases. Callers can extend or replace these per project.
DEFAULT_CANONICAL_METABOLITES: Dict[str, str] = {
    "h2": "h2",
    "h2_e": "h2",
    "co2": "co2",
    "co2_e": "co2",
    "n2": "n2",
    "n2_e": "n2",
    "nh4": "nh4",
    "nh4_e": "nh4",
    "so4": "so4",
    "so4_e": "so4",
    "h2s": "h2s",
    "h2s_e": "h2s",
    "ch4": "ch4",
    "ch4_e": "ch4",
    "ac": "acetate",
    "ac_e": "acetate",
    "acetate": "acetate",
    "acetate_e": "acetate",
}


def _cobra():
    try:
        import cobra
    except ImportError as exc:
        raise ImportError(
            "SBML community modelling requires COBRApy. Install n2bio with "
            "the 'community' extra."
        ) from exc
    return cobra


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    if not clean:
        raise ValueError("Organism IDs must contain at least one letter or number")
    return clean


def _objective_reaction_ids(model) -> List[str]:
    return sorted(
        reaction.id
        for reaction in model.reactions
        if abs(float(reaction.objective_coefficient)) > 1e-12
    )


def _sbml_flux_units(path: Path) -> Optional[str]:
    try:
        import libsbml
    except ImportError:
        return None
    document = libsbml.readSBMLFromFile(str(path))
    model = document.getModel()
    if model is None:
        return None
    units = set()
    for reaction_index in range(model.getNumReactions()):
        reaction = model.getReaction(reaction_index)
        fbc = reaction.getPlugin("fbc")
        if fbc is None:
            continue
        for parameter_id in (fbc.getLowerFluxBound(), fbc.getUpperFluxBound()):
            parameter = model.getParameter(parameter_id)
            if parameter is not None and parameter.isSetUnits():
                units.add(parameter.getUnits())
    return ", ".join(sorted(units)) or None


@dataclass(frozen=True)
class OrganismModelInfo:
    organism_id: str
    model_id: str
    source_file: str
    objective_reactions: Tuple[str, ...]
    biomass_reaction: Optional[str]
    compartments: Mapping[str, str]
    exchange_reactions: Tuple[str, ...]
    flux_units: Optional[str]
    fbc_bounds: Mapping[str, FluxBounds]
    gene_reaction_rules: Mapping[str, str]
    annotations: Mapping[str, Mapping]
    validation_warnings: Tuple[str, ...]


@dataclass
class LoadedOrganism:
    organism_id: str
    model: object
    info: OrganismModelInfo


@dataclass
class CommunityModel:
    model: object
    organisms: Mapping[str, LoadedOrganism]
    biomass_reactions: Mapping[str, str]
    shared_metabolites: Mapping[str, str]
    environment_exchanges: Mapping[str, str]
    abundances: Mapping[str, float]
    growth_variable_name: Optional[str] = None


@dataclass(frozen=True)
class CommunitySolution:
    status: str
    objective_value: float
    organism_growth: Mapping[str, float]
    shared_exchange_fluxes: Mapping[str, float]
    fluxes: Mapping[str, float]


@dataclass(frozen=True)
class DynamicCommunityResult:
    times: np.ndarray
    pools: Mapping[str, np.ndarray]
    biomass: Mapping[str, np.ndarray]
    statuses: Tuple[str, ...]


def load_sbml_organism(
    path: Union[str, Path],
    organism_id: Optional[str] = None,
    biomass_reaction: Optional[str] = None,
    validate: bool = True,
) -> LoadedOrganism:
    """Load one SBML-FBC model and retain its modelling metadata."""
    cobra = _cobra()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    if validate:
        model, validation = cobra.io.validate_sbml_model(str(source))
        if model is None:
            fatal_messages = validation.get("SBML_FATAL", []) + validation.get(
                "COBRA_FATAL", []
            )
            detail = "; ".join(fatal_messages) or "unknown SBML validation error"
            raise ValueError(f"Could not load {source.name}: {detail}")
        warnings = [
            f"{category}: {message}"
            for category, messages in validation.items()
            for message in messages
        ]
    else:
        model = cobra.io.read_sbml_model(str(source))
        warnings = []
    org_id = _safe_id(organism_id or model.id or source.stem)
    objective_ids = _objective_reaction_ids(model)
    if biomass_reaction is None:
        if len(objective_ids) == 1:
            biomass_reaction = objective_ids[0]
        else:
            biomass_reaction = next(
                (r.id for r in model.reactions if "biomass" in r.id.lower()),
                None,
            )
    if biomass_reaction is not None and biomass_reaction not in model.reactions:
        raise ValueError(
            f"Biomass reaction '{biomass_reaction}' is absent from {source.name}"
        )

    flux_units = model.notes.get("flux_units") if hasattr(model, "notes") else None
    flux_units = flux_units or _sbml_flux_units(source)
    info = OrganismModelInfo(
        organism_id=org_id,
        model_id=model.id or source.stem,
        source_file=str(source),
        objective_reactions=tuple(objective_ids),
        biomass_reaction=biomass_reaction,
        compartments=dict(model.compartments),
        exchange_reactions=tuple(sorted(r.id for r in model.exchanges)),
        flux_units=str(flux_units) if flux_units else None,
        fbc_bounds={r.id: (float(r.lower_bound), float(r.upper_bound)) for r in model.reactions},
        gene_reaction_rules={r.id: r.gene_reaction_rule for r in model.reactions if r.gene_reaction_rule},
        annotations={
            **({"model": dict(model.annotation)} if getattr(model, "annotation", None) else {}),
            **{f"metabolite:{m.id}": dict(m.annotation) for m in model.metabolites if m.annotation},
            **{f"reaction:{r.id}": dict(r.annotation) for r in model.reactions if r.annotation},
            **{f"gene:{g.id}": dict(g.annotation) for g in model.genes if g.annotation},
        },
        validation_warnings=tuple(warnings),
    )
    return LoadedOrganism(org_id, model, info)


def _canonical_name(metabolite, mapping: Mapping[str, str]) -> str:
    candidates = [metabolite.id, re.sub(r"_[A-Za-z][A-Za-z0-9]*$", "", metabolite.id)]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return candidates[-1]


def build_community_model(
    organisms: Sequence[LoadedOrganism],
    metabolite_mapping: Optional[Mapping[str, str]] = None,
    medium: Optional[Mapping[str, FluxBounds]] = None,
    objective_weights: Optional[Mapping[str, float]] = None,
    minimum_growth: Optional[Mapping[str, float]] = None,
    fixed_abundances: Optional[Mapping[str, float]] = None,
    extracellular_compartments: Iterable[str] = DEFAULT_EXTRACELLULAR_COMPARTMENTS,
) -> CommunityModel:
    """Assemble namespaced organisms into one mass-balanced community model.

    Boundary reactions from source models are replaced by community-level
    exchanges. Existing organism transport reactions connect intracellular
    metabolites to the shared extracellular metabolites.
    """
    cobra = _cobra()
    if not organisms:
        raise ValueError("At least one organism model is required")
    org_ids = [organism.organism_id for organism in organisms]
    if len(set(org_ids)) != len(org_ids):
        raise ValueError("Organism IDs must be unique")
    for label, values in (
        ("objective_weights", objective_weights),
        ("minimum_growth", minimum_growth),
    ):
        unknown = set(values or {}) - set(org_ids)
        if unknown:
            raise ValueError(f"{label} contains unknown organisms: {sorted(unknown)}")

    mapping = dict(DEFAULT_CANONICAL_METABOLITES)
    if metabolite_mapping:
        mapping.update(metabolite_mapping)
    external = set(extracellular_compartments)
    community = cobra.Model("n2bio_community")
    community.compartments = {"community_e": "shared community extracellular pool"}
    shared = {}
    biomass_reactions = {}

    for loaded in organisms:
        prefix = f"{loaded.organism_id}__"
        source = loaded.model
        boundary_ids = {reaction.id for reaction in source.exchanges}
        metabolite_copies = {}

        for metabolite in source.metabolites:
            if metabolite.compartment in external:
                canonical = _canonical_name(metabolite, mapping)
                if canonical not in shared:
                    shared[canonical] = cobra.Metabolite(
                        f"community_e__{canonical}",
                        name=metabolite.name or canonical,
                        compartment="community_e",
                        formula=metabolite.formula,
                        charge=metabolite.charge,
                    )
                else:
                    existing = shared[canonical]
                    if existing.formula and metabolite.formula and existing.formula != metabolite.formula:
                        raise ValueError(
                            f"Mapped metabolite '{canonical}' has conflicting formulas: "
                            f"{existing.formula} and {metabolite.formula}"
                        )
                    if existing.charge is not None and metabolite.charge is not None and existing.charge != metabolite.charge:
                        raise ValueError(
                            f"Mapped metabolite '{canonical}' has conflicting charges: "
                            f"{existing.charge} and {metabolite.charge}"
                        )
                metabolite_copies[metabolite.id] = shared[canonical]
            else:
                copied = metabolite.copy()
                copied.id = prefix + metabolite.id
                copied.compartment = prefix + (metabolite.compartment or "c")
                metabolite_copies[metabolite.id] = copied
                community.compartments.setdefault(
                    copied.compartment,
                    f"{loaded.organism_id} {metabolite.compartment or 'internal'}",
                )

        for reaction in source.reactions:
            if reaction.id in boundary_ids:
                continue
            copied = cobra.Reaction(prefix + reaction.id)
            copied.name = reaction.name
            copied.lower_bound = reaction.lower_bound
            copied.upper_bound = reaction.upper_bound
            copied.subsystem = reaction.subsystem
            copied.annotation = dict(reaction.annotation)
            copied.notes = dict(reaction.notes)
            stoichiometry = {}
            for metabolite, coefficient in reaction.metabolites.items():
                target = metabolite_copies[metabolite.id]
                stoichiometry[target] = stoichiometry.get(target, 0.0) + coefficient
            copied.add_metabolites(stoichiometry)
            community.add_reactions([copied])
            if reaction.gene_reaction_rule:
                rule = reaction.gene_reaction_rule
                for gene in sorted(reaction.genes, key=lambda item: len(item.id), reverse=True):
                    rule = re.sub(
                        rf"(?<![A-Za-z0-9_.-]){re.escape(gene.id)}(?![A-Za-z0-9_.-])",
                        prefix + gene.id,
                        rule,
                    )
                copied.gene_reaction_rule = rule

        biomass_id = loaded.info.biomass_reaction
        if biomass_id is None:
            raise ValueError(f"No biomass reaction identified for {loaded.organism_id}")
        combined_biomass_id = prefix + biomass_id
        if combined_biomass_id not in community.reactions:
            raise ValueError(
                f"Biomass reaction {biomass_id} is a boundary reaction and cannot drive growth"
            )
        biomass_reactions[loaded.organism_id] = combined_biomass_id

    missing_shared = [
        metabolite for metabolite in shared.values()
        if not community.metabolites.has_id(metabolite.id)
    ]
    community.add_metabolites(missing_shared)
    unknown_medium = set(medium or {}) - set(shared)
    if unknown_medium:
        raise ValueError(
            f"Medium contains metabolites absent from the community: {sorted(unknown_medium)}"
        )
    exchange_ids = {}
    for canonical, metabolite in shared.items():
        exchange = cobra.Reaction(f"EX_community_e__{canonical}")
        exchange.name = f"Community exchange for {canonical}"
        exchange.add_metabolites({metabolite: -1.0})
        lo, hi = (medium or {}).get(canonical, (0.0, 1000.0))
        if float(lo) > float(hi):
            raise ValueError(f"Invalid medium bounds for {canonical}: {(lo, hi)}")
        exchange.bounds = (float(lo), float(hi))
        community.add_reactions([exchange])
        exchange_ids[canonical] = exchange.id

    minimum_growth = minimum_growth or {}
    for organism_id, lower_bound in minimum_growth.items():
        if float(lower_bound) < 0:
            raise ValueError("Minimum growth constraints cannot be negative")
        community.reactions.get_by_id(biomass_reactions[organism_id]).lower_bound = float(lower_bound)

    weights = {organism_id: 1.0 for organism_id in org_ids}
    if objective_weights:
        weights.update({key: float(value) for key, value in objective_weights.items()})
    if any(value < 0 for value in weights.values()):
        raise ValueError("Objective weights cannot be negative")
    abundance_values = {organism_id: 1.0 / len(org_ids) for organism_id in org_ids}
    growth_variable_name = None

    if fixed_abundances is not None:
        if set(fixed_abundances) != set(org_ids):
            raise ValueError("Fixed abundances must specify every organism exactly once")
        total = sum(float(value) for value in fixed_abundances.values())
        if total <= 0 or any(float(value) <= 0 for value in fixed_abundances.values()):
            raise ValueError("Fixed abundances must be positive")
        abundance_values = {key: float(value) / total for key, value in fixed_abundances.items()}
        growth_variable_name = "community_growth_rate"
        mu = community.problem.Variable(growth_variable_name, lb=0)
        community.add_cons_vars([mu])
        for organism_id, abundance in abundance_values.items():
            biomass = community.reactions.get_by_id(biomass_reactions[organism_id])
            constraint = community.problem.Constraint(
                biomass.flux_expression - abundance * mu,
                lb=0,
                ub=0,
                name=f"fixed_abundance__{organism_id}",
            )
            community.add_cons_vars([constraint])
        community.objective = community.problem.Objective(mu, direction="max")
    else:
        community.objective = {
            community.reactions.get_by_id(biomass_reactions[organism_id]): weights[organism_id]
            for organism_id in org_ids
        }
        community.objective_direction = "max"

    return CommunityModel(
        model=community,
        organisms={organism.organism_id: organism for organism in organisms},
        biomass_reactions=biomass_reactions,
        shared_metabolites={key: value.id for key, value in shared.items()},
        environment_exchanges=exchange_ids,
        abundances=abundance_values,
        growth_variable_name=growth_variable_name,
    )


def solve_community(community: CommunityModel) -> CommunitySolution:
    solution = community.model.optimize()
    objective = float(solution.objective_value) if solution.objective_value is not None else np.nan
    growth = {
        organism_id: float(solution.fluxes.get(reaction_id, np.nan))
        for organism_id, reaction_id in community.biomass_reactions.items()
    }
    exchanges = {
        metabolite: float(solution.fluxes.get(reaction_id, np.nan))
        for metabolite, reaction_id in community.environment_exchanges.items()
    }
    return CommunitySolution(
        status=solution.status,
        objective_value=objective,
        organism_growth=growth,
        shared_exchange_fluxes=exchanges,
        fluxes={key: float(value) for key, value in solution.fluxes.items()},
    )


def solve_cooperative_tradeoff(
    community: CommunityModel,
    fraction_of_optimum: float = 0.8,
) -> CommunitySolution:
    """Retain community productivity while maximizing the weakest member."""
    if not 0 < fraction_of_optimum <= 1:
        raise ValueError("fraction_of_optimum must be in (0, 1]")
    model = community.model.copy()
    primary = model.slim_optimize(error_value=np.nan)
    if not np.isfinite(primary):
        return solve_community(community)

    objective_expression = model.objective.expression
    retained = model.problem.Constraint(
        objective_expression,
        lb=float(primary) * fraction_of_optimum,
        name="cooperative_primary_objective",
    )
    minimum = model.problem.Variable("minimum_member_growth", lb=0)
    model.add_cons_vars([retained, minimum])
    for organism_id, reaction_id in community.biomass_reactions.items():
        biomass = model.reactions.get_by_id(reaction_id)
        model.add_cons_vars([
            model.problem.Constraint(
                biomass.flux_expression - minimum,
                lb=0,
                name=f"cooperative_growth__{organism_id}",
            )
        ])
    model.objective = model.problem.Objective(minimum, direction="max")
    copied = CommunityModel(
        model=model,
        organisms=community.organisms,
        biomass_reactions=community.biomass_reactions,
        shared_metabolites=community.shared_metabolites,
        environment_exchanges=community.environment_exchanges,
        abundances=community.abundances,
    )
    return solve_community(copied)


def simulate_dynamic_community(
    community: CommunityModel,
    initial_pools: Mapping[str, float],
    initial_biomass: Mapping[str, float],
    t_end: float,
    dt: float,
    volume: float = 1.0,
) -> DynamicCommunityResult:
    """Euler dynamic FBA for a fixed-composition community and finite pools."""
    if t_end <= 0 or dt <= 0 or volume <= 0:
        raise ValueError("t_end, dt, and volume must be positive")
    if community.growth_variable_name is None:
        raise ValueError(
            "Dynamic community FBA requires a model built with fixed_abundances"
        )
    times = np.arange(0.0, t_end + 0.5 * dt, dt)
    pools = {key: np.zeros(times.size) for key in initial_pools}
    biomass = {key: np.zeros(times.size) for key in community.biomass_reactions}
    for key, value in initial_pools.items():
        pools[key][0] = max(0.0, float(value))
    for key in biomass:
        biomass[key][0] = max(0.0, float(initial_biomass.get(key, 0.0)))
    initial_total = sum(values[0] for values in biomass.values())
    if initial_total > 0:
        observed_abundances = {
            key: values[0] / initial_total for key, values in biomass.items()
        }
        if any(
            not np.isclose(observed_abundances[key], community.abundances[key], atol=1e-6)
            for key in biomass
        ):
            raise ValueError(
                "initial_biomass fractions must match the model's fixed_abundances"
            )

    statuses = []
    for index in range(times.size - 1):
        step_model = community.model.copy()
        total_biomass = sum(biomass[key][index] for key in biomass)
        for metabolite, exchange_id in community.environment_exchanges.items():
            if metabolite in pools:
                exchange = step_model.reactions.get_by_id(exchange_id)
                if total_biomass > 0:
                    exchange.lower_bound = max(
                        exchange.lower_bound,
                        -pools[metabolite][index] * volume / (dt * total_biomass),
                    )
                else:
                    exchange.lower_bound = 0.0
        step = CommunityModel(
            step_model,
            community.organisms,
            community.biomass_reactions,
            community.shared_metabolites,
            community.environment_exchanges,
            community.abundances,
        )
        solution = solve_community(step)
        statuses.append(solution.status)
        if solution.status != "optimal":
            for values in pools.values():
                values[index + 1] = values[index]
            for values in biomass.values():
                values[index + 1] = values[index]
            continue

        for metabolite, values in pools.items():
            exchange_flux = solution.shared_exchange_fluxes.get(metabolite, 0.0)
            values[index + 1] = max(
                0.0,
                values[index] + exchange_flux * total_biomass * dt / volume,
            )
        for organism_id, values in biomass.items():
            growth_rate = solution.objective_value
            values[index + 1] = values[index] * np.exp(growth_rate * dt)

    return DynamicCommunityResult(times, pools, biomass, tuple(statuses))


def aligned_exchange_flux_matrix(
    solutions: Sequence[CommunitySolution],
    metabolite_order: Sequence[str],
) -> np.ndarray:
    """Build samples in a declared, biologically shared metabolite basis."""
    return np.asarray(
        [
            [solution.shared_exchange_fluxes.get(metabolite, 0.0) for metabolite in metabolite_order]
            for solution in solutions
        ],
        dtype=float,
    )


def aligned_flux_covariance(
    solutions: Sequence[CommunitySolution],
    metabolite_order: Sequence[str],
    regularization: float = 1e-8,
) -> np.ndarray:
    """Estimate SPD covariance from repeated flux samples in shared coordinates."""
    matrix = aligned_exchange_flux_matrix(solutions, metabolite_order)
    if matrix.shape[0] < 2:
        raise ValueError("At least two flux samples are required for covariance")
    covariance = np.cov(matrix, rowvar=False)
    return np.atleast_2d(covariance) + regularization * np.eye(matrix.shape[1])
