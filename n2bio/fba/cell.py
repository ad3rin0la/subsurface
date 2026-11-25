"""Reservoir-cell adapter for joint community FBA."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from n2bio.geochemistry.speciation import NH3Speciation

from .community import (
    CommunityModel,
    CommunitySolution,
    LoadedOrganism,
    build_community_model,
    load_organism,
    solve_community,
)


@dataclass(frozen=True)
class AmmoniaResponse:
    """Curated reaction IDs and free-ammonia response coefficients."""

    atp_maintenance: Optional[str] = None
    acetoclastic_mcr: Optional[str] = None
    hydrogenotrophic_mcr: Optional[str] = None
    acetoclastic_ki_mM: float = 4.5
    hydrogenotrophic_ki_mM: float = 20.0
    atp_leak_per_mM: float = 0.45


@dataclass(frozen=True)
class SpeciesConfig:
    """Explicit model and reservoir coupling configuration for one member."""

    organism_id: str
    model_path: str
    exchanges: Mapping[str, str]
    biomass_reaction: Optional[str] = None
    cardinal_temperatures_c: Optional[Tuple[float, float, float]] = None
    ammonia_response: Optional[AmmoniaResponse] = None


@dataclass(frozen=True)
class ReservoirCellState:
    """Solutes are mol/L and biomass pools are gDW/L."""

    solutes: Mapping[str, float]
    mobile_biomass: Mapping[str, float]
    attached_biomass: Mapping[str, float] = field(default_factory=dict)
    temperature_c: Optional[float] = None
    ph: Optional[float] = None
    total_ammonia_n: Optional[float] = None


@dataclass(frozen=True)
class ReservoirCellResult:
    status: str
    solution: CommunitySolution
    state: ReservoirCellState
    solute_rates: Mapping[str, float]
    growth_rates: Mapping[str, float]
    free_ammonia_mM: Optional[float]


def load_species_manifest(path: Union[str, Path]) -> Tuple[SpeciesConfig, ...]:
    """Read a JSON manifest without guessing biological reaction IDs."""
    source = Path(path).expanduser().resolve()
    document = json.loads(source.read_text())
    records = document.get("species", document) if isinstance(document, dict) else document
    if not isinstance(records, list):
        raise ValueError("Species manifest must contain a list")
    configurations = []
    for record in records:
        response = record.get("ammonia_response")
        temperatures = record.get("cardinal_temperatures_c")
        configurations.append(
            SpeciesConfig(
                organism_id=record["organism_id"],
                model_path=str((source.parent / record["model_path"]).resolve()),
                exchanges=dict(record["exchanges"]),
                biomass_reaction=record.get("biomass_reaction"),
                cardinal_temperatures_c=tuple(temperatures) if temperatures else None,
                ammonia_response=AmmoniaResponse(**response) if response else None,
            )
        )
    return tuple(configurations)


def load_configured_organisms(
    configurations: Sequence[SpeciesConfig],
    validate: bool = True,
) -> Tuple[LoadedOrganism, ...]:
    """Load configured members and verify every reservoir exchange mapping."""
    loaded = []
    for config in configurations:
        organism = load_organism(
            config.model_path,
            organism_id=config.organism_id,
            biomass_reaction=config.biomass_reaction,
            validate=validate,
        )
        if len(set(config.exchanges.values())) != len(config.exchanges):
            raise ValueError(f"Duplicate exchange reaction in {config.organism_id}")
        exchange_ids = {reaction.id for reaction in organism.model.exchanges}
        for solute, reaction_id in config.exchanges.items():
            if reaction_id not in exchange_ids:
                raise ValueError(
                    f"{config.organism_id} maps {solute!r} to non-exchange {reaction_id!r}"
                )
            reaction = organism.model.reactions.get_by_id(reaction_id)
            if len(reaction.metabolites) != 1:
                raise ValueError(f"Exchange {reaction_id!r} must contain one metabolite")
        loaded.append(organism)
    return tuple(loaded)


def cardinal_temperature_fitness(
    temperature_c: float,
    minimum_c: float,
    optimum_c: float,
    maximum_c: float,
) -> float:
    """Rosso cardinal-temperature fitness in the closed interval [0, 1]."""
    if not minimum_c < optimum_c < maximum_c:
        raise ValueError("Cardinal temperatures must satisfy minimum < optimum < maximum")
    if temperature_c <= minimum_c or temperature_c >= maximum_c:
        return 0.0
    numerator = (temperature_c - maximum_c) * (temperature_c - minimum_c) ** 2
    denominator = (optimum_c - minimum_c) * (
        (optimum_c - minimum_c) * (temperature_c - optimum_c)
        - (optimum_c - maximum_c)
        * (optimum_c + minimum_c - 2.0 * temperature_c)
    )
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _environment_adjusted_members(
    configurations: Sequence[SpeciesConfig],
    organisms: Sequence[LoadedOrganism],
    state: ReservoirCellState,
) -> Tuple[Tuple[LoadedOrganism, ...], Optional[float]]:
    free_ammonia_mM = None
    if state.total_ammonia_n is not None:
        if state.temperature_c is None or state.ph is None:
            raise ValueError("Ammonia response requires temperature_c and ph")
        _, free_ammonia = NH3Speciation.speciate(
            state.total_ammonia_n, state.temperature_c + 273.15, state.ph
        )
        free_ammonia_mM = free_ammonia * 1000.0

    adjusted = []
    for config, original in zip(configurations, organisms):
        model = original.model.copy()
        biomass_id = original.info.biomass_reaction
        if config.cardinal_temperatures_c is not None:
            if state.temperature_c is None:
                raise ValueError("Cardinal-temperature response requires temperature_c")
            fitness = cardinal_temperature_fitness(
                state.temperature_c, *config.cardinal_temperatures_c
            )
            biomass = model.reactions.get_by_id(biomass_id)
            biomass.upper_bound = max(
                biomass.lower_bound, min(biomass.upper_bound, biomass.upper_bound * fitness)
            )

        response = config.ammonia_response
        if response is not None and free_ammonia_mM is not None:
            for reaction_id in (
                response.atp_maintenance,
                response.acetoclastic_mcr,
                response.hydrogenotrophic_mcr,
            ):
                if reaction_id is not None and reaction_id not in model.reactions:
                    raise ValueError(
                        f"Configured reaction {reaction_id!r} is absent from {config.organism_id}"
                    )
            if response.atp_maintenance:
                reaction = model.reactions.get_by_id(response.atp_maintenance)
                reaction.lower_bound = min(
                    reaction.upper_bound,
                    reaction.lower_bound + response.atp_leak_per_mM * free_ammonia_mM,
                )
            for reaction_id, inhibition_constant in (
                (response.acetoclastic_mcr, response.acetoclastic_ki_mM),
                (response.hydrogenotrophic_mcr, response.hydrogenotrophic_ki_mM),
            ):
                if reaction_id:
                    reaction = model.reactions.get_by_id(reaction_id)
                    factor = inhibition_constant / (inhibition_constant + free_ammonia_mM)
                    inhibited_bound = reaction.upper_bound * factor
                    reaction.upper_bound = max(
                        reaction.lower_bound,
                        min(reaction.upper_bound, inhibited_bound),
                    )
        adjusted.append(LoadedOrganism(original.organism_id, model, original.info))
    return tuple(adjusted), free_ammonia_mM


def _metabolite_mapping(
    configurations: Sequence[SpeciesConfig], organisms: Sequence[LoadedOrganism]
) -> Mapping[str, str]:
    mapping = {}
    for config, organism in zip(configurations, organisms):
        for solute, reaction_id in config.exchanges.items():
            reaction = organism.model.reactions.get_by_id(reaction_id)
            metabolite = next(iter(reaction.metabolites))
            previous = mapping.setdefault(metabolite.id, solute)
            if previous != solute:
                raise ValueError(
                    f"Metabolite {metabolite.id!r} maps to both {previous!r} and {solute!r}"
                )
    return mapping


def solve_reservoir_cell(
    configurations: Sequence[SpeciesConfig],
    organisms: Sequence[LoadedOrganism],
    state: ReservoirCellState,
    dt_hours: float,
) -> ReservoirCellResult:
    """Advance one well-mixed cell using a joint, finite-inventory FBA solve."""
    if dt_hours <= 0:
        raise ValueError("dt_hours must be positive")
    if len(configurations) != len(organisms):
        raise ValueError("Configurations and organisms must have equal length")
    organism_ids = {config.organism_id for config in configurations}
    if organism_ids != {organism.organism_id for organism in organisms}:
        raise ValueError("Configurations and loaded organisms do not match")
    mobile = {key: max(0.0, float(state.mobile_biomass.get(key, 0.0))) for key in organism_ids}
    attached = {key: max(0.0, float(state.attached_biomass.get(key, 0.0))) for key in organism_ids}
    totals = {key: mobile[key] + attached[key] for key in organism_ids}
    total_biomass = sum(totals.values())
    if total_biomass <= 0:
        raise ValueError("At least one organism must have positive biomass")

    organism_by_id = {organism.organism_id: organism for organism in organisms}
    active_configurations = tuple(
        config for config in configurations if totals[config.organism_id] > 0
    )
    active_organisms = tuple(
        organism_by_id[config.organism_id] for config in active_configurations
    )
    adjusted, free_ammonia_mM = _environment_adjusted_members(
        active_configurations, active_organisms, state
    )
    community = build_community_model(
        adjusted,
        metabolite_mapping=_metabolite_mapping(active_configurations, adjusted),
        fixed_abundances={
            config.organism_id: totals[config.organism_id] / total_biomass
            for config in active_configurations
        },
    )
    step_model = community.model.copy()
    configured_solutes = {
        solute for config in active_configurations for solute in config.exchanges
    }
    for solute, exchange_id in community.environment_exchanges.items():
        exchange = step_model.reactions.get_by_id(exchange_id)
        available = (
            max(0.0, float(state.solutes.get(solute, 0.0)))
            if solute in configured_solutes
            else 0.0
        )
        exchange.lower_bound = -available * 1000.0 / (dt_hours * total_biomass)
    step_community = CommunityModel(
        step_model,
        community.organisms,
        community.biomass_reactions,
        community.shared_metabolites,
        community.environment_exchanges,
        community.abundances,
        community.growth_variable_name,
    )
    solution = solve_community(step_community)
    if solution.status != "optimal":
        return ReservoirCellResult(
            solution.status, solution, state, {}, {}, free_ammonia_mM
        )

    solute_rates = {
        solute: flux * total_biomass / 1000.0
        for solute, flux in solution.shared_exchange_fluxes.items()
    }
    next_solutes = dict(state.solutes)
    for solute, rate in solute_rates.items():
        next_solutes[solute] = max(0.0, next_solutes.get(solute, 0.0) + rate * dt_hours)
    growth_rate = solution.objective_value
    factor = float(np.exp(growth_rate * dt_hours))
    next_mobile = {key: value * factor for key, value in mobile.items()}
    next_attached = {key: value * factor for key, value in attached.items()}
    next_state = ReservoirCellState(
        next_solutes,
        next_mobile,
        next_attached,
        state.temperature_c,
        state.ph,
        state.total_ammonia_n,
    )
    return ReservoirCellResult(
        solution.status,
        solution,
        next_state,
        solute_rates,
        {key: growth_rate if totals[key] > 0 else 0.0 for key in organism_ids},
        free_ammonia_mM,
    )

