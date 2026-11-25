"""Reservoir-cell coupling tests for community FBA."""

import json

import pytest

cobra = pytest.importorskip("cobra")

from n2bio.fba.cell import (
    AmmoniaResponse,
    ReservoirCellState,
    SpeciesConfig,
    _environment_adjusted_members,
    cardinal_temperature_fitness,
    load_configured_organisms,
    load_species_manifest,
    solve_reservoir_cell,
)


def _write_member(path, model_id, substrate, product=None):
    model = cobra.Model(model_id)
    substrate_e = cobra.Metabolite(substrate, compartment="e")
    precursor = cobra.Metabolite(f"{model_id}_precursor_c", compartment="c")
    exchange = cobra.Reaction(f"EX_{model_id}_feed", lower_bound=-10, upper_bound=1000)
    exchange.add_metabolites({substrate_e: -1})
    uptake = cobra.Reaction(f"{model_id}_uptake", lower_bound=0, upper_bound=1000)
    uptake.add_metabolites({substrate_e: -1, precursor: 1})
    biomass = cobra.Reaction(f"BIOMASS_{model_id}", lower_bound=0, upper_bound=1000)
    biomass.add_metabolites({precursor: -1})
    reactions = [exchange, uptake, biomass]
    if product:
        product_e = cobra.Metabolite(product, compartment="e")
        biomass.add_metabolites({product_e: 1})
        secretion = cobra.Reaction(
            f"EX_{model_id}_product", lower_bound=-1000, upper_bound=1000
        )
        secretion.add_metabolites({product_e: -1})
        reactions.append(secretion)
    model.add_reactions(reactions)
    model.objective = biomass
    cobra.io.save_json_model(model, str(path))
    return model


def test_manifest_loads_json_models_and_exact_exchange_maps(tmp_path):
    model_path = tmp_path / "member.json"
    _write_member(model_path, "member", "odd_substrate_e")
    manifest = tmp_path / "species.json"
    manifest.write_text(
        json.dumps(
            {
                "species": [
                    {
                        "organism_id": "member",
                        "model_path": "member.json",
                        "biomass_reaction": "BIOMASS_member",
                        "exchanges": {"glucose": "EX_member_feed"},
                    }
                ]
            }
        )
    )

    configurations = load_species_manifest(manifest)
    organisms = load_configured_organisms(configurations)

    assert configurations[0].exchanges == {"glucose": "EX_member_feed"}
    assert organisms[0].info.model_id == "member"


def test_joint_cell_solve_uses_inventory_and_same_step_crossfeeding(tmp_path):
    producer_path = tmp_path / "producer.json"
    consumer_path = tmp_path / "consumer.json"
    _write_member(producer_path, "producer", "unusual_glucose_e", "private_acetate_e")
    _write_member(consumer_path, "consumer", "private_acetate_e")
    configurations = (
        SpeciesConfig(
            "producer",
            str(producer_path),
            {"glucose": "EX_producer_feed", "acetate": "EX_producer_product"},
            "BIOMASS_producer",
        ),
        SpeciesConfig(
            "consumer",
            str(consumer_path),
            {"acetate": "EX_consumer_feed"},
            "BIOMASS_consumer",
        ),
    )
    organisms = load_configured_organisms(configurations)
    state = ReservoirCellState(
        solutes={"glucose": 0.0001, "acetate": 0.0},
        mobile_biomass={"producer": 0.03, "consumer": 0.04},
        attached_biomass={"producer": 0.02, "consumer": 0.01},
    )

    result = solve_reservoir_cell(configurations, organisms, state, dt_hours=1.0)

    assert result.status == "optimal"
    assert result.state.solutes["glucose"] == pytest.approx(0.0, abs=1e-12)
    assert result.state.solutes["acetate"] == pytest.approx(0.0, abs=1e-12)
    assert result.solute_rates["glucose"] == pytest.approx(-0.0001)
    assert result.solution.organism_growth["producer"] > 0
    assert result.solution.organism_growth["consumer"] > 0
    assert result.state.mobile_biomass["producer"] > 0.03
    assert result.state.attached_biomass["producer"] > 0.02


def test_unmapped_substrate_cannot_enter_cell(tmp_path):
    model_path = tmp_path / "member.json"
    _write_member(model_path, "member", "unmapped_e")
    configurations = (
        SpeciesConfig("member", str(model_path), {}, "BIOMASS_member"),
    )
    organisms = load_configured_organisms(configurations)

    result = solve_reservoir_cell(
        configurations,
        organisms,
        ReservoirCellState({"unmapped": 1.0}, {"member": 0.1}),
        dt_hours=1.0,
    )

    assert result.status == "optimal"
    assert result.solution.objective_value == pytest.approx(0.0)


def test_cardinal_temperature_response_has_expected_limits():
    assert cardinal_temperature_fitness(20.0, 20.0, 35.0, 50.0) == 0.0
    assert cardinal_temperature_fitness(35.0, 20.0, 35.0, 50.0) == pytest.approx(1.0)
    assert cardinal_temperature_fitness(50.0, 20.0, 35.0, 50.0) == 0.0


def test_ammonia_response_respects_native_reaction_bounds(tmp_path):
    model_path = tmp_path / "methanogen.json"
    _write_member(model_path, "methanogen", "feed_e")
    configuration = SpeciesConfig(
        "methanogen",
        str(model_path),
        {"feed": "EX_methanogen_feed"},
        "BIOMASS_methanogen",
        ammonia_response=AmmoniaResponse(
            atp_maintenance="ATPM",
            hydrogenotrophic_mcr="MCR",
        ),
    )
    organism = load_configured_organisms((configuration,))[0]
    atp = cobra.Reaction("ATPM", lower_bound=2.0, upper_bound=4.0)
    mcr = cobra.Reaction("MCR", lower_bound=0.0, upper_bound=8.0)
    organism.model.add_reactions([atp, mcr])

    adjusted, free_ammonia_mM = _environment_adjusted_members(
        (configuration,),
        (organism,),
        ReservoirCellState(
            {"feed": 1.0},
            {"methanogen": 0.1},
            temperature_c=35.0,
            ph=9.0,
            total_ammonia_n=0.02,
        ),
    )

    adjusted_model = adjusted[0].model
    assert free_ammonia_mM > 0
    assert 2.0 < adjusted_model.reactions.ATPM.lower_bound <= 4.0
    assert 0.0 <= adjusted_model.reactions.MCR.upper_bound < 8.0

