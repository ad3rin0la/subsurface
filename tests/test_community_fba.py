"""Reference tests for SBML loading and joint community mass balance."""

import numpy as np
import pytest

cobra = pytest.importorskip("cobra")

from n2bio.fba.community import (
    LoadedOrganism,
    OrganismModelInfo,
    build_community_model,
    load_sbml_organism,
    simulate_dynamic_community,
    solve_community,
)


def _organism(model, organism_id, biomass_reaction):
    model.objective = biomass_reaction
    info = OrganismModelInfo(
        organism_id=organism_id,
        model_id=model.id,
        source_file="generated test model",
        objective_reactions=(biomass_reaction,),
        biomass_reaction=biomass_reaction,
        compartments=dict(model.compartments),
        exchange_reactions=tuple(reaction.id for reaction in model.exchanges),
        flux_units=None,
        fbc_bounds={reaction.id: reaction.bounds for reaction in model.reactions},
        gene_reaction_rules={},
        annotations={},
        validation_warnings=(),
    )
    return LoadedOrganism(organism_id, model, info)


def _producer():
    model = cobra.Model("producer")
    glucose_e = cobra.Metabolite("glc_e", compartment="e")
    precursor = cobra.Metabolite("precursor_c", compartment="c")
    energy = cobra.Metabolite("energy_c", compartment="c")
    acetate_e = cobra.Metabolite("ac_e", compartment="e")
    exchange = cobra.Reaction("EX_glc_e", lower_bound=-10, upper_bound=1000)
    exchange.add_metabolites({glucose_e: -1})
    uptake = cobra.Reaction("GLC_UPTAKE", lower_bound=0, upper_bound=1000)
    uptake.add_metabolites({glucose_e: -1, precursor: 1, energy: 1})
    biomass = cobra.Reaction("BIOMASS_PRODUCER", lower_bound=0, upper_bound=1000)
    biomass.add_metabolites({precursor: -1, energy: -1, acetate_e: 1})
    model.add_reactions([exchange, uptake, biomass])
    return _organism(model, "producer", biomass.id)


def _consumer(organism_id="consumer", substrate="ac_e"):
    model = cobra.Model(organism_id)
    substrate_e = cobra.Metabolite(substrate, compartment="e")
    precursor = cobra.Metabolite("precursor_c", compartment="c")
    energy = cobra.Metabolite("energy_c", compartment="c")
    exchange = cobra.Reaction(f"EX_{substrate}", lower_bound=-10, upper_bound=1000)
    exchange.add_metabolites({substrate_e: -1})
    uptake = cobra.Reaction("UPTAKE", lower_bound=0, upper_bound=1000)
    uptake.add_metabolites({substrate_e: -1, precursor: 1, energy: 1})
    biomass = cobra.Reaction("BIOMASS_CONSUMER", lower_bound=0, upper_bound=1000)
    biomass.add_metabolites({precursor: -1, energy: -1})
    model.add_reactions([exchange, uptake, biomass])
    return _organism(model, organism_id, biomass.id)


def test_sbml_round_trip_records_model_metadata(tmp_path):
    source = _producer().model
    source.reactions.BIOMASS_PRODUCER.gene_reaction_rule = "gene_a and gene_b"
    sbml = tmp_path / "producer.xml"
    cobra.io.write_sbml_model(source, str(sbml))

    loaded = load_sbml_organism(sbml, organism_id="p", validate=False)

    assert loaded.info.model_id == "producer"
    assert loaded.info.biomass_reaction == "BIOMASS_PRODUCER"
    assert "EX_glc_e" in loaded.info.exchange_reactions
    assert loaded.info.fbc_bounds["EX_glc_e"] == (-10.0, 1000.0)
    assert loaded.info.gene_reaction_rules["BIOMASS_PRODUCER"] == "gene_a and gene_b"


def test_obligate_cross_feeding_is_enforced_by_shared_pool():
    community = build_community_model(
        [_producer(), _consumer()],
        metabolite_mapping={"glc_e": "glucose", "ac_e": "acetate"},
        medium={"glucose": (-1, 1000), "acetate": (0, 1000)},
    )
    solution = solve_community(community)

    assert solution.status == "optimal"
    assert solution.organism_growth["producer"] == pytest.approx(1.0)
    assert solution.organism_growth["consumer"] == pytest.approx(1.0)
    assert solution.shared_exchange_fluxes["acetate"] == pytest.approx(0.0, abs=1e-8)


def test_joint_objective_matches_direct_cobra_reference():
    community = build_community_model(
        [_producer(), _consumer()],
        metabolite_mapping={"glc_e": "glucose", "ac_e": "acetate"},
        medium={"glucose": (-1, 1000), "acetate": (0, 1000)},
    )
    observed = solve_community(community)

    reference = cobra.Model("reference_cross_feed")
    glucose = cobra.Metabolite("glucose_e", compartment="e")
    acetate = cobra.Metabolite("acetate_e", compartment="e")
    p1 = cobra.Metabolite("p1_c", compartment="p")
    p2 = cobra.Metabolite("p2_c", compartment="p")
    c1 = cobra.Metabolite("c1_c", compartment="c")
    c2 = cobra.Metabolite("c2_c", compartment="c")
    reactions = [
        cobra.Reaction("EX_glucose", lower_bound=-1, upper_bound=1000),
        cobra.Reaction("EX_acetate", lower_bound=0, upper_bound=1000),
        cobra.Reaction("P_UP", lower_bound=0, upper_bound=1000),
        cobra.Reaction("P_BIOMASS", lower_bound=0, upper_bound=1000),
        cobra.Reaction("C_UP", lower_bound=0, upper_bound=1000),
        cobra.Reaction("C_BIOMASS", lower_bound=0, upper_bound=1000),
    ]
    reactions[0].add_metabolites({glucose: -1})
    reactions[1].add_metabolites({acetate: -1})
    reactions[2].add_metabolites({glucose: -1, p1: 1, p2: 1})
    reactions[3].add_metabolites({p1: -1, p2: -1, acetate: 1})
    reactions[4].add_metabolites({acetate: -1, c1: 1, c2: 1})
    reactions[5].add_metabolites({c1: -1, c2: -1})
    reference.add_reactions(reactions)
    reference.objective = {reactions[3]: 1.0, reactions[5]: 1.0}
    expected = reference.optimize()

    assert observed.objective_value == pytest.approx(expected.objective_value)
    assert observed.organism_growth["producer"] == pytest.approx(expected.fluxes.P_BIOMASS)
    assert observed.organism_growth["consumer"] == pytest.approx(expected.fluxes.C_BIOMASS)


def test_competitors_share_one_limiting_substrate():
    first = _consumer("first", substrate="glc_e")
    second = _consumer("second", substrate="glc_e")
    community = build_community_model(
        [first, second],
        metabolite_mapping={"glc_e": "glucose"},
        medium={"glucose": (-1, 1000)},
        minimum_growth={"first": 0.2, "second": 0.2},
    )
    solution = solve_community(community)

    assert solution.status == "optimal"
    assert sum(solution.organism_growth.values()) == pytest.approx(1.0)
    assert all(value >= 0.2 - 1e-8 for value in solution.organism_growth.values())


def test_every_shared_metabolite_has_zero_steady_state_residual():
    community = build_community_model(
        [_producer(), _consumer()],
        metabolite_mapping={"glc_e": "glucose", "ac_e": "acetate"},
        medium={"glucose": (-1, 1000), "acetate": (0, 1000)},
    )
    solution = solve_community(community)

    for metabolite_id in community.shared_metabolites.values():
        metabolite = community.model.metabolites.get_by_id(metabolite_id)
        residual = sum(
            reaction.metabolites[metabolite] * solution.fluxes[reaction.id]
            for reaction in metabolite.reactions
        )
        assert residual == pytest.approx(0.0, abs=1e-8)


def test_required_member_reports_infeasible_when_cross_feed_is_absent():
    community = build_community_model(
        [_consumer()],
        metabolite_mapping={"ac_e": "acetate"},
        medium={"acetate": (0, 1000)},
        minimum_growth={"consumer": 0.1},
    )
    solution = solve_community(community)

    assert solution.status == "infeasible"


def test_fixed_abundance_constrains_biomass_flux_ratios():
    first = _consumer("first", substrate="glc_e")
    second = _consumer("second", substrate="glc_e")
    community = build_community_model(
        [first, second],
        metabolite_mapping={"glc_e": "glucose"},
        medium={"glucose": (-1, 1000)},
        fixed_abundances={"first": 0.75, "second": 0.25},
    )
    solution = solve_community(community)

    assert solution.status == "optimal"
    assert solution.organism_growth["first"] / solution.organism_growth["second"] == pytest.approx(3.0)


def test_dynamic_fba_updates_finite_pool_without_overdraw():
    community = build_community_model(
        [_producer()],
        metabolite_mapping={"glc_e": "glucose", "ac_e": "acetate"},
        medium={"glucose": (-1, 1000), "acetate": (0, 1000)},
        fixed_abundances={"producer": 1.0},
    )
    result = simulate_dynamic_community(
        community,
        initial_pools={"glucose": 0.2, "acetate": 0.0},
        initial_biomass={"producer": 0.1},
        t_end=2.0,
        dt=0.25,
    )

    assert np.all(result.pools["glucose"] >= 0.0)
    assert result.pools["glucose"][-1] < result.pools["glucose"][0]
    assert result.pools["acetate"][-1] > 0.0
    assert result.biomass["producer"][-1] > result.biomass["producer"][0]
