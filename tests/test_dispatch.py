"""Dispatch behaviour of the production model: power balance, limits, curtailment.

Each test builds a scenario, solves it with the production ``run_scenario`` /
``build_network`` path, and checks the solved time series against the declared
inputs. Nothing here re-implements the optimiser's equations.
"""

from __future__ import annotations

import numpy as np
import pytest

from energy_model import build_network, run_scenario, solve

from .conftest import HOURS, POWER_TOL_MW, make_profiles, technologies


def _solve(profiles, techs, days_per_year=365):
    return run_scenario(profiles=profiles, technologies=techs, days_per_year=days_per_year)


# --- Power balance and limits on the shipped time series -----------------------


def test_power_balance_holds_at_every_hour_shipped_profiles(shipped_profiles):
    result = run_scenario(shipped_profiles)
    balance = result.dispatch_mw.sum(axis=1) - shipped_profiles["load_mw"]
    assert len(result.dispatch_mw) == 24
    assert np.abs(balance.to_numpy()).max() < POWER_TOL_MW
    assert result.power_balance_max_residual_mw < POWER_TOL_MW


def test_dispatch_within_availability_and_capacity_every_hour(shipped_profiles):
    result = run_scenario(shipped_profiles)
    p = result.dispatch_mw
    assert (p >= -POWER_TOL_MW).all().all(), "negative generation"
    assert result.capacity_violation_max_mw < POWER_TOL_MW
    for tech, profile in [("Solar_PV", "solar_cf"), ("Wind_Turbine", "wind_cf")]:
        bound = shipped_profiles[profile].to_numpy() * result.p_nom_mw[tech]
        assert (p[tech].to_numpy() <= bound + POWER_TOL_MW).all(), tech


def test_no_negative_or_nonfinite_output(shipped_profiles):
    result = run_scenario(shipped_profiles)
    values = result.dispatch_mw.to_numpy()
    assert np.isfinite(values).all()
    assert (values > -POWER_TOL_MW).all()


# --- Renewable-rich, low-renewable and thermal-necessary scenarios -------------


def test_renewable_rich_scenario_uses_solar_and_no_gas():
    # Solar available at full capacity in every hour; gas is available but is
    # the more expensive option over the year.
    profiles = make_profiles(load=150.0, solar=1.0)
    techs = technologies("Solar_PV", "Gas_Peaker")
    result = _solve(profiles, techs)
    assert result.p_nom_mw["Solar_PV"] == pytest.approx(150.0, abs=1e-4)
    assert result.p_nom_mw["Gas_Peaker"] == pytest.approx(0.0, abs=1e-4)
    assert result.energy_mwh_per_day["Gas_Peaker"] == pytest.approx(0.0, abs=1e-4)
    assert result.energy_mwh_per_day["Solar_PV"] == pytest.approx(3600.0, abs=1e-3)
    assert result.curtailment_mwh_per_day["Solar_PV"] == pytest.approx(0.0, abs=1e-3)


def test_low_renewable_scenario_forces_gas_to_serve_full_load():
    # No renewable resource at all: the only feasible supply is thermal.
    profiles = make_profiles(load=150.0, solar=0.0, wind=0.0)
    techs = technologies("Solar_PV", "Wind_Turbine", "Gas_Peaker")
    result = _solve(profiles, techs)
    assert result.p_nom_mw["Gas_Peaker"] == pytest.approx(150.0, abs=1e-4)
    assert result.p_nom_mw["Solar_PV"] == pytest.approx(0.0, abs=1e-4)
    assert result.p_nom_mw["Wind_Turbine"] == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(result.dispatch_mw["Gas_Peaker"], 150.0, atol=POWER_TOL_MW)


def test_thermal_is_necessary_at_night_when_solar_is_zero():
    solar = [0.0] * 6 + [0.8] * 12 + [0.0] * 6  # daylight 06:00-17:59 only
    profiles = make_profiles(load=150.0, solar=solar, wind=0.0)
    techs = technologies(
        "Solar_PV", "Gas_Peaker", Solar_PV={"p_nom": 150.0, "p_nom_extendable": False}
    )
    result = _solve(profiles, techs)
    gas = result.dispatch_mw["Gas_Peaker"].to_numpy()
    night = np.array(solar) == 0.0
    assert np.allclose(gas[night], 150.0, atol=POWER_TOL_MW), "gas must cover night hours"
    assert (gas[~night] < 150.0 - 1e-3).all(), "solar must displace gas in daylight"
    # 12 night hours at 150 MW plus 12 daylight hours at 150 - 0.8 * 150 = 30 MW.
    assert result.energy_mwh_per_day["Gas_Peaker"] == pytest.approx(
        150.0 * 12 + 30.0 * 12, abs=1e-3
    )
    assert np.allclose(gas[~night], 30.0, atol=POWER_TOL_MW)


def test_solar_output_never_exceeds_hourly_availability():
    solar = [0.0] * 8 + [0.2, 0.9, 0.5] + [0.0] * 13
    profiles = make_profiles(load=150.0, solar=solar)
    techs = technologies(
        "Solar_PV", "Gas_Peaker", Solar_PV={"p_nom": 400.0, "p_nom_extendable": False}
    )
    result = _solve(profiles, techs)
    solar_dispatch = result.dispatch_mw["Solar_PV"].to_numpy()
    cap = 400.0 * np.array(solar)
    assert (solar_dispatch <= cap + POWER_TOL_MW).all()
    # Hours with zero availability must have zero solar output.
    assert np.allclose(solar_dispatch[np.array(solar) == 0.0], 0.0, atol=POWER_TOL_MW)


# --- Curtailment --------------------------------------------------------------


def test_curtailment_equals_available_minus_dispatched_energy():
    solar = [0.0] * 6 + [1.0] * 12 + [0.0] * 6
    profiles = make_profiles(load=150.0, solar=solar)
    techs = technologies(
        "Solar_PV", "Gas_Peaker", Solar_PV={"p_nom": 300.0, "p_nom_extendable": False}
    )
    result = _solve(profiles, techs)

    # Available solar is 300 MW in 12 daylight hours; the load is 150 MW, so
    # 150 MW must be curtailed each daylight hour and gas is not needed.
    assert np.allclose(result.dispatch_mw["Solar_PV"].iloc[6:18], 150.0, atol=POWER_TOL_MW)
    assert np.allclose(result.dispatch_mw["Gas_Peaker"].iloc[6:18], 0.0, atol=POWER_TOL_MW)
    assert result.curtailment_mwh_per_day["Solar_PV"] == pytest.approx(150.0 * 12, abs=1e-3)


def test_curtailment_is_non_negative_with_shipped_profiles(shipped_profiles):
    # With the shipped profiles the solver may size renewables above demand; the
    # surplus is curtailed. Curtailment must never be negative.
    result = run_scenario(shipped_profiles)
    assert (result.curtailment_mwh_per_day >= -1e-3).all()
    assert result.curtailment_mwh_per_day["Wind_Turbine"] >= 0.0


# --- Time-series alignment ----------------------------------------------------


def test_solar_profile_is_aligned_to_timestamps():
    # Single sunny hour at 05:00: a fixed 100 MW solar plant can only produce
    # in that hour, and must do so at full availability.
    solar = np.zeros(24)
    solar[5] = 1.0
    profiles = make_profiles(load=150.0, solar=solar)
    techs = technologies(
        "Solar_PV", "Gas_Peaker", Solar_PV={"p_nom": 100.0, "p_nom_extendable": False}
    )
    result = _solve(profiles, techs)
    solar_dispatch = result.dispatch_mw["Solar_PV"]
    assert solar_dispatch.iloc[5] == pytest.approx(100.0, abs=POWER_TOL_MW)
    assert solar_dispatch.drop(solar_dispatch.index[5]).abs().max() < POWER_TOL_MW
    assert solar_dispatch.index[5] == HOURS[5]


def test_time_varying_load_is_met_hour_by_hour():
    load = np.linspace(100.0, 200.0, 24)
    profiles = make_profiles(load=load)
    result = _solve(profiles, technologies("Gas_Peaker"))
    assert np.allclose(result.dispatch_mw["Gas_Peaker"].to_numpy(), load, atol=POWER_TOL_MW)
    assert result.demand_mwh_per_day == pytest.approx(load.sum(), abs=1e-6)
    assert result.peak_demand_mw == pytest.approx(200.0)
    assert result.p_nom_mw["Gas_Peaker"] == pytest.approx(200.0, abs=1e-4)


# --- Invalid scenario inputs --------------------------------------------------


def test_unknown_profile_reference_is_rejected():
    techs = technologies("Gas_Peaker")
    techs["Hydro"] = {"capital_cost": 1.0, "marginal_cost": 0.0, "profile": "hydro_cf"}
    with pytest.raises(ValueError, match="unknown profile"):
        build_network(make_profiles(), technologies=techs)


def test_technology_missing_cost_is_rejected():
    techs = {"Gas_Peaker": {"marginal_cost": 65.0}}
    with pytest.raises(ValueError, match="missing keys"):
        build_network(make_profiles(), technologies=techs)


def test_empty_technology_set_is_rejected():
    with pytest.raises(ValueError, match="at least one technology"):
        build_network(make_profiles(), technologies={})


def test_solve_returns_status_for_valid_problem():
    network = build_network(make_profiles(), technologies("Gas_Peaker"))
    status, condition = solve(network)
    assert (status, condition) == ("ok", "optimal")


# --- The verification metrics themselves must detect violations -----------------


def test_verification_metrics_detect_injected_violations():
    from energy_model.model import extract_results

    profiles = make_profiles(load=150.0, solar=[0.0] * 6 + [0.5] * 12 + [0.0] * 6)
    techs = technologies(
        "Solar_PV", "Gas_Peaker", Solar_PV={"p_nom": 100.0, "p_nom_extendable": False}
    )
    network = build_network(profiles, technologies=techs)
    status, condition = solve(network)

    clean = extract_results(network, status=status, termination_condition=condition)
    assert clean.capacity_violation_max_mw < POWER_TOL_MW
    assert clean.power_balance_max_residual_mw < POWER_TOL_MW

    # Inject an infeasible solution: solar exceeds its availability at 12:00 ...
    network.generators_t.p.loc[HOURS[12], "Solar_PV"] = 80.0  # available is 50 MW
    violated = extract_results(network, status=status, termination_condition=condition)
    assert violated.capacity_violation_max_mw == pytest.approx(30.0, abs=1e-6)

    # ... and one where generation does not match load at 03:00.
    network.generators_t.p.loc[HOURS[3], "Gas_Peaker"] = 100.0  # load is 150 MW
    unbalanced = extract_results(network, status=status, termination_condition=condition)
    assert unbalanced.power_balance_max_residual_mw == pytest.approx(50.0, abs=1e-6)
