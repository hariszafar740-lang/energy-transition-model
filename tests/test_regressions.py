"""Regression tests for each defect reproduced in the original ``main.py``.

Original behaviour (commit 8e9e0e6) and its root cause:

1. Renewable availability was static: no ``p_max_pu`` time series was assigned, so
   Solar_PV and Wind_Turbine were available at 100 % in every hour (night included).
2. Gas served the entire 150 MW demand. Root cause: the objective combined
   annualised capital costs with a single 24 h operating period (snapshot weight 1),
   so operating costs were under-counted by a factor of 365 relative to capital.
3. The reported "150 MW / 3600 MWh" was not labelled. 150 MW is the instantaneous
   demand (and the gas capacity); 3600 MWh is the daily energy. Both are
   consistent with a constant 150 MW load over 24 h.
4. ``network.optimize`` return values were ignored, so an infeasible or failed
   solve would not be reported.
5. No tests were collected by pytest (exit code 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_model import DEFAULT_TECHNOLOGIES, SolverError, build_network, run_scenario, solve
from energy_model.model import RENEWABLE_CARRIER

from .conftest import make_profiles, technologies

ORIGINAL_OBJECTIVE_24H = 150.0 * 20000.0 + 3600.0 * 65.0  # 3,234,000


def _original_style_technologies():
    """Original main.py inputs: renewables available at 100 % every hour."""
    techs = technologies("Solar_PV", "Wind_Turbine", "Gas_Peaker")
    techs["Solar_PV"]["profile"] = None
    techs["Solar_PV"]["carrier"] = RENEWABLE_CARRIER
    techs["Wind_Turbine"]["profile"] = None
    techs["Wind_Turbine"]["carrier"] = RENEWABLE_CARRIER
    return techs


# --- Defect 1: static renewable availability ---------------------------------


def test_regression_renewable_availability_is_not_static(shipped_profiles):
    network = build_network(shipped_profiles)
    solar = network.generators_t.p_max_pu["Solar_PV"]
    wind = network.generators_t.p_max_pu["Wind_Turbine"]
    assert solar.nunique() > 1 and wind.nunique() > 1
    assert solar.iloc[0] == 0.0, "no solar output at midnight"
    assert np.allclose(solar.to_numpy(), shipped_profiles["solar_cf"].to_numpy())
    assert np.allclose(wind.to_numpy(), shipped_profiles["wind_cf"].to_numpy())


def test_regression_changing_solar_profile_changes_solution():
    flat = make_profiles(solar=0.5)
    shifted = make_profiles(solar=[0.0] * 6 + [0.5] * 12 + [0.0] * 6)
    techs = technologies("Solar_PV", "Gas_Peaker")
    a = run_scenario(flat, techs)
    b = run_scenario(shifted, techs)
    assert not np.allclose(a.dispatch_mw["Gas_Peaker"], b.dispatch_mw["Gas_Peaker"])
    assert b.dispatch_mw["Gas_Peaker"].iloc[0] == pytest.approx(150.0, abs=1e-5)


# --- Defect 2: gas-only dispatch caused by horizon mismatch ------------------


def test_regression_original_gas_only_result_reproduced_with_one_day_weighting():
    """Original inputs with a one-day objective reproduce the reported gas-only result."""
    result = run_scenario(
        make_profiles(load=150.0),
        _original_style_technologies(),
        days_per_year=1,
    )
    assert result.p_nom_mw["Gas_Peaker"] == pytest.approx(150.0, abs=1e-4)
    assert result.energy_mwh_per_day["Gas_Peaker"] == pytest.approx(3600.0, abs=1e-3)
    assert result.total_cost_per_year == pytest.approx(ORIGINAL_OBJECTIVE_24H, rel=1e-7)


def test_regression_annualised_horizon_removes_gas_only_artifact():
    """Same inputs, annual horizon: renewables are cheaper than gas and are built."""
    result = run_scenario(
        make_profiles(load=150.0),
        _original_style_technologies(),
        days_per_year=365,
    )
    assert result.p_nom_mw["Solar_PV"] == pytest.approx(150.0, abs=1e-4)
    assert result.p_nom_mw["Gas_Peaker"] == pytest.approx(0.0, abs=1e-4)
    assert result.energy_mwh_per_day["Gas_Peaker"] == pytest.approx(0.0, abs=1e-4)


def test_regression_shipped_scenario_is_not_gas_only(shipped_profiles):
    result = run_scenario(shipped_profiles)
    assert result.energy_mwh_per_day["Gas_Peaker"] < result.demand_mwh_per_day
    renewable_energy = result.energy_mwh_per_day[["Solar_PV", "Wind_Turbine"]].sum()
    assert renewable_energy > 0.0


# --- Defect 3: meaning of 150 MW / 3600 MWh -----------------------------------


def test_regression_150mw_is_power_and_capacity_3600mwh_is_daily_energy():
    techs = technologies("Gas_Peaker")
    result = run_scenario(make_profiles(load=150.0), techs)

    # 150 MW: instantaneous demand, and the installed gas capacity that serves it.
    assert result.peak_demand_mw == pytest.approx(150.0)
    assert result.p_nom_mw["Gas_Peaker"] == pytest.approx(150.0, abs=1e-4)
    assert np.allclose(result.dispatch_mw["Gas_Peaker"], 150.0, atol=1e-5)

    # 3600 MWh: energy of one representative day = 150 MW x 24 h x 1 h/snapshot.
    assert result.demand_mwh_per_day == pytest.approx(3600.0)
    assert result.energy_mwh_per_day["Gas_Peaker"] == pytest.approx(3600.0, abs=1e-3)
    assert result.n_snapshots == 24 and result.snapshot_hours == 1.0


def test_regression_150mw_3600mwh_annual_and_cost_by_hand():
    result = run_scenario(make_profiles(load=150.0), technologies("Gas_Peaker"))
    capex = 20000.0 * 150.0  # currency / MW / yr x MW
    opex = 65.0 * 3600.0 * 365.0  # currency / MWh x MWh/day x days/yr
    assert result.energy_mwh_per_year["Gas_Peaker"] == pytest.approx(3600.0 * 365.0)
    assert result.capex_per_year == pytest.approx(capex)
    assert result.opex_per_year == pytest.approx(opex, rel=1e-9)
    assert result.total_cost_per_year == pytest.approx(capex + opex, rel=1e-9)


# --- Defect 4: solver status not checked --------------------------------------


def test_regression_infeasible_problem_raises_instead_of_reporting_dispatch():
    # No thermal capacity, no renewables, and a 150 MW load: no feasible dispatch.
    techs = technologies(
        "Gas_Peaker",
        Gas_Peaker={"p_nom_extendable": False, "p_nom": 0.0},
    )
    with pytest.raises(SolverError, match="infeasible"):
        run_scenario(make_profiles(load=150.0), techs)


def test_regression_non_optimal_status_is_not_accepted(monkeypatch):
    network = build_network(make_profiles(load=150.0), technologies("Gas_Peaker"))
    monkeypatch.setattr(network, "optimize", lambda **kwargs: ("warning", "infeasible"))
    with pytest.raises(SolverError):
        solve(network)


def test_regression_missing_objective_is_not_accepted(monkeypatch):
    network = build_network(make_profiles(load=150.0), technologies("Gas_Peaker"))
    # Report optimal status without solving, so no objective value exists.
    monkeypatch.setattr(network, "optimize", lambda **kwargs: ("ok", "optimal"))
    with pytest.raises(SolverError, match="no finite objective"):
        solve(network)


# --- Defaults remain documented --------------------------------------------------


def test_default_technology_inputs_match_original_main_py():
    assert DEFAULT_TECHNOLOGIES["Solar_PV"]["capital_cost"] == 45000
    assert DEFAULT_TECHNOLOGIES["Wind_Turbine"]["capital_cost"] == 70000
    assert DEFAULT_TECHNOLOGIES["Gas_Peaker"]["capital_cost"] == 20000
    assert DEFAULT_TECHNOLOGIES["Gas_Peaker"]["marginal_cost"] == 65
    assert all(DEFAULT_TECHNOLOGIES[t]["marginal_cost"] == 0 for t in ("Solar_PV", "Wind_Turbine"))


def test_regression_no_storage_component_is_modelled(shipped_profiles):
    """Storage is not part of the model; storage validation is therefore not applicable."""
    network = build_network(shipped_profiles)
    assert len(network.storage_units) == 0
    assert len(network.stores) == 0
    assert len(network.links) == 0
    assert isinstance(network.generators, pd.DataFrame)
