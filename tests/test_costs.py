"""Cost accounting, annualisation and an independent LP cross-check.

Expected values are computed here by hand from the declared inputs (costs in
currency/MW/yr for capacity and currency/MWh for energy; one hour per snapshot;
DAYS_PER_YEAR representative days), not by reading the model's own outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from energy_model import DEFAULT_TECHNOLOGIES, run_scenario

from .conftest import COST_REL_TOL, make_profiles, technologies

DAYS = 365


def test_capex_and_opex_by_hand_for_renewable_rich_case():
    result = run_scenario(
        make_profiles(load=150.0, solar=1.0),
        technologies("Solar_PV", "Gas_Peaker"),
        days_per_year=DAYS,
    )
    # 150 MW solar at 45 000/MW/yr; no fuel, no energy cost.
    assert result.capex_per_year == pytest.approx(45000.0 * 150.0, rel=COST_REL_TOL)
    assert result.opex_per_year == pytest.approx(0.0, abs=1e-6)
    assert result.total_cost_per_year == pytest.approx(6_750_000.0, rel=COST_REL_TOL)
    assert result.objective_difference == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("days", [1, 7, 365])
def test_operating_cost_scales_linearly_with_days_capex_does_not(days):
    result = run_scenario(make_profiles(load=150.0), technologies("Gas_Peaker"), days_per_year=days)
    daily_fuel_cost = 65.0 * 3600.0  # currency per representative day
    assert result.capex_per_year == pytest.approx(20000.0 * 150.0, rel=COST_REL_TOL)
    assert result.opex_per_year == pytest.approx(daily_fuel_cost * days, rel=COST_REL_TOL)
    assert result.total_cost_per_year == pytest.approx(
        20000.0 * 150.0 + daily_fuel_cost * days, rel=COST_REL_TOL
    )
    assert result.energy_mwh_per_year["Gas_Peaker"] == pytest.approx(
        3600.0 * days, rel=COST_REL_TOL
    )


def test_energy_per_year_is_days_times_energy_per_day(shipped_profiles):
    result = run_scenario(shipped_profiles, days_per_year=DAYS)
    ratio = result.energy_mwh_per_year / result.energy_mwh_per_day
    assert np.allclose(ratio.to_numpy(), DAYS, rtol=1e-12)


def test_mixed_scenario_cost_matches_manual_recalculation(shipped_profiles):
    result = run_scenario(shipped_profiles, days_per_year=DAYS)
    capital = {name: DEFAULT_TECHNOLOGIES[name]["capital_cost"] for name in result.p_nom_mw.index}
    marginal = {name: DEFAULT_TECHNOLOGIES[name]["marginal_cost"] for name in result.p_nom_mw.index}

    capex = sum(capital[g] * result.p_nom_mw[g] for g in capital)
    daily_energy = result.dispatch_mw.sum(axis=0) * 1.0  # MWh per representative day
    opex = DAYS * sum(marginal[g] * daily_energy[g] for g in marginal)

    assert result.capex_per_year == pytest.approx(capex, rel=COST_REL_TOL)
    assert result.opex_per_year == pytest.approx(opex, rel=COST_REL_TOL)
    assert result.total_cost_per_year == pytest.approx(capex + opex, rel=COST_REL_TOL)


def test_reported_objective_equals_independent_annual_cost(shipped_profiles):
    result = run_scenario(shipped_profiles, days_per_year=DAYS)
    assert result.objective_reported == pytest.approx(result.total_cost_per_year, rel=1e-6)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "365", None])
def test_invalid_days_per_year_rejected(bad):
    with pytest.raises(ValueError, match="days_per_year"):
        run_scenario(make_profiles(), technologies("Gas_Peaker"), days_per_year=bad)


def _independent_lp_objective(profiles, techs, days):
    """Hand-built LP for the same model, solved with scipy's HiGHS interface.

    Variables x = [p_{g,t} (g-major, t-minor), p_nom_g]. Written from the model
    equations directly, independently of PyPSA/linopy.
    """
    names = list(techs)
    G, T = len(names), len(profiles)
    load = profiles["load_mw"].to_numpy()
    cf = {}
    for g in names:
        profile = techs[g].get("profile")
        cf[g] = np.ones(T) if profile is None else profiles[profile].to_numpy()
    n_var = G * T + G

    c = np.zeros(n_var)
    for i, g in enumerate(names):
        c[i * T : (i + 1) * T] = days * techs[g]["marginal_cost"]  # 1 h per snapshot
        c[G * T + i] = techs[g]["capital_cost"]

    A_eq = np.zeros((T, n_var))
    for t in range(T):
        for i in range(G):
            A_eq[t, i * T + t] = 1.0
    b_eq = load

    rows = []
    for i, g in enumerate(names):
        for t in range(T):
            row = np.zeros(n_var)
            row[i * T + t] = 1.0
            row[G * T + i] = -cf[g][t]  # p_gt - cf_gt * p_nom_g <= 0
            rows.append(row)
    A_ub = np.array(rows)
    b_ub = np.zeros(len(rows))

    bounds = [(0, None)] * n_var
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return res.fun


@pytest.mark.parametrize(
    "solar, wind, days",
    [
        (None, None, 365),  # shipped profiles, annual horizon
        (np.zeros(24), np.zeros(24), 365),  # no renewable resource
        (np.ones(24), np.zeros(24), 1),  # renewable-rich, one-day horizon
    ],
)
def test_production_model_matches_independent_lp(shipped_profiles, solar, wind, days):
    if solar is None:
        profiles = shipped_profiles
    else:
        profiles = make_profiles(load=150.0, solar=solar, wind=wind)
    techs = technologies("Solar_PV", "Wind_Turbine", "Gas_Peaker")
    expected = _independent_lp_objective(profiles, techs, days)
    result = run_scenario(profiles, techs, days_per_year=days)
    assert result.objective_reported == pytest.approx(expected, rel=1e-6)
    assert result.total_cost_per_year == pytest.approx(expected, rel=1e-6)
