"""CO2 cap-tax Lagrangian duality, multi-fuel emissions & MACC tests (v0.5.0)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from energy_model import (
    DEFAULT_TECHNOLOGIES,
    analytical_fuel_switching_carbon_price,
    baseline_emission_specs,
    default_multifuel_specs,
    format_emissions_report,
    format_macc_report,
    make_emissions_week_profiles,
    plot_co2_pareto_frontier_and_macc,
    run_emissions_scenario,
    solve_emissions_linprog_crosscheck,
    sweep_carbon_tax_equivalents,
    sweep_co2_pareto_and_macc,
    verify_cap_tax_lagrangian_duality,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def week_profiles():
    return make_emissions_week_profiles()


@pytest.fixture(scope="module")
def macc_table(week_profiles):
    return sweep_co2_pareto_and_macc(week_profiles)


@pytest.fixture(scope="module")
def tax_table(week_profiles):
    return sweep_carbon_tax_equivalents(week_profiles)


class TestEmissionSpecsAndAnalyticalSwitching:
    def test_specific_electric_emissions_factors_exact(self):
        specs = default_multifuel_specs()
        assert specs["Coal_Steam"].specific_emissions_tco2_per_mwh_e == pytest.approx(
            0.340 / 0.40, rel=1e-12
        )
        assert specs["CCGT_Gas"].specific_emissions_tco2_per_mwh_e == pytest.approx(
            0.202 / 0.58, rel=1e-12
        )
        assert specs["OCGT_Peaker"].specific_emissions_tco2_per_mwh_e == pytest.approx(
            0.202 / 0.38, rel=1e-12
        )
        assert specs["Gas_CCS"].specific_emissions_tco2_per_mwh_e == pytest.approx(
            0.0202 / 0.50, rel=1e-12
        )
        assert specs["Solar_PV"].specific_emissions_tco2_per_mwh_e == 0.0
        assert specs["Wind_Turbine"].specific_emissions_tco2_per_mwh_e == 0.0

    def test_analytical_short_run_and_long_run_fuel_switching_prices(self):
        specs = default_multifuel_specs()
        sr_coal_ccgt = analytical_fuel_switching_carbon_price(
            specs["Coal_Steam"], specs["CCGT_Gas"]
        )
        expected_sr = (35.0 - 16.0) / (0.340 / 0.40 - 0.202 / 0.58)
        assert sr_coal_ccgt == pytest.approx(expected_sr, rel=1e-12)

        lr_coal_ccgt = analytical_fuel_switching_carbon_price(
            specs["Coal_Steam"], specs["CCGT_Gas"], capacity_factor=0.75
        )
        expected_lr = ((40_000.0 - 62_000.0) / (8760.0 * 0.75) + (35.0 - 16.0)) / (
            0.340 / 0.40 - 0.202 / 0.58
        )
        assert lr_coal_ccgt == pytest.approx(expected_lr, rel=1e-12)
        assert lr_coal_ccgt < sr_coal_ccgt

        lr_ccgt_ccs = analytical_fuel_switching_carbon_price(
            specs["CCGT_Gas"], specs["Gas_CCS"], capacity_factor=0.55
        )
        assert lr_ccgt_ccs > lr_coal_ccgt

    @pytest.mark.parametrize("bad_cf", [0.0, -0.25, 1.05, True, float("nan")])
    def test_switching_price_rejects_invalid_capacity_factor(self, bad_cf):
        specs = default_multifuel_specs()
        with pytest.raises(ValueError):
            analytical_fuel_switching_carbon_price(
                specs["Coal_Steam"], specs["CCGT_Gas"], capacity_factor=bad_cf
            )

    def test_switching_price_rejects_non_decreasing_emissions_pair(self):
        specs = default_multifuel_specs()
        with pytest.raises(ValueError, match="strictly higher"):
            analytical_fuel_switching_carbon_price(specs["CCGT_Gas"], specs["Coal_Steam"])

    def test_baseline_emission_specs_preserves_default_technologies(self):
        mapped = baseline_emission_specs()
        assert set(mapped) == set(DEFAULT_TECHNOLOGIES)
        for name, raw in DEFAULT_TECHNOLOGIES.items():
            assert mapped[name].capital_cost_per_mw_yr == pytest.approx(raw["capital_cost"])
            assert mapped[name].base_marginal_cost_per_mwh_e == pytest.approx(raw["marginal_cost"])
        assert mapped["Gas_Peaker"].specific_emissions_tco2_per_mwh_e > 0.5
        assert mapped["Solar_PV"].specific_emissions_tco2_per_mwh_e == 0.0
        assert mapped["Wind_Turbine"].specific_emissions_tco2_per_mwh_e == 0.0

    def test_week_profiles_determinism_and_integrity(self, week_profiles):
        again = make_emissions_week_profiles()
        assert len(week_profiles) == 168
        assert week_profiles.equals(again)
        assert float(week_profiles["load_mw"].min()) > 80.0
        s_min = float(week_profiles["solar_cf"].min())
        s_max = float(week_profiles["solar_cf"].max())
        w_min = float(week_profiles["wind_cf"].min())
        w_max = float(week_profiles["wind_cf"].max())
        assert 0.0 <= s_min <= s_max <= 1.0
        assert 0.0 < w_min < w_max <= 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"co2_cap_tco2_per_year": -1.0},
            {"co2_price_per_tco2": -5.0},
            {"co2_price_per_tco2": True},
            {"co2_cap_tco2_per_year": float("nan")},
            {"annual_hours": 0.0},
            {"specs": {}},
        ],
    )
    def test_run_emissions_scenario_parameter_validation(self, shipped_profiles, kwargs):
        with pytest.raises(ValueError):
            run_emissions_scenario(shipped_profiles, include_storage=False, **kwargs)


class TestCapTaxLagrangianDuality:
    @pytest.mark.parametrize("tau", [18.0, 45.0])
    def test_tax_to_cap_primal_and_saddle_point_equivalence(self, week_profiles, tau: float):
        cmp_res = verify_cap_tax_lagrangian_duality(week_profiles, co2_price_per_tco2=tau)
        assert cmp_res.direction == "tax_to_cap"
        assert cmp_res.rel_saddle_gap < 1e-12
        assert cmp_res.emissions_diff_tco2 < 1e-6
        assert cmp_res.max_p_nom_diff_mw < 1e-6
        assert cmp_res.cap_resource_cost_per_year == pytest.approx(
            cmp_res.tax_resource_cost_per_year, rel=1e-10
        )

    @pytest.mark.parametrize("cap_tco2", [600_000.0, 250_000.0, 0.0])
    def test_cap_to_tax_saddle_point_and_nodal_lmp_equivalence(
        self, week_profiles, cap_tco2: float
    ):
        cmp_res = verify_cap_tax_lagrangian_duality(week_profiles, co2_cap_tco2_per_year=cap_tco2)
        assert cmp_res.direction == "cap_to_tax"
        assert cmp_res.rel_saddle_gap < 1e-12
        assert cmp_res.max_lmp_diff_per_mwh < 1e-8

    def test_net_zero_boundary_subdifferential_regularisation(self, week_profiles):
        zero_cap = run_emissions_scenario(week_profiles, co2_cap_tco2_per_year=0.0)
        assert zero_cap.annual_emissions_tco2 == pytest.approx(0.0, abs=1e-7)
        assert zero_cap.shadow_co2_price_per_tco2 == pytest.approx(418.254645, rel=1e-4)
        assert zero_cap.power_balance_max_residual_mw < 1e-9

        above_backstop = run_emissions_scenario(week_profiles, co2_price_per_tco2=420.0)
        assert above_backstop.annual_emissions_tco2 == pytest.approx(0.0, abs=1e-7)
        assert above_backstop.total_resource_cost_per_year == pytest.approx(
            zero_cap.total_resource_cost_per_year, rel=1e-10
        )

    def test_duality_helper_requires_exactly_one_policy_input(self, shipped_profiles):
        with pytest.raises(ValueError, match="exactly one"):
            verify_cap_tax_lagrangian_duality(shipped_profiles)
        with pytest.raises(ValueError, match="exactly one"):
            verify_cap_tax_lagrangian_duality(
                shipped_profiles, co2_cap_tco2_per_year=10_000.0, co2_price_per_tco2=25.0
            )


class TestParetoFrontierAndMACC:
    def test_pareto_convexity_and_macc_monotonicity(self, macc_table):
        e_vals = macc_table["emissions_tco2"].to_numpy(dtype=float)
        c_vals = macc_table["total_resource_cost_per_year"].to_numpy(dtype=float)
        mac_vals = macc_table["marginal_abatement_cost_per_tco2"].to_numpy(dtype=float)
        avg_vals = macc_table["average_abatement_cost_per_tco2"].to_numpy(dtype=float)

        assert e_vals[0] == pytest.approx(1_206_958.07, rel=1e-4)
        assert e_vals[-1] == pytest.approx(0.0, abs=1e-7)
        assert np.all(np.diff(e_vals) < 0.0)
        assert np.all(np.diff(c_vals) > 0.0)
        assert np.all(np.diff(mac_vals) >= -1e-6)
        assert np.all(np.diff(avg_vals) >= -1e-6)
        assert np.all(mac_vals + 1e-6 >= avg_vals)
        assert float(macc_table["lagrangian_saddle_gap"].max()) < 1e-4

    def test_four_stage_fuel_and_storage_transition(self, macc_table):
        # BAU dominated by Coal_Steam
        assert float(macc_table.iloc[0]["coal_share_pct"]) > 90.0
        # Mid-transition (75-92%): CCGT_Gas expands strongly while coal declines
        mid = macc_table.loc[macc_table["abatement_target_pct"] == 85.0].iloc[0]
        assert float(mid["ccgt_mw"]) > 60.0
        assert float(mid["coal_mw"]) < float(macc_table.iloc[0]["coal_mw"]) * 0.25
        # Deep transition (96-99.5%): coal phased out to 0 MW and 90% Gas_CCS enters
        ccs_row = macc_table.loc[macc_table["abatement_target_pct"] == 98.5].iloc[0]
        assert float(ccs_row["coal_mw"]) == pytest.approx(0.0, abs=1e-5)
        assert float(ccs_row["gas_ccs_mw"]) > 25.0
        # Net-Zero (100%): 100% VRE primary generation + Green H2 cavern > 3,500 MWh_H2
        nz_row = macc_table.iloc[-1]
        assert float(nz_row["vre_share_pct"]) == pytest.approx(100.0, abs=1e-5)
        assert float(nz_row["h2_mwh"]) > 3_500.0

    def test_carbon_tax_laffer_revenue_profile(self, tax_table):
        revs = tax_table["carbon_tax_revenue_per_year"].to_numpy(dtype=float)
        emissions = tax_table["emissions_tco2"].to_numpy(dtype=float)
        assert np.all(np.diff(emissions) <= 1e-5)
        assert revs[0] == pytest.approx(0.0, abs=1e-9)
        assert float(revs.max()) > 5.5e6
        assert revs[-1] == pytest.approx(0.0, abs=1e-6)


class TestSciPyEmissionsCrossCheck:
    def test_multifuel_storage_co2_cap_matches_scipy_linprog(self, week_profiles):
        chk = solve_emissions_linprog_crosscheck(
            week_profiles, co2_cap_tco2_per_year=300_000.0, co2_price_per_tco2=0.0
        )
        assert chk.rel_objective_diff < 1e-10
        assert chk.shadow_co2_price_diff < 1e-8
        assert chk.max_p_nom_diff_mw < 1e-8
        assert chk.max_storage_e_nom_diff_mwh < 1e-8
        assert chk.max_lmp_diff_per_mwh < 1e-8

    def test_multifuel_storage_carbon_tax_matches_scipy_linprog(self, week_profiles):
        chk = solve_emissions_linprog_crosscheck(
            week_profiles, co2_cap_tco2_per_year=None, co2_price_per_tco2=65.0
        )
        assert chk.rel_objective_diff < 1e-10
        assert chk.max_p_nom_diff_mw < 1e-8
        assert chk.max_storage_e_nom_diff_mwh < 1e-8
        assert chk.max_lmp_diff_per_mwh < 1e-8

    def test_shipped_24h_fixed_capacity_cap_matches_scipy_linprog(self, shipped_profiles):
        raw_techs = {
            "Solar_PV": {
                "capital_cost": 45_000.0,
                "marginal_cost": 0.0,
                "p_nom_extendable": False,
                "p_nom": 100.0,
                "profile": "solar_cf",
            },
            "Wind_Turbine": {
                "capital_cost": 70_000.0,
                "marginal_cost": 0.0,
                "p_nom_extendable": True,
                "profile": "wind_cf",
            },
            "Gas_Peaker": {
                "capital_cost": 20_000.0,
                "marginal_cost": 65.0,
                "p_nom_extendable": True,
                "profile": None,
            },
        }
        chk = solve_emissions_linprog_crosscheck(
            shipped_profiles,
            specs=baseline_emission_specs(raw_techs),
            co2_cap_tco2_per_year=15_000.0,
            include_storage=False,
        )
        assert chk.rel_objective_diff < 1e-10
        assert chk.shadow_co2_price_diff < 1e-8
        assert chk.max_p_nom_diff_mw < 1e-8
        assert chk.max_lmp_diff_per_mwh < 1e-8


class TestCLIAndMACCFigure:
    def test_committed_macc_figure_and_report_helpers(
        self, shipped_profiles, macc_table, tax_table, tmp_path
    ):
        committed = REPO_ROOT / "results" / "co2_pareto_frontier_and_macc.png"
        assert committed.is_file() and committed.stat().st_size > 120_000

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "results/co2_pareto_frontier_and_macc.png" in readme

        rep_macc = format_macc_report(macc_table)
        assert "Marginal Abatement Cost Curve (MACC)" in rep_macc
        assert "100.00%" in rep_macc

        em_res = run_emissions_scenario(
            shipped_profiles,
            specs=baseline_emission_specs(),
            co2_price_per_tco2=40.0,
            include_storage=False,
        )
        rep_em = format_emissions_report(em_res)
        assert "CO2 Policy & Lagrangian Shadow-Price Diagnostics:" in rep_em

        out_png = plot_co2_pareto_frontier_and_macc(
            output_path=tmp_path / "macc_render.png",
            macc_df=macc_table,
            tax_df=tax_table,
        )
        assert out_png.is_file() and out_png.stat().st_size > 120_000

    def test_cli_co2_flags_and_negative_input_error_exit(self):
        ok_proc = subprocess.run(
            [sys.executable, "main.py", "--co2-price", "50", "--co2-cap", "20000"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert ok_proc.returncode == 0, ok_proc.stderr
        assert "CO2 Policy & Lagrangian Shadow-Price Diagnostics:" in ok_proc.stdout
        assert "hybrid_cap_and_tax" in ok_proc.stdout

        err_proc = subprocess.run(
            [sys.executable, "main.py", "--co2-price", "-10"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert err_proc.returncode == 1
        assert "ERROR:" in err_proc.stderr
