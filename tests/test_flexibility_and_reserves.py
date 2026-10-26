"""Thermal ramp envelopes, must-run floors & spinning reserves tests (v0.6.0)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from energy_model import (
    DEFAULT_TECHNOLOGIES,
    FlexibleTechSpec,
    SolverError,
    SpinningReservePolicy,
    baseline_flexibility_specs,
    compare_flexibility_regimes,
    default_flexible_fleet_specs,
    format_flexibility_report,
    make_duck_curve_profiles,
    plot_duck_curve_ramping_and_reserves,
    run_flexibility_scenario,
    solve_flexibility_linprog_crosscheck,
    sweep_thermal_ramp_tightness,
    verify_ramp_induced_negative_lmp,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def duck_profiles():
    return make_duck_curve_profiles()


@pytest.fixture(scope="module")
def regime_table(duck_profiles):
    return compare_flexibility_regimes(duck_profiles)


class TestFlexibilitySpecsAndProfiles:
    def test_default_flexible_fleet_hierarchy(self):
        specs = default_flexible_fleet_specs()
        assert set(specs) == {
            "Baseload_Thermal",
            "CCGT_MidMerit",
            "OCGT_FastPeaker",
            "Solar_PV",
            "Wind_Turbine",
        }
        assert specs["Baseload_Thermal"].p_min_pu > specs["CCGT_MidMerit"].p_min_pu
        assert (
            specs["Baseload_Thermal"].ramp_limit_up
            < specs["CCGT_MidMerit"].ramp_limit_up
            < specs["OCGT_FastPeaker"].ramp_limit_up
        )
        assert (
            specs["Baseload_Thermal"].reserve_cap_pu
            < specs["CCGT_MidMerit"].reserve_cap_pu
            < specs["OCGT_FastPeaker"].reserve_cap_pu
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"p_min_pu": -0.1},
            {"p_min_pu": 1.1},
            {"ramp_limit_up": 0.0},
            {"ramp_limit_down": 1.2},
            {"reserve_cap_pu": -0.05},
            {"profile": "solar_cf", "p_min_pu": 0.2},
            {"profile": "unknown_cf"},
            {"capital_cost_per_mw_yr": -10.0},
        ],
    )
    def test_flexible_tech_spec_validation(self, kwargs):
        base = {
            "name": "Unit",
            "carrier": "thermal",
            "capital_cost_per_mw_yr": 30_000.0,
            "marginal_cost_per_mwh": 40.0,
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            FlexibleTechSpec(**base)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"load_reserve_fraction": -0.01},
            {"vre_reserve_fraction": 1.2},
            {"bess_sustain_hours": 0.0},
            {"bess_sustain_hours": -1.0},
        ],
    )
    def test_spinning_reserve_policy_validation(self, kwargs):
        with pytest.raises(ValueError):
            SpinningReservePolicy(**kwargs)

    def test_duck_curve_profiles_chronology_and_bounds(self, duck_profiles):
        again = make_duck_curve_profiles(days=2)
        assert len(duck_profiles) == 48
        assert duck_profiles.equals(again)
        assert float(duck_profiles["solar_cf"].max()) > 0.90
        assert float(duck_profiles["load_mw"].max()) > 195.0
        with pytest.raises(ValueError):
            make_duck_curve_profiles(days=0)

    def test_baseline_flexibility_specs_preserves_default_technologies(self):
        mapped = baseline_flexibility_specs()
        assert set(mapped) == set(DEFAULT_TECHNOLOGIES)
        assert mapped["Solar_PV"].p_min_pu == 0.0
        assert mapped["Gas_Peaker"].ramp_limit_up == pytest.approx(0.40)


class TestRampingReservesAndStrongDuality:
    def test_cooptimized_bess_envelopes_headroom_soc_and_power_balance(self, duck_profiles):
        pol = SpinningReservePolicy(
            enabled=True,
            load_reserve_fraction=0.06,
            vre_reserve_fraction=0.12,
            bess_sustain_hours=1.0,
        )
        res = run_flexibility_scenario(
            duck_profiles,
            enforce_ramping=True,
            enforce_must_run=True,
            include_bess=True,
            reserve_policy=pol,
        )
        assert res.power_balance_max_residual_mw < 1e-9
        assert res.strong_duality_revenue_gap < 1e-6

        # Verify system LCOE == load-wtd LMP + alpha_load * load-wtd Reserve Price
        expected_lcoe = (
            res.load_weighted_lmp_per_mwh
            + pol.load_reserve_fraction * res.load_weighted_reserve_price_per_mw_h
        )
        assert res.system_lcoe_per_mwh == pytest.approx(expected_lcoe, rel=1e-10)

        # Verify thermal ramp utilization never exceeds 100% of envelope
        for util_pct in res.max_thermal_ramp_utilization_pct:
            assert util_pct <= 100.0 + 1e-6

        # Verify total spinning reserve supplied >= dynamic requirement at every snapshot
        tot_res = res.reserve_by_unit_mw.sum(axis=1) + res.bess_reserve_mw
        assert np.all(tot_res.to_numpy() + 1e-7 >= res.reserve_requirement_mw.to_numpy())

        # Verify BESS reserve never exceeds stored energy SoC * eta_dis
        assert np.all(res.bess_reserve_mw.to_numpy() <= 0.95 * res.bess_soc_mwh.to_numpy() + 1e-7)

    def test_four_regime_cost_vre_hosting_and_peaker_hierarchy(self, regime_table):
        r1, r2, r3, r4 = [regime_table.iloc[i] for i in range(4)]
        assert float(r1["system_lcoe_per_mwh"]) < float(r2["system_lcoe_per_mwh"])
        assert float(r2["system_lcoe_per_mwh"]) < float(r3["system_lcoe_per_mwh"])
        assert float(r4["system_lcoe_per_mwh"]) < float(r3["system_lcoe_per_mwh"])

        # BESS restores >3x VRE hosting capacity and cuts OCGT peakers & scarcity prices
        assert float(r4["vre_total_mw"]) > 2.5 * float(r3["vre_total_mw"])
        assert float(r4["ocgt_mw"]) < 0.5 * float(r3["ocgt_mw"])
        assert float(r4["max_lmp_per_mwh"]) < float(r3["max_lmp_per_mwh"])
        assert float(r4["max_reserve_price_per_mw_h"]) < float(r3["max_reserve_price_per_mw_h"])
        assert float(regime_table["strong_duality_revenue_gap"].max()) < 1e-6

    def test_nested_polytope_ramp_tightness_monotonicity(self, duck_profiles):
        sweep = sweep_thermal_ramp_tightness(
            duck_profiles, ramp_scales=(1.0, 0.65, 0.35), include_bess=False
        )
        lcoes = sweep["system_lcoe_per_mwh"].to_numpy(dtype=float)
        assert np.all(np.diff(lcoes) >= -1e-7)
        assert lcoes[-1] > lcoes[0]

    def test_ramp_induced_negative_lmp_theorem_with_nonnegative_costs(self):
        proof = verify_ramp_induced_negative_lmp()
        assert proof.min_generator_marginal_cost >= 0.0
        assert proof.trough_lmp_per_mwh == pytest.approx(-138.0, abs=1e-8)
        assert proof.abs_lmp_formula_error < 1e-9

    def test_active_reserves_without_capable_units_raises_solver_error(self, shipped_profiles):
        specs = baseline_flexibility_specs(thermal_reserve_cap_pu=0.0)
        with pytest.raises(SolverError, match="zero reserve-capable"):
            run_flexibility_scenario(
                shipped_profiles,
                specs=specs,
                include_bess=False,
                reserve_policy=SpinningReservePolicy(
                    enabled=True, load_reserve_fraction=0.05, vre_reserve_fraction=0.0
                ),
            )


class TestSciPyFlexibilityCrossCheck:
    @pytest.mark.parametrize("include_bess", [False, True])
    def test_duck_curve_ramping_mustrun_reserves_matches_scipy_linprog(
        self, duck_profiles, include_bess: bool
    ):
        chk = solve_flexibility_linprog_crosscheck(duck_profiles, include_bess=include_bess)
        assert chk.rel_objective_diff < 1e-10
        assert chk.max_p_nom_diff_mw < 1e-8
        assert chk.bess_e_nom_diff_mwh < 1e-8
        assert chk.max_lmp_diff_per_mwh < 1e-8
        assert chk.max_reserve_price_diff_per_mw_h < 1e-8

    def test_shipped_24h_open_boundary_ramping_matches_scipy_linprog(self, shipped_profiles):
        chk = solve_flexibility_linprog_crosscheck(
            shipped_profiles,
            specs=baseline_flexibility_specs(),
            cyclic_ramping=False,
            include_bess=False,
            reserve_policy=SpinningReservePolicy(
                enabled=True, load_reserve_fraction=0.05, vre_reserve_fraction=0.08
            ),
        )
        assert chk.rel_objective_diff < 1e-10
        assert chk.max_p_nom_diff_mw < 1e-8
        assert chk.max_lmp_diff_per_mwh < 1e-8
        assert chk.max_reserve_price_diff_per_mw_h < 1e-8


class TestCLIAndFlexibilityFigure:
    def test_committed_flexibility_figure_and_report(self, duck_profiles, regime_table, tmp_path):
        committed = REPO_ROOT / "results" / "duck_curve_ramping_and_reserves.png"
        assert committed.is_file() and committed.stat().st_size > 120_000

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "results/duck_curve_ramping_and_reserves.png" in readme

        rep = format_flexibility_report(regime_table)
        assert "4_Flexible_BESS_CoOptimized" in rep

        out_png = plot_duck_curve_ramping_and_reserves(
            output_path=tmp_path / "duck_flex.png",
            profiles=duck_profiles,
            regime_df=regime_table,
        )
        assert out_png.is_file() and out_png.stat().st_size > 120_000

    def test_cli_flexibility_flags_and_invalid_input_exit(self):
        ok_proc = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--ramp-limit",
                "0.35",
                "--must-run",
                "0.20",
                "--reserve-margin",
                "0.06",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert ok_proc.returncode == 0, ok_proc.stderr
        assert "Operational Flexibility & Reserve Diagnostics:" in ok_proc.stdout

        err_proc = subprocess.run(
            [sys.executable, "main.py", "--ramp-limit", "0.0"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert err_proc.returncode == 1
        assert "ERROR:" in err_proc.stderr
