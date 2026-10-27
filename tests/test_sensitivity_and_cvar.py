"""Tests for Module 6 (v0.8.0): 2D WACCxCapEx surface, CVaR_95 adequacy & etm-sim CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from energy_model import (
    DEFAULT_TECHNOLOGIES,
    REFERENCE_WACC,
    WeatherRegimeSpec,
    __version__,
    build_scaled_wacc_technologies,
    capital_recovery_factor,
    compute_discrete_cvar,
    crf_wacc_derivative,
    default_weather_year_specs,
    evaluate_stochastic_value_metrics,
    format_cvar_report,
    format_sensitivity_report,
    format_single_cvar_report,
    generate_multiyear_weather_profiles,
    plot_wacc_capex_and_cvar_adequacy,
    run_stochastic_cvar_scenario,
    solve_stochastic_cvar_linprog_crosscheck,
    sweep_cvar_risk_frontier,
    sweep_wacc_capex_surface,
    verify_envelope_and_concavity_theorems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wacc_capex_surface():
    return sweep_wacc_capex_surface()


@pytest.fixture(scope="module")
def envelope_verification():
    return verify_envelope_and_concavity_theorems()


@pytest.fixture(scope="module")
def weather_ensemble():
    return generate_multiyear_weather_profiles()


@pytest.fixture(scope="module")
def hedged_cvar_result(weather_ensemble):
    return run_stochastic_cvar_scenario(weather_ensemble, risk_aversion_beta=0.35)


@pytest.fixture(scope="module")
def cvar_frontier_table(weather_ensemble):
    return sweep_cvar_risk_frontier(weather_ensemble)


@pytest.fixture(scope="module")
def stochastic_value_metrics(weather_ensemble):
    return evaluate_stochastic_value_metrics(weather_ensemble, hedged_beta=0.35)


class TestWACCCapExSurfaceAndEnvelopeTheorem:
    @pytest.mark.parametrize(
        "wacc, lifetime_yr",
        [
            (0.00, 25.0),
            (1e-7, 20.0),
            (0.03, 20.0),
            (0.07, 25.0),
            (0.12, 30.0),
        ],
    )
    def test_crf_wacc_derivative_zero_limit_and_finite_difference_identity(
        self, wacc: float, lifetime_yr: float
    ):
        an = crf_wacc_derivative(wacc, lifetime_yr)
        if wacc == 0.0:
            assert an == pytest.approx((lifetime_yr + 1.0) / (2.0 * lifetime_yr), rel=1e-12)
            h = 1e-4
            fd = (
                -3.0 * capital_recovery_factor(0.0, lifetime_yr)
                + 4.0 * capital_recovery_factor(h, lifetime_yr)
                - capital_recovery_factor(2.0 * h, lifetime_yr)
            ) / (2.0 * h)
            assert an == pytest.approx(fd, rel=1e-6)
        elif wacc < 1e-4:
            exact_taylor = (
                (lifetime_yr + 1.0) / (2.0 * lifetime_yr)
                + ((lifetime_yr**2 - 1.0) / (6.0 * lifetime_yr)) * wacc
                - ((lifetime_yr**2 - 1.0) / (12.0 * lifetime_yr)) * (wacc**2)
            )
            assert an == pytest.approx(exact_taylor, rel=1e-12)
        else:
            h = 1e-6
            fd = (
                capital_recovery_factor(wacc + h, lifetime_yr)
                - capital_recovery_factor(wacc - h, lifetime_yr)
            ) / (2.0 * h)
            assert an == pytest.approx(fd, rel=1e-8)

    def test_scaled_wacc_technologies_reproduce_baseline_and_validate_inputs(self):
        base_techs = build_scaled_wacc_technologies(REFERENCE_WACC, 1.0)
        for name, expected in DEFAULT_TECHNOLOGIES.items():
            assert base_techs[name]["capital_cost"] == pytest.approx(
                expected["capital_cost"], rel=1e-12
            )
            assert base_techs[name]["marginal_cost"] == pytest.approx(
                expected["marginal_cost"], abs=1e-12
            )

        for bad_kwargs in (
            {"wacc": -0.01, "vre_capex_scale": 1.0},
            {"wacc": 1.05, "vre_capex_scale": 1.0},
            {"wacc": True, "vre_capex_scale": 1.0},
            {"wacc": 0.07, "vre_capex_scale": 0.0},
            {"wacc": 0.07, "vre_capex_scale": -0.5},
            {"wacc": 0.07, "vre_capex_scale": 1.0, "thermal_capex_scale": 0.0},
        ):
            with pytest.raises(ValueError):
                build_scaled_wacc_technologies(**bad_kwargs)

    def test_2d_surface_monotonicity_reference_anchor_and_phase_transition(
        self, wacc_capex_surface
    ):
        lcoe_mat = wacc_capex_surface.lcoe_surface_df.to_numpy(dtype=float)
        vre_mat = wacc_capex_surface.vre_share_surface_df.to_numpy(dtype=float)
        sub_mat = wacc_capex_surface.substitution_dividend_surface_df.to_numpy(dtype=float)

        assert wacc_capex_surface.reference_lcoe_per_mwh == pytest.approx(28.0555377, rel=1e-6)
        assert float(wacc_capex_surface.lcoe_surface_df.loc[0.07, 1.00]) == pytest.approx(
            wacc_capex_surface.reference_lcoe_per_mwh, rel=1e-12
        )
        assert float(
            wacc_capex_surface.substitution_dividend_surface_df.loc[0.07, 1.00]
        ) == pytest.approx(0.0, abs=1e-9)

        # Strict LCOE monotonicity wrt both WACC (rows) and VRE CapEx scale (cols)
        assert np.all(np.diff(lcoe_mat, axis=0) > 0.0)
        assert np.all(np.diff(lcoe_mat, axis=1) > 0.0)

        # VRE share monotonically non-increasing wrt WACC and VRE CapEx scale
        assert np.all(np.diff(vre_mat, axis=0) <= 1e-6)
        assert np.all(np.diff(vre_mat, axis=1) <= 1e-6)
        assert vre_mat[0, 0] > 99.0
        assert vre_mat[-1, -1] == pytest.approx(0.0, abs=1e-6)

        # Global substitution dividend >= 0 with >20% dividend at concessional VRE corner
        assert np.all(sub_mat >= 0.0)
        assert sub_mat[0, 0] > 20.0
        assert sub_mat[-1, -1] > 10.0

    def test_hotelling_shephard_envelope_and_global_concavity_theorems(self, envelope_verification):
        assert envelope_verification.rel_error_dcost_dwacc < 1e-6
        assert envelope_verification.rel_error_dcost_dvre_scale < 1e-9
        assert envelope_verification.concavity_midpoint_surplus > 5.0e6
        assert envelope_verification.passive_minus_optimal_min_margin >= 0.0


class TestStochasticWeatherCVaRAndAdequacy:
    def test_discrete_cvar_atom_splitting_and_coherent_risk_axioms(self):
        losses = np.array([10.0, 20.0, 50.0, 100.0])
        probs = np.array([0.50, 0.41, 0.06, 0.03])
        # alpha = 0.95 -> tail mass 0.05 = 0.03 * 100 + 0.02 * 50 -> CVaR = (3 + 1)/0.05 = 80
        var95, cvar95 = compute_discrete_cvar(losses, probs, alpha=0.95)
        assert var95 == pytest.approx(50.0, abs=1e-12)
        assert cvar95 == pytest.approx(80.0, abs=1e-12)

        # Translation equivariance & positive homogeneity
        _, cvar_shift = compute_discrete_cvar(2.5 * losses + 17.0, probs, alpha=0.95)
        assert cvar_shift == pytest.approx(2.5 * cvar95 + 17.0, rel=1e-12)

        # Subadditivity: CVaR(X + Y) <= CVaR(X) + CVaR(Y)
        losses_y = np.array([90.0, 40.0, 15.0, 5.0])
        _, cvar_y = compute_discrete_cvar(losses_y, probs, alpha=0.95)
        _, cvar_xy = compute_discrete_cvar(losses + losses_y, probs, alpha=0.95)
        assert cvar_xy <= cvar95 + cvar_y + 1e-12

        with pytest.raises(ValueError):
            compute_discrete_cvar([1.0, 2.0], [0.4, 0.4], alpha=0.95)

    def test_weather_ensemble_probabilities_and_validation(self, weather_ensemble):
        specs = default_weather_year_specs()
        assert len(specs) == 8
        assert sum(s.probability for s in specs) == pytest.approx(1.0, abs=1e-12)
        assert len(weather_ensemble) == 8

        bad_spec = WeatherRegimeSpec("Bad", 0.50, 1.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="sum to 1.0"):
            generate_multiyear_weather_profiles(weather_specs=(bad_spec,))

    def test_rockafellar_uryasev_epigraph_risk_measure_and_complete_market_duality(
        self, hedged_cvar_result
    ):
        res = hedged_cvar_result
        assert res.lp_cvar_epigraph_cost_per_year == pytest.approx(
            res.cvar_total_cost_per_year, rel=1e-11
        )
        assert abs(res.strong_duality_gap) < 1e-5
        assert res.max_zero_profit_residual < 1e-5

        tbl = res.scenario_table
        probs = tbl["probability"].to_numpy(dtype=float)
        q_adj = tbl["risk_adjusted_prob"].to_numpy(dtype=float)
        assert float(np.sum(q_adj)) == pytest.approx(1.0, abs=1e-12)

        beta = res.risk_aversion_beta
        alpha = res.alpha
        rho = q_adj - (1.0 - beta) * probs
        assert np.all(rho >= -1e-12)
        assert np.all(rho <= beta * probs / (1.0 - alpha) + 1e-12)

        # Exact boundary atom split at WY6_Winter_Cloud_Trough (0.01 / 0.05 of beta)
        assert float(tbl.loc["WY6_Winter_Cloud_Trough", "risk_adjusted_prob"]) == pytest.approx(
            (1.0 - beta) * 0.060 + beta * (0.010 / 0.050), abs=1e-10
        )
        assert float(tbl.loc["WY8_Severe_Dunkelflaute", "risk_adjusted_prob"]) == pytest.approx(
            (1.0 - beta) * 0.015 + beta * (0.015 / 0.050), abs=1e-10
        )

    def test_mean_cvar_pareto_frontier_monotonicity_and_blackout_elimination(
        self, cvar_frontier_table
    ):
        ez = cvar_frontier_table["expected_cost_per_year"].to_numpy(dtype=float)
        cv = cvar_frontier_table["cvar_cost_per_year"].to_numpy(dtype=float)
        eens = cvar_frontier_table["eens_mwh_per_year"].to_numpy(dtype=float)

        assert np.all(np.diff(ez) >= -1e-5)
        assert np.all(np.diff(cv) <= 1e-5)

        # Risk-neutral RP (beta=0) leaves positive Stoft scarcity shedding in WY8,
        # whereas beta >= 0.15 eliminates 100% of load shedding for < 0.15% E[Cost] premium
        assert eens[0] > 20.0
        assert np.all(eens[1:] == pytest.approx(0.0, abs=1e-8))
        rel_cost_bump_b15 = (ez[1] - ez[0]) / ez[0]
        assert 0.0 < rel_cost_bump_b15 < 0.0015

    def test_birge_louveaux_evpi_and_vss_sandwich_inequality(self, stochastic_value_metrics):
        vm = stochastic_value_metrics
        assert vm.wait_and_see_cost_per_year < vm.recourse_problem_cost_per_year
        assert vm.recourse_problem_cost_per_year < vm.expected_value_plan_eev_cost_per_year
        assert vm.evpi_per_year > 5.0e6
        assert vm.vss_per_year > 200.0e6
        assert vm.mean_weather_eens_mwh_per_year > 100_000.0
        assert vm.risk_neutral_eens_mwh_per_year > 20.0
        assert vm.cvar_hedged_eens_mwh_per_year == pytest.approx(0.0, abs=1e-8)


class TestSciPyStochasticCVaRCrossCheck:
    @pytest.mark.parametrize(
        "beta, include_bess",
        [
            (0.00, True),
            (0.35, True),
            (0.35, False),
            (0.85, True),
        ],
    )
    def test_scipy_matrix_matches_pypsa_stochastic_cvar_to_machine_precision(
        self, weather_ensemble, beta: float, include_bess: bool
    ):
        chk = solve_stochastic_cvar_linprog_crosscheck(
            weather_ensemble,
            risk_aversion_beta=beta,
            include_bess=include_bess,
        )
        assert chk.rel_objective_diff < 1e-11
        assert chk.max_p_nom_diff_mw < 1e-8
        assert chk.rel_cvar_cost_diff < 1e-11
        assert chk.max_risk_prob_diff < 1e-10
        assert chk.max_lmp_diff_per_mwh < 1e-8


class TestPackagedETMSimCLIAndRiskFigure:
    def test_committed_risk_figure_readme_and_reports(
        self,
        wacc_capex_surface,
        envelope_verification,
        cvar_frontier_table,
        stochastic_value_metrics,
        hedged_cvar_result,
        tmp_path,
    ):
        committed = REPO_ROOT / "results" / "wacc_capex_surface_and_cvar_adequacy.png"
        assert committed.is_file() and committed.stat().st_size > 120_000

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "results/wacc_capex_surface_and_cvar_adequacy.png" in readme
        assert "etm-sim" in readme

        s_rep = format_sensitivity_report(wacc_capex_surface, envelope_verification)
        c_rep = format_cvar_report(cvar_frontier_table, stochastic_value_metrics)
        sc_rep = format_single_cvar_report(hedged_cvar_result)
        assert "Max Substitution Dividend" in s_rep
        assert "Birge-Louveaux Stochastic Value Sandwich" in c_rep
        assert "Two-Stage Stochastic Mean-CVaR_95" in sc_rep
        assert "-0.0" not in c_rep and "-0.0" not in sc_rep

        out_png = plot_wacc_capex_and_cvar_adequacy(
            output_path=tmp_path / "risk_fig.png",
            surface=wacc_capex_surface,
            env_check=envelope_verification,
            frontier_df=cvar_frontier_table,
            value_metrics=stochastic_value_metrics,
            hedged_result=hedged_cvar_result,
        )
        assert out_png.is_file() and out_png.stat().st_size > 120_000

    def test_pyproject_declares_etm_sim_console_script_and_version(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert pyproject["project"]["scripts"]["etm-sim"] == "energy_model.cli:cli_entrypoint"

        etm_bin = Path(sys.executable).with_name("etm-sim")
        cmd = (
            [str(etm_bin), "--version"]
            if etm_bin.is_file()
            else [
                sys.executable,
                "-m",
                "energy_model.cli",
                "--version",
            ]
        )
        ver_proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
        assert ver_proc.returncode == 0, ver_proc.stderr
        assert f"etm-sim {__version__}" in ver_proc.stdout

    def test_etm_sim_cli_wacc_cvar_and_json_summary(self, tmp_path):
        json_path = tmp_path / "etm_summary.json"
        etm_bin = Path(sys.executable).with_name("etm-sim")
        runner = (
            [str(etm_bin)]
            if etm_bin.is_file()
            else [
                sys.executable,
                "-m",
                "energy_model.cli",
            ]
        )
        proc = subprocess.run(
            [
                *runner,
                "--wacc",
                "0.04",
                "--vre-capex-scale",
                "0.75",
                "--cvar-beta",
                "0.35",
                "--json-summary",
                str(json_path),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Two-Stage Stochastic Mean-CVaR_95 Adequacy Report" in proc.stdout
        assert json_path.is_file()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["version"] == __version__
        assert payload["baseline"]["system_lcoe_per_mwh"] == pytest.approx(17.058, abs=0.05)
        assert payload["cvar_scenario"]["eens_mwh_per_year"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        "bad_args",
        [
            ["--wacc", "1.5"],
            ["--vre-capex-scale", "0.0"],
            ["--cvar-beta", "-0.1"],
            ["--cvar-beta", "0.5", "--cvar-alpha", "1.0"],
        ],
    )
    def test_cli_invalid_risk_and_sensitivity_flags_exit_nonzero(self, bad_args):
        proc = subprocess.run(
            [sys.executable, "main.py", *bad_args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 1
        assert "ERROR:" in proc.stderr
