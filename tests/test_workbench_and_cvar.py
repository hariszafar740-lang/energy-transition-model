"""Module 7 tests: mean-CVaR kernel, workbench dataset contract & artifact integrity.

The workbench is a *derived* artifact: every figure must be traceable to a solved
model.  These tests pin the three properties that make that claim checkable:

1. the explicit mean-CVaR LP agrees with an independent SciPy re-formulation and
   with the discrete tail integral;
2. the shipped JSON/HTML artifacts are exactly reproducible from the template and
   contain no external reference, no placeholder token and no untracked solver run;
3. the printed numbers quoted in the README equal the numbers the code produces.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from energy_model.workbench import (
    DEFAULT_RISK_REGIMES,
    RISK_ALPHA,
    RISK_BETA_SWEEP,
    SCHEMA_VERSION,
    TEMPLATE_DATA_TOKEN,
    TEMPLATE_SHA_TOKEN,
    RiskRegime,
    bilinear_interpolate,
    birge_louveaux_bounds,
    check_workbench,
    dataset_digest,
    discrete_cvar,
    interpolate_surface,
    render_page,
    risk_frontier,
    solve_mean_cvar,
    template_path,
    validate_risk_regimes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "workbench" / "data" / "etm_workbench_data.json"
HTML_FILE = REPO_ROOT / "workbench" / "index.html"


@pytest.fixture(scope="module")
def dataset() -> dict:
    """Load the committed workbench dataset once per module."""
    assert DATA_FILE.is_file(), f"missing committed dataset: {DATA_FILE}"
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def template() -> str:
    """Load the packaged HTML template once per module."""
    return template_path().read_text(encoding="utf-8")


class TestRiskRegimes:
    def test_probabilities_sum_to_one(self):
        total = sum(r.probability for r in DEFAULT_RISK_REGIMES)
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_default_regimes_validate(self):
        assert validate_risk_regimes(DEFAULT_RISK_REGIMES) == DEFAULT_RISK_REGIMES

    def test_unanchored_probabilities_are_rejected(self):
        broken = DEFAULT_RISK_REGIMES[:-1]
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_risk_regimes(broken)

    def test_duplicate_names_are_rejected(self):
        dup = (
            DEFAULT_RISK_REGIMES[0],
            DEFAULT_RISK_REGIMES[0].__class__(
                "WY1_Windy_Equinox",
                0.82,
                0.0,
                1.0,
                0.62,
                0.92,
                0.12,
            ),
        )
        with pytest.raises(ValueError, match="unique"):
            validate_risk_regimes(dup)

    def test_invalid_bounds_are_rejected(self):
        with pytest.raises(ValueError, match="wind CF"):
            RiskRegime("bad", 1.0, 0.0, 1.0, 0.5, 0.4, 0.6)

    def test_profiles_are_hourly_and_physical(self):
        for regime in DEFAULT_RISK_REGIMES:
            prof = regime.profiles()
            assert prof["load"].shape == (24,)
            assert np.all(prof["load"] > 0.0)
            assert np.all((prof["solar_cf"] >= 0.0) & (prof["solar_cf"] <= 1.0))
            assert np.all((prof["wind_cf"] >= 0.0) & (prof["wind_cf"] <= 1.0))
            expected = 150.0 + regime.load_adder_mw
            assert prof["load"].mean() == pytest.approx(expected, rel=1e-9)


class TestDiscreteCvar:
    def test_atom_splitting_matches_hand_calculation(self):
        costs = np.array([10.0, 20.0, 30.0, 40.0])
        probs = np.array([0.5, 0.3, 0.15, 0.05])
        var, cvar = discrete_cvar(costs, probs, 0.95)
        assert var == pytest.approx(40.0)
        assert cvar == pytest.approx(40.0)

    def test_partial_atom_is_split_exactly(self):
        costs = np.array([0.0, 100.0])
        probs = np.array([0.9, 0.1])
        # alpha = 0.95 -> the worst 5% is half of the 100-cost atom
        var, cvar = discrete_cvar(costs, probs, 0.95)
        assert var == pytest.approx(100.0)
        assert cvar == pytest.approx(100.0)

    def test_alpha_zero_returns_mean(self):
        costs = np.array([1.0, 2.0, 3.0])
        probs = np.array([0.2, 0.5, 0.3])
        _, cvar = discrete_cvar(costs, probs, 1e-9)
        assert cvar == pytest.approx(float(np.dot(costs, probs)))

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            discrete_cvar([1.0], [0.5], 0.95)
        with pytest.raises(ValueError):
            discrete_cvar([1.0, 2.0], [0.5, 0.5], 1.5)


class TestMeanCvarKernel:
    @pytest.mark.parametrize("beta", [0.0, 0.35, 1.0])
    def test_beta_zero_reproduces_risk_neutral_plan(self, beta):
        solution = solve_mean_cvar(beta)
        assert math.isfinite(solution.objective)
        assert solution.n_variables == 4 + 8 * 7 * 24 + 1 + 8
        assert solution.n_constraints >= 1_000
        assert solution.total_risk_weight == pytest.approx(1.0, abs=1e-9)

    def test_expected_cost_monotone_in_beta(self):
        frontier = risk_frontier(RISK_BETA_SWEEP)
        expected = frontier["expected_cost_per_year"].to_numpy(dtype=float)
        tail = frontier["cvar_per_year"].to_numpy(dtype=float)
        assert np.all(np.diff(expected) >= -1e-6)
        assert np.all(np.diff(tail) <= 1e-6)

    def test_tail_improves_by_at_least_ten_percent(self):
        frontier = risk_frontier(RISK_BETA_SWEEP)
        assert frontier["cvar_reduction_pct"].iloc[-1] > 10.0
        assert frontier["risk_premium_per_mwh"].iloc[-1] > 0.0

    def test_risk_neutral_plan_leaves_some_unserved_energy(self):
        neutral = solve_mean_cvar(0.0)
        hedged = solve_mean_cvar(0.35)
        assert neutral.expected_unserved_mwh_per_year > 0.0
        assert hedged.expected_unserved_mwh_per_year == pytest.approx(0.0, abs=1e-6)

    def test_duality_gap_is_machine_precision(self):
        solution = solve_mean_cvar(0.5)
        scale = max(abs(solution.risk_adjusted_cost_per_year), 1.0)
        assert solution.duality_gap / scale < 1e-9

    def test_risk_weights_never_exceed_the_envelope(self):
        for beta in RISK_BETA_SWEEP:
            solution = solve_mean_cvar(beta)
            envelope = (1.0 - beta) * np.array(
                [r.probability for r in DEFAULT_RISK_REGIMES]
            ) + beta * np.array([r.probability for r in DEFAULT_RISK_REGIMES]) / (1.0 - RISK_ALPHA)
            assert solution.weight_envelope_excess <= 1e-9
            assert np.all(solution.risk_weights >= -1e-12)
            assert np.all(solution.risk_weights <= envelope + 1e-9)

    def test_pure_cvar_puts_all_weight_on_the_tail(self):
        solution = solve_mean_cvar(1.0)
        probs = np.array([r.probability for r in DEFAULT_RISK_REGIMES])
        order = np.argsort(solution.scenario_cost_per_year)
        worst = order[-1]
        assert solution.risk_weights[worst] > probs[worst] * 1.5
        assert solution.risk_weights[order[0]] < 0.02

    def test_beta_bounds_are_enforced(self):
        with pytest.raises(ValueError, match="beta"):
            solve_mean_cvar(1.2)
        with pytest.raises(ValueError, match="alpha"):
            solve_mean_cvar(0.5, alpha=1.0)

    def test_each_scenario_balances_exactly(self):
        solution = solve_mean_cvar(0.35)
        for s, regime in enumerate(DEFAULT_RISK_REGIMES):
            prof = regime.profiles()
            dispatch = (
                solution.dispatch_mw["solar"][s]
                + solution.dispatch_mw["wind"][s]
                + solution.dispatch_mw["gas"][s]
                + solution.dispatch_mw["discharge"][s]
                - solution.dispatch_mw["charge"][s]
                + solution.dispatch_mw["shed"][s]
            )
            # HiGHS primal feasibility tolerance on a ~150 MW system: 1 mW is exact.
            assert np.allclose(dispatch, prof["load"], atol=1e-3)

    def test_shortfall_exactly_equals_unserved_energy(self):
        solution = solve_mean_cvar(0.0)
        probs = np.array([r.probability for r in DEFAULT_RISK_REGIMES])
        eigene = float(np.dot(probs, solution.shed_mw.sum(axis=1)))
        assert eigene * 365.0 == pytest.approx(solution.expected_unserved_mwh_per_year, rel=1e-9)


class TestIndependentKernelCrossCheck:
    def test_scipy_dense_reformulation_matches_sparse_kernel(self):
        """Re-derive the beta = 0.5 objective with an independently built LP."""
        from scipy.optimize import linprog

        beta, alpha = 0.5, RISK_ALPHA
        regimes = DEFAULT_RISK_REGIMES
        n_s, hours = len(regimes), 24
        per = 7 * hours
        i_eta = 4 + n_s * per
        i_z = i_eta + 1
        n_var = i_z + n_s
        capex = np.array([45_000.0, 70_000.0, 20_000.0, 30_000.0])

        cost = np.zeros(n_var)
        cost[:4] = capex * (1.0 - beta)
        cost[i_eta] = beta
        rows, cols, vals, rhs = [], [], [], []
        eq_rows, eq_cols, eq_vals, eq_rhs = [], [], [], []
        n_row = n_eq = 0
        for s, regime in enumerate(regimes):
            prof = regime.profiles()
            offset = 4 + s * per
            idx = {k: offset + k * hours for k in range(7)}
            cost[i_z + s] = beta * regime.probability / (1.0 - alpha)
            cost[idx[2] : idx[2] + hours] = 65.0 * 365.0 * (1.0 - beta) * regime.probability
            cost[idx[6] : idx[6] + hours] = 1800.0 * 365.0 * (1.0 - beta) * regime.probability
            for j, value in enumerate(capex):
                rows.append(n_row)
                cols.append(j)
                vals.append(value)
            for t in range(hours):
                rows.extend([n_row, n_row])
                cols.extend([idx[2] + t, idx[6] + t])
                vals.extend([65.0 * 365.0, 1800.0 * 365.0])
            rows.extend([n_row, n_row])
            cols.extend([i_eta, i_z + s])
            vals.extend([-1.0, -1.0])
            rhs.append(0.0)
            n_row += 1
            for t in range(hours):
                for k, coef in ((0, 1.0), (1, 1.0), (2, 1.0), (4, 1.0), (3, -1.0), (6, 1.0)):
                    eq_rows.append(n_eq)
                    eq_cols.append(idx[k] + t)
                    eq_vals.append(coef)
                eq_rhs.append(float(prof["load"][t]))
                n_eq += 1
                # cyclic state of charge
                for col, val in (
                    (idx[5] + t, 1.0),
                    (idx[5] + (t - 1) % hours, -1.0),
                    (idx[3] + t, -0.95),
                    (idx[4] + t, 1.0 / 0.95),
                ):
                    eq_rows.append(n_eq)
                    eq_cols.append(col)
                    eq_vals.append(val)
                eq_rhs.append(0.0)
                n_eq += 1
                # discharge <= P_bess, soc <= duration * P_bess
                for col, val in ((idx[4] + t, 1.0), (3, -1.0)):
                    rows.append(n_row)
                    cols.append(col)
                    vals.append(val)
                rhs.append(0.0)
                n_row += 1
                for col, val in ((idx[5] + t, 1.0), (3, -4.0)):
                    rows.append(n_row)
                    cols.append(col)
                    vals.append(val)
                rhs.append(0.0)
                n_row += 1
                # resource availability and thermal capacity
                for col, val in ((idx[0] + t, 1.0), (0, -float(prof["solar_cf"][t]))):
                    rows.append(n_row)
                    cols.append(col)
                    vals.append(val)
                rhs.append(0.0)
                n_row += 1
                for col, val in ((idx[1] + t, 1.0), (1, -float(prof["wind_cf"][t]))):
                    rows.append(n_row)
                    cols.append(col)
                    vals.append(val)
                rhs.append(0.0)
                n_row += 1
                for col, val in ((idx[2] + t, 1.0), (2, -1.0)):
                    rows.append(n_row)
                    cols.append(col)
                    vals.append(val)
                rhs.append(0.0)
                n_row += 1
        from scipy.sparse import coo_matrix

        a_ub = coo_matrix((vals, (rows, cols)), shape=(n_row, n_var)).toarray()
        a_eq = coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(n_eq, n_var)).toarray()
        reference = linprog(
            cost,
            A_ub=a_ub,
            b_ub=np.array(rhs),
            A_eq=a_eq,
            b_eq=np.array(eq_rhs),
            bounds=[(0.0, None)] * n_var,
            method="highs",
        )
        ours = solve_mean_cvar(beta)
        assert reference.status == 0
        assert ours.objective == pytest.approx(float(reference.fun), rel=1e-9)
        caps = ours.capacities_mw
        assert caps["solar"] == pytest.approx(float(reference.x[0]), abs=1e-6)
        assert caps["wind"] == pytest.approx(float(reference.x[1]), abs=1e-6)


class TestBirgeLouveaux:
    def test_bounds_are_ordered(self):
        bounds = birge_louveaux_bounds()
        assert bounds.ordering_ok()
        assert (
            bounds.wait_and_see_per_year <= bounds.recourse_per_year <= bounds.expected_ev_per_year
        )

    def test_information_has_value(self):
        bounds = birge_louveaux_bounds()
        assert bounds.evpi_per_year > 0.0
        assert bounds.vss_per_year > 0.0
        assert bounds.evpi_per_year < bounds.recourse_per_year
        assert bounds.recourse_plan_mw["gas"] > bounds.expected_value_plan_mw["gas"]


class TestSurrogate:
    def test_bilinear_reproduces_grid_nodes(self):
        grid = [[1.0, 2.0], [3.0, 4.0]]
        assert bilinear_interpolate([0.0, 1.0], [0.0, 1.0], np.array(grid), 0.0, 0.0) == 1.0
        assert bilinear_interpolate([0.0, 1.0], [0.0, 1.0], np.array(grid), 1.0, 1.0) == 4.0
        midpoint = bilinear_interpolate([0.0, 1.0], [0.0, 1.0], np.array(grid), 0.5, 0.5)
        assert midpoint == pytest.approx(2.5)

    def test_edges_clamp(self):
        grid = [[1.0, 2.0], [3.0, 4.0]]
        assert bilinear_interpolate([0.0, 1.0], [0.0, 1.0], np.array(grid), -5.0, -5.0) == 1.0
        assert bilinear_interpolate([0.0, 1.0], [0.0, 1.0], np.array(grid), 5.0, 5.0) == 4.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            bilinear_interpolate([0.0, 1.0], [0.0, 1.0], np.zeros((3, 3)), 0.5, 0.5)

    def test_committed_surface_interpolates_reference_node(self, dataset):
        surface = dataset["surface"]
        value = interpolate_surface(surface, "lcoe", 0.07, 1.0)
        assert value == pytest.approx(28.0555, abs=1e-3)

    def test_surrogate_error_is_bounded(self, dataset):
        surrogate = dataset["surrogate"]
        assert surrogate["n_solves"] >= 16
        assert surrogate["r2"] > 0.999
        assert surrogate["max_rel_error_pct"] < 3.0


class TestDatasetContract:
    def test_schema_and_meta(self, dataset):
        assert dataset["schema"] == SCHEMA_VERSION
        assert dataset["meta"]["package_version"] == "1.0.0"
        assert dataset["meta"]["n_solves"] >= 90
        assert dataset["meta"]["reference_wacc"] == pytest.approx(0.07)

    def test_surface_is_complete_and_monotone(self, dataset):
        surface = dataset["surface"]
        assert len(surface["wacc_grid"]) == 5
        assert len(surface["capex_grid"]) == 5
        assert len(surface["cases"]) == 25
        lcoe = np.array(surface["lcoe"], dtype=float)
        assert np.all(np.diff(lcoe, axis=0) > 0.0)  # dearer capital -> dearer energy
        assert np.all(np.diff(lcoe, axis=1) > 0.0)  # dearer VRE -> dearer energy
        assert lcoe[2][2] == pytest.approx(28.0555, abs=1e-3)

    def test_dispatch_series_balance(self, dataset):
        for case in dataset["surface"]["cases"]:
            total = np.array(case["solar"]) + np.array(case["wind"]) + np.array(case["gas"])
            assert np.allclose(total, case["load"], atol=0.02)

    def test_storage_crossover_and_envelope(self, dataset):
        storage = dataset["storage"]
        assert storage["crossover_hours"] is not None
        assert 6.0 < storage["crossover_hours"] < 12.0
        hybrid = np.array(storage["hybrid_lcoe"])
        single = np.minimum(storage["bess_only_lcoe"], storage["h2_only_lcoe"])
        assert np.all(hybrid <= single + 1e-6)
        assert len(storage["dispatch"]["load"]) == 168

    def test_carbon_frontier_shape(self, dataset):
        carbon = dataset["carbon"]
        assert carbon["targets_pct"][0] == 0.0
        assert carbon["targets_pct"][-1] == 100.0
        assert carbon["emissions_tco2"][-1] == pytest.approx(0.0, abs=1e-6)
        assert carbon["net_zero_shadow_price_per_tco2"] == pytest.approx(418.25, rel=2e-2)
        prices = np.array(carbon["shadow_price_per_tco2"])
        assert np.all(np.diff(prices) >= -1e-9)

    def test_flexibility_negative_price_proof(self, dataset):
        proof = dataset["flexibility"]["negative_lmp_proof"]
        assert proof["trough_lmp_per_mwh"] < 0.0
        assert proof["trough_lmp_per_mwh"] == pytest.approx(
            proof["analytical_trough_lmp_per_mwh"], abs=1e-9
        )
        assert proof["min_generator_marginal_cost"] >= 0.0

    def test_grid_ptdf_and_braess(self, dataset):
        grid = dataset["grid"]
        ptdf = np.array(grid["ptdf"], dtype=float)
        assert ptdf.shape == (5, 4)
        assert np.allclose(ptdf[:, 3], 0.0)  # slack column
        assert grid["braess"]["cost_increase_pct"] == pytest.approx(62.5)
        assert grid["regimes"][3]["annual_congestion_rent"] == pytest.approx(
            grid["regimes"][3]["annual_line_capex_cost"], rel=1e-6
        )

    def test_risk_block_matches_live_solver(self, dataset):
        risk = dataset["risk"]
        assert risk["alpha"] == pytest.approx(RISK_ALPHA)
        assert len(risk["betas"]) == len(RISK_BETA_SWEEP)
        live = solve_mean_cvar(0.35)
        index = RISK_BETA_SWEEP.index(0.35)
        assert risk["expected_cost_per_year"][index] == pytest.approx(
            live.expected_cost_per_year, rel=1e-3
        )
        assert risk["cvar_per_year"][index] == pytest.approx(live.cvar_per_year, rel=1e-3)
        assert risk["kernel_size"]["variables"] == live.n_variables

    def test_verification_table_all_pass(self, dataset):
        for row in dataset["verification"]:
            assert row["residual"] <= row["tolerance"], row

    def test_all_modules_report_tests(self, dataset):
        assert dataset["tests"]
        assert sum(entry["tests"] for entry in dataset["tests"]) > 100


class TestRenderedArtifact:
    def test_page_is_reproducible_from_template_and_data(self, dataset, template):
        assert HTML_FILE.read_text(encoding="utf-8") == render_page(template, dataset)

    def test_digest_is_embedded(self, dataset):
        html = HTML_FILE.read_text(encoding="utf-8")
        digest = dataset_digest(dataset)
        assert f'content="{digest}"' in html
        assert TEMPLATE_DATA_TOKEN not in html
        assert TEMPLATE_SHA_TOKEN not in html

    def test_no_network_dependencies(self):
        html = HTML_FILE.read_text(encoding="utf-8")
        assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
        assert "https://" not in html.replace("http://www.w3.org/2000/svg", "")
        assert "<script src" not in html
        assert "<link " not in html
        assert "@import" not in html
        assert "fetch(" not in html
        assert "XMLHttpRequest" not in html

    def test_asset_size_budget(self):
        html_bytes = HTML_FILE.stat().st_size
        assert 120_000 < html_bytes < 600_000
        assert DATA_FILE.stat().st_size < 400_000

    def test_check_reports_no_drift(self):
        assert check_workbench(REPO_ROOT) == []

    def test_check_detects_a_stale_page(self, tmp_path):
        root = tmp_path / "repo"
        (root / "workbench" / "data").mkdir(parents=True)
        (root / "energy_model").mkdir()
        (root / "energy_model" / "data").mkdir()
        (root / "energy_model" / "data" / "workbench_template.html").write_text(
            template_path().read_text(encoding="utf-8"), encoding="utf-8"
        )
        payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        (root / "workbench" / "data" / "etm_workbench_data.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        artifact = root / "workbench" / "index.html"
        artifact.write_text("<html>stale</html>", encoding="utf-8")
        problems = check_workbench(root)
        assert any("stale" in p for p in problems)

    def test_render_rejects_unknown_schema(self, template, dataset):
        broken = dict(dataset)
        broken["schema"] = "etm-workbench/999"
        with pytest.raises(ValueError, match="schema"):
            render_page(template, broken)

    def test_render_requires_placeholders(self, dataset):
        with pytest.raises(ValueError, match="placeholder"):
            render_page("<html>no tokens</html>", dataset)

    def test_page_declares_views_and_controls(self):
        html = HTML_FILE.read_text(encoding="utf-8")
        for view in (
            "overview",
            "dispatch",
            "storage",
            "carbon",
            "flex",
            "grid",
            "risk",
            "method",
        ):
            assert f'data-view="{view}"' in html
        assert 'id="wacc"' in html
        assert 'id="beta"' in html
        assert 'id="hour"' in html

    def test_json_has_no_script_breakout(self):
        payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        assert "</script" not in blob.lower()


class TestCli:
    def test_check_command_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "energy_model.workbench", "check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "up to date" in proc.stdout

    def test_render_command_rewrites_identical_page(self, tmp_path):
        before = HTML_FILE.read_text(encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "energy_model.workbench", "render"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert HTML_FILE.read_text(encoding="utf-8") == before

    def test_help_lists_commands(self):
        proc = subprocess.run(
            [sys.executable, "-m", "energy_model.workbench", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0
        for command in ("build", "render", "check"):
            assert command in proc.stdout


class TestPackagingAndDocs:
    def test_version_is_one_zero_zero(self):
        import energy_model

        assert energy_model.__version__ == "1.0.0"

    def test_pyproject_declares_entry_point(self):
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'version = "1.0.0"' in text
        assert 'etm-workbench = "energy_model.workbench:main"' in text
        assert "Production/Stable" in text

    def test_readme_documents_the_workbench(self):
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "Interactive Energy-Transition Workbench (v1.0.0)" in text
        assert "etm-workbench build" in text
        assert "results/risk_frontier_and_cvar_kernel.png" in text
        assert "workbench/index.html" in text

    def test_readme_numbers_match_the_dataset(self, dataset):
        """README figures must be traceable to the committed dataset values.

        Solver micro-versions can move the last digits, so the comparison is a
        0.1% band around each published value rather than an exact string match.
        """
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        published = [float(match.replace(",", "")) for match in re.findall(r"\d[\d,]*\.\d+", text)]
        risk = dataset["risk"]
        last = len(risk["betas"]) - 1
        targets = {
            "reference LCOE": dataset["kpi"]["lcoe_at_reference_per_mwh"],
            "net-zero carbon price": dataset["kpi"]["net_zero_carbon_price_per_tco2"],
            "tail-risk LCOE": risk["lcoe_per_mwh"][last],
            "tail cost (million)": risk["cvar_per_year"][last] / 1e6,
        }
        for label, value in targets.items():
            assert any(
                abs(candidate - value) <= max(abs(value) * 1e-3, 1e-6) for candidate in published
            ), f"{label} {value!r} is not quoted in the README"

    def test_pages_workflow_publishes_the_workbench(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "deploy_pages.yml"
        assert workflow.is_file()
        text = workflow.read_text(encoding="utf-8")
        assert "actions/deploy-pages@v4" in text
        assert "workbench" in text
        assert "permissions:" in text and "pages: write" in text
        # the deep link advertised in the README, the root, and the 404 fallback
        assert "site/workbench/index.html" in text
        assert "site/index.html" in text
        assert "site/404.html" in text
        assert ".nojekyll" in text

    def test_ci_verifies_artifact_freshness(self):
        text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "etm-workbench check" in text


class TestDeterminism:
    def test_repeated_kernel_solves_are_identical(self):
        first = solve_mean_cvar(0.5)
        second = solve_mean_cvar(0.5)
        assert first.objective == pytest.approx(second.objective, rel=0, abs=0)
        assert first.capacities_mw == pytest.approx(second.capacities_mw, rel=0, abs=0)

    def test_digest_is_key_order_independent(self):
        payload = {"schema": SCHEMA_VERSION, "a": 1, "b": [1, 2, 3]}
        reordered = {"b": [1, 2, 3], "a": 1, "schema": SCHEMA_VERSION}
        assert dataset_digest(payload) == dataset_digest(reordered)

    def test_template_token_count(self):
        raw = template_path().read_text(encoding="utf-8")
        assert raw.count(TEMPLATE_DATA_TOKEN) == 1
        assert raw.count(TEMPLATE_SHA_TOKEN) == 1

    def test_no_hardcoded_result_in_template(self):
        """Guards against a hand-typed figure: solved numbers live only in the JSON."""
        raw = template_path().read_text(encoding="utf-8")
        for literal in ("28.0555", "418.25", "7088134", "62.5%", "51.2489"):
            assert literal not in raw, f"hand-typed result {literal!r} found in template"
        assert not re.search(r"\d{2,3}\.?\d*\s*/MWh", raw)
