"""Governance, packaging, WACC/CRF cash-flow, and KKT nodal LMP duality tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import energy_model
from energy_model import (
    DEFAULT_TECHNOLOGIES,
    REFERENCE_WACC,
    analyze_market_duality_and_lcoe,
    annualize_overnight_capex,
    build_wacc_technologies,
    capital_recovery_factor,
    default_financial_specs,
    plot_baseline_dispatch_and_lmp_duality,
    run_scenario,
    solve_primal_dual_linprog_crosscheck,
    sweep_wacc_lcoe_and_mix,
)

from .conftest import make_profiles, technologies

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestGovernanceAndPackaging:
    def test_community_governance_files_and_py_typed(self):
        for rel in (
            "LICENSE",
            "CITATION.cff",
            "SECURITY.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "pyproject.toml",
            "energy_model/py.typed",
        ):
            path = REPO_ROOT / rel
            assert path.is_file(), f"Missing {rel}"
            assert path.stat().st_size > 20

        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        expected_version = pyproject["project"]["version"]
        assert energy_model.__version__ == expected_version
        cff = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        assert f"version: {expected_version}" in cff

    def test_readme_honest_scope_and_figure_reference(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        lower = readme.lower()
        assert "results/baseline_dispatch_and_lmp_duality.png" in readme
        assert "results/multiduration_storage_dunkelflaute_dispatch.png" in readme
        assert (REPO_ROOT / "results" / "baseline_dispatch_and_lmp_duality.png").is_file()
        assert (REPO_ROOT / "results" / "multiduration_storage_dunkelflaute_dispatch.png").is_file()
        assert "politecnico di milano" in lower
        assert "synthetic" in lower
        assert "engineering certification" in lower
        assert readme.count(chr(96) * 3) % 2 == 0

    def test_built_wheel_contains_py_typed_and_packaged_csv(self, tmp_path):
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--wheel",
                    "--outdir",
                    str(tmp_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert proc.returncode == 0, proc.stderr
            ver = energy_model.__version__
            wheels = list(tmp_path.glob(f"energy_transition_model-{ver}-*.whl"))
            assert len(wheels) == 1
            with zipfile.ZipFile(wheels[0]) as zf:
                names = set(zf.namelist())
            assert "energy_model/py.typed" in names
            assert "energy_model/data/profiles_synthetic_24h.csv" in names
            assert not any(n.startswith("tests/") for n in names)
        finally:
            shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)


class TestDiscountedCashFlowAndWACC:
    @pytest.mark.parametrize(
        "wacc, lifetime_yr",
        [
            (0.00, 25),
            (0.03, 20),
            (0.07, 25),
            (0.10, 30),
            (0.15, 15),
        ],
    )
    def test_crf_zero_discount_limit_and_present_value_annuity_identity(
        self, wacc: float, lifetime_yr: int
    ):
        overnight = 750_000.0
        crf = capital_recovery_factor(wacc, lifetime_yr)
        annual_payment = annualize_overnight_capex(overnight, wacc, lifetime_yr)

        if wacc == 0.0:
            assert crf == pytest.approx(1.0 / lifetime_yr, rel=1e-12)

        years = np.arange(1, lifetime_yr + 1, dtype=float)
        pv = float(np.sum(annual_payment / ((1.0 + wacc) ** years)))
        assert pv == pytest.approx(overnight, rel=1e-11)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"overnight_capex_per_mw": -1.0, "wacc": 0.05, "lifetime_years": 20.0},
            {"overnight_capex_per_mw": 100.0, "wacc": -0.01, "lifetime_years": 20.0},
            {"overnight_capex_per_mw": 100.0, "wacc": 1.05, "lifetime_years": 20.0},
            {"overnight_capex_per_mw": 100.0, "wacc": 0.05, "lifetime_years": 0.0},
            {"overnight_capex_per_mw": 100.0, "wacc": True, "lifetime_years": 20.0},
            {"overnight_capex_per_mw": float("nan"), "wacc": 0.05, "lifetime_years": 20.0},
            {
                "overnight_capex_per_mw": 100.0,
                "wacc": 0.05,
                "lifetime_years": 20.0,
                "fixed_om_per_mw_yr": -5.0,
            },
        ],
    )
    def test_crf_and_annualize_parameter_validation(self, kwargs):
        with pytest.raises(ValueError):
            annualize_overnight_capex(**kwargs)

    def test_wacc_calibration_reproduces_default_technologies_at_reference_wacc(self):
        specs = default_financial_specs(REFERENCE_WACC)
        calibrated = build_wacc_technologies(REFERENCE_WACC, financial_specs=specs)
        for name, expected in DEFAULT_TECHNOLOGIES.items():
            assert calibrated[name]["capital_cost"] == pytest.approx(
                expected["capital_cost"], rel=1e-12
            )
            assert calibrated[name]["marginal_cost"] == pytest.approx(
                expected["marginal_cost"], abs=1e-12
            )
            assert calibrated[name]["profile"] == expected["profile"]

    def test_wacc_comparative_statics_monotonicity(self, shipped_profiles):
        df = sweep_wacc_lcoe_and_mix(
            shipped_profiles,
            wacc_values=(0.00, 0.04, 0.07, 0.10, 0.15),
        )
        assert np.all(np.diff(df["system_lcoe_per_mwh"].to_numpy()) > 0.0)
        assert np.all(np.diff(df["wind_mw"].to_numpy()) <= 1e-6)
        assert np.all(np.diff(df["vre_share_pct"].to_numpy()) <= 1e-6)
        assert np.all(np.diff(df["gas_mw"].to_numpy()) >= -1e-6)


class TestKKTNodalDualityAndMarketEquilibrium:
    def test_kkt_strong_duality_and_zero_profit_equilibrium(self, shipped_profiles):
        res = run_scenario(shipped_profiles)
        dual = analyze_market_duality_and_lcoe(res, profiles=shipped_profiles)

        assert abs(dual.strong_duality_gap) < 1e-4
        assert dual.annual_demand_payment == pytest.approx(res.total_cost_per_year, rel=1e-9)
        assert dual.load_weighted_lmp_per_mwh == pytest.approx(dual.system_lcoe_per_mwh, rel=1e-9)
        assert dual.max_zero_profit_residual < 1e-4

        tbl = dual.technology_table
        for tech in ("Solar_PV", "Wind_Turbine", "Gas_Peaker"):
            assert tbl.loc[tech, "p_nom_mw"] > 1.0
            assert tbl.loc[tech, "annual_lmp_revenue"] == pytest.approx(
                tbl.loc[tech, "annual_total_cost"], rel=1e-8
            )
            assert tbl.loc[tech, "lcoe_per_mwh"] == pytest.approx(
                tbl.loc[tech, "market_value_per_mwh"], rel=1e-8
            )

    def test_scarcity_hour_lmp_matches_analytical_peaker_capex_plus_fuel(self, shipped_profiles):
        res = run_scenario(shipped_profiles)
        dual = analyze_market_duality_and_lcoe(res, profiles=shipped_profiles)

        expected_scarcity_lmp = 65.0 + 20_000.0 / 365.0
        ts_18 = pd.Timestamp("2026-01-01 18:00:00")
        assert res.lmp_per_mwh.loc[ts_18] == pytest.approx(expected_scarcity_lmp, rel=1e-9)
        assert dual.peak_lmp_per_mwh == pytest.approx(expected_scarcity_lmp, rel=1e-9)
        assert dual.scarcity_hours == 1
        assert dual.zero_price_hours == 13

        tbl = dual.technology_table
        assert tbl.loc["Gas_Peaker", "value_factor"] > 2.0
        assert tbl.loc["Solar_PV", "value_factor"] < 1.0
        assert tbl.loc["Wind_Turbine", "value_factor"] < 1.0

    def test_primal_dual_crosscheck_matches_independent_scipy_linprog(self, shipped_profiles):
        chk_ext = solve_primal_dual_linprog_crosscheck(shipped_profiles)
        assert chk_ext.max_p_nom_diff_mw < 1e-8
        assert chk_ext.max_lmp_diff_per_mwh < 1e-8
        assert chk_ext.rel_objective_diff < 1e-10

        fixed_techs = technologies(
            "Solar_PV",
            "Gas_Peaker",
            Solar_PV={"p_nom_extendable": False, "p_nom": 90.0},
        )
        chk_fix = solve_primal_dual_linprog_crosscheck(
            make_profiles(load=150.0, solar=[0.0] * 6 + [0.6] * 12 + [0.0] * 6),
            technologies=fixed_techs,
        )
        assert chk_fix.max_p_nom_diff_mw < 1e-8
        assert chk_fix.max_lmp_diff_per_mwh < 1e-8
        assert chk_fix.rel_objective_diff < 1e-10

    def test_cli_duality_report_and_plot_generation(self, tmp_path):
        out_png = tmp_path / "m1_plot.png"
        proc = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--duality-report",
                "--save-plot",
                str(out_png),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        assert "KKT Nodal Shadow-Price (LMP)" in proc.stdout
        assert "System LCOE (currency/MWh)" in proc.stdout
        assert out_png.is_file() and out_png.stat().st_size > 80_000

        direct = plot_baseline_dispatch_and_lmp_duality(output_path=tmp_path / "direct.png")
        assert direct.is_file() and direct.stat().st_size > 80_000
