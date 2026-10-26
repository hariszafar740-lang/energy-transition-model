"""4-zone meshed DC-OPF, PTDF/B-theta, congestion rent & AC-TEP tests (v0.7.0)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from energy_model import (
    DEFAULT_REFERENCE_ZONE,
    ZONES_4BUS,
    TransmissionCorridorSpec,
    ZonalTechSpec,
    compare_transmission_regimes,
    compute_ptdf_matrix,
    default_transmission_corridors,
    default_zonal_generator_specs,
    format_transmission_report,
    make_four_zone_profiles,
    plot_zonal_dcopf_congestion_and_tep,
    run_transmission_scenario,
    solve_dcopf_btheta_linprog_crosscheck,
    sweep_transmission_capacity_scale,
    validate_zonal_profiles,
    verify_braess_paradox_dcopf,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def zonal_profiles():
    return make_four_zone_profiles()


@pytest.fixture(scope="module")
def trans_regime_table(zonal_profiles):
    return compare_transmission_regimes(zonal_profiles)


class TestZonalSpecsProfilesAndPTDF:
    def test_default_zonal_specs_and_ptdf_properties(self):
        gens = default_zonal_generator_specs()
        corr = default_transmission_corridors()
        assert len(gens) == 8
        assert len(corr) == 5

        ptdf = compute_ptdf_matrix(corr)
        assert ptdf.shape == (5, 4)
        assert np.allclose(ptdf[DEFAULT_REFERENCE_ZONE].to_numpy(), 0.0, atol=1e-15)
        assert float(ptdf.loc["L2_North_East", "North_Wind"]) > 0.45
        assert float(ptdf.loc["L4_South_East", "South_Hub"]) > 0.65

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bus0": "North_Wind", "bus1": "North_Wind"},
            {"length_km": 0.0},
            {"x_ohm": -5.0},
            {"s_nom_mw": 0.0},
            {"capital_cost_per_mw_yr": -100.0},
        ],
    )
    def test_invalid_corridor_spec_rejected(self, kwargs):
        base = {
            "name": "BadLine",
            "bus0": "North_Wind",
            "bus1": "East_Metro",
            "length_km": 100.0,
            "x_ohm": 25.0,
            "s_nom_mw": 80.0,
            "capital_cost_per_mw_yr": 9_000.0,
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            TransmissionCorridorSpec(**base)

    def test_disconnected_graph_ptdf_rejected(self):
        # 4 zones but only 1 edge -> disconnected graph -> singular B_bus
        corr = {
            "OnlyOne": TransmissionCorridorSpec(
                name="OnlyOne",
                bus0="North_Wind",
                bus1="West_Solar",
                length_km=100.0,
                x_ohm=25.0,
                s_nom_mw=100.0,
                capital_cost_per_mw_yr=8_000.0,
            )
        }
        with pytest.raises(ValueError, match="disconnected"):
            compute_ptdf_matrix(corr, zones=ZONES_4BUS)

    def test_zonal_profile_validation_and_determinism(self, zonal_profiles):
        again = make_four_zone_profiles(days=2)
        assert len(zonal_profiles) == 48
        assert zonal_profiles.equals(again)

        bad_neg = zonal_profiles.copy()
        bad_neg.iloc[0, 0] = -5.0
        with pytest.raises(ValueError, match="negative"):
            validate_zonal_profiles(bad_neg)

        bad_cf = zonal_profiles.copy()
        bad_cf["cf_wind_north"] = 1.05
        with pytest.raises(ValueError, match="within \\[0, 1\\]"):
            validate_zonal_profiles(bad_cf)

        with pytest.raises(ValueError):
            ZonalTechSpec(
                name="",
                zone="North_Wind",
                carrier="renewable",
                capital_cost_per_mw_yr=1000.0,
                marginal_cost_per_mwh=0.0,
            )


class TestKVLPTDFCongestionRentAndTEP:
    @pytest.mark.parametrize("extendable", [False, True])
    def test_ptdf_flow_lmp_decomposition_and_merchandising_surplus(
        self, zonal_profiles, extendable: bool
    ):
        res = run_transmission_scenario(zonal_profiles, mode="ac_kvl", extendable_lines=extendable)
        assert res.nodal_balance_max_residual_mw < 1e-9
        assert res.kvl_cycle_max_residual_pu < 1e-12
        assert res.ptdf_flow_max_error_mw < 1e-10
        assert res.ptdf_lmp_decomposition_max_error < 1e-10
        assert res.merchandising_surplus_identity_max_gap < 1e-8
        assert res.generator_revenue_adequacy_gap < 1e-6

        # Total hourly congestion rent is strictly non-negative at every snapshot
        assert np.all(res.hourly_congestion_rent.to_numpy() >= -1e-9)

        # Individual parallel AC corridors experience adverse counter-price KVL loop flow (< 0)
        assert float(res.corridor_merchandising_rent_per_h.min().min()) < -100.0

        if extendable:
            # Joskow-Tirole theorem: annual congestion rent == annualised TEP line CapEx
            assert res.joskow_tirole_rent_recovery_gap < 1e-6
            assert res.annual_congestion_rent == pytest.approx(
                res.annual_line_capex_cost, rel=1e-10
            )
            assert res.annual_load_payment == pytest.approx(res.total_annual_cost, rel=1e-10)

    def test_four_regime_hierarchy_and_kvl_loop_flow_penalty(self, trans_regime_table):
        r_cop, r_fix, r_hvdc, r_tep = [trans_regime_table.iloc[i] for i in range(4)]

        # Copper-Plate <= Co-Opt AC-TEP < Controllable HVDC < Congested Fixed AC
        assert float(r_cop["system_lcoe_per_mwh"]) < float(r_tep["system_lcoe_per_mwh"])
        assert float(r_tep["system_lcoe_per_mwh"]) < float(r_hvdc["system_lcoe_per_mwh"])
        assert float(r_hvdc["system_lcoe_per_mwh"]) < float(r_fix["system_lcoe_per_mwh"])

        # Co-Opt TEP unlocks >2x North Offshore Wind and slashes East Metro OCGT by >90%
        assert float(r_tep["north_wind_mw"]) > 2.0 * float(r_fix["north_wind_mw"])
        assert float(r_tep["east_ocgt_mw"]) < 0.10 * float(r_fix["east_ocgt_mw"])
        assert float(r_tep["mean_lmp_spread_per_mwh"]) < 0.10 * float(
            r_fix["mean_lmp_spread_per_mwh"]
        )

    def test_corridor_scale_monotonic_convergence_to_copper_plate(
        self, zonal_profiles, trans_regime_table
    ):
        sw = sweep_transmission_capacity_scale(zonal_profiles, scales=(0.50, 1.00, 1.50, 2.50))
        lcoes = sw["system_lcoe_per_mwh"].to_numpy(dtype=float)
        assert np.all(np.diff(lcoes) <= 1e-8)
        copper_lcoe = float(trans_regime_table.iloc[0]["system_lcoe_per_mwh"])
        assert lcoes[-1] == pytest.approx(copper_lcoe, rel=1e-9)

    def test_wheatstone_braess_paradox_ac_vs_hvdc_theorem(self):
        bp = verify_braess_paradox_dcopf()
        assert bp.base_transfer_mw == pytest.approx(120.0, abs=1e-8)
        assert bp.ac_5line_transfer_mw == pytest.approx(82.5, abs=1e-8)
        assert bp.hvdc_5line_transfer_mw == pytest.approx(160.0, abs=1e-8)
        assert bp.base_4line_cost_per_h == pytest.approx(4_800.0, abs=1e-8)
        assert bp.ac_5line_cost_per_h == pytest.approx(7_800.0, abs=1e-8)
        assert bp.hvdc_5line_cost_per_h == pytest.approx(1_600.0, abs=1e-8)
        assert bp.braess_cost_increase_pct == pytest.approx(62.5, abs=1e-8)


class TestSciPyBThetaCrossCheck:
    @pytest.mark.parametrize("extendable_lines", [False, True])
    def test_scipy_btheta_matches_pypsa_cycle_kvl_to_machine_precision(
        self, zonal_profiles, extendable_lines: bool
    ):
        chk = solve_dcopf_btheta_linprog_crosscheck(
            zonal_profiles, extendable_lines=extendable_lines
        )
        assert chk.rel_objective_diff < 1e-11
        assert chk.max_gen_p_nom_diff_mw < 1e-8
        assert chk.max_line_s_nom_diff_mw < 1e-8
        assert chk.max_lmp_diff_per_mwh < 1e-8
        assert chk.max_congestion_rent_diff_per_h < 1e-7
        assert chk.scipy_ptdf_flow_residual_mw < 1e-10


class TestCLIAndTransmissionFigure:
    def test_committed_transmission_figure_and_report(
        self, zonal_profiles, trans_regime_table, tmp_path
    ):
        committed = REPO_ROOT / "results" / "zonal_dcopf_congestion_and_tep.png"
        assert committed.is_file() and committed.stat().st_size > 120_000

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "results/zonal_dcopf_congestion_and_tep.png" in readme

        rep = format_transmission_report(trans_regime_table)
        assert "4_CoOptimized_AC_TEP" in rep
        assert "-0" not in rep

        out_png = plot_zonal_dcopf_congestion_and_tep(
            output_path=tmp_path / "zonal_tep.png",
            profiles=zonal_profiles,
            regime_df=trans_regime_table,
        )
        assert out_png.is_file() and out_png.stat().st_size > 120_000

    def test_cli_transmission_flags_and_invalid_scale_exit(self):
        ok_proc = subprocess.run(
            [sys.executable, "main.py", "--cooptimize-tep"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert ok_proc.returncode == 0, ok_proc.stderr
        assert "4-Zone DC-OPF & Transmission Diagnostics" in ok_proc.stdout

        err_proc = subprocess.run(
            [sys.executable, "main.py", "--grid-scale", "0.0"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert err_proc.returncode == 1
        assert "ERROR:" in err_proc.stderr
