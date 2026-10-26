"""Multi-duration storage (Li-Ion BESS + Green H2 P2X) & Dunkelflaute tests (v0.4.0)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from energy_model import (
    BatteryStorageSpec,
    HydrogenP2XSpec,
    build_network,
    build_storage_network,
    compare_storage_adequacy_regimes,
    format_storage_report,
    generate_dunkelflaute_profiles,
    run_storage_scenario,
    solve_storage_linprog_crosscheck,
    sweep_dunkelflaute_duration,
    validate_profiles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestStorageSpecificationsAndProfiles:
    def test_default_bess_and_hydrogen_efficiency_properties(self):
        bess = BatteryStorageSpec()
        h2 = HydrogenP2XSpec()
        assert bess.round_trip_efficiency == pytest.approx(0.95 * 0.95, rel=1e-12)
        assert bess.energy_equivalent_cost_per_mwh_yr == pytest.approx(30_000.0 / 4.0, rel=1e-12)
        assert h2.round_trip_efficiency == pytest.approx(0.68 * 0.50, rel=1e-12)
        assert h2.turbine_link_capital_cost_per_mw_h2_yr == pytest.approx(
            0.50 * 30_000.0, rel=1e-12
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_hours": 0.0},
            {"max_hours": -2.0},
            {"efficiency_store": 0.0},
            {"efficiency_dispatch": 1.05},
            {"standing_loss": -0.01},
            {"standing_loss": 1.0},
            {"capital_cost_per_mw_yr": -100.0},
            {"marginal_cost_per_mwh": -1.0},
            {"max_hours": True},
        ],
    )
    def test_invalid_bess_spec_rejected(self, kwargs):
        with pytest.raises(ValueError):
            BatteryStorageSpec(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"electrolyzer_efficiency": 0.0},
            {"turbine_efficiency": 1.1},
            {"standing_loss": -0.001},
            {"standing_loss": 1.0},
            {"electrolyzer_capital_cost_per_mw_e_yr": -10.0},
            {"turbine_capital_cost_per_mw_e_yr": -10.0},
            {"store_capital_cost_per_mwh_h2_yr": -1.0},
        ],
    )
    def test_invalid_hydrogen_spec_rejected(self, kwargs):
        with pytest.raises(ValueError):
            HydrogenP2XSpec(**kwargs)

    def test_dunkelflaute_profile_chronology_and_lull_signature(self):
        prof = generate_dunkelflaute_profiles(n_days=7, lull_start_hour=72, lull_duration_hours=48)
        assert len(prof) == 168
        validated = validate_profiles(prof)
        assert validated.equals(prof)

        in_lull = prof.iloc[72:120]
        outside = prof.drop(in_lull.index)
        assert in_lull["wind_cf"].mean() < 0.08
        assert outside["wind_cf"].mean() > 0.30
        assert in_lull["solar_cf"].max() <= 0.085
        assert outside["solar_cf"].max() > 0.80
        assert in_lull["load_mw"].mean() > outside["load_mw"].mean() + 20.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_days": 0},
            {"n_days": True},
            {"lull_start_hour": -1},
            {"lull_duration_hours": -5},
            {"n_days": 3, "lull_start_hour": 60, "lull_duration_hours": 24},
        ],
    )
    def test_invalid_dunkelflaute_parameters_rejected(self, kwargs):
        with pytest.raises(ValueError):
            generate_dunkelflaute_profiles(**kwargs)

    def test_baseline_network_remains_strictly_storage_free(self, shipped_profiles):
        base_net = build_network(shipped_profiles)
        assert len(base_net.storage_units) == 0
        assert len(base_net.stores) == 0
        assert len(base_net.links) == 0

        storage_net = build_storage_network(
            generate_dunkelflaute_profiles(n_days=2, lull_start_hour=12, lull_duration_hours=12)
        )
        assert len(storage_net.storage_units) == 1
        assert len(storage_net.stores) == 1
        assert len(storage_net.links) == 2


class TestChronologicalConservationAndKKTArbitrage:
    def test_hybrid_dunkelflaute_power_balance_cyclic_soc_and_kkt_duality(self):
        prof = generate_dunkelflaute_profiles()
        res = run_storage_scenario(prof, include_bess=True, include_h2=True, max_gas_mw=0.0)

        assert res.status == "ok" and res.termination_condition == "optimal"
        assert res.power_balance_max_residual_mw < 1e-8
        assert res.soc_cyclic_residual_mwh < 1e-8
        assert abs(res.strong_duality_gap) < 1e-4
        assert abs(res.bess_net_profit_per_year) < 1e-4
        assert abs(res.h2_net_profit_per_year) < 1e-4
        assert res.max_zero_profit_residual < 1e-4

        # Both short-duration BESS and long-duration H2 are co-optimally built
        assert res.bess_p_nom_mw > 20.0
        assert res.h2_electrolyzer_mw_e > 100.0
        assert res.h2_turbine_mw_e > 100.0
        assert res.h2_store_mwh_h2 > 8_000.0

        # Timescale separation: BESS cycles far more frequently; H2 holds >20x electric MWh
        assert res.bess_equiv_cycles_per_year > 3.0 * res.h2_equiv_cycles_per_year
        assert res.h2_store_mwh_e_equiv > 20.0 * res.bess_e_nom_mwh

    def test_lossless_hydrogen_bus_shadow_price_is_constant_on_interior_soc(self):
        prof = generate_dunkelflaute_profiles()
        res = run_storage_scenario(
            prof,
            include_bess=True,
            include_h2=True,
            max_gas_mw=0.0,
            h2_spec=HydrogenP2XSpec(standing_loss=0.0),
        )
        e_h2 = res.h2_soc_mwh_h2.to_numpy(dtype=float)
        e_max = res.h2_store_mwh_h2
        lmp_h2 = res.lmp_h2_per_mwh.to_numpy(dtype=float)

        # For every hour t where 0 < e_h2[t] < e_max, KKT stationarity wrt e_h2[t]
        # requires lambda_h2[t] == lambda_h2[(t + 1) mod T]
        interior = (e_h2 > 1e-3) & (e_h2 < e_max - 1e-3)
        assert int(interior.sum()) > 100
        next_lmp_h2 = np.roll(lmp_h2, -1)
        assert np.max(np.abs(lmp_h2[interior] - next_lmp_h2[interior])) < 1e-7

    @pytest.mark.parametrize(
        "inc_bess, inc_h2, standing_h2",
        [
            (True, False, 0.0),
            (False, True, 0.0),
            (True, True, 0.00005),
        ],
    )
    def test_independent_scipy_linprog_primal_dual_crosscheck(
        self, inc_bess: bool, inc_h2: bool, standing_h2: float
    ):
        prof = generate_dunkelflaute_profiles(n_days=2, lull_start_hour=18, lull_duration_hours=18)
        chk = solve_storage_linprog_crosscheck(
            prof,
            include_bess=inc_bess,
            include_h2=inc_h2,
            h2_spec=HydrogenP2XSpec(standing_loss=standing_h2),
        )
        assert chk.rel_objective_diff < 1e-10
        assert chk.max_capacity_diff < 1e-7
        assert chk.max_lmp_elec_diff < 1e-7


class TestAdequacyRegimesAndCrossoverTheorem:
    def test_adequacy_regimes_cost_hierarchy_across_48h_dunkelflaute(self):
        df = compare_storage_adequacy_regimes()
        lcoe_vre = df.loc["vre_only_zero_gas", "system_lcoe_per_mwh"]
        lcoe_bess = df.loc["vre_bess_zero_gas", "system_lcoe_per_mwh"]
        lcoe_h2 = df.loc["vre_h2_zero_gas", "system_lcoe_per_mwh"]
        lcoe_hyb = df.loc["vre_hybrid_bess_h2_zero_gas", "system_lcoe_per_mwh"]
        lcoe_gas = df.loc["vre_hybrid_with_unconstrained_gas", "system_lcoe_per_mwh"]

        assert lcoe_vre > 200.0
        assert lcoe_hyb < lcoe_h2 < lcoe_bess < lcoe_vre
        assert lcoe_gas < lcoe_hyb

    def test_dunkelflaute_duration_crossover_and_hybrid_lower_envelope(self):
        sweep = sweep_dunkelflaute_duration(lull_durations=(0, 6, 12, 24, 48))
        b_only = sweep["bess_only_lcoe"].to_numpy(dtype=float)
        h_only = sweep["h2_only_lcoe"].to_numpy(dtype=float)
        hyb = sweep["hybrid_lcoe"].to_numpy(dtype=float)

        # At 0 h and 6 h diurnal/short lull, high-RTE BESS strictly beats H2-only
        assert b_only[0] < h_only[0]
        assert b_only[1] < h_only[1]
        # Beyond ~8 h (12 h, 24 h, 48 h), low-energy-CapEx H2 strictly beats BESS-only
        assert h_only[2] < b_only[2]
        assert h_only[3] < b_only[3]
        assert h_only[4] < b_only[4]
        # Hybrid portfolio forms the Pareto lower envelope everywhere
        assert np.all(hyb <= np.minimum(b_only, h_only) + 1e-7)
        # Monotonicity wrt Dunkelflaute duration
        assert np.all(np.diff(hyb) > 0.0)

    def test_cli_dunkelflaute_report_and_storage_plot(self, tmp_path):
        out_png = tmp_path / "m2_storage.png"
        proc = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--dunkelflaute-report",
                "--save-storage-plot",
                str(out_png),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Multi-Duration Storage & Dunkelflaute" in proc.stdout
        assert "Green H2 Cavern Store" in proc.stdout
        assert out_png.is_file() and out_png.stat().st_size > 100_000

        rep_txt = format_storage_report(
            run_storage_scenario(
                generate_dunkelflaute_profiles(n_days=2, lull_start_hour=12, lull_duration_hours=12)
            )
        )
        assert "Strong-duality gap" in rep_txt

        repo_fig = REPO_ROOT / "results" / "multiduration_storage_dunkelflaute_dispatch.png"
        assert repo_fig.is_file() and repo_fig.stat().st_size > 100_000
