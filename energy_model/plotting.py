"""Publication-grade visualisations for capacity expansion, dispatch, KKT LMPs & storage."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .economics import analyze_market_duality_and_lcoe, sweep_wacc_lcoe_and_mix
from .emissions import (
    sweep_carbon_tax_equivalents,
    sweep_co2_pareto_and_macc,
)
from .flexibility import (
    SpinningReservePolicy,
    compare_flexibility_regimes,
    make_duck_curve_profiles,
    run_flexibility_scenario,
)
from .model import DEFAULT_PROFILE_PATH, DEFAULT_TECHNOLOGIES, ScenarioResult, run_scenario
from .profiles import load_profiles, validate_profiles
from .risk_sensitivity import (
    EnvelopeConcavityVerification,
    SensitivitySurfaceResult,
    StochasticCVaRResult,
    StochasticValueMetrics,
    evaluate_stochastic_value_metrics,
    run_stochastic_cvar_scenario,
    sweep_cvar_risk_frontier,
    sweep_wacc_capex_surface,
    verify_envelope_and_concavity_theorems,
)
from .storage import (
    StorageScenarioResult,
    compare_storage_adequacy_regimes,
    generate_dunkelflaute_profiles,
    run_storage_scenario,
    sweep_dunkelflaute_duration,
)
from .transmission import (
    compare_transmission_regimes,
    make_four_zone_profiles,
    run_transmission_scenario,
)


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
        }
    )


def plot_baseline_dispatch_and_lmp_duality(
    output_path: str | Path = "results/baseline_dispatch_and_lmp_duality.png",
    profiles: pd.DataFrame | None = None,
    result: ScenarioResult | None = None,
) -> Path:
    """Render 4-panel representative-day dispatch, KKT LMP, zero-profit & WACC sensitivity."""
    prof = load_profiles(DEFAULT_PROFILE_PATH) if profiles is None else validate_profiles(profiles)
    res = run_scenario(prof) if result is None else result
    dual = analyze_market_duality_and_lcoe(res, profiles=prof, technologies=DEFAULT_TECHNOLOGIES)
    wacc_df = sweep_wacc_lcoe_and_mix(prof)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    hours = np.arange(len(prof), dtype=float)
    wind_p = res.dispatch_mw["Wind_Turbine"].to_numpy(dtype=float)
    solar_p = res.dispatch_mw["Solar_PV"].to_numpy(dtype=float)
    gas_p = res.dispatch_mw["Gas_Peaker"].to_numpy(dtype=float)
    load_p = prof["load_mw"].to_numpy(dtype=float)

    wind_avail = prof["wind_cf"].to_numpy(dtype=float) * float(res.p_nom_mw["Wind_Turbine"])
    solar_avail = prof["solar_cf"].to_numpy(dtype=float) * float(res.p_nom_mw["Solar_PV"])
    total_avail_vre = wind_avail + solar_avail
    curtailed_mw = np.maximum(0.0, total_avail_vre - (wind_p + solar_p))

    _apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), dpi=180)
    fig.suptitle(
        (
            "Energy Transition Capacity Expansion: "
            "24 h Dispatch, KKT Nodal LMP Duality & WACC Sensitivity"
        ),
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    ax0 = axes[0, 0]
    ax0.stackplot(
        hours,
        wind_p,
        solar_p,
        gas_p,
        labels=[
            f"Wind ({res.p_nom_mw['Wind_Turbine']:.1f} MW)",
            f"Solar PV ({res.p_nom_mw['Solar_PV']:.1f} MW)",
            f"Gas Peaker ({res.p_nom_mw['Gas_Peaker']:.1f} MW)",
        ],
        colors=["#0284c7", "#f59e0b", "#ef4444"],
        alpha=0.86,
    )
    ax0.fill_between(
        hours,
        load_p,
        load_p + curtailed_mw,
        color="#10b981",
        alpha=0.25,
        hatch="//",
        edgecolor="#059669",
        linewidth=0.8,
        label=f"Curtailed VRE ({res.curtailment_mwh_per_day.sum():.0f} MWh/d)",
    )
    ax0.plot(hours, load_p, color="#0f172a", lw=2.0, ls="--", label="Demand (150.0 MW)")
    ax0.set_title("(a) Optimal Hourly Dispatch Stack & Renewable Curtailment")
    ax0.set_xlabel("Representative-Day Hour (h)")
    ax0.set_ylabel("Power (MW)")
    ax0.set_xlim(0, 23)
    ax0.set_ylim(0, max(265.0, float((load_p + curtailed_mw).max()) * 1.24))
    ax0.set_xticks(range(0, 24, 3))
    ax0.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    ax1 = axes[0, 1]
    lmp_vals = res.lmp_per_mwh.to_numpy(dtype=float)
    ax1.step(hours, lmp_vals, where="mid", color="#7c3aed", lw=2.2, label=r"Nodal LMP λ_t")
    ax1.fill_between(hours, 0.0, lmp_vals, step="mid", color="#7c3aed", alpha=0.14)
    ax1.axhline(
        65.0,
        color="#ef4444",
        ls=":",
        lw=1.5,
        label="Gas Marginal Fuel Cost (65.0 /MWh)",
    )
    ax1.axhline(
        dual.system_lcoe_per_mwh,
        color="#059669",
        ls="--",
        lw=1.7,
        label=f"System LCOE = Load-Wtd LMP ({dual.system_lcoe_per_mwh:.2f} /MWh)",
    )
    peak_idx = int(np.argmax(lmp_vals))
    ax1.scatter([hours[peak_idx]], [lmp_vals[peak_idx]], color="#dc2626", zorder=5, s=36)
    ax1.annotate(
        (
            f"Scarcity Hour {peak_idx:02d}:00\n"
            f"λ_{{18}} = {lmp_vals[peak_idx]:.2f} /MWh\n"
            "(= 65 + 20,000/365)"
        ),
        xy=(hours[peak_idx], lmp_vals[peak_idx]),
        xytext=(hours[peak_idx] - 7.8, lmp_vals[peak_idx] - 18.0),
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.0},
    )
    ax1.set_title(r"(b) KKT Nodal Shadow Price λ_t & Scarcity-Rent Regimes")
    ax1.set_xlabel("Representative-Day Hour (h)")
    ax1.set_ylabel("Shadow Price (currency / MWh)")
    ax1.set_xlim(0, 23)
    ax1.set_ylim(-5.0, max(145.0, float(lmp_vals.max()) * 1.22))
    ax1.set_xticks(range(0, 24, 3))
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    ax2 = axes[1, 0]
    tbl = dual.technology_table
    tech_labels = ["Solar PV", "Wind Turbine", "Gas Peaker"]
    x_pos = np.arange(len(tech_labels), dtype=float)
    width = 0.27

    costs_m = tbl["annual_total_cost"].to_numpy(dtype=float) / 1e6
    revs_m = tbl["annual_lmp_revenue"].to_numpy(dtype=float) / 1e6

    ax2.bar(
        x_pos - 0.5 * width,
        costs_m,
        width=width,
        color="#334155",
        label="Annualized CapEx + OpEx (M/yr)",
    )
    ax2.bar(
        x_pos + 0.5 * width,
        revs_m,
        width=width,
        color="#10b981",
        label="Annual LMP Revenue (M/yr)",
    )
    for idx in range(len(x_pos)):
        lcoe_val = tbl["lcoe_per_mwh"].iloc[idx]
        vf_val = tbl["value_factor"].iloc[idx]
        top_y = max(costs_m[idx], revs_m[idx])
        ax2.text(
            x_pos[idx],
            top_y + 1.0,
            f"LCOE=MV={lcoe_val:.1f}/MWh\n(VF = {vf_val:.2f}x)",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#0f172a",
        )
    ax2.set_title("(c) KKT Zero-Profit Equilibrium: Annual Cost == LMP Revenue")
    ax2.set_ylabel("Annual Cash Flow (Million currency / yr)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tech_labels)
    ax2.set_ylim(0.0, max(revs_m.max() * 1.32, 32.0))
    ax2.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    ax3 = axes[1, 1]
    w_pct = wacc_df["wacc_pct"].to_numpy(dtype=float)
    ax3.plot(w_pct, wacc_df["wind_mw"], "o-", color="#0284c7", lw=2.0, label="Wind Capacity (MW)")
    ax3.plot(
        w_pct, wacc_df["solar_mw"], "s-", color="#f59e0b", lw=2.0, label="Solar PV Capacity (MW)"
    )
    ax3.plot(
        w_pct,
        wacc_df["gas_mw"],
        "^-",
        color="#ef4444",
        lw=2.0,
        label="Gas Peaker Capacity (MW)",
    )
    ax3.axvline(7.0, color="#64748b", ls=":", lw=1.3)
    ax3.set_title("(d) Cost-of-Capital (WACC) Shift: Optimal Mix & System LCOE")
    ax3.set_xlabel("Real Discount Rate WACC (%) [Calibrated Baseline = 7.0%]")
    ax3.set_ylabel("Optimal Installed Capacity (MW)")
    ax3.set_ylim(0.0, max(575.0, float(wacc_df["wind_mw"].max()) * 1.22))

    ax3_r = ax3.twinx()
    ax3_r.grid(False)
    ax3_r.plot(
        w_pct,
        wacc_df["system_lcoe_per_mwh"],
        "d--",
        color="#059669",
        lw=2.1,
        label="System LCOE (/MWh)",
    )
    ax3_r.set_ylabel("System LCOE (currency / MWh)", color="#065f46")
    ax3_r.tick_params(axis="y", colors="#065f46")
    ax3_r.set_ylim(0.0, max(72.0, float(wacc_df["system_lcoe_per_mwh"].max()) * 1.55))

    lines_l, labels_l = ax3.get_legend_handles_labels()
    lines_r, labels_r = ax3_r.get_legend_handles_labels()
    ax3.legend(
        lines_l + lines_r, labels_l + labels_r, loc="upper right", fontsize=7.9, framealpha=0.92
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_multiduration_storage_dunkelflaute(
    output_path: str | Path = "results/multiduration_storage_dunkelflaute_dispatch.png",
    profiles: pd.DataFrame | None = None,
    hybrid_result: StorageScenarioResult | None = None,
) -> Path:
    """Render 4-panel chronological Dunkelflaute dispatch, SoC separation & LCOE crossover."""
    prof = generate_dunkelflaute_profiles() if profiles is None else validate_profiles(profiles)
    hyb = (
        run_storage_scenario(prof, include_bess=True, include_h2=True, max_gas_mw=0.0)
        if hybrid_result is None
        else hybrid_result
    )
    regimes_df = compare_storage_adequacy_regimes(prof)
    sweep_df = sweep_dunkelflaute_duration()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    hours = np.arange(len(prof), dtype=float)
    load = prof["load_mw"].to_numpy(dtype=float)
    wind_p = hyb.generator_dispatch_mw["Wind_Turbine"].to_numpy(dtype=float)
    solar_p = hyb.generator_dispatch_mw["Solar_PV"].to_numpy(dtype=float)
    h2_dis = hyb.h2_discharge_mw_e.to_numpy(dtype=float)
    bess_dis = hyb.bess_discharge_mw.to_numpy(dtype=float)
    bess_ch = hyb.bess_charge_mw.to_numpy(dtype=float)
    h2_ch = hyb.h2_charge_mw_e.to_numpy(dtype=float)

    _apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), dpi=180)
    fig.suptitle(
        (
            "Multi-Duration Energy Storage (4 h Li-Ion BESS + Green H2 P2X) "
            "Across a 168 h Chronological Dunkelflaute"
        ),
        fontsize=12.4,
        fontweight="bold",
        y=0.98,
    )

    # Panel (a): 168 h Chronological Dispatch & Charging Stack
    ax0 = axes[0, 0]
    ax0.axvspan(72, 120, color="#fecdd3", alpha=0.36, label="48 h Dunkelflaute Lull")
    ax0.stackplot(
        hours,
        wind_p,
        solar_p,
        h2_dis,
        bess_dis,
        labels=[
            f"Wind ({hyb.generator_p_nom_mw['Wind_Turbine']:.0f} MW)",
            f"Solar PV ({hyb.generator_p_nom_mw['Solar_PV']:.0f} MW)",
            f"H2 Turbine ({hyb.h2_turbine_mw_e:.0f} MW_e)",
            f"Li-Ion BESS Dis. ({hyb.bess_p_nom_mw:.0f} MW)",
        ],
        colors=["#0284c7", "#f59e0b", "#8b5cf6", "#10b981"],
        alpha=0.86,
    )
    ax0.fill_between(
        hours,
        0.0,
        -h2_ch,
        color="#6366f1",
        alpha=0.65,
        label=f"H2 Electrolyzer (-{hyb.h2_electrolyzer_mw_e:.0f} MW_e)",
    )
    ax0.fill_between(
        hours,
        -h2_ch,
        -(h2_ch + bess_ch),
        color="#059669",
        alpha=0.65,
        label="Li-Ion BESS Charge",
    )
    ax0.plot(hours, load, color="#0f172a", lw=1.8, ls="--", label="Demand (MW)")
    ax0.axhline(0.0, color="#475569", lw=0.8)
    ax0.set_title("(a) 168 h Zero-Fossil Dispatch (+Supply / -Storage Charging)")
    ax0.set_xlabel("Chronological Hour of 7-Day Synoptic Window (h)")
    ax0.set_ylabel("Electric Power (MW_e)")
    ax0.set_xlim(0, len(prof) - 1)
    top_supply = float((wind_p + solar_p + h2_dis + bess_dis).max())
    max_sink = float((h2_ch + bess_ch).max())
    ax0.set_ylim(-max(255.0, max_sink * 1.25), max(580.0, top_supply * 1.68))
    ax0.set_xticks(range(0, 169, 24))
    ax0.legend(loc="upper right", ncol=2, fontsize=7.5, framealpha=0.92)

    # Panel (b): Normalized State-of-Charge (SoC %) Timescale Separation
    ax1 = axes[0, 1]
    ax1.axvspan(72, 120, color="#fecdd3", alpha=0.36, label="48 h Dunkelflaute Lull")
    bess_soc_pct = 100.0 * hyb.bess_soc_mwh.to_numpy(dtype=float) / max(1e-6, hyb.bess_e_nom_mwh)
    h2_soc_pct = 100.0 * hyb.h2_soc_mwh_h2.to_numpy(dtype=float) / max(1e-6, hyb.h2_store_mwh_h2)

    ax1.plot(
        hours,
        h2_soc_pct,
        color="#7c3aed",
        lw=2.5,
        label=(
            f"Green H2 Cavern ({hyb.h2_store_mwh_h2:,.0f} MWh_H2 | "
            f"{hyb.h2_equiv_cycles_per_year:.1f} cyc/yr)"
        ),
    )
    ax1.fill_between(hours, 0.0, h2_soc_pct, color="#7c3aed", alpha=0.12)
    ax1.plot(
        hours,
        bess_soc_pct,
        color="#059669",
        lw=1.9,
        ls="-",
        label=(
            f"Li-Ion BESS 4 h ({hyb.bess_e_nom_mwh:,.0f} MWh_e | "
            f"{hyb.bess_equiv_cycles_per_year:.0f} cyc/yr)"
        ),
    )
    ax1.annotate(
        "Multi-day H2 drawdown\nbridges 48 h wind/solar lull",
        xy=(90.0, float(h2_soc_pct[90])),
        xytext=(92.0, 86.0),
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.0},
    )
    ax1.set_title("(b) State-of-Charge Timescale Separation: Diurnal BESS vs. Synoptic H2")
    ax1.set_xlabel("Chronological Hour of 7-Day Synoptic Window (h)")
    ax1.set_ylabel("Normalized Reservoir State of Charge (%)")
    ax1.set_xlim(0, len(prof) - 1)
    ax1.set_ylim(-4.0, 128.0)
    ax1.set_xticks(range(0, 169, 24))
    ax1.legend(loc="upper left", fontsize=7.9, framealpha=0.92)

    # Panel (c): Adequacy Regime LCOE Comparison
    ax2 = axes[1, 0]
    short_labels = [
        "VRE Only\n(0% Gas)",
        "VRE +\n4h BESS",
        "VRE +\nGreen H2",
        "Hybrid\nBESS + H2",
        "Hybrid +\nGas Ref.",
    ]
    lcoes = regimes_df["system_lcoe_per_mwh"].to_numpy(dtype=float)
    bar_colors = ["#dc2626", "#f59e0b", "#6366f1", "#10b981", "#475569"]
    x_idx = np.arange(len(short_labels), dtype=float)
    bars = ax2.bar(x_idx, lcoes, width=0.56, color=bar_colors, alpha=0.9)
    for idx_b, bar in enumerate(bars):
        val = lcoes[idx_b]
        ax2.text(
            bar.get_x() + 0.5 * bar.get_width(),
            val + 5.5,
            f"{val:.1f}\n/MWh",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold" if idx_b == 3 else "normal",
            color="#0f172a",
        )
    reduction_pct = 100.0 * (lcoes[0] - lcoes[3]) / lcoes[0]
    ax2.annotate(
        f"-{reduction_pct:.1f}% zero-fossil LCOE\nvia multi-duration storage",
        xy=(2.76, lcoes[3] + 1.0),
        xytext=(1.05, lcoes[0] * 0.60),
        fontsize=8.1,
        bbox={"boxstyle": "round,pad=0.32", "fc": "#ecfdf5", "ec": "#10b981"},
        arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 1.2},
    )
    ax2.set_title("(c) System LCOE Across 48 h Dunkelflaute Adequacy Regimes")
    ax2.set_ylabel("System LCOE (currency / MWh)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(short_labels, fontsize=8.3)
    ax2.set_ylim(0.0, max(350.0, float(lcoes.max()) * 1.22))

    # Panel (d): Storage-Duration Crossover Curve (0 h .. 72 h Dunkelflaute)
    ax3 = axes[1, 1]
    dur_h = sweep_df["lull_hours"].to_numpy(dtype=float)
    b_lcoe = sweep_df["bess_only_lcoe"].to_numpy(dtype=float)
    h_lcoe = sweep_df["h2_only_lcoe"].to_numpy(dtype=float)
    hyb_lcoe = sweep_df["hybrid_lcoe"].to_numpy(dtype=float)

    ax3.plot(
        dur_h,
        b_lcoe,
        "s--",
        color="#f59e0b",
        lw=2.0,
        label="4 h Li-Ion BESS Only (90.3% RTE)",
    )
    ax3.plot(
        dur_h,
        h_lcoe,
        "^--",
        color="#6366f1",
        lw=2.0,
        label="Green H2 P2X Only (34.0% RTE, Cheap Cavern)",
    )
    ax3.plot(
        dur_h,
        hyb_lcoe,
        "o-",
        color="#059669",
        lw=2.5,
        label="Hybrid BESS + H2 (Lower Envelope)",
    )
    # Interpolate crossover where b_lcoe - h_lcoe crosses zero
    diff = b_lcoe - h_lcoe
    cross_idx = int(np.where(np.diff(np.sign(diff)) != 0)[0][0])
    x0, x1 = dur_h[cross_idx], dur_h[cross_idx + 1]
    d0, d1 = diff[cross_idx], diff[cross_idx + 1]
    x_cross = x0 - d0 * (x1 - x0) / (d1 - d0)
    y_cross = float(np.interp(x_cross, dur_h, b_lcoe))
    ax3.scatter([x_cross], [y_cross], color="#dc2626", s=46, zorder=6)
    ax3.annotate(
        f"RTE vs. Energy-CapEx\nCrossover (~{x_cross:.1f} h)",
        xy=(x_cross, y_cross),
        xytext=(x_cross + 11.0, y_cross - 7.5),
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.0},
    )
    ax3.set_title("(d) Zero-Fossil LCOE Crossover vs. Dunkelflaute Lull Duration")
    ax3.set_xlabel("Chronological Dunkelflaute Lull Duration (hours)")
    ax3.set_ylabel("Zero-Fossil System LCOE (currency / MWh)")
    ax3.set_xlim(-2.0, 74.0)
    ax3.set_ylim(18.0, max(102.0, float(b_lcoe.max()) * 1.22))
    ax3.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax3.legend(loc="upper left", fontsize=7.9, framealpha=0.92)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_co2_pareto_frontier_and_macc(
    output_path: str | Path = "results/co2_pareto_frontier_and_macc.png",
    macc_df: pd.DataFrame | None = None,
    tax_df: pd.DataFrame | None = None,
) -> Path:
    """Render 4-panel CO2 Pareto frontier, MACC, fuel/storage transition & fiscal incidence."""
    macc = sweep_co2_pareto_and_macc() if macc_df is None else macc_df.copy()
    tax = sweep_carbon_tax_equivalents() if tax_df is None else tax_df.copy()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9.3,
            "axes.titlesize": 10.4,
            "axes.labelsize": 9.3,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), dpi=180)
    fig.suptitle(
        (
            "CO2 Cap-Tax Lagrangian Duality: "
            "Pareto Frontier, Marginal Abatement Cost Curve (MACC) & Fuel Switching"
        ),
        fontsize=12.4,
        fontweight="bold",
        y=0.98,
    )

    # --- Panel (a): Cost-Emissions Pareto Frontier & Cap-Tax Saddle-Point Equivalence ---
    ax0 = axes[0, 0]
    e_kt = macc["emissions_tco2"].to_numpy(dtype=float) / 1e3
    c_res_m = macc["total_resource_cost_per_year"].to_numpy(dtype=float) / 1e6

    tax_sorted = tax.sort_values("emissions_tco2")
    t_e_kt = tax_sorted["emissions_tco2"].to_numpy(dtype=float) / 1e3
    t_res_m = tax_sorted["total_resource_cost_per_year"].to_numpy(dtype=float) / 1e6
    t_priv_m = tax_sorted["total_private_cost_per_year"].to_numpy(dtype=float) / 1e6

    ax0.fill_between(
        t_e_kt,
        t_res_m,
        t_priv_m,
        color="#a855f7",
        alpha=0.17,
        label=r"Recycled Tax Transfer τ ·E^*(τ)",
    )
    ax0.plot(
        t_e_kt,
        t_priv_m,
        color="#7e22ce",
        lw=1.8,
        ls="--",
        label=r"Private Cost under Tax J_{tax}^*(τ)",
    )
    ax0.plot(
        e_kt,
        c_res_m,
        color="#0284c7",
        lw=2.4,
        marker="o",
        ms=5.2,
        label=r"Cap Resource Cost J_{cap}^*(Ē)",
    )
    ax0.scatter(
        t_e_kt,
        t_res_m,
        marker="D",
        s=38,
        facecolors="none",
        edgecolors="#dc2626",
        linewidths=1.5,
        zorder=5,
        label=r"Tax Resource Cost J_{res}^*(τ) (Duality)",
    )

    # Annotate BAU, 75% tangent point, and Net-Zero
    idx_75 = int(np.argmin(np.abs(macc["abatement_realised_pct"].to_numpy(dtype=float) - 75.0)))
    e75 = float(e_kt[idx_75])
    c75 = float(c_res_m[idx_75])
    mu75 = float(macc["marginal_abatement_cost_per_tco2"].iloc[idx_75])
    ax0.annotate(
        (
            f"75% Abatement\n$E^*$ = {e75:.0f} ktCO$_2$/yr\n"
            f"-∂ J^*/∂ Ē = {mu75:.2f} /tCO$_2$"
        ),
        xy=(e75, c75),
        xytext=(55.0, 35.0),
        fontsize=7.9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.0},
    )
    ax0.annotate(
        f"Net-Zero (0 kt)\n{c_res_m[-1]:.2f} M/yr",
        xy=(e_kt[-1], c_res_m[-1]),
        xytext=(145.0, c_res_m[-1] + 1.6),
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#f0fdf4", "ec": "#86efac"},
        arrowprops={"arrowstyle": "->", "color": "#15803d", "lw": 1.0},
    )
    ax0.annotate(
        f"BAU ({e_kt[0]:.0f} kt)\n{c_res_m[0]:.2f} M/yr",
        xy=(e_kt[0], c_res_m[0]),
        xytext=(e_kt[0] - 300.0, c_res_m[0] + 7.2),
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#fff7ed", "ec": "#fdba74"},
        arrowprops={"arrowstyle": "->", "color": "#c2410c", "lw": 1.0},
    )
    ax0.set_title("(a) Cost-Emissions Pareto Frontier & Cap-Tax Duality")
    ax0.set_xlabel(r"Annual CO_2 Emissions (10^3tCO_2 / yr)")
    ax0.set_ylabel("Annual System Cost (M currency / yr)")
    ax0.set_xlim(-35.0, float(e_kt.max()) * 1.07)
    ax0.set_ylim(float(c_res_m.min()) * 0.92, float(max(c_res_m.max(), t_priv_m.max())) * 1.11)
    ax0.legend(loc="upper right", fontsize=7.7, framealpha=0.93)

    # --- Panel (b): Marginal Abatement Cost Curve (MACC) & Average Cost ---
    ax1 = axes[0, 1]
    ab_pct = macc["abatement_realised_pct"].to_numpy(dtype=float)
    mac = macc["marginal_abatement_cost_per_tco2"].to_numpy(dtype=float)
    avg_c = macc["average_abatement_cost_per_tco2"].to_numpy(dtype=float)

    ax1.set_yscale("symlog", linthresh=25.0, linscale=1.05)
    ax1.step(
        ab_pct,
        mac,
        where="post",
        color="#dc2626",
        lw=2.3,
        label=r"Marginal Abatement Cost μ_{CO_2}^*",
    )
    ax1.scatter(ab_pct, mac, color="#dc2626", s=26, zorder=4)
    ax1.fill_between(ab_pct, 0.0, mac, step="post", color="#ef4444", alpha=0.12)
    ax1.plot(
        ab_pct,
        avg_c,
        color="#0f766e",
        lw=2.1,
        ls="--",
        marker="s",
        ms=4.2,
        label="Average Abatement Cost (/tCO$_2$)",
    )

    # Annotate physical abatement stages cleanly
    ax1.annotate(
        "Stage I: Wind Entry\n(~6-10 /tCO_2)",
        xy=(
            30.0,
            float(
                macc.loc[
                    macc["abatement_target_pct"] == 30.0,
                    "marginal_abatement_cost_per_tco2",
                ].iloc[0]
            ),
        ),
        xytext=(6.0, 17.5),
        fontsize=7.7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 0.9},
    )
    ax1.annotate(
        "Stage II: Coal -> CCGT\n+ Solar (~19-29 /tCO_2)",
        xy=(85.0, float(mac[np.argmin(np.abs(ab_pct - 85.0))])),
        xytext=(42.0, 48.0),
        fontsize=7.7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 0.9},
    )
    ax1.annotate(
        "Stage III: 90% Gas CCS\n+ BESS (~53-104 /tCO_2)",
        xy=(98.5, float(mac[np.argmin(np.abs(ab_pct - 98.5))])),
        xytext=(47.0, 145.0),
        fontsize=7.7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 0.9},
    )
    ax1.annotate(
        f"Stage IV: Net-Zero H$_2$\nBackstop ({mac[-1]:.1f} /tCO$_2$)",
        xy=(ab_pct[-1], mac[-1]),
        xytext=(52.0, 360.0),
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#fef2f2", "ec": "#fca5a5"},
        arrowprops={"arrowstyle": "->", "color": "#b91c1c", "lw": 1.0},
    )

    yticks = [0, 5, 10, 15, 25, 50, 100, 200, 420]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([str(v) for v in yticks])
    ax1.set_xlim(0.0, 102.5)
    ax1.set_ylim(0.0, 620.0)
    ax1.set_title("(b) Marginal Abatement Cost Curve (MACC) & Regime Staircase")
    ax1.set_xlabel(r"Cumulative CO_2 Abatement (% of BAU)")
    ax1.set_ylabel(r"Carbon Price / Cost (currency / tCO_2, symlog)")
    ax1.legend(loc="upper left", fontsize=7.8, framealpha=0.93)

    # --- Panel (c): Generation Energy Mix (%) & Multi-Duration Storage Trajectory ---
    ax2 = axes[1, 0]
    coal_sh = macc["coal_share_pct"].to_numpy(dtype=float)
    gas_sh = macc["gas_unabated_share_pct"].to_numpy(dtype=float)
    ccs_sh = macc["gas_ccs_share_pct"].to_numpy(dtype=float)
    sol_sh = macc["solar_share_pct"].to_numpy(dtype=float)
    wnd_sh = macc["wind_share_pct"].to_numpy(dtype=float)

    ax2.stackplot(
        ab_pct,
        coal_sh,
        gas_sh,
        ccs_sh,
        sol_sh,
        wnd_sh,
        labels=[
            "Coal Steam (0.85 t/MWh)",
            "CCGT + OCGT Gas",
            "Gas CCS (90% Capture)",
            "Solar PV",
            "Wind Turbine",
        ],
        colors=["#334155", "#f97316", "#8b5cf6", "#facc15", "#0ea5e9"],
        alpha=0.86,
    )
    ax2.set_xlim(0.0, 100.0)
    ax2.set_ylim(0.0, 127.0)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_title("(c) Generation Share (%) & Storage Capacity Across Abatement")
    ax2.set_xlabel(r"Cumulative CO_2 Abatement (% of BAU)")
    ax2.set_ylabel("Annual Generation Share (%)")

    ax2_r = ax2.twinx()
    ax2_r.grid(False)
    h2_gwh = macc["h2_mwh"].to_numpy(dtype=float) / 1e3
    bess_10mwh = macc["bess_mwh"].to_numpy(dtype=float) / 10.0
    ax2_r.plot(
        ab_pct,
        h2_gwh,
        color="#be123c",
        lw=2.2,
        marker="^",
        ms=4.8,
        label=r"Green H_2 Cavern (GWh_{H2})",
    )
    ax2_r.plot(
        ab_pct,
        bess_10mwh,
        color="#15803d",
        lw=1.9,
        ls="--",
        marker="o",
        ms=4.0,
        label="Li-Ion BESS ($10^1$ MWh)",
    )
    ax2_r.set_ylabel(r"Storage Capacity (GWh_{H2} or 10^1 MWh BESS)", color="#9f1239")
    ax2_r.set_ylim(0.0, max(5.2, float(max(h2_gwh.max(), bess_10mwh.max())) * 1.32))

    h_l, l_l = ax2.get_legend_handles_labels()
    h_r, l_r = ax2_r.get_legend_handles_labels()
    ax2.legend(
        h_l + h_r,
        l_l + l_r,
        loc="upper left",
        fontsize=7.3,
        ncol=2,
        framealpha=0.92,
    )

    # --- Panel (d): Fiscal Incidence & Resource LCOE vs Nodal LMP ---
    ax3 = axes[1, 1]
    targets = [0.0, 45.0, 75.0, 96.0, 100.0]
    sel_indices = [
        int(np.argmin(np.abs(macc["abatement_target_pct"].to_numpy(dtype=float) - tg)))
        for tg in targets
    ]
    sub = macc.iloc[sel_indices]
    x_pos = np.arange(len(targets), dtype=float)
    x_labels = ["0%\n(BAU)", "45%\nCap", "75%\nCap", "96%\n(CCS)", "100%\nNet-Zero"]

    gen_cap_m = sub["annual_gen_capex_cost"].to_numpy(dtype=float) / 1e6
    st_cap_m = sub["annual_storage_capex_cost"].to_numpy(dtype=float) / 1e6
    fuel_m = sub["annual_fuel_opex_cost"].to_numpy(dtype=float) / 1e6
    tax_rev_m = sub["equivalent_tax_revenue_per_year"].to_numpy(dtype=float) / 1e6

    w_bar = 0.52
    ax3.bar(x_pos, gen_cap_m, width=w_bar, color="#0284c7", label="Gen CapEx (M/yr)")
    ax3.bar(
        x_pos,
        st_cap_m,
        width=w_bar,
        bottom=gen_cap_m,
        color="#10b981",
        label="BESS + H$_2$ CapEx (M/yr)",
    )
    ax3.bar(
        x_pos,
        fuel_m,
        width=w_bar,
        bottom=gen_cap_m + st_cap_m,
        color="#f59e0b",
        label="Fuel OpEx (M/yr)",
    )
    ax3.bar(
        x_pos,
        tax_rev_m,
        width=w_bar,
        bottom=gen_cap_m + st_cap_m + fuel_m,
        color="#e9d5ff",
        edgecolor="#7e22ce",
        hatch="//",
        linewidth=0.9,
        label=r"Equiv. Tax Revenue μ^* E^* (M/yr)",
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels)
    ax3.set_ylabel("Annual Cost Breakdown (M currency / yr)")
    ax3.set_ylim(0.0, float((gen_cap_m + st_cap_m + fuel_m + tax_rev_m).max()) * 1.30)
    ax3.set_title("(d) Fiscal Cost Incidence, Resource LCOE & Nodal LMP")

    ax3_r = ax3.twinx()
    ax3_r.grid(False)
    res_lcoe = sub["resource_lcoe_per_mwh"].to_numpy(dtype=float)
    tax_incl_lmp = res_lcoe + (
        sub["equivalent_tax_revenue_per_year"].to_numpy(dtype=float)
        / (
            sub["total_resource_cost_per_year"].to_numpy(dtype=float)
            / sub["resource_lcoe_per_mwh"].to_numpy(dtype=float)
        )
    )
    ax3_r.plot(
        x_pos,
        res_lcoe,
        color="#0f172a",
        lw=2.1,
        marker="o",
        ms=5.2,
        label="Resource LCOE (/MWh)",
    )
    ax3_r.plot(
        x_pos,
        tax_incl_lmp,
        color="#7e22ce",
        lw=1.9,
        ls="--",
        marker="D",
        ms=4.8,
        label="Tax-Incl. Load-Wtd LMP (/MWh)",
    )
    ax3_r.set_ylabel("Unit Electricity Cost / Price (currency / MWh)", color="#334155")
    ax3_r.set_ylim(18.0, float(max(res_lcoe.max(), tax_incl_lmp.max())) * 1.28)

    h3_l, l3_l = ax3.get_legend_handles_labels()
    h3_r, l3_r = ax3_r.get_legend_handles_labels()
    ax3.legend(
        h3_l + h3_r,
        l3_l + l3_r,
        loc="upper left",
        fontsize=7.2,
        ncol=2,
        framealpha=0.93,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_duck_curve_ramping_and_reserves(
    output_path: str | Path = "results/duck_curve_ramping_and_reserves.png",
    profiles: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
) -> Path:
    """Render 4-panel Duck-Curve dispatch, ramp envelopes, spinning reserves & regime LCOE."""
    prof = make_duck_curve_profiles() if profiles is None else profiles.copy()
    regimes = compare_flexibility_regimes(prof) if regime_df is None else regime_df.copy()

    inflex_res = run_flexibility_scenario(
        prof,
        enforce_ramping=True,
        enforce_must_run=True,
        include_bess=False,
        reserve_policy=SpinningReservePolicy(),
    )
    flex_bess = run_flexibility_scenario(
        prof,
        enforce_ramping=True,
        enforce_must_run=True,
        include_bess=True,
        reserve_policy=SpinningReservePolicy(),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.3,
            "axes.labelsize": 9.2,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.4), dpi=180)
    fig.suptitle(
        (
            "Thermal Ramp-Rate Envelopes, Must-Run Floors & Co-Optimized Spinning Reserves "
            "(48 h Duck-Curve Benchmark)"
        ),
        fontsize=12.2,
        fontweight="bold",
        y=0.98,
    )

    hrs = np.arange(len(prof.index), dtype=float)
    load_mw = prof["load_mw"].to_numpy(dtype=float)

    # --- Panel (a): 48 h Duck-Curve Dispatch, BESS Belly Charging & Net Load ---
    ax0 = axes[0, 0]
    p_df = flex_bess.dispatch_mw
    base_p = p_df["Baseload_Thermal"].to_numpy(dtype=float)
    ccgt_p = p_df["CCGT_MidMerit"].to_numpy(dtype=float)
    ocgt_p = p_df["OCGT_FastPeaker"].to_numpy(dtype=float)
    wind_p = p_df["Wind_Turbine"].to_numpy(dtype=float)
    sol_p = p_df["Solar_PV"].to_numpy(dtype=float)
    b_dis = flex_bess.bess_discharge_mw.to_numpy(dtype=float)
    b_chg = flex_bess.bess_charge_mw.to_numpy(dtype=float)
    net_l = flex_bess.net_load_mw.to_numpy(dtype=float)

    ax0.stackplot(
        hrs,
        base_p,
        ccgt_p,
        ocgt_p,
        wind_p,
        sol_p,
        b_dis,
        labels=[
            "Baseload (60% Must-Run)",
            "CCGT Mid-Merit",
            "OCGT Fast Peaker",
            "Wind Turbine",
            "Solar PV",
            "BESS Discharge",
        ],
        colors=["#475569", "#f97316", "#ef4444", "#0ea5e9", "#facc15", "#10b981"],
        alpha=0.86,
    )
    ax0.fill_between(
        hrs,
        0.0,
        -b_chg,
        color="#059669",
        alpha=0.38,
        step="mid",
        label="BESS Charge (-MW)",
    )
    ax0.plot(hrs, load_mw, color="#0f172a", lw=2.1, label="Gross Demand (MW)")
    ax0.plot(
        hrs,
        net_l,
        color="#be123c",
        lw=1.8,
        ls="--",
        label="Net Load (Demand - VRE)",
    )
    ax0.axhline(0.0, color="#64748b", lw=0.8)

    belly_idx = int(np.argmin(net_l[:24]))
    ax0.annotate(
        f"Duck Belly ({net_l[belly_idx]:.0f} MW)\nBESS Absorbs Solar",
        xy=(float(hrs[belly_idx]), float(net_l[belly_idx])),
        xytext=(3.0, -72.0),
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#f0fdf4", "ec": "#86efac"},
        arrowprops={"arrowstyle": "->", "color": "#15803d", "lw": 1.0},
    )
    pos_stack_max = float((base_p + ccgt_p + ocgt_p + wind_p + sol_p + b_dis).max())
    ax0.set_xlim(0.0, float(len(hrs) - 1))
    ax0.set_ylim(-95.0, float(max(pos_stack_max, load_mw.max())) * 1.30)
    ax0.set_title("(a) 48 h Duck-Curve Dispatch & BESS Ramp-Shaving")
    ax0.set_xlabel("Chronological Snapshot Hour (h)")
    ax0.set_ylabel("Active Power (MW)")
    ax0.legend(loc="upper left", fontsize=7.0, ncol=3, framealpha=0.92)

    # --- Panel (b): Thermal Hourly Ramp Utilization (%/h) & Nodal LMP Relief ---
    ax1 = axes[0, 1]
    k_base = max(1e-6, float(flex_bess.p_nom_mw["Baseload_Thermal"]))
    k_ccgt = max(1e-6, float(flex_bess.p_nom_mw["CCGT_MidMerit"]))
    dp_base_pct = 100.0 * np.concatenate([[base_p[0] - base_p[-1]], np.diff(base_p)]) / k_base
    dp_ccgt_pct = 100.0 * np.concatenate([[ccgt_p[0] - ccgt_p[-1]], np.diff(ccgt_p)]) / k_ccgt

    ax1.plot(
        hrs,
        dp_ccgt_pct,
        color="#f97316",
        lw=1.7,
        alpha=0.9,
        label="CCGT Ramp (%/h, +-45% lim)",
    )
    ax1.plot(
        hrs,
        dp_base_pct,
        color="#1e293b",
        lw=2.1,
        label="Baseload Ramp (%/h, +-10% lim)",
    )
    ax1.axhline(10.0, color="#1e293b", lw=1.1, ls=":", label="Baseload +-10%/h Envelope")
    ax1.axhline(-10.0, color="#1e293b", lw=1.1, ls=":")
    ax1.axhline(0.0, color="#94a3b8", lw=0.7)
    ax1.set_xlim(0.0, float(len(hrs) - 1))
    ax1.set_ylim(-58.0, 68.0)
    ax1.set_title("(b) Thermal Ramp Trajectories & Sunset LMP Scarcity Relief")
    ax1.set_xlabel("Chronological Snapshot Hour (h)")
    ax1.set_ylabel("Hourly Thermal Ramp Rate (% of $K_g$ / h)")

    ax1_r = ax1.twinx()
    ax1_r.grid(False)
    lmp_no_bess = inflex_res.lmp_per_mwh.to_numpy(dtype=float)
    lmp_bess = flex_bess.lmp_per_mwh.to_numpy(dtype=float)
    ax1_r.plot(
        hrs,
        lmp_no_bess,
        color="#dc2626",
        lw=1.6,
        ls="--",
        alpha=0.85,
        label=f"LMP No BESS (Peak {lmp_no_bess.max():.0f} /MWh)",
    )
    ax1_r.plot(
        hrs,
        lmp_bess,
        color="#7e22ce",
        lw=1.9,
        label=f"LMP With BESS (Peak {lmp_bess.max():.0f} /MWh)",
    )
    ax1_r.set_ylabel("Nodal Energy LMP (currency / MWh)", color="#6b21a8")
    ax1_r.set_ylim(-15.0, float(max(lmp_no_bess.max(), lmp_bess.max())) * 1.34)

    h1_l, l1_l = ax1.get_legend_handles_labels()
    h1_r, l1_r = ax1_r.get_legend_handles_labels()
    ax1.legend(
        h1_l + h1_r,
        l1_l + l1_r,
        loc="upper left",
        fontsize=7.1,
        ncol=2,
        framealpha=0.92,
    )

    # --- Panel (c): Upward Spinning Reserve Provision & KKT Reserve Shadow Price ---
    ax2 = axes[1, 0]
    r_base = flex_bess.reserve_by_unit_mw["Baseload_Thermal"].to_numpy(dtype=float)
    r_ccgt = flex_bess.reserve_by_unit_mw["CCGT_MidMerit"].to_numpy(dtype=float)
    r_ocgt = flex_bess.reserve_by_unit_mw["OCGT_FastPeaker"].to_numpy(dtype=float)
    r_bess = flex_bess.bess_reserve_mw.to_numpy(dtype=float)
    r_req = flex_bess.reserve_requirement_mw.to_numpy(dtype=float)

    ax2.stackplot(
        hrs,
        r_base,
        r_ccgt,
        r_ocgt,
        r_bess,
        labels=[
            "Baseload Headroom (5%)",
            "CCGT Headroom (25%)",
            "OCGT Fast Reserve (55%)",
            "BESS SoC-Backed Reserve",
        ],
        colors=["#64748b", "#fb923c", "#f43f5e", "#14b8a6"],
        alpha=0.86,
    )
    ax2.plot(
        hrs,
        r_req,
        color="#0f172a",
        lw=2.0,
        ls="--",
        label="Dynamic Req (6% Load + 12% VRE)",
    )
    res_stack_max = float((r_base + r_ccgt + r_ocgt + r_bess).max())
    ax2.set_xlim(0.0, float(len(hrs) - 1))
    ax2.set_ylim(0.0, float(max(res_stack_max, r_req.max())) * 1.38)
    ax2.set_title("(c) Co-Optimized Spinning Reserve Stack & Shadow Price")
    ax2.set_xlabel("Chronological Snapshot Hour (h)")
    ax2.set_ylabel("Upward Spinning Reserve (MW)")

    ax2_r = ax2.twinx()
    ax2_r.grid(False)
    rp_no_bess = inflex_res.reserve_price_per_mw_h.to_numpy(dtype=float)
    rp_bess = flex_bess.reserve_price_per_mw_h.to_numpy(dtype=float)
    ax2_r.plot(
        hrs,
        rp_no_bess,
        color="#b91c1c",
        lw=1.6,
        ls=":",
        label=f"Res Price No BESS (Max {rp_no_bess.max():.0f})",
    )
    ax2_r.plot(
        hrs,
        rp_bess,
        color="#0369a1",
        lw=1.9,
        label=f"Res Price With BESS (Max {rp_bess.max():.0f})",
    )
    ax2_r.set_ylabel("Spinning Reserve Price (currency / MW-h)", color="#0369a1")
    ax2_r.set_ylim(0.0, float(max(rp_no_bess.max(), rp_bess.max(), 10.0)) * 1.36)

    h2_l, l2_l = ax2.get_legend_handles_labels()
    h2_r, l2_r = ax2_r.get_legend_handles_labels()
    ax2.legend(
        h2_l + h2_r,
        l2_l + l2_r,
        loc="upper left",
        fontsize=7.0,
        ncol=2,
        framealpha=0.92,
    )

    # --- Panel (d): 4-Regime Portfolio Shift, VRE Hosting & System LCOE ---
    ax3 = axes[1, 1]
    x_pos = np.arange(len(regimes), dtype=float)
    x_labels = [
        "1. Relaxed\n(No Ramp/Res)",
        "2. Must-Run\n+ Ramping",
        "3. Inflexible\n+ Reserves",
        "4. Flexible\n+ BESS Co-Opt",
    ]

    b_mw = regimes["baseload_mw"].to_numpy(dtype=float)
    c_mw = regimes["ccgt_mw"].to_numpy(dtype=float)
    o_mw = regimes["ocgt_mw"].to_numpy(dtype=float)
    s_mw = regimes["solar_mw"].to_numpy(dtype=float)
    w_mw = regimes["wind_mw"].to_numpy(dtype=float)

    w_bar = 0.50
    ax3.bar(x_pos, b_mw, width=w_bar, color="#475569", label="Baseload (MW)")
    ax3.bar(x_pos, c_mw, width=w_bar, bottom=b_mw, color="#f97316", label="CCGT (MW)")
    ax3.bar(
        x_pos,
        o_mw,
        width=w_bar,
        bottom=b_mw + c_mw,
        color="#ef4444",
        label="OCGT Peaker (MW)",
    )
    ax3.bar(
        x_pos,
        s_mw,
        width=w_bar,
        bottom=b_mw + c_mw + o_mw,
        color="#facc15",
        label="Solar PV (MW)",
    )
    ax3.bar(
        x_pos,
        w_mw,
        width=w_bar,
        bottom=b_mw + c_mw + o_mw + s_mw,
        color="#0ea5e9",
        label="Wind (MW)",
    )

    totals = b_mw + c_mw + o_mw + s_mw + w_mw
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, fontsize=8.3)
    ax3.set_ylabel("Optimal Installed Generation Capacity (MW)")
    ax3.set_ylim(0.0, float(totals.max()) * 1.34)
    ax3.set_title("(d) Portfolio Hosting Capacity & System LCOE Across Regimes")

    ax3_r = ax3.twinx()
    ax3_r.grid(False)
    lcoe_vals = regimes["system_lcoe_per_mwh"].to_numpy(dtype=float)
    ax3_r.plot(
        x_pos,
        lcoe_vals,
        color="#7e22ce",
        lw=2.3,
        marker="D",
        ms=6.0,
        label="System LCOE (/MWh)",
    )
    for idx_i, val in enumerate(lcoe_vals):
        ax3_r.annotate(
            f"{val:.2f}",
            xy=(float(x_pos[idx_i]), float(val)),
            xytext=(0.0, 9.0),
            textcoords="offset points",
            ha="center",
            fontsize=7.9,
            fontweight="bold",
            color="#581c87",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#e9d5ff", "alpha": 0.92},
        )
    bess_mwh_last = float(regimes["bess_e_nom_mwh"].iloc[-1])
    ax3.annotate(
        f"+{bess_mwh_last:.0f} MWh BESS\n3.2x VRE vs Reg. 3",
        xy=(float(x_pos[-1]), float(totals[-1])),
        xytext=(float(x_pos[-1]) - 0.85, float(totals.max()) * 1.03),
        fontsize=7.7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#f0fdf4", "ec": "#86efac"},
        arrowprops={"arrowstyle": "->", "color": "#15803d", "lw": 1.0},
    )
    ax3_r.set_ylabel("System LCOE (currency / MWh)", color="#581c87")
    ax3_r.set_ylim(float(lcoe_vals.min()) - 1.2, float(lcoe_vals.max()) + 1.65)

    h3_l, l3_l = ax3.get_legend_handles_labels()
    h3_r, l3_r = ax3_r.get_legend_handles_labels()
    ax3.legend(
        h3_l + h3_r,
        l3_l + l3_r,
        loc="upper right",
        fontsize=7.2,
        ncol=2,
        framealpha=0.92,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_zonal_dcopf_congestion_and_tep(
    output_path: str | Path = "results/zonal_dcopf_congestion_and_tep.png",
    profiles: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
) -> Path:
    """Render 4-panel 4-zone DC-OPF LMP divergence, TEP upgrades, congestion rent & regimes."""
    prof = make_four_zone_profiles() if profiles is None else profiles.copy()
    regimes = compare_transmission_regimes(prof) if regime_df is None else regime_df.copy()

    fixed_ac = run_transmission_scenario(prof, mode="ac_kvl", extendable_lines=False)
    tep_ac = run_transmission_scenario(prof, mode="ac_kvl", extendable_lines=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.3,
            "axes.labelsize": 9.2,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.4), dpi=180)
    fig.suptitle(
        (
            "4-Zone Meshed 380 kV DC-OPF: KVL Loop Flows, Congestion Rent & "
            "Co-Optimized Transmission Expansion (48 h)"
        ),
        fontsize=12.1,
        fontweight="bold",
        y=0.98,
    )

    hrs = np.arange(len(prof.index), dtype=float)

    # --- Panel (a): 4-Zone Spatial LMP Divergence (Fixed AC vs Co-Optimized TEP) ---
    ax0 = axes[0, 0]
    lmp_fix = fixed_ac.zonal_lmp_per_mwh
    lmp_tep = tep_ac.zonal_lmp_per_mwh

    l_east = lmp_fix["East_Metro"].to_numpy(dtype=float)
    l_south = lmp_fix["South_Hub"].to_numpy(dtype=float)
    l_west = lmp_fix["West_Solar"].to_numpy(dtype=float)
    l_north = lmp_fix["North_Wind"].to_numpy(dtype=float)
    l_tep_e = lmp_tep["East_Metro"].to_numpy(dtype=float)

    ax0.fill_between(
        hrs,
        np.minimum.reduce([l_east, l_south, l_west, l_north]),
        np.maximum.reduce([l_east, l_south, l_west, l_north]),
        color="#fecdd3",
        alpha=0.45,
        label=f"Fixed-Grid LMP Spread (Mean {fixed_ac.mean_spatial_lmp_spread_per_mwh:.1f})",
    )
    ax0.plot(hrs, l_east, color="#dc2626", lw=2.0, label="East Metro LMP (Fixed AC)")
    ax0.plot(hrs, l_south, color="#f97316", lw=1.5, ls="--", label="South Hub LMP (Fixed AC)")
    ax0.plot(hrs, l_west, color="#eab308", lw=1.5, ls="-.", label="West Solar LMP (Fixed AC)")
    ax0.plot(hrs, l_north, color="#0284c7", lw=1.8, label="North Wind LMP (Fixed AC)")
    ax0.plot(
        hrs,
        l_tep_e,
        color="#7e22ce",
        lw=2.1,
        ls=":",
        label=f"East Metro Post-TEP (Spread {tep_ac.mean_spatial_lmp_spread_per_mwh:.1f})",
    )

    ax0.set_xlim(0.0, float(len(hrs) - 1))
    ax0.set_ylim(-6.0, float(max(l_east.max(), l_tep_e.max())) * 1.34)
    ax0.set_title("(a) Spatial Nodal LMP Separation & Post-TEP Convergence")
    ax0.set_xlabel("Chronological Snapshot Hour (h)")
    ax0.set_ylabel("Nodal Energy Shadow Price (currency / MWh)")
    ax0.legend(loc="upper left", fontsize=7.0, ncol=2, framealpha=0.92)

    # --- Panel (b): 5-Corridor Thermal Ratings (Brownfield vs Co-Opt TEP) & Congestion Hours ---
    ax1 = axes[0, 1]
    l_names = list(fixed_ac.line_s_nom_init_mw.index)
    short_l = [ln.replace("_", "\n", 1).replace("_", "-") for ln in l_names]
    y_pos = np.arange(len(l_names), dtype=float)
    s_init = fixed_ac.line_s_nom_init_mw.to_numpy(dtype=float)
    s_tep = tep_ac.line_s_nom_opt_mw.to_numpy(dtype=float)
    cong_fix = fixed_ac.congested_corridor_hours.to_numpy(dtype=float)
    cong_tep = tep_ac.congested_corridor_hours.to_numpy(dtype=float)

    w_b = 0.34
    ax1.bar(
        y_pos - w_b / 2.0,
        s_init,
        width=w_b,
        color="#64748b",
        label="Existing Rating (MW)",
    )
    ax1.bar(
        y_pos + w_b / 2.0,
        s_tep,
        width=w_b,
        color="#0ea5e9",
        label="Co-Optimized TEP Rating (MW)",
    )
    for i_l in range(len(l_names)):
        added = s_tep[i_l] - s_init[i_l]
        ax1.annotate(
            f"+{added:.0f} MW",
            xy=(float(y_pos[i_l] + w_b / 2.0), float(s_tep[i_l])),
            xytext=(0.0, 6.0),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            color="#0369a1",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "#bae6fd", "alpha": 0.92},
        )

    ax1.set_xticks(y_pos)
    ax1.set_xticklabels(short_l, fontsize=8.0)
    ax1.set_ylim(0.0, float(s_tep.max()) * 1.36)
    ax1.set_ylabel("Corridor Thermal Rating F_ℓ (MW)")
    ax1.set_title("(b) 380 kV Corridor Expansion & Binding Congestion Hours")

    ax1_r = ax1.twinx()
    ax1_r.grid(False)
    ax1_r.plot(
        y_pos,
        cong_fix,
        color="#dc2626",
        lw=1.9,
        marker="o",
        ms=5.5,
        label="Congested Hours (Fixed AC)",
    )
    ax1_r.plot(
        y_pos,
        cong_tep,
        color="#15803d",
        lw=1.8,
        ls="--",
        marker="s",
        ms=5.0,
        label="Congested Hours (Post-TEP)",
    )
    ax1_r.set_ylabel("Binding Thermal Limit Hours (out of 48 h)", color="#b91c1c")
    ax1_r.set_ylim(-2.0, 62.0)

    h1_l, l1_l = ax1.get_legend_handles_labels()
    h1_r, l1_r = ax1_r.get_legend_handles_labels()
    ax1.legend(
        h1_l + h1_r,
        l1_l + l1_r,
        loc="upper right",
        fontsize=7.1,
        ncol=2,
        framealpha=0.92,
    )

    # --- Panel (c): ISO Merchandising Surplus (Congestion Rent) & KVL Adverse Loop Flow ---
    ax2 = axes[1, 0]
    cr_fix_k = fixed_ac.hourly_congestion_rent.to_numpy(dtype=float) / 1000.0
    cr_tep_k = tep_ac.hourly_congestion_rent.to_numpy(dtype=float) / 1000.0
    min_corridor_rent_k = (
        fixed_ac.corridor_merchandising_rent_per_h.min(axis=1).to_numpy(dtype=float) / 1000.0
    )

    ax2.fill_between(
        hrs,
        0.0,
        cr_fix_k,
        color="#f97316",
        alpha=0.30,
        label="ISO Congestion Rent (Fixed AC)",
    )
    ax2.plot(hrs, cr_fix_k, color="#ea580c", lw=1.9)
    ax2.plot(
        hrs,
        cr_tep_k,
        color="#0d9488",
        lw=2.0,
        label="Post-TEP Rent (= 100% Line CapEx)",
    )
    ax2.plot(
        hrs,
        min_corridor_rent_k,
        color="#be123c",
        lw=1.5,
        ls="--",
        label="Min Single-Line Rent (KVL Loop < 0)",
    )
    ax2.axhline(0.0, color="#475569", lw=0.8)

    tep_pk = int(np.argmax(cr_tep_k))
    rent_ymax = float(max(cr_fix_k.max(), cr_tep_k.max()))
    ax2.annotate(
        (
            "Joskow-Tirole Equilibrium:\n"
            f"Rent = Line CapEx ({tep_ac.annual_line_capex_cost / 1e6:.2f}M/yr)"
        ),
        xy=(float(hrs[tep_pk]), float(cr_tep_k[tep_pk])),
        xytext=(float(hrs[tep_pk]) + 3.5, rent_ymax * 0.84),
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#f0fdfa", "ec": "#5eead4"},
        arrowprops={"arrowstyle": "->", "color": "#0f766e", "lw": 1.0},
    )
    ax2.set_xlim(0.0, float(len(hrs) - 1))
    ax2.set_ylim(min(-1.8, float(min_corridor_rent_k.min()) * 2.8), rent_ymax * 1.28)
    ax2.set_title("(c) Hourly ISO Congestion Rent & KVL Counter-Price Loop Flow")
    ax2.set_xlabel("Chronological Snapshot Hour (h)")
    ax2.set_ylabel("Merchandising Surplus (k-currency / h)")
    ax2.legend(loc="upper left", fontsize=7.1, framealpha=0.92)

    # --- Panel (d): 4-Regime Spatial Fleet Relocation & System LCOE Hierarchy ---
    ax3 = axes[1, 1]
    x_pos = np.arange(len(regimes), dtype=float)
    x_labels = [
        "1. Copper-Plate\n(Single Bus)",
        "2. Congested AC\n(KVL Fixed)",
        "3. Controllable\nHVDC Transport",
        "4. Co-Optimized\nAC-TEP (KVL)",
    ]

    nw_mw = regimes["north_wind_mw"].to_numpy(dtype=float)
    ws_mw = regimes["west_solar_mw"].to_numpy(dtype=float)
    eo_mw = regimes["east_ocgt_mw"].to_numpy(dtype=float)

    wb = 0.48
    ax3.bar(x_pos, nw_mw, width=wb, color="#0284c7", label="North Offshore Wind (MW)")
    ax3.bar(
        x_pos,
        ws_mw,
        width=wb,
        bottom=nw_mw,
        color="#facc15",
        label="West Utility Solar (MW)",
    )
    ax3.bar(
        x_pos,
        eo_mw,
        width=wb,
        bottom=nw_mw + ws_mw,
        color="#ef4444",
        label="East Metro OCGT (MW)",
    )

    tot_bars = nw_mw + ws_mw + eo_mw
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, fontsize=8.2)
    ax3.set_ylabel("Key Zonal Installed Capacity (MW)")
    ax3.set_ylim(0.0, float(tot_bars.max()) * 1.36)
    ax3.set_title("(d) Spatial VRE Unlocking, KVL Penalty & System LCOE")

    ax3_r = ax3.twinx()
    ax3_r.grid(False)
    lcoes = regimes["system_lcoe_per_mwh"].to_numpy(dtype=float)
    ax3_r.plot(
        x_pos,
        lcoes,
        color="#7e22ce",
        lw=2.3,
        marker="D",
        ms=6.0,
        label="System LCOE (/MWh)",
    )
    for idx_i, val in enumerate(lcoes):
        ax3_r.annotate(
            f"{val:.2f}",
            xy=(float(x_pos[idx_i]), float(val)),
            xytext=(0.0, 9.0),
            textcoords="offset points",
            ha="center",
            fontsize=7.9,
            fontweight="bold",
            color="#581c87",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#e9d5ff", "alpha": 0.92},
        )

    kvl_pen = float(lcoes[1] - lcoes[2])
    ax3_r.annotate(
        f"KVL Loop-Flow\nPenalty +{kvl_pen:.2f} /MWh",
        xy=(1.0, float(lcoes[1])),
        xytext=(1.55, float(lcoes[1]) + 0.6),
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#fff1f2", "ec": "#fda4af"},
        arrowprops={"arrowstyle": "->", "color": "#be123c", "lw": 1.0},
    )
    ax3_r.set_ylabel("System LCOE (currency / MWh)", color="#581c87")
    ax3_r.set_ylim(float(lcoes.min()) - 2.2, float(lcoes.max()) + 4.2)

    h3_l, l3_l = ax3.get_legend_handles_labels()
    h3_r, l3_r = ax3_r.get_legend_handles_labels()
    ax3.legend(
        h3_l + h3_r,
        l3_l + l3_r,
        loc="upper right",
        fontsize=7.1,
        ncol=2,
        framealpha=0.92,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_wacc_capex_and_cvar_adequacy(
    output_path: str | Path,
    *,
    surface: SensitivitySurfaceResult | None = None,
    env_check: EnvelopeConcavityVerification | None = None,
    frontier_df: pd.DataFrame | None = None,
    value_metrics: StochasticValueMetrics | None = None,
    hedged_result: StochasticCVaRResult | None = None,
    hedged_beta: float = 0.35,
) -> Path:
    """Render 4-panel WACCxCapEx sensitivity surface & weather CVaR_95 adequacy figure."""
    _apply_plot_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    surf = sweep_wacc_capex_surface() if surface is None else surface
    env_chk = verify_envelope_and_concavity_theorems() if env_check is None else env_check
    front = sweep_cvar_risk_frontier() if frontier_df is None else frontier_df
    val_m = (
        evaluate_stochastic_value_metrics(hedged_beta=hedged_beta)
        if value_metrics is None
        else value_metrics
    )
    hedged_res = (
        run_stochastic_cvar_scenario(risk_aversion_beta=hedged_beta)
        if hedged_result is None
        else hedged_result
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2))
    fig.suptitle(
        "2D WACC x CapEx Sensitivity Surface, Envelope Concavity & "
        "Multi-Year Weather CVaR_95 Adequacy",
        fontsize=13.2,
        fontweight="bold",
        y=0.982,
    )

    # --- Panel (a): 2D WACC x VRE CapEx System LCOE Heatmap ---
    ax0 = axes[0, 0]
    ax0.grid(False)
    lcoe_mat = surf.lcoe_surface_df.to_numpy(dtype=float)
    im = ax0.imshow(
        lcoe_mat,
        origin="lower",
        aspect="auto",
        cmap="YlGnBu",
        vmin=float(np.min(lcoe_mat)) * 0.95,
        vmax=float(np.max(lcoe_mat)) * 1.03,
    )
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    cbar.set_label("System LCOE (currency / MWh)", fontsize=8.8)

    n_w, n_k = lcoe_mat.shape
    ax0.set_xticks(np.arange(n_k))
    ax0.set_xticklabels([f"{k:.2f}x" for k in surf.vre_capex_scale_grid], fontsize=8.5)
    ax0.set_yticks(np.arange(n_w))
    ax0.set_yticklabels([f"{100.0 * r:.0f}%" for r in surf.wacc_grid], fontsize=8.5)
    ax0.set_xlabel("VRE Overnight CapEx Multiplier (Solar PV & Wind)")
    ax0.set_ylabel("Real Financing Cost (WACC)")
    ax0.set_title("(a) 2D WACC x VRE CapEx System LCOE Surface (/MWh)")

    mid_thresh = 0.49 * (float(np.min(lcoe_mat)) + float(np.max(lcoe_mat)))
    for i in range(n_w):
        for j in range(n_k):
            val = float(lcoe_mat[i, j])
            txt_col = "white" if val >= mid_thresh else "#0f172a"
            is_ref = (
                abs(surf.wacc_grid[i] - surf.reference_wacc) < 1e-6
                and abs(surf.vre_capex_scale_grid[j] - surf.reference_vre_scale) < 1e-6
            )
            label_txt = f"{val:.2f}\n(Ref)" if is_ref else f"{val:.2f}"
            ax0.text(
                j,
                i,
                label_txt,
                ha="center",
                va="center",
                fontsize=8.2 if is_ref else 8.4,
                fontweight="bold",
                color=txt_col,
            )
            if is_ref:
                rect = Rectangle(
                    (j - 0.48, i - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor="#dc2626",
                    lw=2.2,
                    ls="--",
                )
                ax0.add_patch(rect)

    # --- Panel (b): VRE Share Phase Transition & Substitution Dividend ---
    ax1 = axes[0, 1]
    k_arr = np.array(surf.vre_capex_scale_grid, dtype=float)
    w_low = surf.wacc_grid[0]
    w_mid = surf.wacc_grid[len(surf.wacc_grid) // 2]
    w_high = surf.wacc_grid[-1]

    palette_w = [
        (w_low, "#0284c7", "o", f"VRE Share @ WACC={100 * w_low:.0f}%"),
        (w_mid, "#0f766e", "s", f"VRE Share @ WACC={100 * w_mid:.0f}% (Ref)"),
        (w_high, "#dc2626", "^", f"VRE Share @ WACC={100 * w_high:.0f}%"),
    ]
    for w_val, col, mk, lbl in palette_w:
        vre_y = surf.vre_share_surface_df.loc[w_val].to_numpy(dtype=float)
        ax1.plot(k_arr, vre_y, color=col, lw=2.2, marker=mk, ms=5.8, label=lbl)

    ax1.set_xlabel("VRE Overnight CapEx Multiplier")
    ax1.set_ylabel("Optimal VRE Generation Share (%)")
    ax1.set_ylim(-5.0, 110.0)
    ax1.set_xlim(float(k_arr.min()) - 0.06, float(k_arr.max()) + 0.06)
    ax1.set_title("(b) VRE Share Regime Transition & Substitution Dividend")

    ax1_r = ax1.twinx()
    ax1_r.grid(False)
    sub_low = surf.substitution_dividend_surface_df.loc[w_low].to_numpy(dtype=float)
    sub_high = surf.substitution_dividend_surface_df.loc[w_high].to_numpy(dtype=float)
    ax1_r.plot(
        k_arr,
        sub_low,
        color="#7e22ce",
        lw=1.9,
        ls="--",
        marker="D",
        ms=4.8,
        label=f"Subst. Dividend @ {100 * w_low:.0f}% WACC (%)",
    )
    ax1_r.plot(
        k_arr,
        sub_high,
        color="#d97706",
        lw=1.9,
        ls=":",
        marker="P",
        ms=5.5,
        label=f"Subst. Dividend @ {100 * w_high:.0f}% WACC (%)",
    )
    ax1_r.set_ylabel("Substitution Dividend vs Frozen Mix (%)", color="#6b21a8")
    ax1_r.set_ylim(-1.25, max(28.0, float(surf.max_substitution_dividend_pct) * 1.28))

    ax1.text(
        0.03,
        0.37,
        (
            "Shephard-Hotelling Envelope:\n"
            f"  dJ*/dr err = {env_chk.rel_error_dcost_dwacc:.1e}\n"
            f"  dJ*/dk err = {env_chk.rel_error_dcost_dvre_scale:.1e}\n"
            f"  Concavity = +{env_chk.concavity_midpoint_surplus / 1e6:.2f}M/yr"
        ),
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.26",
            "fc": "#f8fafc",
            "ec": "#94a3b8",
            "alpha": 0.95,
        },
    )

    h1_l, l1_l = ax1.get_legend_handles_labels()
    h1_r, l1_r = ax1_r.get_legend_handles_labels()
    ax1.legend(
        h1_l + h1_r,
        l1_l + l1_r,
        loc="upper left",
        bbox_to_anchor=(0.27, 0.77),
        fontsize=6.9,
        framealpha=0.94,
    )

    # --- Panel (c): Mean-CVaR_95 Pareto Risk Frontier ---
    ax2 = axes[1, 0]
    ez_m = front["expected_cost_per_year"].to_numpy(dtype=float) / 1e6
    cv_m = front["cvar_cost_per_year"].to_numpy(dtype=float) / 1e6
    betas = front["beta"].to_numpy(dtype=float)
    eens_arr = front["eens_mwh_per_year"].to_numpy(dtype=float)

    ax2.plot(
        ez_m,
        cv_m,
        color="#1d4ed8",
        lw=2.4,
        marker="o",
        ms=6.5,
        zorder=3,
        label="Mean-CVaR_95 Efficient Frontier",
    )
    ax2.scatter(
        [ez_m[0]],
        [cv_m[0]],
        color="#ef4444",
        s=78,
        zorder=5,
        label=f"Risk-Neutral RP (beta=0, EENS={eens_arr[0]:.1f} MWh/yr)",
    )
    if len(ez_m) > 2:
        ax2.scatter(
            [ez_m[2]],
            [cv_m[2]],
            color="#10b981",
            s=82,
            marker="D",
            zorder=5,
            label=f"CVaR-Hedged (beta={betas[2]:.2f}, EENS=0.0 MWh/yr)",
        )

    offsets = [
        (10.0, 6.0),
        (46.0, 9.0),
        (50.0, -3.0),
        (38.0, -16.0),
        (-28.0, -14.0),
        (-36.0, 7.0),
    ]
    for idx_b, (xv, yv, bv) in enumerate(zip(ez_m, cv_m, betas, strict=True)):
        dx, dy = offsets[idx_b] if idx_b < len(offsets) else (6.0, 4.0)
        use_leader = idx_b in (1, 2, 3)
        ax2.annotate(
            f"b={bv:.2f}",
            xy=(float(xv), float(yv)),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.6,
            fontweight="bold",
            color="#1e3a8a",
            arrowprops=(
                {"arrowstyle": "-", "color": "#94a3b8", "lw": 0.75} if use_leader else None
            ),
        )

    ax2.text(
        0.97,
        0.84,
        (
            "Birge-Louveaux Stochastic Value:\n"
            f"  WS  = {val_m.wait_and_see_cost_per_year / 1e6:6.2f}M / yr\n"
            f"  RP  = {val_m.recourse_problem_cost_per_year / 1e6:6.2f}M / yr\n"
            f"  EEV = {val_m.expected_value_plan_eev_cost_per_year / 1e6:6.2f}M / yr\n"
            f"  EVPI = {val_m.evpi_per_year / 1e6:.2f}M | "
            f"VSS = {val_m.vss_per_year / 1e6:.1f}M/yr\n"
            f"  Det. EENS: {val_m.mean_weather_eens_mwh_per_year:,.0f} -> 0.0 MWh/yr"
        ),
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.32", "fc": "#eff6ff", "ec": "#93c5fd"},
    )

    x_span = max(1.0, float(ez_m.max() - ez_m.min()))
    y_span = max(1.0, float(cv_m.max() - cv_m.min()))
    ax2.set_xlim(float(ez_m.min()) - 0.08 * x_span, float(ez_m.max()) + 0.16 * x_span)
    ax2.set_ylim(float(cv_m.min()) - 0.28 * y_span, float(cv_m.max()) + 0.16 * y_span)
    ax2.set_xlabel("Expected Annual System Cost E[Z] (M-currency / yr)")
    ax2.set_ylabel("Tail Adequacy Cost CVaR_95(Z) (M-currency / yr)")
    ax2.set_title("(c) Mean-CVaR_95 Pareto Frontier & Blackout Elimination")
    ax2.legend(loc="lower left", fontsize=7.2, framealpha=0.94)

    # --- Panel (d): Physical vs Risk-Adjusted Kernel (q*) & Scenario Tail Cost ---
    ax3 = axes[1, 1]
    sc_tbl = hedged_res.scenario_table
    x_idx = np.arange(len(sc_tbl), dtype=float)
    short_labels = [
        "WY1\nWindy",
        "WY2\nSunny",
        "WY3\nNormal",
        "WY4\nCloudy",
        "WY5\nHeat",
        "WY6*\nTrough",
        "WY7\nFreeze",
        "WY8\nDunkel.",
    ]
    if len(short_labels) != len(sc_tbl):
        short_labels = [s.split("_")[0] for s in sc_tbl.index]

    pi_pct = 100.0 * sc_tbl["probability"].to_numpy(dtype=float)
    q_pct = 100.0 * sc_tbl["risk_adjusted_prob"].to_numpy(dtype=float)
    z_scen_m = sc_tbl["total_cost_per_year"].to_numpy(dtype=float) / 1e6

    bw = 0.36
    ax3.bar(
        x_idx - bw / 2.0,
        pi_pct,
        width=bw,
        color="#64748b",
        label="Physical Weather Prob pi_w (%)",
    )
    ax3.bar(
        x_idx + bw / 2.0,
        q_pct,
        width=bw,
        color="#ea580c",
        label=f"CVaR_95 Dual Kernel q*_w (b={hedged_beta:.2f}) (%)",
    )
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(short_labels, fontsize=8.0)
    ax3.set_ylabel("Scenario Probability Weight (%)")
    ax3.set_ylim(0.0, max(float(pi_pct.max()), float(q_pct.max())) * 1.38)
    ax3.set_title("(d) Physical vs CVaR_95 Dual Measure (q*) & Scenario Cost")

    ax3_r = ax3.twinx()
    ax3_r.grid(False)
    ax3_r.plot(
        x_idx,
        z_scen_m,
        color="#7e22ce",
        lw=2.2,
        marker="D",
        ms=5.5,
        label="Scenario Cost Z_w (M/yr)",
    )
    var_m = hedged_res.var_total_cost_per_year / 1e6
    ax3_r.axhline(
        var_m,
        color="#be123c",
        ls="--",
        lw=1.3,
        label=f"VaR_95 Threshold ({var_m:.1f}M/yr)",
    )
    ax3_r.set_ylabel("Scenario Total Cost Z(K*, w) (M-currency / yr)", color="#581c87")
    ax3_r.set_ylim(float(z_scen_m.min()) * 0.78, float(z_scen_m.max()) * 1.24)

    ax3.annotate(
        "7.65x Tail Weight\nAmplification (WY7-WY8)",
        xy=(6.0 + bw / 2.0, float(q_pct[6])),
        xytext=(3.4, float(max(pi_pct.max(), q_pct.max())) * 1.12),
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#fff7ed", "ec": "#fdba74"},
        arrowprops={"arrowstyle": "->", "color": "#c2410c", "lw": 1.0},
    )

    h3_l, l3_l = ax3.get_legend_handles_labels()
    h3_r, l3_r = ax3_r.get_legend_handles_labels()
    ax3.legend(
        h3_l + h3_r,
        l3_l + l3_r,
        loc="upper left",
        fontsize=6.9,
        ncol=2,
        framealpha=0.92,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
