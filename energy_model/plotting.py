"""Publication-grade visualisations for capacity expansion, dispatch, KKT LMPs & storage."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .economics import analyze_market_duality_and_lcoe, sweep_wacc_lcoe_and_mix
from .emissions import (
    sweep_carbon_tax_equivalents,
    sweep_co2_pareto_and_macc,
)
from .model import DEFAULT_PROFILE_PATH, DEFAULT_TECHNOLOGIES, ScenarioResult, run_scenario
from .profiles import load_profiles, validate_profiles
from .storage import (
    StorageScenarioResult,
    compare_storage_adequacy_regimes,
    generate_dunkelflaute_profiles,
    run_storage_scenario,
    sweep_dunkelflaute_duration,
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
    ax1.step(hours, lmp_vals, where="mid", color="#7c3aed", lw=2.2, label=r"Nodal LMP $\lambda_t$")
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
            f"$\\lambda_{{18}}$ = {lmp_vals[peak_idx]:.2f} /MWh\n"
            "(= 65 + 20,000/365)"
        ),
        xy=(hours[peak_idx], lmp_vals[peak_idx]),
        xytext=(hours[peak_idx] - 7.8, lmp_vals[peak_idx] - 18.0),
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.0},
    )
    ax1.set_title(r"(b) KKT Nodal Shadow Price $\lambda_t$ & Scarcity-Rent Regimes")
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
        label=r"Recycled Tax Transfer $\tau \cdot E^*(\tau)$",
    )
    ax0.plot(
        t_e_kt,
        t_priv_m,
        color="#7e22ce",
        lw=1.8,
        ls="--",
        label=r"Private Cost under Tax $J_{\mathrm{tax}}^*(\tau)$",
    )
    ax0.plot(
        e_kt,
        c_res_m,
        color="#0284c7",
        lw=2.4,
        marker="o",
        ms=5.2,
        label=r"Cap Resource Cost $J_{\mathrm{cap}}^*(\bar{E})$",
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
        label=r"Tax Resource Cost $J_{\mathrm{res}}^*(\tau)$ (Duality)",
    )

    # Annotate BAU, 75% tangent point, and Net-Zero
    idx_75 = int(np.argmin(np.abs(macc["abatement_realised_pct"].to_numpy(dtype=float) - 75.0)))
    e75 = float(e_kt[idx_75])
    c75 = float(c_res_m[idx_75])
    mu75 = float(macc["marginal_abatement_cost_per_tco2"].iloc[idx_75])
    ax0.annotate(
        (
            f"75% Abatement\n$E^*$ = {e75:.0f} ktCO$_2$/yr\n"
            f"$-\\partial J^*/\\partial \\bar{{E}}$ = {mu75:.2f} /tCO$_2$"
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
    ax0.set_xlabel(r"Annual $\mathrm{CO_2}$ Emissions ($10^3\ \mathrm{tCO_2 / yr}$)")
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
        label=r"Marginal Abatement Cost $\mu_{\mathrm{CO_2}}^*$",
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
        "Stage I: Wind Entry\n(~6-10 /tCO$_2$)",
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
        "Stage II: Coal -> CCGT\n+ Solar (~19-29 /tCO$_2$)",
        xy=(85.0, float(mac[np.argmin(np.abs(ab_pct - 85.0))])),
        xytext=(42.0, 48.0),
        fontsize=7.7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 0.9},
    )
    ax1.annotate(
        "Stage III: 90% Gas CCS\n+ BESS (~53-104 /tCO$_2$)",
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
    ax1.set_xlabel(r"Cumulative $\mathrm{CO_2}$ Abatement (% of BAU)")
    ax1.set_ylabel(r"Carbon Price / Cost (currency / $\mathrm{tCO_2}$, symlog)")
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
    ax2.set_xlabel(r"Cumulative $\mathrm{CO_2}$ Abatement (% of BAU)")
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
        label=r"Green $\mathrm{H_2}$ Cavern ($\mathrm{GWh_{H2}}$)",
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
    ax2_r.set_ylabel(r"Storage Capacity ($\mathrm{GWh_{H2}}$ or $10^1$ MWh BESS)", color="#9f1239")
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
        label=r"Equiv. Tax Revenue $\mu^* E^*$ (M/yr)",
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
