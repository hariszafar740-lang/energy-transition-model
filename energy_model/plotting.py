"""Publication-grade visualisations for capacity expansion, dispatch & KKT nodal LMPs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .economics import analyze_market_duality_and_lcoe, sweep_wacc_lcoe_and_mix
from .model import DEFAULT_PROFILE_PATH, DEFAULT_TECHNOLOGIES, ScenarioResult, run_scenario
from .profiles import load_profiles, validate_profiles


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

    # Panel (a): Dispatch Stack + Curtailment
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

    # Panel (b): Hourly KKT Nodal Shadow Price (LMP)
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

    # Panel (c): KKT Zero-Profit & LCOE == Market Value Equilibrium
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

    # Panel (d): WACC Sensitivity of Optimal Capacity Mix & System LCOE
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
