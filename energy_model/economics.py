"""Discounted cash-flow economics (WACC / CRF / LCOE) and KKT nodal price duality.

Mathematical identities verified by this module
-----------------------------------------------
1. **Capital Recovery Factor (CRF) & Present-Value Exactness:**
   For real discount rate ``r >= 0`` and economic lifetime ``N > 0`` years,

       CRF(r, N) = r * (1 + r)**N / ((1 + r)**N - 1)   for r > 0,
       CRF(0, N) = 1 / N.

   For integer ``N``, the present value of ``N`` equal end-of-year payments
   of ``CRF(r, N) * I_0`` discounted at ``r`` equals ``I_0`` identically.

2. **LP Strong Duality (System Cost == Nodal Shadow-Price Demand Payment):**
   For a linear capacity-expansion problem with extendable capacities starting at
   zero and hourly power balance ``sum_g p_{g,t} = d_t`` (dual ``w_t * lambda_t``),

       C_total,yr = sum_t w_t * lambda_t * d_t * dt,
       System_LCOE = C_total,yr / E_demand,yr = lambda_load_weighted.

3. **Boiteux-Turvey-Stoft Zero-Profit & Brown-Hirth Market-Value Equilibrium:**
   By KKT complementary slackness, every technology built with ``p_nom,g* > 0``
   earns exact zero economic profit at the hourly nodal shadow prices:

       R_g = sum_t w_t * lambda_t * p_{g,t}* * dt = c_cap,g * p_nom,g* + OpEx_g,
       MarketValue_g = R_g / E_g,yr = LCOE_g.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .model import (
    DAYS_PER_YEAR,
    DEFAULT_PROFILE_PATH,
    DEFAULT_TECHNOLOGIES,
    ScenarioResult,
    run_scenario,
)
from .profiles import SNAPSHOT_HOURS, load_profiles, validate_profiles

REFERENCE_WACC = 0.07
DEFAULT_WACC_SWEEP: tuple[float, ...] = (0.00, 0.02, 0.04, 0.07, 0.10, 0.12, 0.15)


def _validate_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite")
    return val


def capital_recovery_factor(wacc: float, lifetime_years: float) -> float:
    """Return the exact Capital Recovery Factor ``CRF(wacc, lifetime_years)`` (1/yr)."""
    r = _validate_finite_float(wacc, "wacc")
    n = _validate_finite_float(lifetime_years, "lifetime_years")
    if r < 0.0 or r > 1.0:
        raise ValueError("wacc must lie within [0.0, 1.0]")
    if n <= 0.0:
        raise ValueError("lifetime_years must be strictly positive")
    if r < 1e-12:
        return 1.0 / n
    growth = (1.0 + r) ** n
    return (r * growth) / (growth - 1.0)


def annualize_overnight_capex(
    overnight_capex_per_mw: float,
    wacc: float,
    lifetime_years: float,
    fixed_om_per_mw_yr: float = 0.0,
) -> float:
    """Convert overnight CapEx (currency/MW) + fixed O&M into annualized capital_cost."""
    i0 = _validate_finite_float(overnight_capex_per_mw, "overnight_capex_per_mw")
    fom = _validate_finite_float(fixed_om_per_mw_yr, "fixed_om_per_mw_yr")
    if i0 < 0.0:
        raise ValueError("overnight_capex_per_mw must be non-negative")
    if fom < 0.0:
        raise ValueError("fixed_om_per_mw_yr must be non-negative")
    return i0 * capital_recovery_factor(wacc, lifetime_years) + fom


@dataclass(frozen=True)
class TechnologyFinancialSpec:
    """Parametric financial specification calibrated to DEFAULT_TECHNOLOGIES at r_ref=7%."""

    name: str
    overnight_capex_per_mw: float
    lifetime_years: float
    fixed_om_per_mw_yr: float
    marginal_cost_per_mwh: float
    profile: str | None


def default_financial_specs(
    reference_wacc: float = REFERENCE_WACC,
) -> dict[str, TechnologyFinancialSpec]:
    """Return overnight CapEx specs calibrated to DEFAULT_TECHNOLOGIES at reference_wacc."""
    lifetimes = {"Solar_PV": 25.0, "Wind_Turbine": 25.0, "Gas_Peaker": 20.0}
    specs: dict[str, TechnologyFinancialSpec] = {}
    for name, raw in DEFAULT_TECHNOLOGIES.items():
        n_yr = lifetimes[name]
        crf_ref = capital_recovery_factor(reference_wacc, n_yr)
        specs[name] = TechnologyFinancialSpec(
            name=name,
            overnight_capex_per_mw=float(raw["capital_cost"]) / crf_ref,
            lifetime_years=n_yr,
            fixed_om_per_mw_yr=0.0,
            marginal_cost_per_mwh=float(raw["marginal_cost"]),
            profile=raw["profile"],
        )
    return specs


def build_wacc_technologies(
    wacc: float,
    financial_specs: dict[str, TechnologyFinancialSpec] | None = None,
) -> dict[str, dict]:
    """Construct a PyPSA technology dictionary annualized at discount rate ``wacc``."""
    specs = default_financial_specs() if financial_specs is None else financial_specs
    if not specs:
        raise ValueError("financial_specs must not be empty")
    techs: dict[str, dict] = {}
    for name, sp in specs.items():
        techs[name] = {
            "capital_cost": annualize_overnight_capex(
                sp.overnight_capex_per_mw,
                wacc=wacc,
                lifetime_years=sp.lifetime_years,
                fixed_om_per_mw_yr=sp.fixed_om_per_mw_yr,
            ),
            "marginal_cost": sp.marginal_cost_per_mwh,
            "p_nom_extendable": True,
            "profile": sp.profile,
        }
    return techs


@dataclass(frozen=True)
class MarketDualitySummary:
    """System & technology-level LCOE, Market Value, and KKT strong-duality diagnostics."""

    system_lcoe_per_mwh: float
    load_weighted_lmp_per_mwh: float
    time_weighted_lmp_per_mwh: float
    peak_lmp_per_mwh: float
    annual_demand_payment: float
    strong_duality_gap: float
    max_zero_profit_residual: float
    zero_price_hours: int
    scarcity_hours: int
    technology_table: pd.DataFrame


def analyze_market_duality_and_lcoe(
    result: ScenarioResult,
    profiles: pd.DataFrame | None = None,
    technologies: dict[str, dict] | None = None,
) -> MarketDualitySummary:
    """Evaluate technology LCOE, Market Value, Value Factor, and KKT duality residuals."""
    if result.lmp_per_mwh.empty or result.lmp_per_mwh.isna().any():
        raise ValueError("ScenarioResult does not contain finite nodal shadow prices (lmp_per_mwh)")

    techs = DEFAULT_TECHNOLOGIES if technologies is None else technologies
    dt = result.snapshot_hours
    weight = float(result.days_per_year) * dt
    lmp = result.lmp_per_mwh.astype(float)

    if profiles is not None:
        validated = validate_profiles(profiles)
        load_mw = validated["load_mw"].astype(float)
    else:
        load_mw = result.dispatch_mw.sum(axis=1).astype(float)

    annual_demand_mwh = float(load_mw.sum() * weight)
    if annual_demand_mwh <= 0.0:
        raise ValueError("annual demand energy must be strictly positive to compute LCOE")

    annual_demand_payment = float((load_mw * lmp).sum() * weight)
    system_lcoe = result.total_cost_per_year / annual_demand_mwh
    load_weighted_lmp = annual_demand_payment / annual_demand_mwh
    time_weighted_lmp = float(lmp.mean())
    peak_lmp = float(lmp.max())
    strong_duality_gap = annual_demand_payment - result.total_cost_per_year

    rows: list[dict[str, float]] = []
    built_profit_residuals: list[float] = []
    max_thermal_mc = max(
        (float(spec["marginal_cost"]) for spec in techs.values()),
        default=0.0,
    )

    for g in result.p_nom_mw.index:
        p_nom = float(result.p_nom_mw[g])
        e_yr = float(result.energy_mwh_per_year[g])
        cap_cost = float(techs[g]["capital_cost"]) * p_nom
        op_cost = float(techs[g]["marginal_cost"]) * e_yr
        tot_cost = cap_cost + op_cost
        rev_yr = float((result.dispatch_mw[g] * lmp).sum() * weight)
        net_profit = rev_yr - tot_cost

        if p_nom > 1e-6 and bool(techs[g].get("p_nom_extendable", True)):
            built_profit_residuals.append(abs(net_profit))

        if e_yr > 1e-6:
            lcoe_g = tot_cost / e_yr
            mv_g = rev_yr / e_yr
            vf_g = mv_g / load_weighted_lmp if load_weighted_lmp > 1e-12 else math.nan
        else:
            lcoe_g = math.nan
            mv_g = math.nan
            vf_g = math.nan

        rows.append(
            {
                "p_nom_mw": p_nom,
                "energy_mwh_per_year": e_yr,
                "annual_capex": cap_cost,
                "annual_opex": op_cost,
                "annual_total_cost": tot_cost,
                "annual_lmp_revenue": rev_yr,
                "net_profit": net_profit,
                "lcoe_per_mwh": lcoe_g,
                "market_value_per_mwh": mv_g,
                "value_factor": vf_g,
            }
        )

    table = pd.DataFrame(rows, index=result.p_nom_mw.index)
    zero_hours = int((lmp.abs() <= 1e-6).sum())
    scarcity_hours = int((lmp > max_thermal_mc + 1e-5).sum())

    return MarketDualitySummary(
        system_lcoe_per_mwh=system_lcoe,
        load_weighted_lmp_per_mwh=load_weighted_lmp,
        time_weighted_lmp_per_mwh=time_weighted_lmp,
        peak_lmp_per_mwh=peak_lmp,
        annual_demand_payment=annual_demand_payment,
        strong_duality_gap=strong_duality_gap,
        max_zero_profit_residual=max(built_profit_residuals, default=0.0),
        zero_price_hours=zero_hours,
        scarcity_hours=scarcity_hours,
        technology_table=table,
    )


@dataclass(frozen=True)
class PrimalDualCrossCheck:
    """Independent SciPy HiGHS primal + KKT dual cross-check result."""

    objective_per_year: float
    p_nom_mw: pd.Series
    lmp_per_mwh: pd.Series
    max_p_nom_diff_mw: float
    max_lmp_diff_per_mwh: float
    rel_objective_diff: float


def solve_primal_dual_linprog_crosscheck(
    profiles: pd.DataFrame | None = None,
    technologies: dict[str, dict] | None = None,
    days_per_year: int = DAYS_PER_YEAR,
) -> PrimalDualCrossCheck:
    """Solve the exact primal/dual LP via ``scipy.optimize.linprog`` and compare to PyPSA."""
    prof = load_profiles(DEFAULT_PROFILE_PATH) if profiles is None else validate_profiles(profiles)
    techs = DEFAULT_TECHNOLOGIES if technologies is None else technologies

    pypsa_res = run_scenario(prof, techs, days_per_year=days_per_year)

    names = list(techs)
    n_gen, n_t = len(names), len(prof)
    load = prof["load_mw"].to_numpy(dtype=float)
    cf: dict[str, np.ndarray] = {}
    for g in names:
        col = techs[g].get("profile")
        cf[g] = np.ones(n_t, dtype=float) if col is None else prof[col].to_numpy(dtype=float)

    n_var = n_gen * n_t + n_gen
    c = np.zeros(n_var, dtype=float)
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_var

    for i, g in enumerate(names):
        c[i * n_t : (i + 1) * n_t] = (
            float(days_per_year) * float(techs[g]["marginal_cost"]) * SNAPSHOT_HOURS
        )
        c[n_gen * n_t + i] = float(techs[g]["capital_cost"])
        if not bool(techs[g].get("p_nom_extendable", True)):
            fixed_cap = float(techs[g].get("p_nom", 0.0))
            bounds[n_gen * n_t + i] = (fixed_cap, fixed_cap)

    a_eq = np.zeros((n_t, n_var), dtype=float)
    for t in range(n_t):
        for i in range(n_gen):
            a_eq[t, i * n_t + t] = 1.0

    ub_rows = []
    for i, g in enumerate(names):
        for t in range(n_t):
            row = np.zeros(n_var, dtype=float)
            row[i * n_t + t] = 1.0
            row[n_gen * n_t + i] = -cf[g][t]
            ub_rows.append(row)

    lp = linprog(
        c,
        A_ub=np.asarray(ub_rows, dtype=float),
        b_ub=np.zeros(len(ub_rows), dtype=float),
        A_eq=a_eq,
        b_eq=load,
        bounds=bounds,
        method="highs",
    )
    if lp.status != 0:
        raise RuntimeError(f"scipy linprog cross-check failed: {lp.message}")

    p_nom_scipy = pd.Series(lp.x[n_gen * n_t :], index=names, dtype=float)
    lmp_scipy_raw = np.asarray(lp.eqlin.marginals, dtype=float) / (
        float(days_per_year) * SNAPSHOT_HOURS
    )
    lmp_scipy_clean = np.where(np.abs(lmp_scipy_raw) > 1e-12, lmp_scipy_raw, 0.0)
    lmp_scipy = pd.Series(lmp_scipy_clean, index=prof.index, dtype=float)

    max_p_diff = float((pypsa_res.p_nom_mw[names] - p_nom_scipy).abs().max())
    max_lmp_diff = float((pypsa_res.lmp_per_mwh - lmp_scipy).abs().max())
    denom = max(1.0, abs(float(lp.fun)))
    rel_obj_diff = abs(pypsa_res.total_cost_per_year - float(lp.fun)) / denom

    return PrimalDualCrossCheck(
        objective_per_year=float(lp.fun),
        p_nom_mw=p_nom_scipy,
        lmp_per_mwh=lmp_scipy,
        max_p_nom_diff_mw=max_p_diff,
        max_lmp_diff_per_mwh=max_lmp_diff,
        rel_objective_diff=rel_obj_diff,
    )


def sweep_wacc_lcoe_and_mix(
    profiles: pd.DataFrame | None = None,
    wacc_values: Sequence[float] = DEFAULT_WACC_SWEEP,
    days_per_year: int = DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Sweep real discount rate ``WACC`` and record optimal capacities, VRE share & System LCOE."""
    if not wacc_values:
        raise ValueError("wacc_values must contain at least one discount rate")
    prof = load_profiles(DEFAULT_PROFILE_PATH) if profiles is None else validate_profiles(profiles)
    fin_specs = default_financial_specs()

    records: list[dict[str, float]] = []
    for wacc in wacc_values:
        techs = build_wacc_technologies(wacc, financial_specs=fin_specs)
        res = run_scenario(prof, techs, days_per_year=days_per_year)
        dual = analyze_market_duality_and_lcoe(res, profiles=prof, technologies=techs)
        vre_mwh = float(
            res.energy_mwh_per_day.get("Solar_PV", 0.0)
            + res.energy_mwh_per_day.get("Wind_Turbine", 0.0)
        )
        records.append(
            {
                "wacc": float(wacc),
                "wacc_pct": float(wacc) * 100.0,
                "solar_mw": float(res.p_nom_mw.get("Solar_PV", 0.0)),
                "wind_mw": float(res.p_nom_mw.get("Wind_Turbine", 0.0)),
                "gas_mw": float(res.p_nom_mw.get("Gas_Peaker", 0.0)),
                "vre_share_pct": 100.0 * vre_mwh / res.demand_mwh_per_day,
                "system_lcoe_per_mwh": dual.system_lcoe_per_mwh,
                "peak_lmp_per_mwh": dual.peak_lmp_per_mwh,
                "total_cost_per_year": res.total_cost_per_year,
            }
        )
    return pd.DataFrame(records)


def format_duality_report(duality: MarketDualitySummary) -> str:
    """Format KKT strong-duality, nodal LMP, and technology LCOE/Market-Value diagnostics."""
    lines = [
        "KKT Nodal Shadow-Price (LMP) & LCOE Equilibrium Diagnostics:",
        f"  System LCOE (currency/MWh):           {duality.system_lcoe_per_mwh:>12,.4f}",
        f"  Demand-weighted LMP (currency/MWh):   {duality.load_weighted_lmp_per_mwh:>12,.4f}",
        f"  Time-weighted mean LMP (currency/MWh):{duality.time_weighted_lmp_per_mwh:>12,.4f}",
        f"  Peak scarcity LMP (currency/MWh):     {duality.peak_lmp_per_mwh:>12,.4f}",
        (
            f"  Zero-price / scarcity hours (of 24):  "
            f"{duality.zero_price_hours:>5d} / {duality.scarcity_hours:<5d}"
        ),
        f"  Strong-duality gap (currency/yr):     {duality.strong_duality_gap:>12.4e}",
        f"  Max built zero-profit residual:       {duality.max_zero_profit_residual:>12.4e}",
        "",
        "  Technology       LCOE (/MWh)   Market Value (/MWh)   Value Factor   Net Profit (/yr)",
    ]
    for name, row in duality.technology_table.iterrows():
        lines.append(
            f"  {name:<14} {row['lcoe_per_mwh']:>13,.4f}   "
            f"{row['market_value_per_mwh']:>19,.4f}   "
            f"{row['value_factor']:>12,.4f}   "
            f"{row['net_profit']:>16.4e}"
        )
    return "\n".join(lines)
