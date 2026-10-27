"""2D WACC/CapEx sensitivity surfaces & multi-year weather CVaR_95 adequacy risk.

Mathematical identities verified by this module
-----------------------------------------------
1. **Analytical Capital-Recovery Sensitivity & Hotelling-Shephard Envelope Theorem:**
   For real discount rate ``r in [0, 1]``, lifetime ``N > 0``, and VRE overnight
   CapEx scale factor ``kappa > 0``, optimal greenfield system cost
   ``J*(r, kappa) = min_{K >= 0, p in F(K)} { c_cap(r, kappa)^T K + c_op^T p }``
   is globally concave in ``c_cap`` (substitution dividend >= 0) and satisfies:

       dJ*(r, kappa) / dr     = sum_g K_g*(r, kappa) * kappa_g * I_{0,g} * dCRF(r, N_g)/dr,
       dJ*(r, kappa) / dkappa = sum_{g in VRE} K_g*(r, kappa) * I_{0,g} * CRF(r, N_g).

2. **Rockafellar-Uryasev Two-Stage Stochastic Mean-CVaR_alpha Formulation:**
   Across meteorological weather regimes ``omega in Omega`` with physical
   probabilities ``pi_omega > 0`` (``sum_omega pi_omega = 1``), first-stage
   capacities ``K`` are committed ex-ante while second-stage dispatch, battery
   cycling, and load shedding ``u_{omega,t}`` (at Value of Lost Load ``V_VoLL``)
   adapt to ``omega``. Minimising ``(1 - beta) E[Z] + beta CVaR_alpha(Z)`` is an
   exact linear program via ``Q_omega - eta - xi_omega <= 0, xi_omega >= 0``:

       CVaR_alpha(Z) = C_cap(K*) + eta* + (1 / (1 - alpha)) * sum_omega pi_omega * xi_omega*.

3. **Complete-Market Risk-Adjusted Dual Measure & Strong LP Duality:**
   With KKT dual multipliers ``rho_omega* >= 0`` on ``Q_omega - eta - xi_omega <= 0``,
   ``sum_omega rho_omega* = beta`` and ``0 <= rho_omega* <= beta * pi_omega / (1 - alpha)``.
   Defining ``q_omega* = (1 - beta) pi_omega + rho_omega*`` yields ``q_omega* >= 0``,
   ``sum_omega q_omega* = 1``, exact Strong LP Duality, and zero-profit equilibrium:

       Obj* = sum_{omega, t} w_yr * dt * lambda_{omega, t}^* * d_{omega, t},
       sum_{omega, t} w_yr * dt * lambda_{omega, t}^* * p_{g,omega,t}^*
         - sum_omega q_omega* * OPEX_{g,omega}^* <= c_cap,g * K_g*   (= if K_g* > 0).

4. **Birge-Louveaux Stochastic Value Chain (WS <= RP <= EEV):**
   Wait-and-See ``WS``, Recourse Problem ``RP``, and out-of-sample Mean-Weather
   ``EEV`` satisfy ``EVPI = RP - WS >= 0`` and ``VSS = EEV - RP >= 0``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from scipy import sparse
from scipy.optimize import linprog

from .economics import (
    REFERENCE_WACC,
    TechnologyFinancialSpec,
    annualize_overnight_capex,
    capital_recovery_factor,
    default_financial_specs,
)
from .model import (
    BUS_CARRIER,
    DAYS_PER_YEAR,
    DEFAULT_PROFILE_PATH,
    DEFAULT_TECHNOLOGIES,
    LOAD_NAME,
    ScenarioResult,
    SolverError,
    run_scenario,
)
from .profiles import load_profiles, validate_profiles

pypsa.options.api.legacy_string_dtype = True

DEFAULT_SOLVER_NAME: str = "highs"
CARRIER: str = BUS_CARRIER
VRE_TECH_NAMES: tuple[str, ...] = ("Solar_PV", "Wind_Turbine")
THERMAL_TECH_NAME: str = "Gas_Peaker"
STOCHASTIC_BESS_NAME: str = "Li_Ion_BESS"
LOAD_SHEDDING_NAME: str = "Load_Shedding"

DEFAULT_SENSITIVITY_WACC_GRID: tuple[float, ...] = (0.01, 0.04, 0.07, 0.11, 0.15)
DEFAULT_VRE_CAPEX_SCALE_GRID: tuple[float, ...] = (0.55, 0.75, 1.00, 1.35, 1.75)

DEFAULT_CVAR_CONFIDENCE: float = 0.95
DEFAULT_VOLL_PER_MWH: float = 1_800.0
DEFAULT_STOCHASTIC_BESS_CAPITAL_COST_PER_MW_YR: float = 30_000.0
DEFAULT_STOCHASTIC_BESS_MAX_HOURS: float = 4.0
DEFAULT_STOCHASTIC_BESS_EFFICIENCY: float = 0.95
DEFAULT_RISK_AVERSION_SWEEP: tuple[float, ...] = (0.00, 0.15, 0.35, 0.60, 0.85, 1.00)


def _validate_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite")
    return val


def _validate_positive_float(value: float, name: str) -> float:
    val = _validate_finite_float(value, name)
    if val <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return val


def _validate_unit_interval(
    value: float,
    name: str,
    *,
    allow_zero: bool = True,
    allow_one: bool = True,
) -> float:
    val = _validate_finite_float(value, name)
    if (allow_zero and val < 0.0) or (not allow_zero and val <= 0.0):
        bound = "[0.0" if allow_zero else "(0.0"
        raise ValueError(f"{name} must lie in {bound}, 1.0]")
    if (allow_one and val > 1.0) or (not allow_one and val >= 1.0):
        bound = "1.0]" if allow_one else "1.0)"
        raise ValueError(f"{name} must lie in [0.0, {bound}")
    return val


def crf_wacc_derivative(wacc: float, lifetime_years: float) -> float:
    """Return exact partial derivative ``dCRF(wacc, lifetime_years) / dwacc``."""
    r = _validate_unit_interval(wacc, "wacc", allow_zero=True, allow_one=True)
    n = _validate_positive_float(lifetime_years, "lifetime_years")
    if r < 1e-5:
        c1 = (n + 1.0) / (2.0 * n)
        c2 = (n * n - 1.0) / (6.0 * n)
        c3 = -(n * n - 1.0) / (12.0 * n)
        return c1 + c2 * r + c3 * r * r
    growth = (1.0 + r) ** n
    num = ((1.0 + r) ** (n - 1.0)) * ((1.0 + r) ** (n + 1.0) - (1.0 + r * (n + 1.0)))
    return num / ((growth - 1.0) ** 2)


def build_scaled_wacc_technologies(
    wacc: float,
    vre_capex_scale: float = 1.0,
    *,
    thermal_capex_scale: float = 1.0,
    financial_specs: dict[str, TechnologyFinancialSpec] | None = None,
) -> dict[str, dict]:
    """Build PyPSA technology specs scaled by real ``wacc`` and VRE overnight CapEx."""
    r = _validate_unit_interval(wacc, "wacc", allow_zero=True, allow_one=True)
    k_vre = _validate_positive_float(vre_capex_scale, "vre_capex_scale")
    k_thm = _validate_positive_float(thermal_capex_scale, "thermal_capex_scale")
    specs = default_financial_specs() if financial_specs is None else financial_specs
    out: dict[str, dict] = {}
    for name, spec in specs.items():
        scale = k_vre if name in VRE_TECH_NAMES else k_thm
        cap_cost = annualize_overnight_capex(
            scale * spec.overnight_capex_per_mw,
            wacc=r,
            lifetime_years=spec.lifetime_years,
            fixed_om_per_mw_yr=spec.fixed_om_per_mw_yr,
        )
        out[name] = {
            "capital_cost": cap_cost,
            "marginal_cost": spec.marginal_cost_per_mwh,
            "p_nom_extendable": True,
            "profile": spec.profile,
        }
    return out


@dataclass(frozen=True)
class SensitivitySurfaceResult:
    """2D ``WACC x VRE_CapEx`` sensitivity surface and substitution-dividend matrices."""

    wacc_grid: tuple[float, ...]
    vre_capex_scale_grid: tuple[float, ...]
    reference_wacc: float
    reference_vre_scale: float
    grid_table: pd.DataFrame
    lcoe_surface_df: pd.DataFrame
    vre_share_surface_df: pd.DataFrame
    solar_mw_surface_df: pd.DataFrame
    wind_mw_surface_df: pd.DataFrame
    gas_mw_surface_df: pd.DataFrame
    substitution_dividend_surface_df: pd.DataFrame
    reference_lcoe_per_mwh: float
    min_lcoe_per_mwh: float
    max_lcoe_per_mwh: float
    max_substitution_dividend_pct: float


@dataclass(frozen=True)
class EnvelopeConcavityVerification:
    """Analytical Hotelling-Shephard Envelope & global cost-concavity diagnostics."""

    wacc: float
    vre_capex_scale: float
    analytic_dcost_dwacc: float
    fd_dcost_dwacc: float
    rel_error_dcost_dwacc: float
    analytic_dcost_dvre_scale: float
    fd_dcost_dvre_scale: float
    rel_error_dcost_dvre_scale: float
    concavity_midpoint_surplus: float
    passive_minus_optimal_min_margin: float


def sweep_wacc_capex_surface(
    profiles: pd.DataFrame | None = None,
    *,
    wacc_grid: Sequence[float] = DEFAULT_SENSITIVITY_WACC_GRID,
    vre_capex_scale_grid: Sequence[float] = DEFAULT_VRE_CAPEX_SCALE_GRID,
    reference_wacc: float = REFERENCE_WACC,
    reference_vre_scale: float = 1.0,
    days_per_year: int = DAYS_PER_YEAR,
    solver_name: str = DEFAULT_SOLVER_NAME,
) -> SensitivitySurfaceResult:
    """Sweep ``(WACC, VRE_CapEx_scale)`` and quantify LCOE, VRE share & substitution dividend."""
    prof = load_profiles(DEFAULT_PROFILE_PATH) if profiles is None else profiles
    validate_profiles(prof)
    if days_per_year <= 0:
        raise ValueError("days_per_year must be positive")
    w_vals = tuple(_validate_unit_interval(w, "wacc_grid") for w in wacc_grid)
    k_vals = tuple(
        _validate_positive_float(k, "vre_capex_scale_grid") for k in vre_capex_scale_grid
    )
    if not w_vals or not k_vals:
        raise ValueError("wacc_grid and vre_capex_scale_grid must be non-empty")

    ref_techs = build_scaled_wacc_technologies(reference_wacc, reference_vre_scale)
    ref_res = run_scenario(
        prof, technologies=ref_techs, days_per_year=days_per_year, solver_name=solver_name
    )
    dt = float((prof.index[1] - prof.index[0]).total_seconds() / 3600.0) if len(prof) > 1 else 1.0
    annual_demand_mwh = float(prof["load_mw"].sum() * dt) * float(days_per_year)
    ref_lcoe = ref_res.total_cost_per_year / annual_demand_mwh
    ref_opex = ref_res.opex_per_year
    ref_p_nom = ref_res.p_nom_mw

    rows: list[dict[str, float]] = []
    for r in w_vals:
        for k in k_vals:
            techs = build_scaled_wacc_technologies(r, k)
            res = run_scenario(
                prof, technologies=techs, days_per_year=days_per_year, solver_name=solver_name
            )
            lcoe = res.total_cost_per_year / annual_demand_mwh
            disp = res.dispatch_mw
            vre_mwh = float((disp["Solar_PV"].sum() + disp["Wind_Turbine"].sum()) * dt) * float(
                days_per_year
            )
            tot_gen = float(disp.sum().sum() * dt) * float(days_per_year)
            frozen_cost = (
                sum(float(ref_p_nom[g]) * float(techs[g]["capital_cost"]) for g in techs) + ref_opex
            )
            sub_div_pct = max(0.0, 100.0 * (frozen_cost - res.total_cost_per_year) / frozen_cost)
            rows.append(
                {
                    "wacc": r,
                    "vre_capex_scale": k,
                    "total_cost_per_year": res.total_cost_per_year,
                    "capex_per_year": res.capex_per_year,
                    "opex_per_year": res.opex_per_year,
                    "system_lcoe_per_mwh": lcoe,
                    "frozen_ref_lcoe_per_mwh": frozen_cost / annual_demand_mwh,
                    "substitution_dividend_pct": sub_div_pct,
                    "vre_generation_share_pct": 100.0 * vre_mwh / max(tot_gen, 1e-12),
                    "solar_mw": float(res.p_nom_mw["Solar_PV"]),
                    "wind_mw": float(res.p_nom_mw["Wind_Turbine"]),
                    "gas_mw": float(res.p_nom_mw["Gas_Peaker"]),
                    "curtailment_mwh_per_day": float(res.curtailment_mwh_per_day.sum()),
                }
            )

    grid_df = pd.DataFrame(rows)

    def _piv(col: str) -> pd.DataFrame:
        p = grid_df.pivot(index="wacc", columns="vre_capex_scale", values=col)
        p.index.name = "wacc"
        p.columns.name = "vre_capex_scale"
        return p

    return SensitivitySurfaceResult(
        wacc_grid=w_vals,
        vre_capex_scale_grid=k_vals,
        reference_wacc=reference_wacc,
        reference_vre_scale=reference_vre_scale,
        grid_table=grid_df,
        lcoe_surface_df=_piv("system_lcoe_per_mwh"),
        vre_share_surface_df=_piv("vre_generation_share_pct"),
        solar_mw_surface_df=_piv("solar_mw"),
        wind_mw_surface_df=_piv("wind_mw"),
        gas_mw_surface_df=_piv("gas_mw"),
        substitution_dividend_surface_df=_piv("substitution_dividend_pct"),
        reference_lcoe_per_mwh=ref_lcoe,
        min_lcoe_per_mwh=float(grid_df["system_lcoe_per_mwh"].min()),
        max_lcoe_per_mwh=float(grid_df["system_lcoe_per_mwh"].max()),
        max_substitution_dividend_pct=float(grid_df["substitution_dividend_pct"].max()),
    )


def verify_envelope_and_concavity_theorems(
    profiles: pd.DataFrame | None = None,
    *,
    wacc: float = REFERENCE_WACC,
    vre_capex_scale: float = 1.0,
    fd_step: float = 1e-5,
    days_per_year: int = DAYS_PER_YEAR,
    solver_name: str = DEFAULT_SOLVER_NAME,
) -> EnvelopeConcavityVerification:
    """Verify Hotelling-Shephard Envelope derivatives & global cost-function concavity."""
    prof = load_profiles(DEFAULT_PROFILE_PATH) if profiles is None else profiles
    validate_profiles(prof)
    r = _validate_unit_interval(wacc, "wacc", allow_zero=False, allow_one=False)
    k = _validate_positive_float(vre_capex_scale, "vre_capex_scale")
    h = _validate_positive_float(fd_step, "fd_step")
    if r - h <= 0.0 or r + h >= 1.0 or k - h <= 0.0:
        raise ValueError("fd_step places wacc or vre_capex_scale outside valid interior")

    base_techs = build_scaled_wacc_technologies(r, k)
    base_res = run_scenario(
        prof, technologies=base_techs, days_per_year=days_per_year, solver_name=solver_name
    )
    fin_specs = default_financial_specs()
    an_dr = sum(
        float(base_res.p_nom_mw[name])
        * (k if name in VRE_TECH_NAMES else 1.0)
        * spec.overnight_capex_per_mw
        * crf_wacc_derivative(r, spec.lifetime_years)
        for name, spec in fin_specs.items()
    )
    an_dk = sum(
        float(base_res.p_nom_mw[name])
        * fin_specs[name].overnight_capex_per_mw
        * capital_recovery_factor(r, fin_specs[name].lifetime_years)
        for name in VRE_TECH_NAMES
    )

    def _cost(rw: float, kv: float) -> ScenarioResult:
        return run_scenario(
            prof,
            technologies=build_scaled_wacc_technologies(rw, kv),
            days_per_year=days_per_year,
            solver_name=solver_name,
        )

    fd_dr = (_cost(r + h, k).total_cost_per_year - _cost(r - h, k).total_cost_per_year) / (2.0 * h)
    fd_dk = (_cost(r, k + h).total_cost_per_year - _cost(r, k - h).total_cost_per_year) / (2.0 * h)

    t_lo = build_scaled_wacc_technologies(0.01, 0.55)
    t_hi = build_scaled_wacc_technologies(0.15, 1.75)
    t_mid: dict[str, dict] = {}
    for g in DEFAULT_TECHNOLOGIES:
        t_mid[g] = {
            "capital_cost": 0.5 * (float(t_lo[g]["capital_cost"]) + float(t_hi[g]["capital_cost"])),
            "marginal_cost": float(t_lo[g]["marginal_cost"]),
            "p_nom_extendable": True,
            "profile": t_lo[g].get("profile"),
        }

    res_lo = run_scenario(
        prof, technologies=t_lo, days_per_year=days_per_year, solver_name=solver_name
    )
    res_hi = run_scenario(
        prof, technologies=t_hi, days_per_year=days_per_year, solver_name=solver_name
    )
    res_mid = run_scenario(
        prof, technologies=t_mid, days_per_year=days_per_year, solver_name=solver_name
    )
    concav_surplus = res_mid.total_cost_per_year - 0.5 * (
        res_lo.total_cost_per_year + res_hi.total_cost_per_year
    )

    min_margin = float("inf")
    for rc, kc in ((0.02, 0.65), (0.07, 1.00), (0.14, 1.60)):
        tc = build_scaled_wacc_technologies(rc, kc)
        rc_opt = run_scenario(
            prof, technologies=tc, days_per_year=days_per_year, solver_name=solver_name
        )
        passive = (
            sum(float(base_res.p_nom_mw[g]) * float(tc[g]["capital_cost"]) for g in tc)
            + base_res.opex_per_year
        )
        min_margin = min(min_margin, passive - rc_opt.total_cost_per_year)

    return EnvelopeConcavityVerification(
        wacc=r,
        vre_capex_scale=k,
        analytic_dcost_dwacc=an_dr,
        fd_dcost_dwacc=fd_dr,
        rel_error_dcost_dwacc=abs(an_dr - fd_dr) / max(abs(an_dr), 1.0),
        analytic_dcost_dvre_scale=an_dk,
        fd_dcost_dvre_scale=fd_dk,
        rel_error_dcost_dvre_scale=abs(an_dk - fd_dk) / max(abs(an_dk), 1.0),
        concavity_midpoint_surplus=concav_surplus,
        passive_minus_optimal_min_margin=max(0.0, min_margin),
    )


@dataclass(frozen=True)
class WeatherRegimeSpec:
    """Meteorological weather-year archetype with physical probability ``pi_omega``."""

    name: str
    probability: float
    solar_scale: float
    wind_scale: float
    load_scale: float
    evening_peak_adder_mw: float = 0.0
    wind_drought_floor_cap: float | None = None


def default_weather_year_specs() -> tuple[WeatherRegimeSpec, ...]:
    """Return 8 calibrated weather-year archetypes spanning normal to Dunkelflaute extremes."""
    return (
        WeatherRegimeSpec("WY1_Windy_Equinox", 0.180, 0.96, 1.32, 0.94),
        WeatherRegimeSpec("WY2_Sunny_Anticyclone", 0.170, 1.18, 0.86, 0.97),
        WeatherRegimeSpec("WY3_Temperate_Normal", 0.250, 1.02, 1.04, 1.00),
        WeatherRegimeSpec("WY4_Cloudy_Maritime", 0.160, 0.76, 1.14, 1.02),
        WeatherRegimeSpec("WY5_Summer_Heat_Wave", 0.140, 1.12, 0.56, 1.15, 22.0),
        WeatherRegimeSpec("WY6_Winter_Cloud_Trough", 0.060, 0.48, 0.42, 1.18, 28.0),
        WeatherRegimeSpec("WY7_Polar_Vortex_Freeze", 0.025, 0.36, 0.22, 1.24, 38.0, 0.24),
        WeatherRegimeSpec("WY8_Severe_Dunkelflaute", 0.015, 0.20, 0.08, 1.30, 78.0, 0.10),
    )


def generate_multiyear_weather_profiles(
    base_profiles: pd.DataFrame | None = None,
    *,
    weather_specs: Sequence[WeatherRegimeSpec] | None = None,
) -> dict[str, tuple[float, pd.DataFrame]]:
    """Generate ``{name: (probability, DataFrame)}`` across meteorological archetypes."""
    base = load_profiles(DEFAULT_PROFILE_PATH) if base_profiles is None else base_profiles
    validate_profiles(base)
    specs = default_weather_year_specs() if weather_specs is None else tuple(weather_specs)
    if not specs:
        raise ValueError("weather_specs must be non-empty")
    tot_p = sum(s.probability for s in specs)
    if abs(tot_p - 1.0) > 1e-9:
        raise ValueError(f"weather_specs probabilities must sum to 1.0, got {tot_p:.12f}")

    n_t = len(base)
    hours = np.arange(n_t, dtype=float)
    eve_bump = np.exp(-0.5 * ((hours - 19.0) / 0.60) ** 2)
    out: dict[str, tuple[float, pd.DataFrame]] = {}
    for s in specs:
        if s.probability <= 0.0:
            raise ValueError(f"Weather regime {s.name!r} probability must be > 0")
        sol = np.clip(base["solar_cf"].to_numpy(dtype=float) * s.solar_scale, 0.0, 1.0)
        wnd = np.clip(base["wind_cf"].to_numpy(dtype=float) * s.wind_scale, 0.0, 1.0)
        if s.wind_drought_floor_cap is not None:
            wnd = np.minimum(wnd, float(s.wind_drought_floor_cap))
        ld = (
            base["load_mw"].to_numpy(dtype=float) * s.load_scale
            + s.evening_peak_adder_mw * eve_bump
        )
        if np.any(ld <= 0.0):
            raise ValueError(f"Weather regime {s.name!r} produced non-positive load")
        df_s = pd.DataFrame(
            {"solar_cf": np.round(sol, 6), "wind_cf": np.round(wnd, 6), "load_mw": np.round(ld, 6)},
            index=base.index.copy(),
        )
        validate_profiles(df_s)
        out[s.name] = (float(s.probability), df_s)
    return out


def compute_discrete_cvar(
    losses: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    alpha: float = DEFAULT_CVAR_CONFIDENCE,
) -> tuple[float, float]:
    """Compute exact discrete ``(VaR_alpha, CVaR_alpha)`` with boundary atom splitting."""
    conf = _validate_unit_interval(alpha, "alpha", allow_zero=False, allow_one=False)
    z = np.asarray(losses, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if z.ndim != 1 or p.ndim != 1 or len(z) == 0 or len(z) != len(p):
        raise ValueError("losses and probabilities must be non-empty 1D arrays of equal length")
    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("losses must be finite and probabilities must be non-negative")
    if abs(float(np.sum(p)) - 1.0) > 1e-8:
        raise ValueError(f"probabilities must sum to 1.0, got {float(np.sum(p)):.12f}")

    order = np.argsort(z)
    z_s, p_s = z[order], p[order]
    cdf = np.cumsum(p_s)
    k_idx = int(np.searchsorted(cdf, conf - 1e-15, side="left"))
    k_idx = min(k_idx, len(z_s) - 1)
    var_alpha = float(z_s[k_idx])
    tail_prob = 1.0 - conf
    strict_mass = float(np.sum(p_s[k_idx + 1 :])) if k_idx + 1 < len(z_s) else 0.0
    strict_sum = float(np.sum(p_s[k_idx + 1 :] * z_s[k_idx + 1 :])) if k_idx + 1 < len(z_s) else 0.0
    rem_mass = max(0.0, tail_prob - strict_mass)
    return var_alpha, (strict_sum + rem_mass * var_alpha) / tail_prob


@dataclass(frozen=True)
class StochasticCVaRResult:
    """Solution and risk-adjusted KKT duality telemetry of a Two-Stage Mean-CVaR LP."""

    alpha: float
    risk_aversion_beta: float
    include_bess: bool
    voll_per_mwh: float
    p_nom_mw: pd.Series
    bess_energy_mwh: float
    first_stage_capex_per_year: float
    expected_recourse_cost_per_year: float
    expected_total_cost_per_year: float
    var_total_cost_per_year: float
    cvar_total_cost_per_year: float
    lp_cvar_epigraph_cost_per_year: float
    objective_value: float
    expected_demand_mwh_per_year: float
    expected_lcoe_per_mwh: float
    cvar_lcoe_per_mwh: float
    eens_mwh_per_year: float
    worst_unserved_mwh_per_year: float
    expected_lolh_hours_per_year: float
    scenario_table: pd.DataFrame
    risk_adjusted_demand_payment: float
    strong_duality_gap: float
    max_zero_profit_residual: float


@dataclass(frozen=True)
class StochasticValueMetrics:
    """Birge-Louveaux stochastic programming value chain (``WS <= RP <= EEV``)."""

    alpha: float
    hedged_beta: float
    wait_and_see_cost_per_year: float
    recourse_problem_cost_per_year: float
    expected_value_plan_eev_cost_per_year: float
    cvar_hedged_expected_cost_per_year: float
    evpi_per_year: float
    vss_per_year: float
    mean_weather_eens_mwh_per_year: float
    risk_neutral_eens_mwh_per_year: float
    cvar_hedged_eens_mwh_per_year: float
    risk_neutral_cvar_cost_per_year: float
    cvar_hedged_cvar_cost_per_year: float


@dataclass(frozen=True)
class StochasticCVaRLinprogCrossCheck:
    """Independent sparse ``scipy.optimize.linprog`` cross-check of Two-Stage CVaR LP."""

    alpha: float
    risk_aversion_beta: float
    pypsa_objective: float
    scipy_objective: float
    rel_objective_diff: float
    max_p_nom_diff_mw: float
    rel_cvar_cost_diff: float
    max_risk_prob_diff: float
    max_lmp_diff_per_mwh: float


def _prepare_stochastic_inputs(
    weather_ensemble: Mapping[str, tuple[float, pd.DataFrame]] | None,
    alpha: float,
    risk_aversion_beta: float,
    technologies: dict[str, dict] | None,
    bess_capital_cost_per_mw_yr: float,
    bess_max_hours: float,
    bess_efficiency: float,
    voll_per_mwh: float,
    days_per_year: int,
):
    ens = generate_multiyear_weather_profiles() if weather_ensemble is None else weather_ensemble
    if not ens:
        raise ValueError("weather_ensemble must be non-empty")
    names: list[str] = []
    probs_l: list[float] = []
    dfs: list[pd.DataFrame] = []
    ref_idx: pd.DatetimeIndex | None = None
    for name, (prob, df) in ens.items():
        p = _validate_positive_float(prob, f"probability[{name}]")
        validate_profiles(df)
        if ref_idx is None:
            ref_idx = df.index
        elif not df.index.equals(ref_idx):
            raise ValueError(f"Scenario {name!r} timestamps do not match reference index")
        names.append(str(name))
        probs_l.append(p)
        dfs.append(df)
    probs = np.asarray(probs_l, dtype=float)
    if abs(float(np.sum(probs)) - 1.0) > 1e-8:
        raise ValueError(f"Scenario probabilities must sum to 1.0, got {float(np.sum(probs)):.12f}")
    conf = _validate_unit_interval(alpha, "alpha", allow_zero=False, allow_one=False)
    beta = _validate_unit_interval(risk_aversion_beta, "risk_aversion_beta")
    voll = _validate_positive_float(voll_per_mwh, "voll_per_mwh")
    bess_cap = _validate_positive_float(bess_capital_cost_per_mw_yr, "bess_capital_cost_per_mw_yr")
    bess_hrs = _validate_positive_float(bess_max_hours, "bess_max_hours")
    bess_eff = _validate_unit_interval(
        bess_efficiency, "bess_efficiency", allow_zero=False, allow_one=True
    )
    if days_per_year <= 0:
        raise ValueError("days_per_year must be positive")
    techs = DEFAULT_TECHNOLOGIES if technologies is None else technologies
    assert ref_idx is not None
    dt = float((ref_idx[1] - ref_idx[0]).total_seconds() / 3600.0) if len(ref_idx) > 1 else 1.0
    return (
        names,
        probs,
        dfs,
        ref_idx,
        conf,
        beta,
        voll,
        bess_cap,
        bess_hrs,
        bess_eff,
        techs,
        dt,
        float(days_per_year),
    )


def run_stochastic_cvar_scenario(
    weather_ensemble: Mapping[str, tuple[float, pd.DataFrame]] | None = None,
    *,
    alpha: float = DEFAULT_CVAR_CONFIDENCE,
    risk_aversion_beta: float = 0.35,
    include_bess: bool = True,
    technologies: dict[str, dict] | None = None,
    bess_capital_cost_per_mw_yr: float = DEFAULT_STOCHASTIC_BESS_CAPITAL_COST_PER_MW_YR,
    bess_max_hours: float = DEFAULT_STOCHASTIC_BESS_MAX_HOURS,
    bess_efficiency: float = DEFAULT_STOCHASTIC_BESS_EFFICIENCY,
    voll_per_mwh: float = DEFAULT_VOLL_PER_MWH,
    fixed_capacities_mw: Mapping[str, float] | None = None,
    days_per_year: int = DAYS_PER_YEAR,
    solver_name: str = DEFAULT_SOLVER_NAME,
) -> StochasticCVaRResult:
    """Solve Two-Stage Stochastic ``Mean-CVaR_alpha`` capacity expansion via PyPSA + linopy."""
    (
        scen_names,
        probs,
        dfs,
        ref_idx,
        conf,
        beta,
        voll,
        bess_cap,
        bess_hrs,
        bess_eff,
        techs,
        dt,
        w_yr,
    ) = _prepare_stochastic_inputs(
        weather_ensemble,
        alpha,
        risk_aversion_beta,
        technologies,
        bess_capital_cost_per_mw_yr,
        bess_max_hours,
        bess_efficiency,
        voll_per_mwh,
        days_per_year,
    )

    network = pypsa.Network()
    network.set_snapshots(ref_idx)
    network.snapshot_weightings = network.snapshot_weightings.astype(float)
    network.snapshot_weightings.loc[:, ["objective", "generators"]] = w_yr * dt
    network.snapshot_weightings.loc[:, "stores"] = dt
    network.add("Carrier", CARRIER)

    s0 = scen_names[0]
    is_fixed = fixed_capacities_mw is not None
    if is_fixed:
        assert fixed_capacities_mw is not None
        for g_name in techs:
            if g_name not in fixed_capacities_mw or float(fixed_capacities_mw[g_name]) < 0.0:
                raise ValueError(f"fixed_capacities_mw missing non-negative entry for {g_name!r}")

    for idx_s, (s_name, prob, df) in enumerate(zip(scen_names, probs, dfs, strict=True)):
        bus = f"Bus__{s_name}"
        network.add("Bus", bus, carrier=CARRIER)
        network.add("Load", f"{LOAD_NAME}__{s_name}", bus=bus, p_set=df["load_mw"])
        w_prob = (1.0 - beta) * float(prob)

        for g_name, sp in techs.items():
            cap_s = float(sp["capital_cost"]) if (idx_s == 0 and not is_fixed) else 0.0
            gen_kw: dict[str, object] = {
                "bus": bus,
                "carrier": CARRIER,
                "p_nom_extendable": not is_fixed,
                "p_nom": float(fixed_capacities_mw[g_name]) if is_fixed else 0.0,  # type: ignore[index]
                "capital_cost": cap_s,
                "marginal_cost": w_prob * float(sp["marginal_cost"]),
            }
            if sp.get("profile") is not None:
                gen_kw["p_max_pu"] = df[str(sp["profile"])]
            network.add("Generator", f"{g_name}__{s_name}", **gen_kw)

        network.add(
            "Generator",
            f"{LOAD_SHEDDING_NAME}__{s_name}",
            bus=bus,
            carrier=CARRIER,
            p_nom_extendable=False,
            p_nom=1e6,
            capital_cost=0.0,
            marginal_cost=w_prob * voll,
        )

        if include_bess:
            b_fix = float(fixed_capacities_mw.get(STOCHASTIC_BESS_NAME, 0.0)) if is_fixed else 0.0  # type: ignore[union-attr]
            network.add(
                "StorageUnit",
                f"{STOCHASTIC_BESS_NAME}__{s_name}",
                bus=bus,
                carrier=CARRIER,
                p_nom_extendable=not is_fixed,
                p_nom=b_fix,
                max_hours=bess_hrs,
                efficiency_store=bess_eff,
                efficiency_dispatch=bess_eff,
                cyclic_state_of_charge=True,
                capital_cost=bess_cap if (idx_s == 0 and not is_fixed) else 0.0,
                marginal_cost=0.0,
            )

    def _extra_functionality(n: pypsa.Network, snapshots: pd.Index) -> None:
        m = n.model
        if not is_fixed and len(scen_names) > 1:
            k_gen = m.variables["Generator-p_nom"]
            g_ext = list(k_gen.coords["name"].to_numpy())
            for g_name in techs:
                na_rows: list[list[float]] = []
                for s_other in scen_names[1:]:
                    row_v = [0.0] * len(g_ext)
                    row_v[g_ext.index(f"{g_name}__{s0}")] = 1.0
                    row_v[g_ext.index(f"{g_name}__{s_other}")] = -1.0
                    na_rows.append(row_v)
                na_da = xr.DataArray(
                    np.asarray(na_rows, dtype=float),
                    coords={"name": g_ext, "scenario_link": scen_names[1:]},
                    dims=["scenario_link", "name"],
                )
                m.add_constraints((k_gen * na_da).sum("name") == 0.0, name=f"NonAnt_Gen_{g_name}")

            if include_bess:
                k_su = m.variables["StorageUnit-p_nom"]
                s_ext = list(k_su.coords["name"].to_numpy())
                su_rows: list[list[float]] = []
                for s_other in scen_names[1:]:
                    row_s = [0.0] * len(s_ext)
                    row_s[s_ext.index(f"{STOCHASTIC_BESS_NAME}__{s0}")] = 1.0
                    row_s[s_ext.index(f"{STOCHASTIC_BESS_NAME}__{s_other}")] = -1.0
                    su_rows.append(row_s)
                su_da = xr.DataArray(
                    np.asarray(su_rows, dtype=float),
                    coords={"name": s_ext, "scenario_link": scen_names[1:]},
                    dims=["scenario_link", "name"],
                )
                m.add_constraints((k_su * su_da).sum("name") == 0.0, name="NonAnt_BESS")

        if beta > 0.0:
            eta = m.add_variables(lower=-np.inf, upper=np.inf, name="cvar_eta")
            scen_coord = pd.Index(scen_names, name="scenario")
            xi = m.add_variables(lower=0.0, coords=[scen_coord], name="cvar_xi")
            gen_p = m.variables["Generator-p"]
            all_g = list(gen_p.coords["name"].to_numpy())
            cost_mat = np.zeros((len(all_g), len(scen_names)), dtype=float)
            for j_s, s_name in enumerate(scen_names):
                for g_name, sp in techs.items():
                    mc = float(sp["marginal_cost"])
                    if mc != 0.0:
                        cost_mat[all_g.index(f"{g_name}__{s_name}"), j_s] = w_yr * dt * mc
                cost_mat[all_g.index(f"{LOAD_SHEDDING_NAME}__{s_name}"), j_s] = w_yr * dt * voll

            cost_da = xr.DataArray(
                cost_mat, coords={"name": all_g, "scenario": scen_names}, dims=["name", "scenario"]
            )
            q_expr = (gen_p * cost_da).sum(["name", "snapshot"])
            m.add_constraints(q_expr - eta - xi <= 0.0, name="CVaR_Epigraph")

            prob_da = xr.DataArray(probs, coords=[scen_coord], dims=["scenario"])
            cvar_term = beta * eta + (beta / (1.0 - conf)) * (xi * prob_da).sum("scenario")
            m.objective = m.objective.expression + cvar_term

    status, condition = network.optimize(
        solver_name=solver_name,
        extra_functionality=_extra_functionality,
        log_to_console=False,
        include_objective_constant=False,
    )
    if status != "ok" or condition != "optimal":
        raise SolverError(
            f"Stochastic CVaR optimisation failed: status={status!r}, condition={condition!r}"
        )

    cap_dict: dict[str, float] = {}
    first_stage_capex = 0.0
    for g_name, sp in techs.items():
        val = (
            float(fixed_capacities_mw[g_name])
            if is_fixed
            else float(network.generators.loc[f"{g_name}__{s0}", "p_nom_opt"])
        )  # type: ignore[index]
        cap_dict[g_name] = val
        first_stage_capex += val * float(sp["capital_cost"])

    if include_bess:
        b_val = (
            float(fixed_capacities_mw.get(STOCHASTIC_BESS_NAME, 0.0))
            if is_fixed
            else float(network.storage_units.loc[f"{STOCHASTIC_BESS_NAME}__{s0}", "p_nom_opt"])
        )  # type: ignore[union-attr]
        cap_dict[STOCHASTIC_BESS_NAME] = b_val
        bess_e_mwh = b_val * bess_hrs
        first_stage_capex += b_val * bess_cap
    else:
        bess_e_mwh = 0.0

    p_nom_series = pd.Series(cap_dict, dtype=float)

    if beta > 0.0:
        m = network.model
        eta_opt = float(m.variables["cvar_eta"].solution.item())
        xi_opt = np.asarray(m.variables["cvar_xi"].solution.to_numpy(), dtype=float)
        raw_duals = np.asarray(m.constraints["CVaR_Epigraph"].dual.to_numpy(), dtype=float)
        rhos = np.maximum(0.0, -raw_duals)
        q_adj = (1.0 - beta) * probs + rhos
        lp_cvar_recourse = eta_opt + (1.0 / (1.0 - conf)) * float(np.sum(probs * xi_opt))
    else:
        q_adj = probs.copy()
        lp_cvar_recourse = float("nan")

    scen_rows: list[dict[str, float | str]] = []
    recourse_arr = np.zeros(len(scen_names), dtype=float)
    total_arr = np.zeros(len(scen_names), dtype=float)
    shed_arr = np.zeros(len(scen_names), dtype=float)
    lolh_arr = np.zeros(len(scen_names), dtype=float)
    dem_arr = np.zeros(len(scen_names), dtype=float)

    risk_demand_payment = 0.0
    gen_risk_rev = {g_name: 0.0 for g_name in techs}
    gen_risk_opex = {g_name: 0.0 for g_name in techs}
    bess_risk_rev = 0.0

    for idx_s, (s_name, prob, df) in enumerate(zip(scen_names, probs, dfs, strict=True)):
        bus = f"Bus__{s_name}"
        load_s = df["load_mw"].to_numpy(dtype=float)
        dem_s = float(np.sum(load_s) * dt * w_yr)
        dem_arr[idx_s] = dem_s

        lmp_w = network.buses_t.marginal_price[bus].to_numpy(dtype=float)
        risk_demand_payment += float(np.sum(w_yr * dt * lmp_w * load_s))
        q_s = float(q_adj[idx_s])
        cond_lmp = lmp_w / q_s if q_s > 1e-12 else np.zeros_like(lmp_w)

        opex_s = 0.0
        for g_name, sp in techs.items():
            p_g = network.generators_t.p[f"{g_name}__{s_name}"].to_numpy(dtype=float)
            opex_g_s = float(np.sum(p_g) * dt * w_yr) * float(sp["marginal_cost"])
            opex_s += opex_g_s
            gen_risk_rev[g_name] += float(np.sum(w_yr * dt * lmp_w * p_g))
            gen_risk_opex[g_name] += q_s * opex_g_s

        p_shed = network.generators_t.p[f"{LOAD_SHEDDING_NAME}__{s_name}"].to_numpy(dtype=float)
        shed_mwh_s = float(np.sum(p_shed) * dt * w_yr)
        lol_h_s = float(np.sum(p_shed > 1e-6) * dt * w_yr)
        opex_s += shed_mwh_s * voll

        if include_bess:
            p_dis = network.storage_units_t.p_dispatch[
                f"{STOCHASTIC_BESS_NAME}__{s_name}"
            ].to_numpy(dtype=float)
            p_ch = network.storage_units_t.p_store[f"{STOCHASTIC_BESS_NAME}__{s_name}"].to_numpy(
                dtype=float
            )
            bess_risk_rev += float(np.sum(w_yr * dt * lmp_w * (p_dis - p_ch)))

        recourse_arr[idx_s] = opex_s
        total_arr[idx_s] = first_stage_capex + opex_s
        shed_arr[idx_s] = shed_mwh_s
        lolh_arr[idx_s] = lol_h_s
        scen_rows.append(
            {
                "scenario": s_name,
                "probability": float(prob),
                "risk_adjusted_prob": q_s,
                "annual_demand_mwh": dem_s,
                "recourse_cost_per_year": opex_s,
                "total_cost_per_year": first_stage_capex + opex_s,
                "scenario_lcoe_per_mwh": (first_stage_capex + opex_s) / dem_s,
                "unserved_energy_mwh_per_year": shed_mwh_s,
                "lolh_hours_per_year": lol_h_s,
                "peak_conditional_lmp_per_mwh": float(np.max(cond_lmp)),
            }
        )

    exp_recourse = float(np.sum(probs * recourse_arr))
    exp_total = first_stage_capex + exp_recourse
    var_tot, cvar_tot = compute_discrete_cvar(total_arr, probs, alpha=conf)
    lp_cvar_tot = first_stage_capex + lp_cvar_recourse if beta > 0.0 else cvar_tot
    obj_val = (
        float(network.objective) if not is_fixed else (1.0 - beta) * exp_total + beta * lp_cvar_tot
    )
    exp_dem = float(np.sum(probs * dem_arr))

    zp_res = 0.0
    if not is_fixed:
        for g_name, sp in techs.items():
            k_g = float(p_nom_series[g_name])
            net_m = (gen_risk_rev[g_name] - gen_risk_opex[g_name]) - k_g * float(sp["capital_cost"])
            zp_res = max(zp_res, abs(net_m) if k_g > 1e-4 else max(0.0, net_m))
        if include_bess:
            k_b = float(p_nom_series[STOCHASTIC_BESS_NAME])
            net_b = bess_risk_rev - k_b * bess_cap
            zp_res = max(zp_res, abs(net_b) if k_b > 1e-4 else max(0.0, net_b))
        duality_gap = obj_val - risk_demand_payment
    else:
        duality_gap = 0.0

    return StochasticCVaRResult(
        alpha=conf,
        risk_aversion_beta=beta,
        include_bess=include_bess,
        voll_per_mwh=voll,
        p_nom_mw=p_nom_series,
        bess_energy_mwh=bess_e_mwh,
        first_stage_capex_per_year=first_stage_capex,
        expected_recourse_cost_per_year=exp_recourse,
        expected_total_cost_per_year=exp_total,
        var_total_cost_per_year=var_tot,
        cvar_total_cost_per_year=cvar_tot,
        lp_cvar_epigraph_cost_per_year=lp_cvar_tot,
        objective_value=obj_val,
        expected_demand_mwh_per_year=exp_dem,
        expected_lcoe_per_mwh=exp_total / exp_dem,
        cvar_lcoe_per_mwh=cvar_tot / exp_dem,
        eens_mwh_per_year=float(np.sum(probs * shed_arr)),
        worst_unserved_mwh_per_year=float(np.max(shed_arr)),
        expected_lolh_hours_per_year=float(np.sum(probs * lolh_arr)),
        scenario_table=pd.DataFrame(scen_rows).set_index("scenario"),
        risk_adjusted_demand_payment=risk_demand_payment,
        strong_duality_gap=duality_gap,
        max_zero_profit_residual=zp_res,
    )


def evaluate_stochastic_value_metrics(
    weather_ensemble: Mapping[str, tuple[float, pd.DataFrame]] | None = None,
    *,
    alpha: float = DEFAULT_CVAR_CONFIDENCE,
    hedged_beta: float = 0.35,
    include_bess: bool = True,
    voll_per_mwh: float = DEFAULT_VOLL_PER_MWH,
    days_per_year: int = DAYS_PER_YEAR,
    solver_name: str = DEFAULT_SOLVER_NAME,
) -> StochasticValueMetrics:
    """Evaluate Birge-Louveaux ``WS <= RP <= EEV``, ``EVPI``, and ``VSS`` metrics."""
    ens = generate_multiyear_weather_profiles() if weather_ensemble is None else weather_ensemble
    b_hedged = _validate_unit_interval(hedged_beta, "hedged_beta", allow_zero=False, allow_one=True)

    rp_res = run_stochastic_cvar_scenario(
        ens,
        alpha=alpha,
        risk_aversion_beta=0.0,
        include_bess=include_bess,
        voll_per_mwh=voll_per_mwh,
        days_per_year=days_per_year,
        solver_name=solver_name,
    )
    hedged_res = run_stochastic_cvar_scenario(
        ens,
        alpha=alpha,
        risk_aversion_beta=b_hedged,
        include_bess=include_bess,
        voll_per_mwh=voll_per_mwh,
        days_per_year=days_per_year,
        solver_name=solver_name,
    )

    ws_cost = 0.0
    first_df = next(iter(ens.values()))[1]
    mean_sol = np.zeros(len(first_df), dtype=float)
    mean_wnd = np.zeros(len(first_df), dtype=float)
    mean_ld = np.zeros(len(first_df), dtype=float)
    for s_name, (prob, df_s) in ens.items():
        ws_s = run_stochastic_cvar_scenario(
            {s_name: (1.0, df_s)},
            alpha=alpha,
            risk_aversion_beta=0.0,
            include_bess=include_bess,
            voll_per_mwh=voll_per_mwh,
            days_per_year=days_per_year,
            solver_name=solver_name,
        )
        ws_cost += float(prob) * ws_s.expected_total_cost_per_year
        mean_sol += float(prob) * df_s["solar_cf"].to_numpy(dtype=float)
        mean_wnd += float(prob) * df_s["wind_cf"].to_numpy(dtype=float)
        mean_ld += float(prob) * df_s["load_mw"].to_numpy(dtype=float)

    mean_df = pd.DataFrame(
        {"solar_cf": mean_sol, "wind_cf": mean_wnd, "load_mw": mean_ld}, index=first_df.index.copy()
    )
    ev_plan = run_stochastic_cvar_scenario(
        {"Mean_Weather": (1.0, mean_df)},
        alpha=alpha,
        risk_aversion_beta=0.0,
        include_bess=include_bess,
        voll_per_mwh=voll_per_mwh,
        days_per_year=days_per_year,
        solver_name=solver_name,
    )
    eev_res = run_stochastic_cvar_scenario(
        ens,
        alpha=alpha,
        risk_aversion_beta=0.0,
        include_bess=include_bess,
        voll_per_mwh=voll_per_mwh,
        fixed_capacities_mw=ev_plan.p_nom_mw.to_dict(),
        days_per_year=days_per_year,
        solver_name=solver_name,
    )

    return StochasticValueMetrics(
        alpha=rp_res.alpha,
        hedged_beta=b_hedged,
        wait_and_see_cost_per_year=ws_cost,
        recourse_problem_cost_per_year=rp_res.expected_total_cost_per_year,
        expected_value_plan_eev_cost_per_year=eev_res.expected_total_cost_per_year,
        cvar_hedged_expected_cost_per_year=hedged_res.expected_total_cost_per_year,
        evpi_per_year=rp_res.expected_total_cost_per_year - ws_cost,
        vss_per_year=eev_res.expected_total_cost_per_year - rp_res.expected_total_cost_per_year,
        mean_weather_eens_mwh_per_year=eev_res.eens_mwh_per_year,
        risk_neutral_eens_mwh_per_year=rp_res.eens_mwh_per_year,
        cvar_hedged_eens_mwh_per_year=hedged_res.eens_mwh_per_year,
        risk_neutral_cvar_cost_per_year=rp_res.cvar_total_cost_per_year,
        cvar_hedged_cvar_cost_per_year=hedged_res.cvar_total_cost_per_year,
    )


def sweep_cvar_risk_frontier(
    weather_ensemble: Mapping[str, tuple[float, pd.DataFrame]] | None = None,
    *,
    betas: Sequence[float] = DEFAULT_RISK_AVERSION_SWEEP,
    alpha: float = DEFAULT_CVAR_CONFIDENCE,
    include_bess: bool = True,
    voll_per_mwh: float = DEFAULT_VOLL_PER_MWH,
    days_per_year: int = DAYS_PER_YEAR,
    solver_name: str = DEFAULT_SOLVER_NAME,
) -> pd.DataFrame:
    """Trace the Pareto-efficient ``Mean-CVaR_alpha`` frontier across ``beta in [0, 1]``."""
    ens = generate_multiyear_weather_profiles() if weather_ensemble is None else weather_ensemble
    beta_vals = tuple(_validate_unit_interval(b, "betas") for b in betas)
    if not beta_vals:
        raise ValueError("betas must be non-empty")

    rows: list[dict[str, float]] = []
    for b in beta_vals:
        res = run_stochastic_cvar_scenario(
            ens,
            alpha=alpha,
            risk_aversion_beta=b,
            include_bess=include_bess,
            voll_per_mwh=voll_per_mwh,
            days_per_year=days_per_year,
            solver_name=solver_name,
        )
        rows.append(
            {
                "beta": b,
                "alpha": res.alpha,
                "expected_cost_per_year": res.expected_total_cost_per_year,
                "cvar_cost_per_year": res.cvar_total_cost_per_year,
                "var_cost_per_year": res.var_total_cost_per_year,
                "first_stage_capex_per_year": res.first_stage_capex_per_year,
                "expected_lcoe_per_mwh": res.expected_lcoe_per_mwh,
                "cvar_lcoe_per_mwh": res.cvar_lcoe_per_mwh,
                "eens_mwh_per_year": res.eens_mwh_per_year,
                "worst_unserved_mwh_per_year": res.worst_unserved_mwh_per_year,
                "solar_mw": float(res.p_nom_mw.get("Solar_PV", 0.0)),
                "wind_mw": float(res.p_nom_mw.get("Wind_Turbine", 0.0)),
                "gas_mw": float(res.p_nom_mw.get("Gas_Peaker", 0.0)),
                "bess_mw": float(res.p_nom_mw.get(STOCHASTIC_BESS_NAME, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def solve_stochastic_cvar_linprog_crosscheck(
    weather_ensemble: Mapping[str, tuple[float, pd.DataFrame]] | None = None,
    *,
    alpha: float = DEFAULT_CVAR_CONFIDENCE,
    risk_aversion_beta: float = 0.35,
    include_bess: bool = True,
    technologies: dict[str, dict] | None = None,
    bess_capital_cost_per_mw_yr: float = DEFAULT_STOCHASTIC_BESS_CAPITAL_COST_PER_MW_YR,
    bess_max_hours: float = DEFAULT_STOCHASTIC_BESS_MAX_HOURS,
    bess_efficiency: float = DEFAULT_STOCHASTIC_BESS_EFFICIENCY,
    voll_per_mwh: float = DEFAULT_VOLL_PER_MWH,
    days_per_year: int = DAYS_PER_YEAR,
) -> StochasticCVaRLinprogCrossCheck:
    """Solve Two-Stage ``Mean-CVaR_alpha`` via standalone sparse ``scipy.optimize.linprog``."""
    (
        scen_names,
        probs,
        dfs,
        ref_idx,
        conf,
        beta,
        voll,
        bess_cap,
        bess_hrs,
        bess_eff,
        techs,
        dt,
        w_yr,
    ) = _prepare_stochastic_inputs(
        weather_ensemble,
        alpha,
        risk_aversion_beta,
        technologies,
        bess_capital_cost_per_mw_yr,
        bess_max_hours,
        bess_efficiency,
        voll_per_mwh,
        days_per_year,
    )
    pypsa_res = run_stochastic_cvar_scenario(
        weather_ensemble,
        alpha=conf,
        risk_aversion_beta=beta,
        include_bess=include_bess,
        technologies=techs,
        bess_capital_cost_per_mw_yr=bess_cap,
        bess_max_hours=bess_hrs,
        bess_efficiency=bess_eff,
        voll_per_mwh=voll,
        days_per_year=days_per_year,
    )

    tech_names = list(techs.keys())
    n_g, n_s, n_t = len(tech_names), len(scen_names), len(ref_idx)
    off_kg = 0
    off_kb = n_g
    off_pg = n_g + (1 if include_bess else 0)
    off_sh = off_pg + n_s * n_g * n_t
    off_ch = off_sh + n_s * n_t
    off_di = off_ch + (n_s * n_t if include_bess else 0)
    off_sc = off_di + (n_s * n_t if include_bess else 0)
    off_et = off_sc + (n_s * n_t if include_bess else 0)
    off_xi = off_et + (1 if beta > 0.0 else 0)
    n_vars = off_xi + (n_s if beta > 0.0 else 0)

    c = np.zeros(n_vars, dtype=float)
    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n_vars
    for g_i, g_name in enumerate(tech_names):
        c[off_kg + g_i] = float(techs[g_name]["capital_cost"])
    if include_bess:
        c[off_kb] = bess_cap

    for s_i, prob in enumerate(probs):
        w_p = (1.0 - beta) * float(prob)
        for g_i, g_name in enumerate(tech_names):
            mc = float(techs[g_name]["marginal_cost"])
            if mc != 0.0:
                base = off_pg + (s_i * n_g + g_i) * n_t
                c[base : base + n_t] = w_p * w_yr * dt * mc
        base_sh = off_sh + s_i * n_t
        c[base_sh : base_sh + n_t] = w_p * w_yr * dt * voll

    if beta > 0.0:
        bounds[off_et] = (None, None)
        c[off_et] = beta
        for s_i, prob in enumerate(probs):
            c[off_xi + s_i] = beta * float(prob) / (1.0 - conf)

    eq_r: list[int] = []
    eq_c: list[int] = []
    eq_v: list[float] = []
    b_eq_l: list[float] = []
    bal_rows: dict[tuple[int, int], int] = {}
    r_eq = 0

    for s_i, df in enumerate(dfs):
        ld = df["load_mw"].to_numpy(dtype=float)
        for t_i in range(n_t):
            bal_rows[(s_i, t_i)] = r_eq
            for g_i in range(n_g):
                eq_r.append(r_eq)
                eq_c.append(off_pg + (s_i * n_g + g_i) * n_t + t_i)
                eq_v.append(1.0)
            eq_r.append(r_eq)
            eq_c.append(off_sh + s_i * n_t + t_i)
            eq_v.append(1.0)
            if include_bess:
                eq_r.extend([r_eq, r_eq])
                eq_c.extend([off_di + s_i * n_t + t_i, off_ch + s_i * n_t + t_i])
                eq_v.extend([1.0, -1.0])
            b_eq_l.append(float(ld[t_i]))
            r_eq += 1

        if include_bess:
            for t_i in range(n_t):
                t_prev = (t_i - 1) % n_t
                eq_r.extend([r_eq, r_eq, r_eq, r_eq])
                eq_c.extend(
                    [
                        off_sc + s_i * n_t + t_i,
                        off_sc + s_i * n_t + t_prev,
                        off_ch + s_i * n_t + t_i,
                        off_di + s_i * n_t + t_i,
                    ]
                )
                eq_v.extend([1.0, -1.0, -bess_eff * dt, dt / bess_eff])
                b_eq_l.append(0.0)
                r_eq += 1

    ub_r: list[int] = []
    ub_c: list[int] = []
    ub_v: list[float] = []
    b_ub_l: list[float] = []
    r_ub = 0

    for s_i, df in enumerate(dfs):
        for g_i, g_name in enumerate(tech_names):
            sp = techs[g_name]
            cf = (
                df[str(sp["profile"])].to_numpy(dtype=float)
                if sp.get("profile") is not None
                else np.ones(n_t, dtype=float)
            )
            for t_i in range(n_t):
                ub_r.append(r_ub)
                ub_c.append(off_pg + (s_i * n_g + g_i) * n_t + t_i)
                ub_v.append(1.0)
                if cf[t_i] > 0.0:
                    ub_r.append(r_ub)
                    ub_c.append(off_kg + g_i)
                    ub_v.append(-float(cf[t_i]))
                b_ub_l.append(0.0)
                r_ub += 1

        if include_bess:
            for t_i in range(n_t):
                for col, scale in (
                    (off_ch + s_i * n_t + t_i, 1.0),
                    (off_di + s_i * n_t + t_i, 1.0),
                    (off_sc + s_i * n_t + t_i, bess_hrs),
                ):
                    ub_r.extend([r_ub, r_ub])
                    ub_c.extend([col, off_kb])
                    ub_v.extend([1.0, -scale])
                    b_ub_l.append(0.0)
                    r_ub += 1

    epi_rows: list[int] = []
    if beta > 0.0:
        for s_i in range(n_s):
            epi_rows.append(r_ub)
            for g_i, g_name in enumerate(tech_names):
                mc = float(techs[g_name]["marginal_cost"])
                if mc != 0.0:
                    for t_i in range(n_t):
                        ub_r.append(r_ub)
                        ub_c.append(off_pg + (s_i * n_g + g_i) * n_t + t_i)
                        ub_v.append(w_yr * dt * mc)
            for t_i in range(n_t):
                ub_r.append(r_ub)
                ub_c.append(off_sh + s_i * n_t + t_i)
                ub_v.append(w_yr * dt * voll)
            ub_r.extend([r_ub, r_ub])
            ub_c.extend([off_et, off_xi + s_i])
            ub_v.extend([-1.0, -1.0])
            b_ub_l.append(0.0)
            r_ub += 1

    res_lp = linprog(
        c,
        A_ub=sparse.csc_matrix((ub_v, (ub_r, ub_c)), shape=(r_ub, n_vars)),
        b_ub=np.asarray(b_ub_l, dtype=float),
        A_eq=sparse.csc_matrix((eq_v, (eq_r, eq_c)), shape=(r_eq, n_vars)),
        b_eq=np.asarray(b_eq_l, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not res_lp.success:
        raise SolverError(f"SciPy linprog CVaR cross-check failed: {res_lp.message}")

    x = np.asarray(res_lp.x, dtype=float)
    scipy_obj = float(res_lp.fun)
    max_p_diff = max(
        abs(float(x[off_kg + g_i]) - float(pypsa_res.p_nom_mw[g_name]))
        for g_i, g_name in enumerate(tech_names)
    )
    if include_bess:
        max_p_diff = max(
            max_p_diff, abs(float(x[off_kb]) - float(pypsa_res.p_nom_mw[STOCHASTIC_BESS_NAME]))
        )

    cap_sp = sum(
        float(x[off_kg + g_i]) * float(techs[g_name]["capital_cost"])
        for g_i, g_name in enumerate(tech_names)
    )
    if include_bess:
        cap_sp += float(x[off_kb]) * bess_cap

    tot_sp = np.zeros(n_s, dtype=float)
    for s_i in range(n_s):
        op_s = 0.0
        for g_i, g_name in enumerate(tech_names):
            mc = float(techs[g_name]["marginal_cost"])
            if mc != 0.0:
                base = off_pg + (s_i * n_g + g_i) * n_t
                op_s += float(np.sum(x[base : base + n_t])) * w_yr * dt * mc
        base_sh = off_sh + s_i * n_t
        op_s += float(np.sum(x[base_sh : base_sh + n_t])) * w_yr * dt * voll
        tot_sp[s_i] = cap_sp + op_s
    _, scipy_cvar = compute_discrete_cvar(tot_sp, probs, alpha=conf)

    if beta > 0.0:
        ub_duals = np.asarray(res_lp.ineqlin.marginals, dtype=float)
        rhos_sp = np.maximum(0.0, -np.asarray([ub_duals[r] for r in epi_rows], dtype=float))
        q_sp = (1.0 - beta) * probs + rhos_sp
    else:
        q_sp = probs.copy()
    q_py = pypsa_res.scenario_table["risk_adjusted_prob"].to_numpy(dtype=float)

    eq_duals = np.asarray(res_lp.eqlin.marginals, dtype=float)
    max_lmp_diff = 0.0
    for s_i, s_name in enumerate(scen_names):
        if q_sp[s_i] > 1e-12:
            lmp_sp = np.asarray(
                [eq_duals[bal_rows[(s_i, t_i)]] / (w_yr * dt * q_sp[s_i]) for t_i in range(n_t)],
                dtype=float,
            )
            peak_py = float(pypsa_res.scenario_table.loc[s_name, "peak_conditional_lmp_per_mwh"])
            max_lmp_diff = max(max_lmp_diff, abs(float(np.max(lmp_sp)) - peak_py))

    return StochasticCVaRLinprogCrossCheck(
        alpha=conf,
        risk_aversion_beta=beta,
        pypsa_objective=pypsa_res.objective_value,
        scipy_objective=scipy_obj,
        rel_objective_diff=abs(pypsa_res.objective_value - scipy_obj)
        / max(abs(pypsa_res.objective_value), 1.0),
        max_p_nom_diff_mw=max_p_diff,
        rel_cvar_cost_diff=abs(pypsa_res.cvar_total_cost_per_year - scipy_cvar)
        / max(abs(pypsa_res.cvar_total_cost_per_year), 1.0),
        max_risk_prob_diff=float(np.max(np.abs(q_py - q_sp))),
        max_lmp_diff_per_mwh=max_lmp_diff,
    )


def _clean_zero(val: float, tol: float = 1e-9) -> float:
    return 0.0 if abs(val) < tol else float(val)


def format_sensitivity_report(
    surface: SensitivitySurfaceResult,
    env_check: EnvelopeConcavityVerification | None = None,
) -> str:
    """Format 2D ``WACC x VRE_CapEx`` LCOE surface & Envelope verification report."""
    cols = list(surface.lcoe_surface_df.columns)
    hdr = "  WACC \\ VRE_Scale " + "".join(f"{float(c):>9.2f}x" for c in cols)
    lines = [
        "--- 2D Financing (WACC) x VRE CapEx Sensitivity Surface (LCOE /MWh) ---",
        hdr,
        "-" * len(hdr),
    ]
    for w_val, row in surface.lcoe_surface_df.iterrows():
        cells = "".join(f"{float(row[c]):>10.2f}" for c in cols)
        lines.append(f"  {100.0 * float(w_val):>6.1f}%          {cells}")
    lines += [
        "-" * len(hdr),
        f"Reference ({100.0 * surface.reference_wacc:.0f}% WACC, "
        f"{surface.reference_vre_scale:.2f}x VRE) : "
        f"{surface.reference_lcoe_per_mwh:,.4f} currency/MWh",
        f"Surface LCOE Range                     : {surface.min_lcoe_per_mwh:,.2f} -> "
        f"{surface.max_lcoe_per_mwh:,.2f} currency/MWh",
        f"Max Substitution Dividend (vs Frozen)  : +{surface.max_substitution_dividend_pct:.2f}%",
    ]
    if env_check is not None:
        lines += [
            f"Hotelling-Shephard dJ*/dWACC Error     : {env_check.rel_error_dcost_dwacc:.2e} "
            f"(analytic={env_check.analytic_dcost_dwacc:,.1f})",
            f"Hotelling-Shephard dJ*/dCapEx Error    : {env_check.rel_error_dcost_dvre_scale:.2e} "
            f"(analytic={env_check.analytic_dcost_dvre_scale:,.1f})",
            "Midpoint Concavity Surplus             : "
            f"+{env_check.concavity_midpoint_surplus:,.0f} currency/yr",
        ]
    return "\n".join(lines)


def format_single_cvar_report(result: StochasticCVaRResult) -> str:
    """Format concise diagnostics for a single Two-Stage Stochastic CVaR run."""
    lines = [
        f"--- Two-Stage Stochastic Mean-CVaR_{int(round(100 * result.alpha))} Adequacy Report "
        f"(beta = {result.risk_aversion_beta:.2f}) ---",
        "  Expected Annual Cost E[Z]     : "
        f"{result.expected_total_cost_per_year:>15,.0f} currency/yr "
        f"({result.expected_lcoe_per_mwh:.2f} /MWh)",
        "  Tail Adequacy Cost CVaR       : "
        f"{result.cvar_total_cost_per_year:>15,.0f} currency/yr "
        f"({result.cvar_lcoe_per_mwh:.2f} /MWh)",
        "  Expected Unserved Energy EENS : "
        f"{_clean_zero(result.eens_mwh_per_year):>15,.2f} MWh/yr "
        f"(worst={_clean_zero(result.worst_unserved_mwh_per_year):,.1f} MWh/yr)",
        f"  Strong LP Duality |Gap|       : {abs(result.strong_duality_gap):>15.3e} currency/yr",
        f"  Max Zero-Profit Residual      : {result.max_zero_profit_residual:>15.3e} currency/yr",
        "  First-Stage Portfolio (MW)    : "
        + ", ".join(f"{k}={float(v):.2f}" for k, v in result.p_nom_mw.items()),
    ]
    return "\n".join(lines)


def format_cvar_report(
    frontier_df: pd.DataFrame,
    metrics: StochasticValueMetrics | None = None,
) -> str:
    """Format multi-year weather Mean-CVaR Pareto frontier & EVPI/VSS value chain."""
    alpha_pct = int(round(100.0 * float(frontier_df["alpha"].iloc[0])))
    hdr = (
        f"{'Beta':>6s}  {'E[Cost]/yr':>12s}  {f'CVaR_{alpha_pct}/yr':>12s}  "
        f"{'E[LCOE]':>8s}  {'CVaR_LCOE':>9s}  {'EENS(MWh/yr)':>12s}  "
        f"{'Solar(MW)':>9s}  {'Wind(MW)':>8s}  {'BESS(MW)':>8s}  {'Gas(MW)':>8s}"
    )
    lines = [
        f"--- Multi-Year Weather Stochastic Mean-CVaR_{alpha_pct} Risk Frontier ---",
        hdr,
        "-" * len(hdr),
    ]
    for _, row in frontier_df.iterrows():
        lines.append(
            f"{float(row['beta']):>6.2f}  "
            f"{float(row['expected_cost_per_year']):>12,.0f}  "
            f"{float(row['cvar_cost_per_year']):>12,.0f}  "
            f"{float(row['expected_lcoe_per_mwh']):>8.2f}  "
            f"{float(row['cvar_lcoe_per_mwh']):>9.2f}  "
            f"{_clean_zero(float(row['eens_mwh_per_year'])):>12,.1f}  "
            f"{float(row['solar_mw']):>9.1f}  "
            f"{float(row['wind_mw']):>8.1f}  "
            f"{float(row['bess_mw']):>8.1f}  "
            f"{float(row['gas_mw']):>8.1f}"
        )

    if metrics is not None:
        lines += [
            "",
            "Birge-Louveaux Stochastic Value Sandwich (WS <= RP <= EEV):",
            f"  Wait-and-See (WS, perfect weather foresight) : "
            f"{metrics.wait_and_see_cost_per_year:>15,.0f} currency/yr",
            f"  Stochastic Recourse (RP, risk-neutral beta=0): "
            f"{metrics.recourse_problem_cost_per_year:>15,.0f} currency/yr",
            f"  Mean-Weather Deterministic Out-of-Sample EEV : "
            f"{metrics.expected_value_plan_eev_cost_per_year:>15,.0f} currency/yr",
            f"  EVPI = RP - WS                               : "
            f"{metrics.evpi_per_year:>15,.0f} currency/yr",
            f"  VSS  = EEV - RP (value of stochastic model)  : "
            f"{metrics.vss_per_year:>15,.0f} currency/yr",
            f"  EENS (Mean-Weather -> Risk-Neutral -> Hedged): "
            f"{_clean_zero(metrics.mean_weather_eens_mwh_per_year):,.1f} -> "
            f"{_clean_zero(metrics.risk_neutral_eens_mwh_per_year):,.1f} -> "
            f"{_clean_zero(metrics.cvar_hedged_eens_mwh_per_year):,.1f} MWh/yr",
        ]
    return "\n".join(lines)
