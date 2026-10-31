"""Thermal ramp-rate envelopes, minimum stable must-run output & spinning reserves.

Mathematical identities verified by this module
-----------------------------------------------
1. Thermal Operational Envelope (Must-Run & Intertemporal Ramp Limits):
   For dispatchable thermal unit ``g`` with installed capacity ``K_g``, minimum
   stable loading ``p_min_pu in [0, 1]``, and upward/downward hourly ramp rates
   ``r_up, r_dn in (0, 1]`` (p.u. of ``K_g`` per hour):

       p_min_pu * K_g <= p_{g,t} <= K_g,
       -r_dn * K_g <= p_{g,t} - p_{g,t-1} <= r_up * K_g.

2. Co-Optimized Upward Spinning Operating Reserves & BESS SoC Coupling:
   Each reserve-eligible thermal unit ``g`` provides upward spinning reserve
   ``r_{g,t} >= 0`` bounded by online headroom and sub-hourly ramp deliverability:

       p_{g,t} + r_{g,t} <= K_g,       0 <= r_{g,t} <= rho_res,g * K_g.

   When 4-hour Li-Ion BESS is co-optimized, battery upward reserve ``r_{bess,t} >= 0``
   is simultaneously bounded by inverter discharge headroom and stored energy SoC:

       eta_dis * p_{dis,t} + r_{bess,t} <= eta_dis * K_{dis},
       r_{bess,t} * dt_sustain <= eta_dis * e_{bess,t}.

   System-wide upward spinning reserve adequacy enforces at every snapshot ``t``:

       sum_g r_{g,t} + r_{bess,t} >= alpha_load * L_t + alpha_vre * sum_{v in VRE} p_{v,t}
       (with KKT reserve shadow price mu_res,t >= 0 [currency / MW-h]).

3. Co-Optimized Energy + Reserve Revenue Adequacy (Strong LP Duality):
   In any greenfield capacity-expansion LP with homogeneous operational, ramping,
   must-run, and cyclic storage constraints, Strong LP Duality equates total
   annualised capital + fuel cost to nodal energy revenue plus exogenous load-reserve
   revenue to solver tolerance (~1e-15 relative):

       C_total* = sum_t w_t * L_t * (lambda_t + alpha_load * mu_res,t) * dt,
       System_LCOE = LoadWeighted_LMP + alpha_load * LoadWeighted_Reserve_Price.

4. Intertemporal Ramp-Induced Negative Nodal LMP Theorem (Zero Subsidies):
   Even when every generator has strictly non-negative fuel cost (``c_g >= 0``),
   binding downward and upward ramp limits across a deep net-load trough force
   inflexible baseload to hold output at ``t_0`` so it can displace high-cost
   peakers in adjacent hours ``t_0 +/- k``, yielding an exact negative nodal
   shadow price ``lambda_{t_0} < 0`` via KKT ramp-multiplier telescoping.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, lil_matrix

from .model import (
    BUS_CARRIER,
    BUS_NAME,
    DEFAULT_TECHNOLOGIES,
    LOAD_NAME,
    RENEWABLE_CARRIER,
    THERMAL_CARRIER,
    SolverError,
)
from .profiles import CAPACITY_FACTOR_COLUMNS, SNAPSHOT_HOURS, validate_profiles

if hasattr(pypsa, "options") and hasattr(pypsa.options, "api"):
    pypsa.options.api.legacy_string_dtype = True

ANNUAL_HOURS_PER_YEAR = 8760.0
SPINNING_RESERVE_CONSTRAINT = "SystemSpinningReserve"

DEFAULT_BESS_STORE_CAPITAL_COST_PER_MWH_YR = 13_500.0
DEFAULT_BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR = 7_000.0
DEFAULT_BESS_CHARGE_EFFICIENCY = 0.95
DEFAULT_BESS_DISCHARGE_EFFICIENCY = 0.95

DEFAULT_RAMP_SENSITIVITY_SCALES: tuple[float, ...] = (
    1.00,
    0.75,
    0.50,
    0.35,
    0.25,
)


def _validate_non_negative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return val


def _validate_unit_interval(value: float, name: str, *, allow_zero: bool = True) -> float:
    val = _validate_non_negative_float(value, name)
    if not allow_zero and val <= 0.0:
        raise ValueError(f"{name} must lie in (0.0, 1.0]")
    if val > 1.0:
        raise ValueError(f"{name} must not exceed 1.0")
    return val


@dataclass(frozen=True)
class FlexibleTechSpec:
    """Generator cost, must-run floor, hourly ramp envelope & spinning-reserve capability."""

    name: str
    carrier: str
    capital_cost_per_mw_yr: float
    marginal_cost_per_mwh: float
    p_min_pu: float = 0.0
    ramp_limit_up: float | None = None
    ramp_limit_down: float | None = None
    reserve_cap_pu: float = 0.0
    profile: str | None = None
    p_nom_extendable: bool = True
    p_nom_mw: float = 0.0

    def __post_init__(self) -> None:
        _validate_non_negative_float(self.capital_cost_per_mw_yr, "capital_cost_per_mw_yr")
        _validate_non_negative_float(self.marginal_cost_per_mwh, "marginal_cost_per_mwh")
        _validate_unit_interval(self.p_min_pu, "p_min_pu", allow_zero=True)
        _validate_unit_interval(self.reserve_cap_pu, "reserve_cap_pu", allow_zero=True)
        _validate_non_negative_float(self.p_nom_mw, "p_nom_mw")
        if self.ramp_limit_up is not None:
            _validate_unit_interval(self.ramp_limit_up, "ramp_limit_up", allow_zero=False)
        if self.ramp_limit_down is not None:
            _validate_unit_interval(self.ramp_limit_down, "ramp_limit_down", allow_zero=False)
        if self.profile is not None:
            if self.profile not in CAPACITY_FACTOR_COLUMNS:
                raise ValueError(
                    f"unknown profile {self.profile!r}; expected one of {CAPACITY_FACTOR_COLUMNS}"
                )
            if self.p_min_pu > 0.0:
                raise ValueError("weather-driven VRE generator cannot set p_min_pu > 0")


@dataclass(frozen=True)
class SpinningReservePolicy:
    """System-wide upward spinning operating reserve requirement rule."""

    enabled: bool = True
    load_reserve_fraction: float = 0.06
    vre_reserve_fraction: float = 0.12
    bess_sustain_hours: float = 1.0

    def __post_init__(self) -> None:
        _validate_unit_interval(
            self.load_reserve_fraction, "load_reserve_fraction", allow_zero=True
        )
        _validate_unit_interval(self.vre_reserve_fraction, "vre_reserve_fraction", allow_zero=True)
        sustain = _validate_non_negative_float(self.bess_sustain_hours, "bess_sustain_hours")
        if sustain <= 0.0:
            raise ValueError("bess_sustain_hours must be strictly positive")

    @property
    def is_active(self) -> bool:
        return bool(
            self.enabled and (self.load_reserve_fraction > 0.0 or self.vre_reserve_fraction > 0.0)
        )


def make_duck_curve_profiles(days: int = 2) -> pd.DataFrame:
    """Chronological multi-day high-solar Duck-Curve profile, deterministic.

    Day 1 is a clear spring day: solar CF peaks near 0.94, the midday net-load belly is
    deep, the 16:00-19:00 sunset ramp is steep. Day 2 is a partly cloudy afternoon with
    higher evening demand and less wind, which is what puts pressure on reserves and on
    ramping.
    """
    if isinstance(days, bool) or not isinstance(days, int) or days < 1 or days > 14:
        raise ValueError("days must be an integer between 1 and 14")
    n_h = 24 * days
    idx = pd.date_range("2026-04-01", periods=n_h, freq="h")
    h = np.arange(n_h, dtype=float)
    hod = h % 24.0
    day_idx = np.floor(h / 24.0).astype(int)

    load_day_mult = np.where(day_idx % 2 == 0, 0.98, 1.05)
    load_mw = (
        140.0
        + 24.0 * np.exp(-0.5 * ((hod - 8.0) / 1.8) ** 2)
        + 62.0 * np.exp(-0.5 * ((hod - 19.0) / 2.0) ** 2)
        - 26.0 * np.exp(-0.5 * ((hod - 3.5) / 2.3) ** 2)
    ) * load_day_mult

    solar_day_mult = np.where(day_idx % 2 == 0, 0.94, 0.64)
    raw_sin = np.sin(np.pi * (hod - 6.0) / 12.0)
    solar_cf = np.where(
        (hod >= 6.0) & (hod <= 18.0),
        np.clip(solar_day_mult * np.maximum(0.0, raw_sin) ** 1.15, 0.0, 1.0),
        0.0,
    )

    wind_day_mult = np.where(day_idx % 2 == 0, 1.05, 0.68)
    wind_cf = np.clip(
        (
            0.39
            + 0.16 * np.cos(2.0 * np.pi * (hod - 2.5) / 24.0)
            - 0.18 * np.exp(-0.5 * ((hod - 18.5) / 2.3) ** 2)
        )
        * wind_day_mult,
        0.06,
        0.76,
    )

    df = pd.DataFrame(
        {"load_mw": load_mw, "solar_cf": solar_cf, "wind_cf": wind_cf},
        index=idx,
    )
    return validate_profiles(df)


def default_flexible_fleet_specs() -> dict[str, FlexibleTechSpec]:
    """Return the 5-technology thermal + VRE portfolio with realistic ramp/must-run bounds."""
    items = (
        FlexibleTechSpec(
            name="Baseload_Thermal",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=92_000.0,
            marginal_cost_per_mwh=20.0,
            p_min_pu=0.60,
            ramp_limit_up=0.10,
            ramp_limit_down=0.10,
            reserve_cap_pu=0.05,
            profile=None,
        ),
        FlexibleTechSpec(
            name="CCGT_MidMerit",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=36_000.0,
            marginal_cost_per_mwh=38.0,
            p_min_pu=0.20,
            ramp_limit_up=0.45,
            ramp_limit_down=0.45,
            reserve_cap_pu=0.25,
            profile=None,
        ),
        FlexibleTechSpec(
            name="OCGT_FastPeaker",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=20_000.0,
            marginal_cost_per_mwh=78.0,
            p_min_pu=0.00,
            ramp_limit_up=1.00,
            ramp_limit_down=1.00,
            reserve_cap_pu=0.55,
            profile=None,
        ),
        FlexibleTechSpec(
            name="Solar_PV",
            carrier=RENEWABLE_CARRIER,
            capital_cost_per_mw_yr=36_000.0,
            marginal_cost_per_mwh=0.0,
            p_min_pu=0.00,
            ramp_limit_up=None,
            ramp_limit_down=None,
            reserve_cap_pu=0.00,
            profile="solar_cf",
        ),
        FlexibleTechSpec(
            name="Wind_Turbine",
            carrier=RENEWABLE_CARRIER,
            capital_cost_per_mw_yr=62_000.0,
            marginal_cost_per_mwh=0.0,
            p_min_pu=0.00,
            ramp_limit_up=None,
            ramp_limit_down=None,
            reserve_cap_pu=0.00,
            profile="wind_cf",
        ),
    )
    return {sp.name: sp for sp in items}


def baseline_flexibility_specs(
    technologies: dict[str, dict] | None = None,
    *,
    thermal_p_min_pu: float = 0.25,
    thermal_ramp_limit_pu: float = 0.40,
    thermal_reserve_cap_pu: float = 0.35,
) -> dict[str, FlexibleTechSpec]:
    """Map DEFAULT_TECHNOLOGIES (or custom dict) onto FlexibleTechSpec."""
    techs = DEFAULT_TECHNOLOGIES if technologies is None else technologies
    if not techs:
        raise ValueError("technologies must not be empty")
    pmin = _validate_unit_interval(thermal_p_min_pu, "thermal_p_min_pu", allow_zero=True)
    rlim = _validate_unit_interval(thermal_ramp_limit_pu, "thermal_ramp_limit_pu", allow_zero=False)
    rcap = _validate_unit_interval(
        thermal_reserve_cap_pu, "thermal_reserve_cap_pu", allow_zero=True
    )

    out: dict[str, FlexibleTechSpec] = {}
    for name, raw in techs.items():
        prof = raw.get("profile")
        is_vre = prof is not None
        out[name] = FlexibleTechSpec(
            name=name,
            carrier=RENEWABLE_CARRIER if is_vre else THERMAL_CARRIER,
            capital_cost_per_mw_yr=float(raw["capital_cost"]),
            marginal_cost_per_mwh=float(raw["marginal_cost"]),
            p_min_pu=0.0 if is_vre else float(raw.get("p_min_pu", pmin)),
            ramp_limit_up=None if is_vre else float(raw.get("ramp_limit_up", rlim)),
            ramp_limit_down=None if is_vre else float(raw.get("ramp_limit_down", rlim)),
            reserve_cap_pu=0.0 if is_vre else float(raw.get("reserve_cap_pu", rcap)),
            profile=prof,
            p_nom_extendable=bool(raw.get("p_nom_extendable", True)),
            p_nom_mw=float(raw.get("p_nom", 0.0)),
        )
    return out


@dataclass
class FlexibilityScenarioResult:
    """Primal, dual, ramping, curtailment & spinning-reserve adequacy summary."""

    enforce_ramping: bool
    enforce_must_run: bool
    cyclic_ramping: bool
    include_bess: bool
    reserve_policy: SpinningReservePolicy
    annual_demand_mwh: float
    p_nom_mw: pd.Series
    bess_e_nom_mwh: float
    bess_charge_p_nom_mw: float
    bess_discharge_p_nom_mw: float
    dispatch_mw: pd.DataFrame
    bess_charge_mw: pd.Series
    bess_discharge_mw: pd.Series
    bess_soc_mwh: pd.Series
    net_load_mw: pd.Series
    max_evening_net_load_ramp_mw_per_h: float
    max_thermal_ramp_utilization_pct: pd.Series
    reserve_requirement_mw: pd.Series
    reserve_by_unit_mw: pd.DataFrame
    bess_reserve_mw: pd.Series
    lmp_per_mwh: pd.Series
    reserve_price_per_mw_h: pd.Series
    load_weighted_lmp_per_mwh: float
    load_weighted_reserve_price_per_mw_h: float
    annual_gen_capex_cost: float
    annual_bess_capex_cost: float
    annual_fuel_opex_cost: float
    total_annual_cost: float
    system_lcoe_per_mwh: float
    annual_energy_market_revenue: float
    annual_reserve_market_revenue: float
    strong_duality_revenue_gap: float
    annual_curtailment_mwh: float
    vre_curtailment_pct: float
    power_balance_max_residual_mw: float


def build_flexibility_network(
    profiles: pd.DataFrame,
    *,
    specs: dict[str, FlexibleTechSpec] | None = None,
    enforce_ramping: bool = True,
    enforce_must_run: bool = True,
    include_bess: bool = False,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> pypsa.Network:
    """Build a single-bus PyPSA network with optional must-run, ramp envelopes & BESS."""
    prof = validate_profiles(profiles)
    gen_specs = default_flexible_fleet_specs() if specs is None else specs
    if not gen_specs:
        raise ValueError("specs must contain at least one technology")
    ann_h = _validate_non_negative_float(annual_hours, "annual_hours")
    if ann_h <= 0.0:
        raise ValueError("annual_hours must be strictly positive")

    n_t = len(prof.index)
    w = ann_h / (float(n_t) * SNAPSHOT_HOURS)

    network = pypsa.Network()
    network.set_snapshots(prof.index)
    network.snapshot_weightings = network.snapshot_weightings.astype(float)
    network.snapshot_weightings.loc[:, "objective"] = w
    network.snapshot_weightings.loc[:, "generators"] = w
    network.snapshot_weightings.loc[:, "stores"] = SNAPSHOT_HOURS

    network.add("Carrier", BUS_CARRIER)
    added_carriers = {BUS_CARRIER}
    for sp in gen_specs.values():
        if sp.carrier not in added_carriers:
            network.add("Carrier", sp.carrier)
            added_carriers.add(sp.carrier)

    network.add("Bus", BUS_NAME, carrier=BUS_CARRIER)
    network.add("Load", LOAD_NAME, bus=BUS_NAME, p_set=prof["load_mw"])

    for name, sp in gen_specs.items():
        kwargs: dict[str, object] = {
            "bus": BUS_NAME,
            "carrier": sp.carrier,
            "capital_cost": float(sp.capital_cost_per_mw_yr),
            "marginal_cost": float(sp.marginal_cost_per_mwh),
            "p_nom_extendable": bool(sp.p_nom_extendable),
        }
        if not sp.p_nom_extendable:
            kwargs["p_nom"] = float(sp.p_nom_mw)
        if enforce_must_run and sp.p_min_pu > 0.0:
            kwargs["p_min_pu"] = float(sp.p_min_pu)
        if enforce_ramping:
            if sp.ramp_limit_up is not None:
                kwargs["ramp_limit_up"] = float(sp.ramp_limit_up)
            if sp.ramp_limit_down is not None:
                kwargs["ramp_limit_down"] = float(sp.ramp_limit_down)
        network.add("Generator", name, **kwargs)
        if sp.profile is not None:
            network.generators_t.p_max_pu[name] = prof[sp.profile]

    if include_bess:
        if "battery" not in added_carriers:
            network.add("Carrier", "battery")
        network.add("Bus", "Battery_Bus", carrier="battery")
        network.add(
            "Store",
            "LiIon_BESS",
            bus="Battery_Bus",
            carrier="battery",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=DEFAULT_BESS_STORE_CAPITAL_COST_PER_MWH_YR,
            marginal_cost=0.0,
        )
        network.add(
            "Link",
            "BESS_Charge",
            bus0=BUS_NAME,
            bus1="Battery_Bus",
            efficiency=DEFAULT_BESS_CHARGE_EFFICIENCY,
            p_nom_extendable=True,
            capital_cost=DEFAULT_BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR,
            marginal_cost=0.0,
        )
        network.add(
            "Link",
            "BESS_Discharge",
            bus0="Battery_Bus",
            bus1=BUS_NAME,
            efficiency=DEFAULT_BESS_DISCHARGE_EFFICIENCY,
            p_nom_extendable=True,
            capital_cost=(
                DEFAULT_BESS_DISCHARGE_EFFICIENCY * DEFAULT_BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR
            ),
            marginal_cost=0.0,
        )

    return network


def run_flexibility_scenario(
    profiles: pd.DataFrame | None = None,
    *,
    specs: dict[str, FlexibleTechSpec] | None = None,
    enforce_ramping: bool = True,
    enforce_must_run: bool = True,
    cyclic_ramping: bool = True,
    include_bess: bool = False,
    reserve_policy: SpinningReservePolicy | None = None,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
    solver_name: str = "highs",
) -> FlexibilityScenarioResult:
    """Solve capacity expansion / dispatch with ramping, must-run & spinning reserves."""
    prof = make_duck_curve_profiles() if profiles is None else validate_profiles(profiles)
    gen_specs = default_flexible_fleet_specs() if specs is None else specs
    if not gen_specs:
        raise ValueError("specs must contain at least one technology")
    policy = SpinningReservePolicy() if reserve_policy is None else reserve_policy
    ann_h = _validate_non_negative_float(annual_hours, "annual_hours")
    if ann_h <= 0.0:
        raise ValueError("annual_hours must be strictly positive")

    network = build_flexibility_network(
        prof,
        specs=gen_specs,
        enforce_ramping=enforce_ramping,
        enforce_must_run=enforce_must_run,
        include_bess=include_bess,
        annual_hours=ann_h,
    )

    names = list(gen_specs)
    res_units = [g for g in names if gen_specs[g].reserve_cap_pu > 0.0]
    vre_units = [g for g in names if gen_specs[g].profile is not None]
    load_arr = prof["load_mw"].to_numpy(dtype=float)

    def _extra_constraints(n: pypsa.Network, snapshots: pd.Index) -> None:
        m = n.model
        p_gen = m["Generator-p"]
        ext_names = set(n.generators.index[n.generators["p_nom_extendable"].astype(bool)])
        k_ext = m["Generator-p_nom"] if ext_names else None

        # 1. Optional periodic wrap-around ramp constraint at snapshot 0 (p[0] - p[-1])
        if enforce_ramping and cyclic_ramping and len(snapshots) > 1:
            for g in names:
                sp = gen_specs[g]
                p0 = p_gen.sel(name=g).isel(snapshot=0, drop=True)
                p_last = p_gen.sel(name=g).isel(snapshot=-1, drop=True)
                if sp.ramp_limit_up is not None:
                    if g in ext_names and k_ext is not None:
                        kg = k_ext.sel(name=g, drop=True)
                        m.add_constraints(
                            p0 - p_last - sp.ramp_limit_up * kg <= 0,
                            name=f"WrapRampUp_{g}",
                        )
                    else:
                        m.add_constraints(
                            p0 - p_last <= sp.ramp_limit_up * sp.p_nom_mw,
                            name=f"WrapRampUp_{g}",
                        )
                if sp.ramp_limit_down is not None:
                    if g in ext_names and k_ext is not None:
                        kg = k_ext.sel(name=g, drop=True)
                        m.add_constraints(
                            p0 - p_last + sp.ramp_limit_down * kg >= 0,
                            name=f"WrapRampDn_{g}",
                        )
                    else:
                        m.add_constraints(
                            p0 - p_last >= -sp.ramp_limit_down * sp.p_nom_mw,
                            name=f"WrapRampDn_{g}",
                        )

        # 2. Co-optimized upward spinning operating reserves
        if policy.is_active:
            total_r = None
            if res_units:
                th_idx = pd.Index(res_units, name="reserve_unit")
                r_gen = m.add_variables(
                    lower=0.0,
                    coords=[snapshots, th_idx],
                    name="reserve_gen_up",
                )
                for g in res_units:
                    sp = gen_specs[g]
                    pg = p_gen.sel(name=g, drop=True)
                    rg = r_gen.sel(reserve_unit=g, drop=True)
                    if g in ext_names and k_ext is not None:
                        kg = k_ext.sel(name=g, drop=True)
                        m.add_constraints(pg + rg - kg <= 0, name=f"ResHeadroom_{g}")
                        m.add_constraints(
                            rg - sp.reserve_cap_pu * kg <= 0,
                            name=f"ResRampCap_{g}",
                        )
                    else:
                        m.add_constraints(pg + rg <= sp.p_nom_mw, name=f"ResHeadroom_{g}")
                        m.add_constraints(
                            rg <= sp.reserve_cap_pu * sp.p_nom_mw,
                            name=f"ResRampCap_{g}",
                        )
                total_r = r_gen.sum("reserve_unit")

            if include_bess:
                r_bess = m.add_variables(
                    lower=0.0,
                    coords=[snapshots],
                    name="reserve_bess_up",
                )
                p_dis = m["Link-p"].sel(name="BESS_Discharge", drop=True)
                k_dis = m["Link-p_nom"].sel(name="BESS_Discharge", drop=True)
                e_soc = m["Store-e"].sel(name="LiIon_BESS", drop=True)
                eta_d = DEFAULT_BESS_DISCHARGE_EFFICIENCY
                m.add_constraints(
                    eta_d * p_dis + r_bess - eta_d * k_dis <= 0,
                    name="ResBESSHeadroom",
                )
                m.add_constraints(
                    policy.bess_sustain_hours * r_bess - eta_d * e_soc <= 0,
                    name="ResBESSSoC",
                )
                total_r = r_bess if total_r is None else (total_r + r_bess)

            if total_r is None:
                raise SolverError(
                    "Active spinning reserve policy has zero reserve-capable units or storage"
                )

            lhs_expr = total_r
            if vre_units and policy.vre_reserve_fraction > 0.0:
                vre_p = p_gen.sel(name=vre_units).sum("name")
                lhs_expr = lhs_expr - policy.vre_reserve_fraction * vre_p

            req_rhs = xr.DataArray(
                policy.load_reserve_fraction * load_arr,
                coords=[snapshots],
                dims=["snapshot"],
            )
            m.add_constraints(lhs_expr >= req_rhs, name=SPINNING_RESERVE_CONSTRAINT)

    try:
        result = network.optimize(
            solver_name=solver_name,
            solver_options={"log_to_console": False},
            extra_functionality=_extra_constraints,
            include_objective_constant=False,
        )
    except Exception as exc:
        raise SolverError(f"Flexibility optimization failed: {exc}") from exc

    status, condition = result if isinstance(result, tuple) else (str(result), "")
    if status != "ok" or condition not in ("optimal", ""):
        raise SolverError(
            f"Solver did not find an optimal solution: status={status!r}, condition={condition!r}"
        )

    n_t = len(prof.index)
    w = ann_h / (float(n_t) * SNAPSHOT_HOURS)
    weight_mwh = w * SNAPSHOT_HOURS
    load_s = prof["load_mw"].astype(float)
    annual_demand = float(load_s.sum() * weight_mwh)

    p = network.generators_t.p[names].astype(float).copy()
    p_nom = pd.Series(
        {
            g: (
                float(network.generators.at[g, "p_nom_opt"])
                if gen_specs[g].p_nom_extendable
                else float(gen_specs[g].p_nom_mw)
            )
            for g in names
        },
        dtype=float,
    ).clip(lower=0.0)

    if include_bess:
        bess_e_nom = max(0.0, float(network.stores.at["LiIon_BESS", "e_nom_opt"]))
        bess_chg_nom = max(0.0, float(network.links.at["BESS_Charge", "p_nom_opt"]))
        bess_dis_nom = max(
            0.0,
            DEFAULT_BESS_DISCHARGE_EFFICIENCY
            * float(network.links.at["BESS_Discharge", "p_nom_opt"]),
        )
        bess_chg = network.links_t.p0["BESS_Charge"].astype(float).clip(lower=0.0)
        bess_dis = (-network.links_t.p1["BESS_Discharge"]).astype(float).clip(lower=0.0)
        bess_soc = network.stores_t.e["LiIon_BESS"].astype(float).clip(lower=0.0)
        bess_capex = float(
            (network.stores["capital_cost"] * network.stores["e_nom_opt"]).sum()
            + (network.links["capital_cost"] * network.links["p_nom_opt"]).sum()
        )
    else:
        bess_e_nom = 0.0
        bess_chg_nom = 0.0
        bess_dis_nom = 0.0
        bess_chg = pd.Series(0.0, index=prof.index, dtype=float)
        bess_dis = pd.Series(0.0, index=prof.index, dtype=float)
        bess_soc = pd.Series(0.0, index=prof.index, dtype=float)
        bess_capex = 0.0

    # Net load & evening ramp diagnostics
    vre_gen = p[vre_units].sum(axis=1) if vre_units else pd.Series(0.0, index=prof.index)
    net_load = load_s - vre_gen
    max_net_ramp = float(net_load.diff().dropna().max()) if len(net_load) > 1 else 0.0

    # Thermal ramp utilization (% of hourly envelope)
    ramp_util: dict[str, float] = {}
    for g in names:
        sp = gen_specs[g]
        kg = float(p_nom[g])
        if sp.ramp_limit_up is not None and kg > 1e-6 and len(p) > 1:
            dp = p[g].diff().dropna()
            if cyclic_ramping:
                wrap_dp = pd.Series([float(p[g].iloc[0] - p[g].iloc[-1])])
                dp = pd.concat([wrap_dp, dp], ignore_index=True)
            max_up_ratio = float(dp.clip(lower=0.0).max()) / (sp.ramp_limit_up * kg)
            r_dn = sp.ramp_limit_down if sp.ramp_limit_down is not None else sp.ramp_limit_up
            max_dn_ratio = float((-dp).clip(lower=0.0).max()) / (r_dn * kg)
            ramp_util[g] = 100.0 * max(max_up_ratio, max_dn_ratio)
        else:
            ramp_util[g] = 0.0

    # Reserve solutions & shadow prices
    if policy.is_active:
        req_series = (
            policy.load_reserve_fraction * load_s + policy.vre_reserve_fraction * vre_gen
        ).astype(float)
        if res_units:
            r_gen_df = (
                network.model.variables["reserve_gen_up"]
                .solution.to_pandas()
                .reindex(index=prof.index, columns=res_units)
                .astype(float)
                .clip(lower=0.0)
            )
        else:
            r_gen_df = pd.DataFrame(index=prof.index)
        if include_bess:
            r_bess_s = (
                network.model.variables["reserve_bess_up"]
                .solution.to_pandas()
                .reindex(prof.index)
                .astype(float)
                .clip(lower=0.0)
            )
        else:
            r_bess_s = pd.Series(0.0, index=prof.index, dtype=float)
        raw_res_dual = (
            network.model.constraints[SPINNING_RESERVE_CONSTRAINT]
            .dual.to_pandas()
            .reindex(prof.index)
            .astype(float)
        )
        res_price = (raw_res_dual / weight_mwh).where(
            (raw_res_dual / weight_mwh).abs() > 1e-12, 0.0
        )
    else:
        req_series = pd.Series(0.0, index=prof.index, dtype=float)
        r_gen_df = pd.DataFrame(0.0, index=prof.index, columns=res_units)
        r_bess_s = pd.Series(0.0, index=prof.index, dtype=float)
        res_price = pd.Series(0.0, index=prof.index, dtype=float)

    raw_lmp = network.buses_t.marginal_price[BUS_NAME].astype(float).copy()
    lmp = raw_lmp.where(raw_lmp.abs() > 1e-12, 0.0)
    load_wtd_lmp = float((load_s * lmp).sum() * weight_mwh / annual_demand)
    load_wtd_res_p = float((load_s * res_price).sum() * weight_mwh / annual_demand)

    # Curtailment
    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")[names]
    avail = p_max_pu * p_nom
    if vre_units:
        avail_vre_mwh = float(avail[vre_units].sum().sum() * weight_mwh)
        curt_mwh = float((avail[vre_units] - p[vre_units]).clip(lower=0.0).sum().sum() * weight_mwh)
        curt_pct = 100.0 * curt_mwh / avail_vre_mwh if avail_vre_mwh > 1e-9 else 0.0
    else:
        curt_mwh = 0.0
        curt_pct = 0.0

    # Costs & Strong LP Revenue Adequacy
    gen_capex = float(
        sum(
            gen_specs[g].capital_cost_per_mw_yr * float(p_nom[g])
            for g in names
            if gen_specs[g].p_nom_extendable
        )
    )
    annual_gen_mwh = p.sum(axis=0) * weight_mwh
    fuel_opex = float(
        sum(gen_specs[g].marginal_cost_per_mwh * float(annual_gen_mwh[g]) for g in names)
    )
    total_cost = gen_capex + bess_capex + fuel_opex
    energy_rev = float((load_s * lmp).sum() * weight_mwh)
    exog_res_mw = (policy.load_reserve_fraction if policy.is_active else 0.0) * load_s
    reserve_rev = float((exog_res_mw * res_price).sum() * weight_mwh)
    all_extendable = all(gen_specs[g].p_nom_extendable for g in names)
    duality_gap = abs(total_cost - (energy_rev + reserve_rev)) if all_extendable else 0.0

    power_res = float((p.sum(axis=1) - bess_chg + bess_dis - load_s).abs().max())

    return FlexibilityScenarioResult(
        enforce_ramping=enforce_ramping,
        enforce_must_run=enforce_must_run,
        cyclic_ramping=cyclic_ramping,
        include_bess=include_bess,
        reserve_policy=policy,
        annual_demand_mwh=annual_demand,
        p_nom_mw=p_nom,
        bess_e_nom_mwh=bess_e_nom,
        bess_charge_p_nom_mw=bess_chg_nom,
        bess_discharge_p_nom_mw=bess_dis_nom,
        dispatch_mw=p,
        bess_charge_mw=bess_chg,
        bess_discharge_mw=bess_dis,
        bess_soc_mwh=bess_soc,
        net_load_mw=net_load,
        max_evening_net_load_ramp_mw_per_h=max_net_ramp,
        max_thermal_ramp_utilization_pct=pd.Series(ramp_util, dtype=float),
        reserve_requirement_mw=req_series,
        reserve_by_unit_mw=r_gen_df,
        bess_reserve_mw=r_bess_s,
        lmp_per_mwh=lmp,
        reserve_price_per_mw_h=res_price,
        load_weighted_lmp_per_mwh=load_wtd_lmp,
        load_weighted_reserve_price_per_mw_h=load_wtd_res_p,
        annual_gen_capex_cost=gen_capex,
        annual_bess_capex_cost=bess_capex,
        annual_fuel_opex_cost=fuel_opex,
        total_annual_cost=total_cost,
        system_lcoe_per_mwh=total_cost / annual_demand,
        annual_energy_market_revenue=energy_rev,
        annual_reserve_market_revenue=reserve_rev,
        strong_duality_revenue_gap=duality_gap,
        annual_curtailment_mwh=curt_mwh,
        vre_curtailment_pct=curt_pct,
        power_balance_max_residual_mw=power_res,
    )


def compare_flexibility_regimes(
    profiles: pd.DataFrame | None = None,
    *,
    specs: dict[str, FlexibleTechSpec] | None = None,
    reserve_policy: SpinningReservePolicy | None = None,
) -> pd.DataFrame:
    """Compare the 4 canonical operational flexibility & spinning-reserve regimes."""
    prof = make_duck_curve_profiles() if profiles is None else validate_profiles(profiles)
    gen_specs = default_flexible_fleet_specs() if specs is None else specs
    pol_on = SpinningReservePolicy() if reserve_policy is None else reserve_policy
    pol_off = SpinningReservePolicy(enabled=False)

    configs = (
        ("1_Relaxed_NoReserves", False, False, pol_off, False),
        ("2_MustRun_And_Ramping", True, True, pol_off, False),
        ("3_Inflexible_With_Reserves", True, True, pol_on, False),
        ("4_Flexible_BESS_CoOptimized", True, True, pol_on, True),
    )

    rows: list[dict[str, object]] = []
    for name, ramp, must_run, pol, bess in configs:
        res = run_flexibility_scenario(
            prof,
            specs=gen_specs,
            enforce_ramping=ramp,
            enforce_must_run=must_run,
            include_bess=bess,
            reserve_policy=pol,
        )
        vre_mw = float(res.p_nom_mw.get("Solar_PV", 0.0) + res.p_nom_mw.get("Wind_Turbine", 0.0))
        rows.append(
            {
                "regime": name,
                "system_lcoe_per_mwh": res.system_lcoe_per_mwh,
                "load_weighted_lmp_per_mwh": res.load_weighted_lmp_per_mwh,
                "load_weighted_reserve_price_per_mw_h": res.load_weighted_reserve_price_per_mw_h,
                "max_lmp_per_mwh": float(res.lmp_per_mwh.max()),
                "max_reserve_price_per_mw_h": float(res.reserve_price_per_mw_h.max()),
                "baseload_mw": float(res.p_nom_mw.get("Baseload_Thermal", 0.0)),
                "ccgt_mw": float(res.p_nom_mw.get("CCGT_MidMerit", 0.0)),
                "ocgt_mw": float(res.p_nom_mw.get("OCGT_FastPeaker", 0.0)),
                "solar_mw": float(res.p_nom_mw.get("Solar_PV", 0.0)),
                "wind_mw": float(res.p_nom_mw.get("Wind_Turbine", 0.0)),
                "vre_total_mw": vre_mw,
                "bess_e_nom_mwh": res.bess_e_nom_mwh,
                "vre_curtailment_pct": res.vre_curtailment_pct,
                "max_evening_net_load_ramp_mw_per_h": res.max_evening_net_load_ramp_mw_per_h,
                "strong_duality_revenue_gap": res.strong_duality_revenue_gap,
            }
        )
    return pd.DataFrame(rows)


def sweep_thermal_ramp_tightness(
    profiles: pd.DataFrame | None = None,
    *,
    ramp_scales: Sequence[float] = DEFAULT_RAMP_SENSITIVITY_SCALES,
    include_bess: bool = False,
) -> pd.DataFrame:
    """Scale thermal ramp limits and verify nested-polytope LCOE monotonicity."""
    if not ramp_scales:
        raise ValueError("ramp_scales must contain at least one scale factor")
    prof = make_duck_curve_profiles() if profiles is None else validate_profiles(profiles)
    base = default_flexible_fleet_specs()

    rows: list[dict[str, float]] = []
    for scale in sorted(ramp_scales, reverse=True):
        sc = _validate_non_negative_float(scale, "ramp_scale")
        if sc <= 0.0 or sc > 2.5:
            raise ValueError("ramp_scale must lie in (0.0, 2.5]")
        scaled_specs: dict[str, FlexibleTechSpec] = {}
        for g, sp in base.items():
            rup = None if sp.ramp_limit_up is None else min(1.0, sp.ramp_limit_up * sc)
            rdn = None if sp.ramp_limit_down is None else min(1.0, sp.ramp_limit_down * sc)
            scaled_specs[g] = FlexibleTechSpec(
                name=sp.name,
                carrier=sp.carrier,
                capital_cost_per_mw_yr=sp.capital_cost_per_mw_yr,
                marginal_cost_per_mwh=sp.marginal_cost_per_mwh,
                p_min_pu=sp.p_min_pu,
                ramp_limit_up=rup,
                ramp_limit_down=rdn,
                reserve_cap_pu=sp.reserve_cap_pu,
                profile=sp.profile,
                p_nom_extendable=sp.p_nom_extendable,
                p_nom_mw=sp.p_nom_mw,
            )
        res = run_flexibility_scenario(
            prof,
            specs=scaled_specs,
            enforce_ramping=True,
            enforce_must_run=True,
            include_bess=include_bess,
            reserve_policy=SpinningReservePolicy(),
        )
        rows.append(
            {
                "ramp_scale": sc,
                "baseload_ramp_pu_per_h": float(
                    scaled_specs["Baseload_Thermal"].ramp_limit_up or 1.0
                ),
                "ccgt_ramp_pu_per_h": float(scaled_specs["CCGT_MidMerit"].ramp_limit_up or 1.0),
                "system_lcoe_per_mwh": res.system_lcoe_per_mwh,
                "vre_total_mw": float(res.p_nom_mw["Solar_PV"] + res.p_nom_mw["Wind_Turbine"]),
                "ocgt_mw": float(res.p_nom_mw["OCGT_FastPeaker"]),
                "bess_e_nom_mwh": res.bess_e_nom_mwh,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class RampNegativeLMPProof:
    """Analytical & numerical verification of ramp-induced negative nodal LMP."""

    min_generator_marginal_cost: float
    trough_lmp_per_mwh: float
    analytical_trough_lmp_per_mwh: float
    abs_lmp_formula_error: float
    lmp_series_per_mwh: pd.Series
    dispatch_mw: pd.DataFrame


def verify_ramp_induced_negative_lmp() -> RampNegativeLMPProof:
    """Negative nodal LMP out of thermal ramp limits alone, all c_g >= 0.

    Six hours of brownfield Duck-Belly dispatch, marginal costs strictly non-negative
    (Baseload_Coal +18 /MWh, OCGT_Peaker +76 /MWh, Solar_PV 0 /MWh), coal open-boundary
    ramping at +-25%/h on 130 MW. At t = 2, the 55 MW midday trough, coal is ramp-down
    bound from t = 0, 1 and ramp-up bound into t = 3, 4 at once. Five multi-period fuel
    substitutions telescope and lambda_2 lands at exactly -138.0 /MWh.
    """
    idx = pd.date_range("2026-04-01", periods=6, freq="h")
    prof = pd.DataFrame(
        {
            "load_mw": [140.0, 175.0, 55.0, 145.0, 185.0, 140.0],
            "solar_cf": [0.0, 0.0, 0.95, 0.40, 0.0, 0.0],
            "wind_cf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    c_coal = 18.0
    c_peaker = 76.0
    c_solar = 0.0
    specs = {
        "Baseload_Coal": FlexibleTechSpec(
            name="Baseload_Coal",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=0.0,
            marginal_cost_per_mwh=c_coal,
            p_min_pu=0.35,
            ramp_limit_up=0.25,
            ramp_limit_down=0.25,
            p_nom_extendable=False,
            p_nom_mw=130.0,
        ),
        "OCGT_Peaker": FlexibleTechSpec(
            name="OCGT_Peaker",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=0.0,
            marginal_cost_per_mwh=c_peaker,
            p_min_pu=0.0,
            ramp_limit_up=1.0,
            ramp_limit_down=1.0,
            p_nom_extendable=False,
            p_nom_mw=120.0,
        ),
        "Solar_PV": FlexibleTechSpec(
            name="Solar_PV",
            carrier=RENEWABLE_CARRIER,
            capital_cost_per_mw_yr=0.0,
            marginal_cost_per_mwh=c_solar,
            profile="solar_cf",
            p_nom_extendable=False,
            p_nom_mw=150.0,
        ),
    }
    res = run_flexibility_scenario(
        prof,
        specs=specs,
        enforce_ramping=True,
        enforce_must_run=True,
        cyclic_ramping=False,
        include_bess=False,
        reserve_policy=SpinningReservePolicy(enabled=False),
        annual_hours=6.0,
    )
    # Analytical 5-hour ramp-chain marginal cost of +1 MWh load at t=2:
    # +c_coal(t=2) + (c_coal - c_peaker)(t=1) + (c_coal - c_peaker)(t=0)
    # + (c_coal - c_solar)(t=3) + (c_coal - c_peaker)(t=4) = -138.0 /MWh
    analytical_lmp_t2 = (
        c_coal
        + (c_coal - c_peaker)
        + (c_coal - c_peaker)
        + (c_coal - c_solar)
        + (c_coal - c_peaker)
    )
    trough_lmp = float(res.lmp_per_mwh.iloc[2])
    min_mc = min(sp.marginal_cost_per_mwh for sp in specs.values())

    return RampNegativeLMPProof(
        min_generator_marginal_cost=min_mc,
        trough_lmp_per_mwh=trough_lmp,
        analytical_trough_lmp_per_mwh=analytical_lmp_t2,
        abs_lmp_formula_error=abs(trough_lmp - analytical_lmp_t2),
        lmp_series_per_mwh=res.lmp_per_mwh,
        dispatch_mw=res.dispatch_mw,
    )


@dataclass(frozen=True)
class FlexibilityLinprogCrossCheck:
    """Direct sparse SciPy HiGHS verification of ramping, must-run, BESS & reserves LP."""

    pypsa_objective: float
    scipy_objective: float
    rel_objective_diff: float
    max_p_nom_diff_mw: float
    bess_e_nom_diff_mwh: float
    max_lmp_diff_per_mwh: float
    max_reserve_price_diff_per_mw_h: float


def solve_flexibility_linprog_crosscheck(
    profiles: pd.DataFrame | None = None,
    *,
    specs: dict[str, FlexibleTechSpec] | None = None,
    enforce_ramping: bool = True,
    enforce_must_run: bool = True,
    cyclic_ramping: bool = True,
    include_bess: bool = True,
    reserve_policy: SpinningReservePolicy | None = None,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> FlexibilityLinprogCrossCheck:
    """Formulate the complete ramping + must-run + BESS + spinning-reserve LP in SciPy."""
    prof = make_duck_curve_profiles() if profiles is None else validate_profiles(profiles)
    gen_specs = default_flexible_fleet_specs() if specs is None else specs
    policy = SpinningReservePolicy() if reserve_policy is None else reserve_policy

    pypsa_res = run_flexibility_scenario(
        prof,
        specs=gen_specs,
        enforce_ramping=enforce_ramping,
        enforce_must_run=enforce_must_run,
        cyclic_ramping=cyclic_ramping,
        include_bess=include_bess,
        reserve_policy=policy,
        annual_hours=annual_hours,
    )

    names = list(gen_specs)
    n_gen = len(names)
    n_t = len(prof.index)
    w = annual_hours / (float(n_t) * SNAPSHOT_HOURS)
    weight_mwh = w * SNAPSHOT_HOURS
    load = prof["load_mw"].to_numpy(dtype=float)

    res_units = [g for g in names if gen_specs[g].reserve_cap_pu > 0.0]
    vre_indices = [i for i, g in enumerate(names) if gen_specs[g].profile is not None]

    # Variable layout:
    # [0 .. n_gen*n_t): p_{g,t}
    # [offset_kg .. offset_kg + n_gen): K_g
    offset_kg = n_gen * n_t
    n_var = offset_kg + n_gen

    offset_bess = n_var
    if include_bess:
        # p_chg(n_t), p_dis(n_t), e_soc(n_t), K_chg, K_dis, E_nom
        n_var += 3 * n_t + 3

    offset_r_gen = n_var
    if policy.is_active and res_units:
        n_var += len(res_units) * n_t

    offset_r_bess = n_var
    if policy.is_active and include_bess:
        n_var += n_t

    c = np.zeros(n_var, dtype=float)
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_var

    for i, g in enumerate(names):
        sp = gen_specs[g]
        c[i * n_t : (i + 1) * n_t] = weight_mwh * sp.marginal_cost_per_mwh
        if sp.p_nom_extendable:
            c[offset_kg + i] = sp.capital_cost_per_mw_yr
        else:
            c[offset_kg + i] = 0.0
            bounds[offset_kg + i] = (sp.p_nom_mw, sp.p_nom_mw)

    if include_bess:
        c[offset_bess + 3 * n_t] = DEFAULT_BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR
        c[offset_bess + 3 * n_t + 1] = (
            DEFAULT_BESS_DISCHARGE_EFFICIENCY * DEFAULT_BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR
        )
        c[offset_bess + 3 * n_t + 2] = DEFAULT_BESS_STORE_CAPITAL_COST_PER_MWH_YR

    # Equality constraints: n_t power balance + (n_t if include_bess) cyclic SoC
    n_eq = n_t + (n_t if include_bess else 0)
    a_eq = lil_matrix((n_eq, n_var), dtype=float)
    b_eq = np.zeros(n_eq, dtype=float)
    b_eq[:n_t] = load

    for t in range(n_t):
        for i in range(n_gen):
            a_eq[t, i * n_t + t] = 1.0
        if include_bess:
            a_eq[t, offset_bess + t] = -1.0
            a_eq[t, offset_bess + n_t + t] = DEFAULT_BESS_DISCHARGE_EFFICIENCY

            eq_row = n_t + t
            t_prev = (t - 1) % n_t
            a_eq[eq_row, offset_bess + 2 * n_t + t] = 1.0
            a_eq[eq_row, offset_bess + 2 * n_t + t_prev] = -1.0
            a_eq[eq_row, offset_bess + t] = -DEFAULT_BESS_CHARGE_EFFICIENCY * SNAPSHOT_HOURS
            a_eq[eq_row, offset_bess + n_t + t] = SNAPSHOT_HOURS

    # Build inequality rows dynamically for clarity and exact row counting
    ub_rows: list[tuple[dict[int, float], float]] = []

    # 1. Generator upper availability & must-run lower bounds
    for i, g in enumerate(names):
        sp = gen_specs[g]
        cf = (
            np.ones(n_t, dtype=float)
            if sp.profile is None
            else prof[sp.profile].to_numpy(dtype=float)
        )
        for t in range(n_t):
            if cf[t] <= 0.0:
                bounds[i * n_t + t] = (0.0, 0.0)
            else:
                ub_rows.append(({i * n_t + t: 1.0, offset_kg + i: -float(cf[t])}, 0.0))
            if enforce_must_run and sp.p_min_pu > 0.0:
                ub_rows.append(({i * n_t + t: -1.0, offset_kg + i: float(sp.p_min_pu)}, 0.0))

    # 2. Generator upward & downward ramp envelopes
    if enforce_ramping and n_t > 1:
        t_start = 0 if cyclic_ramping else 1
        for i, g in enumerate(names):
            sp = gen_specs[g]
            if sp.ramp_limit_up is not None:
                for t in range(t_start, n_t):
                    t_prev = (t - 1) % n_t
                    ub_rows.append(
                        (
                            {
                                i * n_t + t: 1.0,
                                i * n_t + t_prev: -1.0,
                                offset_kg + i: -float(sp.ramp_limit_up),
                            },
                            0.0,
                        )
                    )
            if sp.ramp_limit_down is not None:
                for t in range(t_start, n_t):
                    t_prev = (t - 1) % n_t
                    ub_rows.append(
                        (
                            {
                                i * n_t + t: -1.0,
                                i * n_t + t_prev: 1.0,
                                offset_kg + i: -float(sp.ramp_limit_down),
                            },
                            0.0,
                        )
                    )

    # 3. BESS rating bounds
    if include_bess:
        for t in range(n_t):
            ub_rows.append(({offset_bess + t: 1.0, offset_bess + 3 * n_t: -1.0}, 0.0))
            ub_rows.append(({offset_bess + n_t + t: 1.0, offset_bess + 3 * n_t + 1: -1.0}, 0.0))
            ub_rows.append(({offset_bess + 2 * n_t + t: 1.0, offset_bess + 3 * n_t + 2: -1.0}, 0.0))

    # 4. Co-optimized upward spinning reserves
    reserve_sys_row_start = len(ub_rows)
    if policy.is_active:
        name_to_idx = {g: i for i, g in enumerate(names)}
        for u_pos, g in enumerate(res_units):
            i = name_to_idx[g]
            rho = float(gen_specs[g].reserve_cap_pu)
            for t in range(n_t):
                r_col = offset_r_gen + u_pos * n_t + t
                ub_rows.append(({i * n_t + t: 1.0, r_col: 1.0, offset_kg + i: -1.0}, 0.0))
                ub_rows.append(({r_col: 1.0, offset_kg + i: -rho}, 0.0))

        if include_bess:
            eta_d = DEFAULT_BESS_DISCHARGE_EFFICIENCY
            for t in range(n_t):
                rb_col = offset_r_bess + t
                ub_rows.append(
                    (
                        {
                            offset_bess + n_t + t: eta_d,
                            rb_col: 1.0,
                            offset_bess + 3 * n_t + 1: -eta_d,
                        },
                        0.0,
                    )
                )
                ub_rows.append(
                    (
                        {
                            rb_col: float(policy.bess_sustain_hours),
                            offset_bess + 2 * n_t + t: -eta_d,
                        },
                        0.0,
                    )
                )

        reserve_sys_row_start = len(ub_rows)
        for t in range(n_t):
            row_dict: dict[int, float] = {}
            for u_pos in range(len(res_units)):
                row_dict[offset_r_gen + u_pos * n_t + t] = -1.0
            if include_bess:
                row_dict[offset_r_bess + t] = -1.0
            if policy.vre_reserve_fraction > 0.0:
                for v_i in vre_indices:
                    row_dict[v_i * n_t + t] = float(policy.vre_reserve_fraction)
            ub_rows.append((row_dict, -float(policy.load_reserve_fraction * load[t])))

    a_ub = lil_matrix((len(ub_rows), n_var), dtype=float)
    b_ub = np.zeros(len(ub_rows), dtype=float)
    for r_idx, (coeffs, rhs_val) in enumerate(ub_rows):
        for col_idx, val in coeffs.items():
            a_ub[r_idx, col_idx] = val
        b_ub[r_idx] = rhs_val

    lp = linprog(
        c,
        A_ub=csc_matrix(a_ub),
        b_ub=b_ub,
        A_eq=csc_matrix(a_eq),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if lp.status != 0:
        raise SolverError(f"SciPy linprog flexibility cross-check failed: {lp.message}")

    p_nom_sp = pd.Series(lp.x[offset_kg : offset_kg + n_gen], index=names, dtype=float)
    lmp_raw = np.asarray(lp.eqlin.marginals[:n_t], dtype=float) / weight_mwh
    lmp_sp = pd.Series(
        np.where(np.abs(lmp_raw) > 1e-12, lmp_raw, 0.0),
        index=prof.index,
        dtype=float,
    )

    bess_diff = (
        abs(pypsa_res.bess_e_nom_mwh - float(lp.x[offset_bess + 3 * n_t + 2]))
        if include_bess
        else 0.0
    )

    if policy.is_active:
        res_marg = (
            -np.asarray(
                lp.ineqlin.marginals[reserve_sys_row_start : reserve_sys_row_start + n_t],
                dtype=float,
            )
            / weight_mwh
        )
        res_p_sp = pd.Series(
            np.where(np.abs(res_marg) > 1e-12, res_marg, 0.0),
            index=prof.index,
            dtype=float,
        )
        max_res_p_diff = float((pypsa_res.reserve_price_per_mw_h - res_p_sp).abs().max())
    else:
        max_res_p_diff = 0.0

    scipy_obj = float(lp.fun)
    pypsa_obj = pypsa_res.total_annual_cost
    rel_diff = abs(pypsa_obj - scipy_obj) / max(1.0, abs(scipy_obj))

    return FlexibilityLinprogCrossCheck(
        pypsa_objective=pypsa_obj,
        scipy_objective=scipy_obj,
        rel_objective_diff=rel_diff,
        max_p_nom_diff_mw=float((pypsa_res.p_nom_mw[names] - p_nom_sp).abs().max()),
        bess_e_nom_diff_mwh=bess_diff,
        max_lmp_diff_per_mwh=float((pypsa_res.lmp_per_mwh - lmp_sp).abs().max()),
        max_reserve_price_diff_per_mw_h=max_res_p_diff,
    )


def format_flexibility_report(regime_df: pd.DataFrame) -> str:
    """Format the 4-regime Duck-Curve ramping, must-run & spinning-reserve comparison table."""
    lines = [
        "Thermal Flexibility, Must-Run & Spinning-Reserve Regime Comparison (48 h Duck Curve):",
        (
            "  Regime                        LCOE (/MWh)   Max LMP   Max Res (/MW-h)   "
            "VRE (MW)   OCGT (MW)   BESS (MWh)   Duality Gap"
        ),
    ]
    for _, row in regime_df.iterrows():
        lines.append(
            f"  {str(row['regime']):<28s}  "
            f"{float(row['system_lcoe_per_mwh']):>11.4f}   "
            f"{float(row['max_lmp_per_mwh']):>7.2f}   "
            f"{float(row['max_reserve_price_per_mw_h']):>15.2f}   "
            f"{float(row['vre_total_mw']):>8.2f}   "
            f"{float(row['ocgt_mw']):>9.2f}   "
            f"{float(row['bess_e_nom_mwh']):>10.2f}   "
            f"{float(row['strong_duality_revenue_gap']):>11.2e}"
        )
    return "\n".join(lines)


def format_single_flexibility_report(result: FlexibilityScenarioResult) -> str:
    """Format single-scenario operational flexibility & spinning-reserve summary."""
    lines = [
        "Operational Flexibility & Reserve Diagnostics:",
        f"  System LCOE (currency/MWh):           {result.system_lcoe_per_mwh:>12,.4f}",
        f"  Demand-weighted LMP (currency/MWh):   {result.load_weighted_lmp_per_mwh:>12,.4f}",
        (
            "  Max reserve price (currency/MW-h):    "
            f"{float(result.reserve_price_per_mw_h.max()):>12,.4f}"
        ),
        f"  Strong duality revenue gap:           {result.strong_duality_revenue_gap:>12.2e}",
    ]
    return "\n".join(lines)
