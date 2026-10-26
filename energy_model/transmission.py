"""Meshed 4-zone DC-OPF (KVL loop flows, PTDF/B-theta, congestion rent & co-optimized TEP).

Mathematical identities verified by this module
-----------------------------------------------
1. **Equivalence of Cycle-Basis KVL (PyPSA) & Nodal Phase-Angle B-Theta / PTDF:**
   For a lossless AC transmission network with node-branch incidence matrix
   ``K in {-1, 0, +1}^{|N| x |L|}``, series per-unit reactances ``x_l > 0``, and
   branch susceptances ``b_l = 1 / x_l`` (``B_diag = diag(b)``):

       f_{l,t} = b_l * (theta_{i(l),t} - theta_{j(l),t}),   theta_{slack,t} = 0,
       B_bus * theta_t = (K * B_diag * K^T) * theta_t = g_t - d_t,
       f_t = PTDF * (g_t - d_t),    C^T * X * f_t = 0  (Kirchhoff's Voltage Law).

2. **Nodal LMP Decomposition via PTDF & Thermal Congestion Duals:**
   With shadow prices ``mu^+_{l,t}, mu^-_{l,t} >= 0`` on thermal line limits
   ``-F_l <= f_{l,t} <= F_l`` and reference-bus energy shadow price ``lambda_{ref,t}``:

       lambda_{z,t} = lambda_{ref,t} - sum_{l in L} PTDF_{l,z} * (mu^+_{l,t} - mu^-_{l,t}).

3. **Triple Merchandising Surplus (ISO Congestion Rent) Equivalence:**
   At every snapshot ``t``, ISO Merchandising Surplus equals price-spread branch
   rent and shadow-price thermal capacity rent identically:

       CR_t = sum_z lambda_{z,t} * (d_{z,t} - g_{z,t})
            = sum_{l=(i->j)} (lambda_{j,t} - lambda_{i,t}) * f_{l,t}
            = sum_{l in L} F_l * (mu^+_{l,t} + mu^-_{l,t}) >= 0,

   even when individual uncongested parallel AC corridors exhibit adverse counter-price
   KVL loop flows ``(lambda_{j,t} - lambda_{i,t}) * f_{l,t} < 0``.

4. **Generator Revenue Adequacy & Joskow-Tirole TEP Cost Recovery (Strong Duality):**
   Across all greenfield generation regimes, ``sum_{z,t} w_t * lambda_{z,t} * g_{z,t}``
   recovers 100% of annualised generation CapEx + Fuel OpEx. When transmission
   capacities ``F_l`` are co-optimized without binding lower floors, annual congestion
   rent ``sum_t w_t * CR_t`` recovers 100% of optimal transmission CapEx.

5. **Wheatstone-Bridge Braess's Paradox in Passive AC DC-OPF vs HVDC Control:**
   Adding a zero-cost, high-capacity passive AC tie-line across a symmetric bridge
   adds a KVL loop constraint that reduces maximum cheap power transfer (``120 -> 82.5 MW``)
   and strictly raises dispatch cost (``4,800 -> 7,800 currency/h``, ``+62.5%``),
   whereas a controllable HVDC link on the identical corridor eliminates congestion
   (``160 MW`` transfer, ``1,600 currency/h``, ``-66.7%``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pypsa
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, lil_matrix

from .model import RENEWABLE_CARRIER, THERMAL_CARRIER, SolverError
from .profiles import SNAPSHOT_HOURS

if hasattr(pypsa, "options") and hasattr(pypsa.options, "api"):
    pypsa.options.api.legacy_string_dtype = True

ANNUAL_HOURS_PER_YEAR = 8760.0
AC_BUS_CARRIER = "AC"
DEFAULT_NOMINAL_VOLTAGE_KV = 380.0

ZONES_4BUS: tuple[str, ...] = (
    "North_Wind",
    "West_Solar",
    "South_Hub",
    "East_Metro",
)
DEFAULT_REFERENCE_ZONE = "East_Metro"

DEFAULT_LINE_CAPACITY_SCALES: tuple[float, ...] = (
    0.50,
    0.75,
    1.00,
    1.50,
    2.50,
)

ZONAL_CF_COLUMNS: tuple[str, ...] = (
    "cf_wind_north",
    "cf_solar_west",
    "cf_wind_south",
    "cf_solar_east",
)


def _validate_non_negative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return val


def _validate_positive_float(value: float, name: str) -> float:
    val = _validate_non_negative_float(value, name)
    if val <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return val


@dataclass(frozen=True)
class ZonalTechSpec:
    """Zone-sited generator cost, carrier, and weather availability specification."""

    name: str
    zone: str
    carrier: str
    capital_cost_per_mw_yr: float
    marginal_cost_per_mwh: float
    profile_col: str | None = None
    p_nom_extendable: bool = True
    p_nom_mw: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.zone:
            raise ValueError("name and zone must be non-empty strings")
        _validate_non_negative_float(self.capital_cost_per_mw_yr, "capital_cost_per_mw_yr")
        _validate_non_negative_float(self.marginal_cost_per_mwh, "marginal_cost_per_mwh")
        _validate_non_negative_float(self.p_nom_mw, "p_nom_mw")


@dataclass(frozen=True)
class TransmissionCorridorSpec:
    """380 kV AC / HVDC transmission corridor physical & economic specification."""

    name: str
    bus0: str
    bus1: str
    length_km: float
    x_ohm: float
    s_nom_mw: float
    capital_cost_per_mw_yr: float
    s_nom_extendable: bool = False
    s_nom_min_mw: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.bus0 or not self.bus1:
            raise ValueError("corridor name, bus0, and bus1 must be non-empty")
        if self.bus0 == self.bus1:
            raise ValueError(f"self-loop corridor {self.name!r} ({self.bus0}) is invalid")
        _validate_positive_float(self.length_km, "length_km")
        _validate_positive_float(self.x_ohm, "x_ohm")
        _validate_positive_float(self.s_nom_mw, "s_nom_mw")
        _validate_non_negative_float(self.capital_cost_per_mw_yr, "capital_cost_per_mw_yr")
        _validate_non_negative_float(self.s_nom_min_mw, "s_nom_min_mw")


def validate_zonal_profiles(
    profiles: pd.DataFrame,
    zones: Sequence[str] = ZONES_4BUS,
) -> pd.DataFrame:
    """Validate hourly chronological multi-zone load and VRE capacity-factor profiles."""
    if not isinstance(profiles, pd.DataFrame) or profiles.empty:
        raise ValueError("zonal profiles must be a non-empty pandas DataFrame")
    if not isinstance(profiles.index, pd.DatetimeIndex):
        raise ValueError("zonal profiles index must be a pandas DatetimeIndex")
    if len(profiles.index) > 1:
        diffs = profiles.index.to_series().diff().dropna()
        if not (diffs == pd.Timedelta(hours=1)).all():
            raise ValueError("zonal profiles must have strictly hourly timestamps")

    load_cols = [f"load_{z}" for z in zones]
    missing = [c for c in load_cols if c not in profiles.columns]
    if missing:
        raise ValueError(f"missing zonal load columns: {missing}")

    clean = profiles.copy()
    for col in clean.columns:
        arr = clean[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"column {col!r} contains non-finite values")
        if col.startswith("load_") and np.any(arr < 0.0):
            raise ValueError(f"load column {col!r} cannot contain negative demand")
        if col.startswith("cf_") and (np.any(arr < 0.0) or np.any(arr > 1.0)):
            raise ValueError(f"capacity-factor column {col!r} must lie within [0, 1]")
    return clean


def make_four_zone_profiles(days: int = 2) -> pd.DataFrame:
    """Return deterministic chronological 4-zone load and VRE resource profiles."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 1 or days > 14:
        raise ValueError("days must be an integer between 1 and 14")
    n_t = 24 * days
    idx = pd.date_range("2026-07-01", periods=n_t, freq="h")
    h = np.arange(n_t, dtype=float)
    hod = h % 24.0
    day = np.floor(h / 24.0).astype(int)

    load_east = (
        210.0
        + 75.0 * np.exp(-0.5 * ((hod - 19.0) / 2.4) ** 2)
        + 30.0 * np.exp(-0.5 * ((hod - 9.0) / 2.0) ** 2)
        - 35.0 * np.exp(-0.5 * ((hod - 3.5) / 2.2) ** 2)
    ) * np.where(day % 2 == 0, 0.97, 1.04)

    load_south = (
        115.0
        + 32.0 * np.exp(-0.5 * ((hod - 18.5) / 2.5) ** 2)
        - 18.0 * np.exp(-0.5 * ((hod - 4.0) / 2.2) ** 2)
    ) * np.where(day % 2 == 0, 0.98, 1.03)

    load_west = (
        75.0
        + 25.0 * np.exp(-0.5 * ((hod - 15.0) / 2.6) ** 2)
        - 12.0 * np.exp(-0.5 * ((hod - 3.5) / 2.2) ** 2)
    )
    load_north = 55.0 + 10.0 * np.exp(-0.5 * ((hod - 12.0) / 3.5) ** 2)

    cf_wind_north = np.clip(
        (
            0.56
            + 0.24 * np.cos(2.0 * np.pi * (hod - 3.0) / 24.0)
            + 0.08 * np.sin(2.0 * np.pi * h / 48.0)
        )
        * np.where(day % 2 == 0, 1.04, 0.88),
        0.12,
        0.92,
    )
    raw_solar_w = np.sin(np.pi * (hod - 6.0) / 13.0)
    cf_solar_west = np.where(
        (hod >= 6.0) & (hod <= 19.0),
        np.clip(
            np.where(day % 2 == 0, 0.95, 0.78) * np.maximum(0.0, raw_solar_w) ** 1.12,
            0.0,
            1.0,
        ),
        0.0,
    )
    cf_wind_south = np.clip(
        0.26 + 0.10 * np.cos(2.0 * np.pi * (hod - 5.0) / 24.0),
        0.08,
        0.45,
    )
    raw_solar_e = np.sin(np.pi * (hod - 6.5) / 12.0)
    cf_solar_east = np.where(
        (hod >= 6.5) & (hod <= 18.5),
        np.clip(0.58 * np.maximum(0.0, raw_solar_e) ** 1.20, 0.0, 1.0),
        0.0,
    )

    df = pd.DataFrame(
        {
            "load_North_Wind": load_north,
            "load_West_Solar": load_west,
            "load_South_Hub": load_south,
            "load_East_Metro": load_east,
            "cf_wind_north": cf_wind_north,
            "cf_solar_west": cf_solar_west,
            "cf_wind_south": cf_wind_south,
            "cf_solar_east": cf_solar_east,
        },
        index=idx,
    )
    return validate_zonal_profiles(df, zones=ZONES_4BUS)


def default_zonal_generator_specs() -> dict[str, ZonalTechSpec]:
    """Return the 8-plant spatially differentiated 4-zone generation fleet."""
    items = (
        ZonalTechSpec(
            name="North_Offshore_Wind",
            zone="North_Wind",
            carrier=RENEWABLE_CARRIER,
            capital_cost_per_mw_yr=56_000.0,
            marginal_cost_per_mwh=0.0,
            profile_col="cf_wind_north",
        ),
        ZonalTechSpec(
            name="North_Backup_Gas",
            zone="North_Wind",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=22_000.0,
            marginal_cost_per_mwh=72.0,
        ),
        ZonalTechSpec(
            name="West_Utility_Solar",
            zone="West_Solar",
            carrier=RENEWABLE_CARRIER,
            capital_cost_per_mw_yr=33_000.0,
            marginal_cost_per_mwh=0.0,
            profile_col="cf_solar_west",
        ),
        ZonalTechSpec(
            name="West_Peaker_Gas",
            zone="West_Solar",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=22_000.0,
            marginal_cost_per_mwh=68.0,
        ),
        ZonalTechSpec(
            name="South_CCGT",
            zone="South_Hub",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=38_000.0,
            marginal_cost_per_mwh=40.0,
        ),
        ZonalTechSpec(
            name="South_Onshore_Wind",
            zone="South_Hub",
            carrier=RENEWABLE_CARRIER,
            capital_cost_per_mw_yr=64_000.0,
            marginal_cost_per_mwh=0.0,
            profile_col="cf_wind_south",
        ),
        ZonalTechSpec(
            name="East_Rooftop_Solar",
            zone="East_Metro",
            carrier=RENEWABLE_CARRIER,
            capital_cost_per_mw_yr=46_000.0,
            marginal_cost_per_mwh=0.0,
            profile_col="cf_solar_east",
        ),
        ZonalTechSpec(
            name="East_Metro_OCGT",
            zone="East_Metro",
            carrier=THERMAL_CARRIER,
            capital_cost_per_mw_yr=21_000.0,
            marginal_cost_per_mwh=84.0,
        ),
    )
    return {sp.name: sp for sp in items}


def default_transmission_corridors(
    *,
    extendable: bool = False,
    capacity_scale: float = 1.0,
    use_existing_floor_when_extendable: bool = True,
) -> dict[str, TransmissionCorridorSpec]:
    """Return the 5-corridor 380 kV meshed network (2 fundamental KVL loops)."""
    scale = _validate_positive_float(capacity_scale, "capacity_scale")
    raw = (
        ("L1_North_West", "North_Wind", "West_Solar", 110.0, 26.0, 85.0, 8_500.0),
        ("L2_North_East", "North_Wind", "East_Metro", 150.0, 36.0, 75.0, 11_500.0),
        ("L3_West_South", "West_Solar", "South_Hub", 100.0, 24.0, 80.0, 8_000.0),
        ("L4_South_East", "South_Hub", "East_Metro", 85.0, 20.0, 90.0, 9_500.0),
        ("L5_North_South", "North_Wind", "South_Hub", 130.0, 31.0, 70.0, 10_000.0),
    )
    out: dict[str, TransmissionCorridorSpec] = {}
    for name, b0, b1, km, x_ohm, snom, ccost in raw:
        scaled_snom = snom * scale
        s_min = scaled_snom if (extendable and use_existing_floor_when_extendable) else 0.0
        out[name] = TransmissionCorridorSpec(
            name=name,
            bus0=b0,
            bus1=b1,
            length_km=km,
            x_ohm=x_ohm,
            s_nom_mw=scaled_snom,
            capital_cost_per_mw_yr=ccost,
            s_nom_extendable=extendable,
            s_nom_min_mw=s_min,
        )
    return out


def compute_ptdf_matrix(
    corridors: dict[str, TransmissionCorridorSpec] | None = None,
    *,
    zones: Sequence[str] = ZONES_4BUS,
    slack_zone: str = DEFAULT_REFERENCE_ZONE,
    v_nom_kv: float = DEFAULT_NOMINAL_VOLTAGE_KV,
) -> pd.DataFrame:
    """Compute exact DC-OPF Power Transfer Distribution Factor (PTDF) matrix."""
    corr = default_transmission_corridors() if corridors is None else corridors
    if not corr:
        raise ValueError("corridors must not be empty")
    if slack_zone not in zones:
        raise ValueError(f"slack_zone {slack_zone!r} not in zones {list(zones)}")
    v_kv = _validate_positive_float(v_nom_kv, "v_nom_kv")

    z_list = list(zones)
    l_list = list(corr)
    n_z = len(z_list)
    n_l = len(l_list)
    z_idx = {z: i for i, z in enumerate(z_list)}

    k_inc = np.zeros((n_z, n_l), dtype=float)
    b_vec = np.zeros(n_l, dtype=float)
    for l_i, lname in enumerate(l_list):
        sp = corr[lname]
        if sp.bus0 not in z_idx or sp.bus1 not in z_idx:
            raise ValueError(f"corridor {lname!r} references unknown zone")
        k_inc[z_idx[sp.bus0], l_i] = 1.0
        k_inc[z_idx[sp.bus1], l_i] = -1.0
        x_pu = sp.x_ohm / (v_kv**2)
        b_vec[l_i] = 1.0 / x_pu

    b_diag = np.diag(b_vec)
    b_bus = k_inc @ b_diag @ k_inc.T
    slack_i = z_idx[slack_zone]
    non_slack = [i for i in range(n_z) if i != slack_i]

    b_red = b_bus[np.ix_(non_slack, non_slack)]
    try:
        b_red_inv = np.linalg.inv(b_red)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Transmission graph is disconnected; B_bus is singular") from exc

    ptdf = np.zeros((n_l, n_z), dtype=float)
    ptdf[:, non_slack] = b_diag @ k_inc[non_slack, :].T @ b_red_inv
    return pd.DataFrame(ptdf, index=l_list, columns=z_list)


@dataclass
class TransmissionScenarioResult:
    """Primal, dual, PTDF, KVL loop-flow & merchandising-surplus summary."""

    mode: str
    extendable_lines: bool
    annual_demand_mwh: float
    gen_p_nom_mw: pd.Series
    line_s_nom_init_mw: pd.Series
    line_s_nom_opt_mw: pd.Series
    line_expansion_mw: pd.Series
    gen_dispatch_mw: pd.DataFrame
    zonal_generation_mw: pd.DataFrame
    zonal_load_mw: pd.DataFrame
    line_flows_mw: pd.DataFrame
    voltage_angles_rad: pd.DataFrame
    zonal_lmp_per_mwh: pd.DataFrame
    line_mu_upper_per_mw_h: pd.DataFrame
    line_mu_lower_per_mw_h: pd.DataFrame
    hourly_congestion_rent: pd.Series
    corridor_merchandising_rent_per_h: pd.DataFrame
    annual_gen_capex_cost: float
    annual_line_capex_cost: float
    annual_incremental_line_capex_cost: float
    annual_fuel_opex_cost: float
    total_annual_cost: float
    system_lcoe_per_mwh: float
    annual_load_payment: float
    annual_generator_revenue: float
    annual_congestion_rent: float
    generator_revenue_adequacy_gap: float
    merchandising_surplus_identity_max_gap: float
    ptdf_flow_max_error_mw: float
    ptdf_lmp_decomposition_max_error: float
    kvl_cycle_max_residual_pu: float
    joskow_tirole_rent_recovery_gap: float
    max_spatial_lmp_spread_per_mwh: float
    mean_spatial_lmp_spread_per_mwh: float
    congested_corridor_hours: pd.Series
    vre_curtailment_pct: float
    nodal_balance_max_residual_mw: float


def build_zonal_transmission_network(
    profiles: pd.DataFrame,
    *,
    gen_specs: dict[str, ZonalTechSpec] | None = None,
    corridors: dict[str, TransmissionCorridorSpec] | None = None,
    zones: Sequence[str] = ZONES_4BUS,
    mode: str = "ac_kvl",
    v_nom_kv: float = DEFAULT_NOMINAL_VOLTAGE_KV,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> pypsa.Network:
    """Build a 4-zone AC-KVL, HVDC-transport, or single-bus copper-plate PyPSA network."""
    if mode not in ("copper_plate", "ac_kvl", "hvdc_transport"):
        raise ValueError(
            f"unknown mode {mode!r}; expected 'copper_plate', 'ac_kvl', or 'hvdc_transport'"
        )
    z_list = list(zones)
    prof = validate_zonal_profiles(profiles, zones=z_list)
    gens = default_zonal_generator_specs() if gen_specs is None else gen_specs
    corr = default_transmission_corridors() if corridors is None else corridors
    v_kv = _validate_positive_float(v_nom_kv, "v_nom_kv")
    ann_h = _validate_positive_float(annual_hours, "annual_hours")

    n_t = len(prof.index)
    w = ann_h / (float(n_t) * SNAPSHOT_HOURS)

    network = pypsa.Network()
    network.set_snapshots(prof.index)
    network.snapshot_weightings = network.snapshot_weightings.astype(float)
    network.snapshot_weightings.loc[:, "objective"] = w
    network.snapshot_weightings.loc[:, "generators"] = w

    network.add("Carrier", AC_BUS_CARRIER)
    added_carriers = {AC_BUS_CARRIER}
    for gsp in gens.values():
        if gsp.carrier not in added_carriers:
            network.add("Carrier", gsp.carrier)
            added_carriers.add(gsp.carrier)

    if mode == "copper_plate":
        network.add("Bus", "Copper_Plate_Hub", carrier=AC_BUS_CARRIER, v_nom=v_kv)
        total_load = sum(prof[f"load_{z}"] for z in z_list)
        network.add("Load", "Total_Load", bus="Copper_Plate_Hub", p_set=total_load)
        for gname, gsp in gens.items():
            kwargs: dict[str, object] = {
                "bus": "Copper_Plate_Hub",
                "carrier": gsp.carrier,
                "capital_cost": float(gsp.capital_cost_per_mw_yr),
                "marginal_cost": float(gsp.marginal_cost_per_mwh),
                "p_nom_extendable": bool(gsp.p_nom_extendable),
            }
            if not gsp.p_nom_extendable:
                kwargs["p_nom"] = float(gsp.p_nom_mw)
            network.add("Generator", gname, **kwargs)
            if gsp.profile_col is not None:
                network.generators_t.p_max_pu[gname] = prof[gsp.profile_col]
        return network

    for z in z_list:
        network.add("Bus", z, carrier=AC_BUS_CARRIER, v_nom=v_kv)
        network.add("Load", f"Load_{z}", bus=z, p_set=prof[f"load_{z}"])

    for gname, gsp in gens.items():
        if gsp.zone not in z_list:
            raise ValueError(f"generator {gname!r} zone {gsp.zone!r} not in {z_list}")
        kwargs = {
            "bus": gsp.zone,
            "carrier": gsp.carrier,
            "capital_cost": float(gsp.capital_cost_per_mw_yr),
            "marginal_cost": float(gsp.marginal_cost_per_mwh),
            "p_nom_extendable": bool(gsp.p_nom_extendable),
        }
        if not gsp.p_nom_extendable:
            kwargs["p_nom"] = float(gsp.p_nom_mw)
        network.add("Generator", gname, **kwargs)
        if gsp.profile_col is not None:
            network.generators_t.p_max_pu[gname] = prof[gsp.profile_col]

    if mode == "ac_kvl":
        for lname, lsp in corr.items():
            network.add(
                "Line",
                lname,
                bus0=lsp.bus0,
                bus1=lsp.bus1,
                x=float(lsp.x_ohm),
                r=1e-4 * float(lsp.x_ohm),
                s_nom=float(lsp.s_nom_mw),
                s_nom_min=float(lsp.s_nom_min_mw) if lsp.s_nom_extendable else 0.0,
                capital_cost=(float(lsp.capital_cost_per_mw_yr) if lsp.s_nom_extendable else 0.0),
                s_nom_extendable=bool(lsp.s_nom_extendable),
            )
    else:
        for lname, lsp in corr.items():
            network.add(
                "Link",
                lname,
                bus0=lsp.bus0,
                bus1=lsp.bus1,
                efficiency=1.0,
                p_nom=float(lsp.s_nom_mw),
                p_nom_min=float(lsp.s_nom_min_mw) if lsp.s_nom_extendable else 0.0,
                p_min_pu=-1.0,
                p_max_pu=1.0,
                p_nom_extendable=bool(lsp.s_nom_extendable),
                capital_cost=(float(lsp.capital_cost_per_mw_yr) if lsp.s_nom_extendable else 0.0),
                marginal_cost=0.0,
            )

    return network


def run_transmission_scenario(
    profiles: pd.DataFrame | None = None,
    *,
    gen_specs: dict[str, ZonalTechSpec] | None = None,
    corridors: dict[str, TransmissionCorridorSpec] | None = None,
    zones: Sequence[str] = ZONES_4BUS,
    slack_zone: str = DEFAULT_REFERENCE_ZONE,
    mode: str = "ac_kvl",
    extendable_lines: bool = False,
    capacity_scale: float = 1.0,
    v_nom_kv: float = DEFAULT_NOMINAL_VOLTAGE_KV,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
    solver_name: str = "highs",
) -> TransmissionScenarioResult:
    """Solve 4-zone capacity expansion & DC-OPF / TEP and verify KKT & PTDF theorems."""
    z_list = list(zones)
    prof = (
        make_four_zone_profiles()
        if profiles is None
        else validate_zonal_profiles(profiles, zones=z_list)
    )
    gens = default_zonal_generator_specs() if gen_specs is None else gen_specs
    if corridors is None:
        corr = default_transmission_corridors(
            extendable=extendable_lines,
            capacity_scale=capacity_scale,
        )
    else:
        corr = corridors
        extendable_lines = any(c.s_nom_extendable for c in corr.values())

    ann_h = _validate_positive_float(annual_hours, "annual_hours")
    network = build_zonal_transmission_network(
        prof,
        gen_specs=gens,
        corridors=corr,
        zones=z_list,
        mode=mode,
        v_nom_kv=v_nom_kv,
        annual_hours=ann_h,
    )

    try:
        res = network.optimize(
            solver_name=solver_name,
            solver_options={"log_to_console": False},
            include_objective_constant=False,
        )
    except Exception as exc:
        raise SolverError(f"Transmission optimization failed: {exc}") from exc

    status, condition = res if isinstance(res, tuple) else (str(res), "")
    if status != "ok" or condition not in ("optimal", ""):
        raise SolverError(
            f"Solver did not find optimal solution: status={status!r}, condition={condition!r}"
        )

    n_t = len(prof.index)
    w = ann_h / (float(n_t) * SNAPSHOT_HOURS)
    weight_mwh = w * SNAPSHOT_HOURS

    g_names = list(gens)
    l_names = list(corr)
    zonal_load = pd.DataFrame(
        {z: prof[f"load_{z}"].astype(float) for z in z_list},
        index=prof.index,
    )
    annual_demand = float(zonal_load.sum().sum() * weight_mwh)

    p_gen = network.generators_t.p[g_names].astype(float).copy()
    p_nom = pd.Series(
        {
            g: (
                float(network.generators.at[g, "p_nom_opt"])
                if gens[g].p_nom_extendable
                else float(gens[g].p_nom_mw)
            )
            for g in g_names
        },
        dtype=float,
    ).clip(lower=0.0)

    zonal_gen = pd.DataFrame(0.0, index=prof.index, columns=z_list)
    for g in g_names:
        zonal_gen[gens[g].zone] += p_gen[g]

    s_init = pd.Series({ln: float(corr[ln].s_nom_mw) for ln in l_names}, dtype=float)

    if mode == "copper_plate":
        s_opt = pd.Series(0.0, index=l_names, dtype=float)
        s_exp = pd.Series(0.0, index=l_names, dtype=float)
        flows = pd.DataFrame(0.0, index=prof.index, columns=l_names)
        angles = pd.DataFrame(0.0, index=prof.index, columns=z_list)
        hub_lmp = network.buses_t.marginal_price["Copper_Plate_Hub"].astype(float)
        hub_lmp = hub_lmp.where(hub_lmp.abs() > 1e-12, 0.0)
        zonal_lmp = pd.DataFrame({z: hub_lmp for z in z_list}, index=prof.index)
        mu_up = pd.DataFrame(0.0, index=prof.index, columns=l_names)
        mu_lo = pd.DataFrame(0.0, index=prof.index, columns=l_names)
        ptdf_flow_err = 0.0
        ptdf_lmp_err = 0.0
        kvl_res_max = 0.0
        nodal_res = float((zonal_gen.sum(axis=1) - zonal_load.sum(axis=1)).abs().max())
    elif mode == "ac_kvl":
        s_opt = pd.Series(
            {
                ln: (
                    float(network.lines.at[ln, "s_nom_opt"])
                    if corr[ln].s_nom_extendable
                    else float(corr[ln].s_nom_mw)
                )
                for ln in l_names
            },
            dtype=float,
        )
        s_exp = (s_opt - s_init).clip(lower=0.0)
        flows = network.lines_t.p0[l_names].astype(float).copy()
        raw_lmp = network.buses_t.marginal_price[z_list].astype(float).copy()
        zonal_lmp = raw_lmp.where(raw_lmp.abs() > 1e-12, 0.0)

        mu_up = pd.DataFrame(0.0, index=prof.index, columns=l_names)
        mu_lo = pd.DataFrame(0.0, index=prof.index, columns=l_names)
        for cname_up in ("Line-fix-s-upper", "Line-ext-s-upper"):
            if cname_up in network.model.constraints:
                du = network.model.constraints[cname_up].dual.to_pandas()
                cols = [c for c in l_names if c in du.columns]
                if cols:
                    mu_up[cols] = (-du[cols].reindex(prof.index) / weight_mwh).clip(lower=0.0)
        for cname_lo in ("Line-fix-s-lower", "Line-ext-s-lower"):
            if cname_lo in network.model.constraints:
                du = network.model.constraints[cname_lo].dual.to_pandas()
                cols = [c for c in l_names if c in du.columns]
                if cols:
                    mu_lo[cols] = (du[cols].reindex(prof.index) / weight_mwh).clip(lower=0.0)

        ptdf = compute_ptdf_matrix(corr, zones=z_list, slack_zone=slack_zone, v_nom_kv=v_nom_kv)
        net_inj = (zonal_gen - zonal_load)[z_list].to_numpy(dtype=float)
        f_ptdf = net_inj @ ptdf.to_numpy(dtype=float).T
        ptdf_flow_err = float(np.max(np.abs(flows.to_numpy(dtype=float) - f_ptdf)))

        lam_ref = zonal_lmp[slack_zone].to_numpy(dtype=float)[:, None]
        lmp_ptdf = lam_ref - (
            mu_up.to_numpy(dtype=float) - mu_lo.to_numpy(dtype=float)
        ) @ ptdf.to_numpy(dtype=float)
        ptdf_lmp_err = float(np.max(np.abs(zonal_lmp.to_numpy(dtype=float) - lmp_ptdf)))

        # Recover nodal voltage phase angles (rad) with theta_slack = 0
        z_idx = {z: i for i, z in enumerate(z_list)}
        k_inc = np.zeros((len(z_list), len(l_names)), dtype=float)
        b_vec = np.zeros(len(l_names), dtype=float)
        for l_i, ln in enumerate(l_names):
            k_inc[z_idx[corr[ln].bus0], l_i] = 1.0
            k_inc[z_idx[corr[ln].bus1], l_i] = -1.0
            b_vec[l_i] = (v_nom_kv**2) / corr[ln].x_ohm
        b_bus = k_inc @ np.diag(b_vec) @ k_inc.T
        slack_i = z_idx[slack_zone]
        non_slack = [i for i in range(len(z_list)) if i != slack_i]
        theta_arr = np.zeros((n_t, len(z_list)), dtype=float)
        theta_arr[:, non_slack] = (
            net_inj[:, non_slack] @ np.linalg.inv(b_bus[np.ix_(non_slack, non_slack)]).T
        )
        angles = pd.DataFrame(theta_arr, index=prof.index, columns=z_list)

        # Verify KVL around every fundamental cycle (f_l * x_pu_l == theta_i - theta_j)
        dtheta = theta_arr @ k_inc
        x_pu_vec = 1.0 / b_vec
        kvl_res_max = float(
            np.max(np.abs(flows.to_numpy(dtype=float) * x_pu_vec[None, :] - dtheta))
        )
        nodal_res = float(np.max(np.abs(net_inj - flows.to_numpy(dtype=float) @ k_inc.T)))
    else:
        s_opt = pd.Series(
            {
                ln: (
                    float(network.links.at[ln, "p_nom_opt"])
                    if corr[ln].s_nom_extendable
                    else float(corr[ln].s_nom_mw)
                )
                for ln in l_names
            },
            dtype=float,
        )
        s_exp = (s_opt - s_init).clip(lower=0.0)
        flows = network.links_t.p0[l_names].astype(float).copy()
        angles = pd.DataFrame(0.0, index=prof.index, columns=z_list)
        raw_lmp = network.buses_t.marginal_price[z_list].astype(float).copy()
        zonal_lmp = raw_lmp.where(raw_lmp.abs() > 1e-12, 0.0)

        mu_up = pd.DataFrame(0.0, index=prof.index, columns=l_names)
        mu_lo = pd.DataFrame(0.0, index=prof.index, columns=l_names)
        for cname_up in ("Link-fix-p-upper", "Link-ext-p-upper"):
            if cname_up in network.model.constraints:
                du = network.model.constraints[cname_up].dual.to_pandas()
                cols = [c for c in l_names if c in du.columns]
                if cols:
                    mu_up[cols] = (-du[cols].reindex(prof.index) / weight_mwh).clip(lower=0.0)
        for cname_lo in ("Link-fix-p-lower", "Link-ext-p-lower"):
            if cname_lo in network.model.constraints:
                du = network.model.constraints[cname_lo].dual.to_pandas()
                cols = [c for c in l_names if c in du.columns]
                if cols:
                    mu_lo[cols] = (du[cols].reindex(prof.index) / weight_mwh).clip(lower=0.0)
        ptdf_flow_err = 0.0
        ptdf_lmp_err = 0.0
        kvl_res_max = 0.0
        z_idx = {z: i for i, z in enumerate(z_list)}
        k_inc = np.zeros((len(z_list), len(l_names)), dtype=float)
        for l_i, ln in enumerate(l_names):
            k_inc[z_idx[corr[ln].bus0], l_i] = 1.0
            k_inc[z_idx[corr[ln].bus1], l_i] = -1.0
        net_inj = (zonal_gen - zonal_load)[z_list].to_numpy(dtype=float)
        nodal_res = float(np.max(np.abs(net_inj - flows.to_numpy(dtype=float) @ k_inc.T)))

    # Merchandising surplus (Congestion Rent) across all 3 mathematical forms
    cr_nodal = ((zonal_load - zonal_gen) * zonal_lmp).sum(axis=1)
    corridor_rent = pd.DataFrame(0.0, index=prof.index, columns=l_names)
    if mode != "copper_plate":
        for ln in l_names:
            b0, b1 = corr[ln].bus0, corr[ln].bus1
            corridor_rent[ln] = (zonal_lmp[b1] - zonal_lmp[b0]) * flows[ln]
    cr_spread = corridor_rent.sum(axis=1)
    cr_shadow = ((mu_up + mu_lo) * s_opt).sum(axis=1)
    ms_gap = float(
        max(
            (cr_nodal - cr_spread).abs().max(),
            (cr_nodal - cr_shadow).abs().max(),
        )
    )

    # Costs, Revenues & Strong Duality Checks
    gen_capex = float(
        sum(
            gens[g].capital_cost_per_mw_yr * float(p_nom[g])
            for g in g_names
            if gens[g].p_nom_extendable
        )
    )
    fuel_opex = float(
        sum(gens[g].marginal_cost_per_mwh * float(p_gen[g].sum() * weight_mwh) for g in g_names)
    )
    if mode != "copper_plate":
        line_capex_tot = float(
            sum(
                corr[ln].capital_cost_per_mw_yr * float(s_opt[ln])
                for ln in l_names
                if corr[ln].s_nom_extendable
            )
        )
        line_capex_inc = float(
            sum(
                corr[ln].capital_cost_per_mw_yr * float(s_exp[ln])
                for ln in l_names
                if corr[ln].s_nom_extendable
            )
        )
    else:
        line_capex_tot = 0.0
        line_capex_inc = 0.0

    total_cost = gen_capex + line_capex_tot + fuel_opex
    load_payment = float(((zonal_load * zonal_lmp).sum(axis=1) * weight_mwh).sum())
    gen_revenue = float(((zonal_gen * zonal_lmp).sum(axis=1) * weight_mwh).sum())
    annual_cr = float((cr_nodal * weight_mwh).sum())
    if abs(annual_cr) < 1e-9:
        annual_cr = 0.0

    all_gen_ext = all(gens[g].p_nom_extendable for g in g_names)
    gen_rev_gap = abs((gen_capex + fuel_opex) - gen_revenue) if all_gen_ext else 0.0

    interior_tep = (
        mode != "copper_plate"
        and all(corr[ln].s_nom_extendable for ln in l_names)
        and all(float(s_opt[ln]) > float(corr[ln].s_nom_min_mw) + 1e-5 for ln in l_names)
    )
    jt_gap = abs(annual_cr - line_capex_tot) if interior_tep else 0.0

    # Spatial LMP spreads & congested corridor hours
    lmp_spread = zonal_lmp.max(axis=1) - zonal_lmp.min(axis=1)
    if mode == "copper_plate":
        cong_hrs = pd.Series(0, index=l_names, dtype=int)
    else:
        cong_hrs = pd.Series(
            {ln: int((flows[ln].abs() >= float(s_opt[ln]) - 1e-4).sum()) for ln in l_names},
            dtype=int,
        )

    # VRE curtailment
    vre_names = [g for g in g_names if gens[g].profile_col is not None]
    if vre_names:
        p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")[vre_names]
        avail_vre = (p_max_pu * p_nom[vre_names]).sum().sum() * weight_mwh
        disp_vre = p_gen[vre_names].sum().sum() * weight_mwh
        curt_pct = (
            100.0 * max(0.0, float(avail_vre - disp_vre)) / float(avail_vre)
            if avail_vre > 1e-9
            else 0.0
        )
    else:
        curt_pct = 0.0

    return TransmissionScenarioResult(
        mode=mode,
        extendable_lines=extendable_lines,
        annual_demand_mwh=annual_demand,
        gen_p_nom_mw=p_nom,
        line_s_nom_init_mw=s_init,
        line_s_nom_opt_mw=s_opt,
        line_expansion_mw=s_exp,
        gen_dispatch_mw=p_gen,
        zonal_generation_mw=zonal_gen,
        zonal_load_mw=zonal_load,
        line_flows_mw=flows,
        voltage_angles_rad=angles,
        zonal_lmp_per_mwh=zonal_lmp,
        line_mu_upper_per_mw_h=mu_up,
        line_mu_lower_per_mw_h=mu_lo,
        hourly_congestion_rent=cr_nodal,
        corridor_merchandising_rent_per_h=corridor_rent,
        annual_gen_capex_cost=gen_capex,
        annual_line_capex_cost=line_capex_tot,
        annual_incremental_line_capex_cost=line_capex_inc,
        annual_fuel_opex_cost=fuel_opex,
        total_annual_cost=total_cost,
        system_lcoe_per_mwh=total_cost / annual_demand,
        annual_load_payment=load_payment,
        annual_generator_revenue=gen_revenue,
        annual_congestion_rent=annual_cr,
        generator_revenue_adequacy_gap=gen_rev_gap,
        merchandising_surplus_identity_max_gap=ms_gap,
        ptdf_flow_max_error_mw=ptdf_flow_err,
        ptdf_lmp_decomposition_max_error=ptdf_lmp_err,
        kvl_cycle_max_residual_pu=kvl_res_max,
        joskow_tirole_rent_recovery_gap=jt_gap,
        max_spatial_lmp_spread_per_mwh=float(lmp_spread.max()),
        mean_spatial_lmp_spread_per_mwh=float(lmp_spread.mean()),
        congested_corridor_hours=cong_hrs,
        vre_curtailment_pct=curt_pct,
        nodal_balance_max_residual_mw=nodal_res,
    )


def compare_transmission_regimes(
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare Copper-Plate, Congested AC-KVL, Controllable HVDC, and Co-Optimized AC-TEP."""
    prof = make_four_zone_profiles() if profiles is None else validate_zonal_profiles(profiles)
    configs = (
        ("1_CopperPlate_SingleBus", "copper_plate", False),
        ("2_Congested_AC_KVL_FixedGrid", "ac_kvl", False),
        ("3_Controllable_HVDC_Transport", "hvdc_transport", False),
        ("4_CoOptimized_AC_TEP", "ac_kvl", True),
    )

    rows: list[dict[str, object]] = []
    for label, mode, ext in configs:
        res = run_transmission_scenario(prof, mode=mode, extendable_lines=ext)
        rows.append(
            {
                "regime": label,
                "system_lcoe_per_mwh": res.system_lcoe_per_mwh,
                "mean_lmp_spread_per_mwh": res.mean_spatial_lmp_spread_per_mwh,
                "max_lmp_spread_per_mwh": res.max_spatial_lmp_spread_per_mwh,
                "north_wind_mw": float(res.gen_p_nom_mw["North_Offshore_Wind"]),
                "west_solar_mw": float(res.gen_p_nom_mw["West_Utility_Solar"]),
                "east_ocgt_mw": float(res.gen_p_nom_mw["East_Metro_OCGT"]),
                "total_line_mw": float(res.line_s_nom_opt_mw.sum()),
                "added_line_mw": float(res.line_expansion_mw.sum()),
                "annual_congestion_rent": res.annual_congestion_rent,
                "annual_line_capex_cost": res.annual_line_capex_cost,
                "vre_curtailment_pct": res.vre_curtailment_pct,
                "gen_revenue_adequacy_gap": res.generator_revenue_adequacy_gap,
                "merchandising_surplus_gap": res.merchandising_surplus_identity_max_gap,
            }
        )
    return pd.DataFrame(rows)


def sweep_transmission_capacity_scale(
    profiles: pd.DataFrame | None = None,
    *,
    scales: Sequence[float] = DEFAULT_LINE_CAPACITY_SCALES,
) -> pd.DataFrame:
    """Scale fixed AC thermal corridor capacities and verify monotonic LCOE convergence."""
    if not scales:
        raise ValueError("scales must contain at least one positive scale factor")
    prof = make_four_zone_profiles() if profiles is None else validate_zonal_profiles(profiles)
    rows: list[dict[str, float]] = []
    for sc in sorted(scales):
        val = _validate_positive_float(sc, "scale")
        res = run_transmission_scenario(
            prof, mode="ac_kvl", extendable_lines=False, capacity_scale=val
        )
        rows.append(
            {
                "capacity_scale": val,
                "total_corridor_mw": float(res.line_s_nom_opt_mw.sum()),
                "system_lcoe_per_mwh": res.system_lcoe_per_mwh,
                "mean_lmp_spread_per_mwh": res.mean_spatial_lmp_spread_per_mwh,
                "annual_congestion_rent": res.annual_congestion_rent,
                "north_wind_mw": float(res.gen_p_nom_mw["North_Offshore_Wind"]),
                "east_ocgt_mw": float(res.gen_p_nom_mw["East_Metro_OCGT"]),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BraessParadoxProof:
    """Analytical & numerical proof of Wheatstone-Bridge Braess's Paradox in AC DC-OPF."""

    base_4line_cost_per_h: float
    ac_5line_cost_per_h: float
    hvdc_5line_cost_per_h: float
    base_transfer_mw: float
    ac_5line_transfer_mw: float
    hvdc_5line_transfer_mw: float
    analytical_ac_5line_transfer_mw: float
    analytical_ac_5line_cost_per_h: float
    braess_cost_increase_pct: float


def verify_braess_paradox_dcopf() -> BraessParadoxProof:
    """Prove adding a zero-cost 500 MW AC tie-line strictly raises dispatch cost (+62.5%)."""
    idx = pd.date_range("2026-07-01", periods=1, freq="h")
    zones = ("B1_North", "B2_West", "B3_East", "B4_South")

    def _solve_bridge(bridge_mode: str) -> tuple[float, float]:
        n = pypsa.Network()
        n.set_snapshots(idx)
        for c in (AC_BUS_CARRIER, RENEWABLE_CARRIER, THERMAL_CARRIER):
            n.add("Carrier", c)
        for b in zones:
            n.add("Bus", b, carrier=AC_BUS_CARRIER, v_nom=DEFAULT_NOMINAL_VOLTAGE_KV)
        n.add("Load", "Metro_Load", bus="B3_East", p_set=[160.0])
        n.add(
            "Generator",
            "Cheap_Wind",
            bus="B1_North",
            carrier=RENEWABLE_CARRIER,
            marginal_cost=10.0,
            p_nom=500.0,
        )
        n.add(
            "Generator",
            "Expensive_OCGT",
            bus="B3_East",
            carrier=THERMAL_CARRIER,
            marginal_cost=90.0,
            p_nom=500.0,
        )
        base_lines = (
            ("L12", "B1_North", "B2_West", 10.0, 60.0),
            ("L23", "B2_West", "B3_East", 30.0, 200.0),
            ("L14", "B1_North", "B4_South", 30.0, 200.0),
            ("L43", "B4_South", "B3_East", 10.0, 60.0),
        )
        for lname, b0, b1, x_ohm, snom in base_lines:
            n.add(
                "Line",
                lname,
                bus0=b0,
                bus1=b1,
                x=x_ohm,
                r=1e-4 * x_ohm,
                s_nom=snom,
            )
        if bridge_mode == "ac":
            n.add(
                "Line",
                "L24_Bridge",
                bus0="B2_West",
                bus1="B4_South",
                x=2.0,
                r=2e-4,
                s_nom=500.0,
            )
        elif bridge_mode == "hvdc":
            n.add(
                "Link",
                "L24_HVDC",
                bus0="B2_West",
                bus1="B4_South",
                efficiency=1.0,
                p_nom=500.0,
                p_min_pu=-1.0,
                p_max_pu=1.0,
            )
        n.optimize(
            solver_name="highs",
            solver_options={"log_to_console": False},
            include_objective_constant=False,
        )
        return float(n.objective), float(n.generators_t.p.at[idx[0], "Cheap_Wind"])

    c_base, tr_base = _solve_bridge("none")
    c_ac, tr_ac = _solve_bridge("ac")
    c_hvdc, tr_hvdc = _solve_bridge("hvdc")

    # Closed-form Wheatstone KVL: 10*f12 + 2*(f12 - f23) - 30*f23 = 0 => f23 = (3/8)*60 = 22.5 MW
    analytical_tr_ac = 60.0 + (3.0 / 8.0) * 60.0  # 82.5 MW
    analytical_cost_ac = 10.0 * analytical_tr_ac + 90.0 * (160.0 - analytical_tr_ac)

    return BraessParadoxProof(
        base_4line_cost_per_h=c_base,
        ac_5line_cost_per_h=c_ac,
        hvdc_5line_cost_per_h=c_hvdc,
        base_transfer_mw=tr_base,
        ac_5line_transfer_mw=tr_ac,
        hvdc_5line_transfer_mw=tr_hvdc,
        analytical_ac_5line_transfer_mw=analytical_tr_ac,
        analytical_ac_5line_cost_per_h=analytical_cost_ac,
        braess_cost_increase_pct=100.0 * (c_ac - c_base) / c_base,
    )


@dataclass(frozen=True)
class BThetaLinprogCrossCheck:
    """Direct sparse SciPy HiGHS B-theta phase-angle LP cross-check against PyPSA KVL."""

    pypsa_objective: float
    scipy_objective: float
    rel_objective_diff: float
    max_gen_p_nom_diff_mw: float
    max_line_s_nom_diff_mw: float
    max_lmp_diff_per_mwh: float
    max_congestion_rent_diff_per_h: float
    scipy_ptdf_flow_residual_mw: float


def solve_dcopf_btheta_linprog_crosscheck(
    profiles: pd.DataFrame | None = None,
    *,
    extendable_lines: bool = True,
    capacity_scale: float = 1.0,
    v_nom_kv: float = DEFAULT_NOMINAL_VOLTAGE_KV,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> BThetaLinprogCrossCheck:
    """Formulate nodal phase-angle B-theta DC-OPF / TEP in SciPy and compare with PyPSA."""
    prof = make_four_zone_profiles() if profiles is None else validate_zonal_profiles(profiles)
    gens = default_zonal_generator_specs()
    corr = default_transmission_corridors(
        extendable=extendable_lines, capacity_scale=capacity_scale
    )

    pypsa_res = run_transmission_scenario(
        prof,
        gen_specs=gens,
        corridors=corr,
        mode="ac_kvl",
        extendable_lines=extendable_lines,
        v_nom_kv=v_nom_kv,
        annual_hours=annual_hours,
    )

    z_list = list(ZONES_4BUS)
    g_names = list(gens)
    l_names = list(corr)
    n_z = len(z_list)
    n_g = len(g_names)
    n_l = len(l_names)
    n_t = len(prof.index)
    z_idx = {z: i for i, z in enumerate(z_list)}
    slack_i = z_idx[DEFAULT_REFERENCE_ZONE]

    w = annual_hours / (float(n_t) * SNAPSHOT_HOURS)
    weight_mwh = w * SNAPSHOT_HOURS

    # Variable layout:
    # [0 .. n_g*n_t): p_{g,t}
    # [off_kg .. off_kg + n_g): K_g
    # [off_fl .. off_fl + n_l*n_t): f_{l,t}
    # [off_th .. off_th + n_z*n_t): theta_{z,t}
    # [off_sl .. off_sl + n_l): F_l (line capacities)
    off_kg = n_g * n_t
    off_fl = off_kg + n_g
    off_th = off_fl + n_l * n_t
    off_sl = off_th + n_z * n_t
    n_var = off_sl + n_l

    c = np.zeros(n_var, dtype=float)
    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n_var

    for gi, gname in enumerate(g_names):
        gsp = gens[gname]
        c[gi * n_t : (gi + 1) * n_t] = weight_mwh * gsp.marginal_cost_per_mwh
        if gsp.p_nom_extendable:
            c[off_kg + gi] = gsp.capital_cost_per_mw_yr
        else:
            bounds[off_kg + gi] = (gsp.p_nom_mw, gsp.p_nom_mw)

    for idx_f in range(off_fl, off_th):
        bounds[idx_f] = (None, None)

    for zi in range(n_z):
        for t in range(n_t):
            col = off_th + zi * n_t + t
            if zi == slack_i:
                bounds[col] = (0.0, 0.0)
            else:
                bounds[col] = (None, None)

    for li, lname in enumerate(l_names):
        lsp = corr[lname]
        col = off_sl + li
        if lsp.s_nom_extendable:
            c[col] = lsp.capital_cost_per_mw_yr
            bounds[col] = (lsp.s_nom_min_mw, None)
        else:
            c[col] = 0.0
            bounds[col] = (lsp.s_nom_mw, lsp.s_nom_mw)

    # Equalities:
    # 1) n_z * n_t nodal KCL balances: net_gen_z - net_outflow_z = load_{z,t}
    # 2) n_l * n_t branch B-theta equations: f_{l,t} - b_l*(theta_{b0,t} - theta_{b1,t}) = 0
    n_eq = n_z * n_t + n_l * n_t
    a_eq = lil_matrix((n_eq, n_var), dtype=float)
    b_eq = np.zeros(n_eq, dtype=float)

    for zi, zname in enumerate(z_list):
        d_arr = prof[f"load_{zname}"].to_numpy(dtype=float)
        for t in range(n_t):
            row = zi * n_t + t
            b_eq[row] = d_arr[t]
            for gi, gname in enumerate(g_names):
                if gens[gname].zone == zname:
                    a_eq[row, gi * n_t + t] = 1.0
            for li, lname in enumerate(l_names):
                if corr[lname].bus0 == zname:
                    a_eq[row, off_fl + li * n_t + t] = -1.0
                elif corr[lname].bus1 == zname:
                    a_eq[row, off_fl + li * n_t + t] = 1.0

    base_btheta = n_z * n_t
    for li, lname in enumerate(l_names):
        lsp = corr[lname]
        b_sus = (v_nom_kv**2) / lsp.x_ohm
        i0 = z_idx[lsp.bus0]
        i1 = z_idx[lsp.bus1]
        for t in range(n_t):
            row = base_btheta + li * n_t + t
            a_eq[row, off_fl + li * n_t + t] = 1.0
            a_eq[row, off_th + i0 * n_t + t] = -b_sus
            a_eq[row, off_th + i1 * n_t + t] = b_sus

    # Inequalities: generator availability + symmetric thermal line ratings +-f_{l,t} <= F_l
    ub_rows: list[tuple[dict[int, float], float]] = []
    for gi, gname in enumerate(g_names):
        gsp = gens[gname]
        cf = (
            np.ones(n_t, dtype=float)
            if gsp.profile_col is None
            else prof[gsp.profile_col].to_numpy(dtype=float)
        )
        for t in range(n_t):
            if cf[t] <= 0.0:
                bounds[gi * n_t + t] = (0.0, 0.0)
            else:
                ub_rows.append(({gi * n_t + t: 1.0, off_kg + gi: -float(cf[t])}, 0.0))

    for li in range(n_l):
        for t in range(n_t):
            ub_rows.append(({off_fl + li * n_t + t: 1.0, off_sl + li: -1.0}, 0.0))
            ub_rows.append(({off_fl + li * n_t + t: -1.0, off_sl + li: -1.0}, 0.0))

    a_ub = lil_matrix((len(ub_rows), n_var), dtype=float)
    b_ub = np.zeros(len(ub_rows), dtype=float)
    for r_i, (coeffs, rhs_v) in enumerate(ub_rows):
        for c_i, val in coeffs.items():
            a_ub[r_i, c_i] = val
        b_ub[r_i] = rhs_v

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
        raise SolverError(f"SciPy B-theta DC-OPF cross-check failed: {lp.message}")

    kg_sp = pd.Series(lp.x[off_kg : off_kg + n_g], index=g_names, dtype=float)
    sl_sp = pd.Series(lp.x[off_sl : off_sl + n_l], index=l_names, dtype=float)
    fl_sp = pd.DataFrame(
        {ln: lp.x[off_fl + li * n_t : off_fl + (li + 1) * n_t] for li, ln in enumerate(l_names)},
        index=prof.index,
    )
    th_sp = pd.DataFrame(
        {zn: lp.x[off_th + zi * n_t : off_th + (zi + 1) * n_t] for zi, zn in enumerate(z_list)},
        index=prof.index,
    )
    lmp_sp = pd.DataFrame(
        {
            zn: np.where(
                np.abs(lp.eqlin.marginals[zi * n_t : (zi + 1) * n_t] / weight_mwh) > 1e-12,
                lp.eqlin.marginals[zi * n_t : (zi + 1) * n_t] / weight_mwh,
                0.0,
            )
            for zi, zn in enumerate(z_list)
        },
        index=prof.index,
    )

    zonal_gen_sp = pd.DataFrame(0.0, index=prof.index, columns=z_list)
    for gi, gname in enumerate(g_names):
        zonal_gen_sp[gens[gname].zone] += lp.x[gi * n_t : (gi + 1) * n_t]
    zonal_load_df = pypsa_res.zonal_load_mw[z_list]
    cr_sp = ((zonal_load_df - zonal_gen_sp) * lmp_sp).sum(axis=1)

    ptdf_mat = compute_ptdf_matrix(
        corr, zones=z_list, slack_zone=DEFAULT_REFERENCE_ZONE, v_nom_kv=v_nom_kv
    )
    f_ptdf_sp = (zonal_gen_sp - zonal_load_df).to_numpy(dtype=float) @ ptdf_mat.to_numpy(
        dtype=float
    ).T
    ptdf_res_sp = float(np.max(np.abs(fl_sp.to_numpy(dtype=float) - f_ptdf_sp)))

    # Touch th_sp so phase-angle slack boundary is explicitly verified
    if float(th_sp[DEFAULT_REFERENCE_ZONE].abs().max()) > 1e-12:
        raise SolverError("Reference bus angle must be identically zero")

    scipy_obj = float(lp.fun)
    pypsa_obj = pypsa_res.total_annual_cost

    return BThetaLinprogCrossCheck(
        pypsa_objective=pypsa_obj,
        scipy_objective=scipy_obj,
        rel_objective_diff=abs(pypsa_obj - scipy_obj) / max(1.0, abs(scipy_obj)),
        max_gen_p_nom_diff_mw=float((pypsa_res.gen_p_nom_mw - kg_sp).abs().max()),
        max_line_s_nom_diff_mw=float((pypsa_res.line_s_nom_opt_mw - sl_sp).abs().max()),
        max_lmp_diff_per_mwh=float((pypsa_res.zonal_lmp_per_mwh - lmp_sp).abs().max().max()),
        max_congestion_rent_diff_per_h=float(
            (pypsa_res.hourly_congestion_rent - cr_sp).abs().max()
        ),
        scipy_ptdf_flow_residual_mw=ptdf_res_sp,
    )


def format_transmission_report(regime_df: pd.DataFrame) -> str:
    """Format the 4-regime 4-zone DC-OPF, HVDC & co-optimized TEP comparison table."""
    lines = [
        "Meshed 4-Zone DC-OPF, KVL Loop-Flow & Transmission Expansion Comparison (48 h):",
        (
            "  Regime                          LCOE (/MWh)   Mean Spread   "
            "North Wind   East OCGT   Grid (MW)   Cong. Rent (/yr)"
        ),
    ]
    for _, row in regime_df.iterrows():
        raw_cr = float(row["annual_congestion_rent"])
        cr_val = 0.0 if abs(raw_cr) < 0.5 else raw_cr
        lines.append(
            f"  {str(row['regime']):<30s}  "
            f"{float(row['system_lcoe_per_mwh']):>11.4f}   "
            f"{float(row['mean_lmp_spread_per_mwh']):>11.2f}   "
            f"{float(row['north_wind_mw']):>10.2f}   "
            f"{float(row['east_ocgt_mw']):>9.2f}   "
            f"{float(row['total_line_mw']):>9.2f}   "
            f"{cr_val:>16,.0f}"
        )
    return "\n".join(lines)


def format_single_transmission_report(result: TransmissionScenarioResult) -> str:
    """Format single-scenario 4-zone DC-OPF / TEP diagnostics."""
    tot_grid = float(result.line_s_nom_opt_mw.sum())
    ms_gap = result.merchandising_surplus_identity_max_gap
    spread = result.mean_spatial_lmp_spread_per_mwh
    lines = [
        (
            "4-Zone DC-OPF & Transmission Diagnostics "
            f"(mode={result.mode}, tep={result.extendable_lines}):"
        ),
        f"  System LCOE (currency/MWh):             {result.system_lcoe_per_mwh:>14,.4f}",
        f"  Mean spatial LMP spread (currency/MWh): {spread:>14,.4f}",
        f"  Annual congestion rent (currency/yr):   {result.annual_congestion_rent:>14,.2f}",
        f"  Total corridor capacity (MW):           {tot_grid:>14,.2f}",
        f"  Merchandising surplus identity gap:     {ms_gap:>14.2e}",
    ]
    return "\n".join(lines)
