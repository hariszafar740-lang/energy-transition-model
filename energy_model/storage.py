"""Multi-duration energy storage (Li-Ion BESS + Green H2 P2X) & Dunkelflaute adequacy.

Mathematical formulation
------------------------
Over a chronological horizon ``t = 0, ..., T - 1`` of step ``dt = 1 h`` with annual
scaling weight ``w = hours_per_year / (T * dt)``:

1. **AC Nodal Power Balance (bus ``Regional_Grid``, dual ``w * lambda_t^elec``):**

       sum_g p_{g,t} + p_{bess,dis,t} - p_{bess,ch,t}
                     + p_{h2,dis,t}^{elec} - p_{h2,ch,t}^{elec} = d_t

2. Coupled Short-Duration Battery Storage (PyPSA ``StorageUnit``: ``Li_Ion_BESS``):
   Power rating ``P_bess`` (MW) and energy reservoir ``E_bess = max_hours * P_bess``
   (MWh) scale at fixed duration ``max_hours`` with cyclic State-of-Charge (SoC):

       e_{bess,t} = (1 - sigma_bess) * e_{bess,(t-1) mod T}
                    + eta_ch * p_{bess,ch,t} * dt
                    - (1 / eta_dis) * p_{bess,dis,t} * dt,
       0 <= e_{bess,t} <= max_hours * P_bess.

3. Decoupled Long-Duration Power-to-X Storage (PyPSA ``Bus`` + ``Store`` + ``Link``s):
   Independent power capacities ``P_el`` (MW_e), ``P_fc_elec`` (MW_e) and underground
   hydrogen cavern energy capacity ``E_h2`` (MWh_H2) satisfy cyclic H2 balance:

       e_{h2,t} = (1 - sigma_h2) * e_{h2,(t-1) mod T}
                  + eta_el * p_{h2,ch,t}^{elec} * dt
                  - (1 / eta_fc) * p_{h2,dis,t}^{elec} * dt,
       0 <= e_{h2,t} <= E_h2.

4. KKT Intertemporal Arbitrage & Zero-Profit Equilibrium:
   Every extendable storage asset built at positive capacity earns exact zero net
   economic profit from nodal price arbitrage across the chronological cycle:

       sum_t w * lambda_t^elec * (p_{dis,t}^* - p_{ch,t}^*) * dt = CapEx_storage^* + OpEx_storage^*.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pypsa
from scipy.optimize import linprog

from .model import (
    BUS_CARRIER,
    BUS_NAME,
    DEFAULT_TECHNOLOGIES,
    LOAD_NAME,
    RENEWABLE_CARRIER,
    THERMAL_CARRIER,
    _technology_spec,
    solve,
)
from .profiles import CAPACITY_FACTOR_COLUMNS, SNAPSHOT_HOURS, validate_profiles

HOURS_PER_YEAR: float = 8760.0
H2_BUS_NAME = "Hydrogen_Hub"
H2_CARRIER = "H2"
BATTERY_CARRIER = "battery"
H2_ELECTROLYSIS_CARRIER = "H2_electrolysis"
H2_RECONVERSION_CARRIER = "H2_reconversion"

BESS_NAME = "Li_Ion_BESS"
H2_STORE_NAME = "H2_Cavern"
H2_ELECTROLYZER_NAME = "H2_Electrolyzer"
H2_TURBINE_NAME = "H2_Turbine"

DEFAULT_LULL_DURATION_SWEEP: tuple[int, ...] = (0, 6, 12, 24, 36, 48, 72)


def _check_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


@dataclass(frozen=True)
class BatteryStorageSpec:
    """Short-duration coupled battery specification (PyPSA StorageUnit)."""

    name: str = BESS_NAME
    max_hours: float = 4.0
    efficiency_store: float = 0.95
    efficiency_dispatch: float = 0.95
    standing_loss: float = 0.0001
    capital_cost_per_mw_yr: float = 30_000.0
    marginal_cost_per_mwh: float = 0.0

    def __post_init__(self) -> None:
        mh = _check_real(self.max_hours, "max_hours")
        es = _check_real(self.efficiency_store, "efficiency_store")
        ed = _check_real(self.efficiency_dispatch, "efficiency_dispatch")
        sl = _check_real(self.standing_loss, "standing_loss")
        cc = _check_real(self.capital_cost_per_mw_yr, "capital_cost_per_mw_yr")
        mc = _check_real(self.marginal_cost_per_mwh, "marginal_cost_per_mwh")
        if mh <= 0.0:
            raise ValueError("max_hours must be strictly positive")
        if not (0.0 < es <= 1.0) or not (0.0 < ed <= 1.0):
            raise ValueError("charging and discharging efficiencies must lie in (0, 1]")
        if not (0.0 <= sl < 1.0):
            raise ValueError("standing_loss must lie in [0, 1)")
        if cc < 0.0 or mc < 0.0:
            raise ValueError("storage costs must be non-negative")

    @property
    def round_trip_efficiency(self) -> float:
        return float(self.efficiency_store) * float(self.efficiency_dispatch)

    @property
    def energy_equivalent_cost_per_mwh_yr(self) -> float:
        return float(self.capital_cost_per_mw_yr) / float(self.max_hours)


@dataclass(frozen=True)
class HydrogenP2XSpec:
    """Decoupled Power-to-X hydrogen storage specification (Bus + Store + 2 Links)."""

    electrolyzer_efficiency: float = 0.68
    turbine_efficiency: float = 0.50
    standing_loss: float = 0.0
    electrolyzer_capital_cost_per_mw_e_yr: float = 26_000.0
    turbine_capital_cost_per_mw_e_yr: float = 30_000.0
    store_capital_cost_per_mwh_h2_yr: float = 350.0

    def __post_init__(self) -> None:
        ee = _check_real(self.electrolyzer_efficiency, "electrolyzer_efficiency")
        te = _check_real(self.turbine_efficiency, "turbine_efficiency")
        sl = _check_real(self.standing_loss, "standing_loss")
        ec = _check_real(
            self.electrolyzer_capital_cost_per_mw_e_yr,
            "electrolyzer_capital_cost_per_mw_e_yr",
        )
        tc = _check_real(
            self.turbine_capital_cost_per_mw_e_yr,
            "turbine_capital_cost_per_mw_e_yr",
        )
        sc = _check_real(
            self.store_capital_cost_per_mwh_h2_yr,
            "store_capital_cost_per_mwh_h2_yr",
        )
        if not (0.0 < ee <= 1.0) or not (0.0 < te <= 1.0):
            raise ValueError("hydrogen conversion efficiencies must lie in (0, 1]")
        if not (0.0 <= sl < 1.0):
            raise ValueError("standing_loss must lie in [0, 1)")
        if ec < 0.0 or tc < 0.0 or sc < 0.0:
            raise ValueError("hydrogen component capital costs must be non-negative")

    @property
    def round_trip_efficiency(self) -> float:
        return float(self.electrolyzer_efficiency) * float(self.turbine_efficiency)

    @property
    def turbine_link_capital_cost_per_mw_h2_yr(self) -> float:
        """PyPSA Link capital_cost is denominated per MW of bus0 (H2 thermal) input."""
        return float(self.turbine_efficiency) * float(self.turbine_capital_cost_per_mw_e_yr)


@dataclass
class StorageScenarioResult:
    """Solved chronological multi-duration storage & generation expansion result."""

    status: str
    termination_condition: str
    n_snapshots: int
    annual_weight: float
    peak_demand_mw: float
    annual_demand_mwh: float
    generator_p_nom_mw: pd.Series
    generator_dispatch_mw: pd.DataFrame
    bess_p_nom_mw: float
    bess_e_nom_mwh: float
    bess_charge_mw: pd.Series
    bess_discharge_mw: pd.Series
    bess_soc_mwh: pd.Series
    bess_equiv_cycles_per_year: float
    h2_electrolyzer_mw_e: float
    h2_turbine_mw_e: float
    h2_store_mwh_h2: float
    h2_store_mwh_e_equiv: float
    h2_charge_mw_e: pd.Series
    h2_discharge_mw_e: pd.Series
    h2_soc_mwh_h2: pd.Series
    h2_equiv_cycles_per_year: float
    curtailment_mw: pd.Series
    annual_curtailment_mwh: float
    lmp_elec_per_mwh: pd.Series
    lmp_h2_per_mwh: pd.Series
    capex_per_year: float
    opex_per_year: float
    total_cost_per_year: float
    system_lcoe_per_mwh: float
    annual_demand_payment: float
    strong_duality_gap: float
    bess_net_profit_per_year: float
    h2_net_profit_per_year: float
    max_zero_profit_residual: float
    power_balance_max_residual_mw: float
    soc_cyclic_residual_mwh: float


def generate_dunkelflaute_profiles(
    n_days: int = 7,
    lull_start_hour: int = 72,
    lull_duration_hours: int = 48,
    start_timestamp: str = "2026-01-01 00:00:00",
) -> pd.DataFrame:
    """Synthetic multi-day Dunkelflaute chronology, no randomness in it.

    Outside [lull_start_hour, lull_start_hour + lull_duration_hours) wind averages ~0.35
    and clear-sky solar tops out at 0.85. Inside the cold anticyclonic lull wind falls to
    0.035, overcast solar to 0.08, and space heating adds +24 MW of demand.
    """
    if isinstance(n_days, bool) or not isinstance(n_days, int) or n_days < 1:
        raise ValueError("n_days must be a positive integer")
    if (
        isinstance(lull_start_hour, bool)
        or not isinstance(lull_start_hour, int)
        or lull_start_hour < 0
    ):
        raise ValueError("lull_start_hour must be a non-negative integer")
    if (
        isinstance(lull_duration_hours, bool)
        or not isinstance(lull_duration_hours, int)
        or lull_duration_hours < 0
    ):
        raise ValueError("lull_duration_hours must be a non-negative integer")

    total_hours = 24 * n_days
    if lull_start_hour + lull_duration_hours > total_hours:
        raise ValueError(
            f"lull window [{lull_start_hour}, {lull_start_hour + lull_duration_hours}) "
            f"exceeds horizon of {total_hours} hours"
        )

    idx = pd.date_range(start_timestamp, periods=total_hours, freq="1h")
    hours = np.arange(total_hours, dtype=float)
    hour_of_day = hours % 24.0
    lull_end = lull_start_hour + lull_duration_hours
    in_lull = (hours >= lull_start_hour) & (hours < lull_end)

    base_load = (
        132.0
        + 18.0 * np.exp(-0.5 * ((hour_of_day - 9.0) / 2.5) ** 2)
        + 42.0 * np.exp(-0.5 * ((hour_of_day - 19.0) / 2.3) ** 2)
    )
    load_mw = base_load + np.where(in_lull, 24.0, 0.0)

    daylight = (hour_of_day > 6.0) & (hour_of_day < 18.0)
    raw_solar = np.where(daylight, np.sin(np.pi * (hour_of_day - 6.0) / 12.0), 0.0)
    solar_cf = np.clip(raw_solar * np.where(in_lull, 0.08, 0.85), 0.0, 1.0)

    diurnal_wind = 0.35 + 0.09 * np.cos(2.0 * np.pi * (hour_of_day - 2.0) / 24.0)
    if lull_duration_hours > 0:
        center = lull_start_hour + 0.5 * lull_duration_hours
        half_width = max(3.5, 0.45 * lull_duration_hours)
        lull_mask = np.exp(-0.5 * ((hours - center) / half_width) ** 4)
        wind_cf = np.clip(diurnal_wind * (1.0 - 0.90 * lull_mask), 0.035, 0.95)
    else:
        wind_cf = np.clip(diurnal_wind, 0.035, 0.95)

    df = pd.DataFrame(
        {"load_mw": load_mw, "solar_cf": solar_cf, "wind_cf": wind_cf},
        index=idx,
    )
    return validate_profiles(df)


def build_storage_network(
    profiles: pd.DataFrame,
    technologies: dict[str, dict] | None = None,
    *,
    include_bess: bool = True,
    include_h2: bool = True,
    max_gas_mw: float | None = 0.0,
    bess_spec: BatteryStorageSpec | None = None,
    h2_spec: HydrogenP2XSpec | None = None,
    hours_per_year: float = HOURS_PER_YEAR,
) -> pypsa.Network:
    """Build a chronological multi-carrier PyPSA network with optional BESS & Green H2."""
    validated = validate_profiles(profiles)
    hrs_yr = _check_real(hours_per_year, "hours_per_year")
    if hrs_yr <= 0.0:
        raise ValueError("hours_per_year must be strictly positive")
    if max_gas_mw is not None:
        cap_gas = _check_real(max_gas_mw, "max_gas_mw")
        if cap_gas < 0.0:
            raise ValueError("max_gas_mw must be non-negative or None")
    else:
        cap_gas = None

    bess = BatteryStorageSpec() if bess_spec is None else bess_spec
    h2 = HydrogenP2XSpec() if h2_spec is None else h2_spec

    raw_techs = copy.deepcopy(DEFAULT_TECHNOLOGIES if technologies is None else technologies)
    if cap_gas == 0.0 and "Gas_Peaker" in raw_techs and technologies is None:
        raw_techs.pop("Gas_Peaker")
    elif cap_gas is not None and "Gas_Peaker" in raw_techs:
        raw_techs["Gas_Peaker"]["p_nom_max"] = cap_gas

    if not raw_techs:
        raise ValueError("at least one generator technology is required")

    n_snapshots = len(validated)
    annual_weight = hrs_yr / (float(n_snapshots) * SNAPSHOT_HOURS)

    network = pypsa.Network()
    network.set_snapshots(validated.index)
    network.snapshot_weightings = network.snapshot_weightings.astype(float)
    network.snapshot_weightings.loc[:, "objective"] = annual_weight
    network.snapshot_weightings.loc[:, "generators"] = annual_weight
    network.snapshot_weightings.loc[:, "stores"] = SNAPSHOT_HOURS

    for carrier in (
        BUS_CARRIER,
        H2_CARRIER,
        RENEWABLE_CARRIER,
        THERMAL_CARRIER,
        BATTERY_CARRIER,
        H2_ELECTROLYSIS_CARRIER,
        H2_RECONVERSION_CARRIER,
    ):
        network.add("Carrier", carrier)

    network.add("Bus", BUS_NAME, carrier=BUS_CARRIER)
    network.add("Load", LOAD_NAME, bus=BUS_NAME, p_set=validated["load_mw"])

    for name, raw_spec in raw_techs.items():
        spec = _technology_spec(raw_spec)
        profile = spec.get("profile")
        if profile is not None and profile not in CAPACITY_FACTOR_COLUMNS:
            raise ValueError(f"technology '{name}' references unknown profile '{profile}'")
        extendable = bool(spec.get("p_nom_extendable", True))
        kwargs: dict[str, object] = {
            "bus": BUS_NAME,
            "p_nom_extendable": extendable,
            "capital_cost": float(spec["capital_cost"]),
            "marginal_cost": float(spec["marginal_cost"]),
            "carrier": spec.get(
                "carrier", RENEWABLE_CARRIER if profile is not None else THERMAL_CARRIER
            ),
        }
        if not extendable:
            kwargs["p_nom"] = float(spec.get("p_nom", 0.0))
        elif "p_nom_max" in spec and spec["p_nom_max"] is not None:
            kwargs["p_nom_max"] = float(spec["p_nom_max"])
        network.add("Generator", name, **kwargs)
        if profile is not None:
            network.generators_t.p_max_pu[name] = validated[profile]

    if include_bess:
        network.add(
            "StorageUnit",
            bess.name,
            bus=BUS_NAME,
            carrier=BATTERY_CARRIER,
            p_nom_extendable=True,
            max_hours=bess.max_hours,
            efficiency_store=bess.efficiency_store,
            efficiency_dispatch=bess.efficiency_dispatch,
            standing_loss=bess.standing_loss,
            cyclic_state_of_charge=True,
            capital_cost=bess.capital_cost_per_mw_yr,
            marginal_cost=bess.marginal_cost_per_mwh,
        )

    if include_h2:
        network.add("Bus", H2_BUS_NAME, carrier=H2_CARRIER)
        network.add(
            "Store",
            H2_STORE_NAME,
            bus=H2_BUS_NAME,
            carrier=H2_CARRIER,
            e_nom_extendable=True,
            e_cyclic=True,
            standing_loss=h2.standing_loss,
            capital_cost=h2.store_capital_cost_per_mwh_h2_yr,
            marginal_cost=0.0,
        )
        network.add(
            "Link",
            H2_ELECTROLYZER_NAME,
            bus0=BUS_NAME,
            bus1=H2_BUS_NAME,
            carrier=H2_ELECTROLYSIS_CARRIER,
            p_nom_extendable=True,
            efficiency=h2.electrolyzer_efficiency,
            capital_cost=h2.electrolyzer_capital_cost_per_mw_e_yr,
            marginal_cost=0.0,
        )
        network.add(
            "Link",
            H2_TURBINE_NAME,
            bus0=H2_BUS_NAME,
            bus1=BUS_NAME,
            carrier=H2_RECONVERSION_CARRIER,
            p_nom_extendable=True,
            efficiency=h2.turbine_efficiency,
            capital_cost=h2.turbine_link_capital_cost_per_mw_h2_yr,
            marginal_cost=0.0,
        )

    return network


def run_storage_scenario(
    profiles: pd.DataFrame | None = None,
    technologies: dict[str, dict] | None = None,
    *,
    include_bess: bool = True,
    include_h2: bool = True,
    max_gas_mw: float | None = 0.0,
    bess_spec: BatteryStorageSpec | None = None,
    h2_spec: HydrogenP2XSpec | None = None,
    hours_per_year: float = HOURS_PER_YEAR,
    solver_name: str = "highs",
) -> StorageScenarioResult:
    """Build, solve, and verify a chronological multi-duration storage scenario."""
    prof = generate_dunkelflaute_profiles() if profiles is None else validate_profiles(profiles)
    bess = BatteryStorageSpec() if bess_spec is None else bess_spec
    h2 = HydrogenP2XSpec() if h2_spec is None else h2_spec

    network = build_storage_network(
        prof,
        technologies=technologies,
        include_bess=include_bess,
        include_h2=include_h2,
        max_gas_mw=max_gas_mw,
        bess_spec=bess,
        h2_spec=h2,
        hours_per_year=hours_per_year,
    )
    status, condition = solve(network, solver_name=solver_name)

    dt = SNAPSHOT_HOURS
    idx = prof.index
    n_t = len(idx)
    annual_weight = float(hours_per_year) / (float(n_t) * dt)
    load = prof["load_mw"].astype(float)
    annual_demand_mwh = float(load.sum() * dt * annual_weight)

    gens = network.generators
    gen_p_nom = gens["p_nom_opt"].astype(float).copy()
    gen_p = network.generators_t.p[gens.index].astype(float).copy()

    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")[gens.index]
    avail = p_max_pu * gen_p_nom
    ren_cols = gens.index[gens["carrier"] == RENEWABLE_CARRIER]
    if len(ren_cols) > 0:
        curtailment_mw = (avail[ren_cols] - gen_p[ren_cols]).clip(lower=0.0).sum(axis=1)
    else:
        curtailment_mw = pd.Series(0.0, index=idx, dtype=float)
    annual_curtailment_mwh = float(curtailment_mw.sum() * dt * annual_weight)

    raw_lmp_e = network.buses_t.marginal_price[BUS_NAME].astype(float)
    lmp_elec = raw_lmp_e.where(raw_lmp_e.abs() > 1e-12, 0.0)

    # Short-duration BESS extraction & cyclic SoC check
    zeros = pd.Series(0.0, index=idx, dtype=float)
    cyclic_residuals: list[float] = [0.0]
    zero_profit_residuals: list[float] = []

    for g in gen_p_nom.index:
        p_g = float(gen_p_nom[g])
        has_cap_bound = math.isfinite(float(gens.loc[g, "p_nom_max"]))
        if p_g > 1e-6 and bool(gens.loc[g, "p_nom_extendable"]) and not has_cap_bound:
            rev_g = float((gen_p[g] * lmp_elec).sum() * dt * annual_weight)
            cost_g = float(gens.loc[g, "capital_cost"]) * p_g + (
                float(gens.loc[g, "marginal_cost"]) * float(gen_p[g].sum() * dt * annual_weight)
            )
            zero_profit_residuals.append(abs(rev_g - cost_g))

    if include_bess and bess.name in network.storage_units.index:
        bess_p_nom = float(network.storage_units.loc[bess.name, "p_nom_opt"])
        bess_e_nom = bess_p_nom * bess.max_hours
        bess_ch = network.storage_units_t.p_store[bess.name].astype(float).copy()
        bess_dis = network.storage_units_t.p_dispatch[bess.name].astype(float).copy()
        bess_soc = network.storage_units_t.state_of_charge[bess.name].astype(float).copy()
        if bess_e_nom > 1e-6:
            bess_cycles = float(bess_dis.sum() * dt * annual_weight) / bess_e_nom
        else:
            bess_cycles = 0.0

        soc_prev = np.roll(bess_soc.to_numpy(dtype=float), 1)
        soc_expected = (
            (1.0 - bess.standing_loss) * soc_prev
            + bess.efficiency_store * bess_ch.to_numpy(dtype=float) * dt
            - (1.0 / bess.efficiency_dispatch) * bess_dis.to_numpy(dtype=float) * dt
        )
        cyclic_residuals.append(
            float(np.max(np.abs(bess_soc.to_numpy(dtype=float) - soc_expected)))
        )

        bess_capex = bess_p_nom * bess.capital_cost_per_mw_yr
        bess_opex = float(bess_dis.sum() * dt * annual_weight) * bess.marginal_cost_per_mwh
        bess_rev = float(((bess_dis - bess_ch) * lmp_elec).sum() * dt * annual_weight)
        bess_net_profit = bess_rev - (bess_capex + bess_opex)
        if bess_p_nom > 1e-6:
            zero_profit_residuals.append(abs(bess_net_profit))
    else:
        bess_p_nom = 0.0
        bess_e_nom = 0.0
        bess_ch = zeros.copy()
        bess_dis = zeros.copy()
        bess_soc = zeros.copy()
        bess_cycles = 0.0
        bess_capex = 0.0
        bess_opex = 0.0
        bess_net_profit = 0.0

    # Long-duration Green H2 extraction & cyclic SoC check
    if include_h2 and H2_STORE_NAME in network.stores.index:
        h2_el_mw = float(network.links.loc[H2_ELECTROLYZER_NAME, "p_nom_opt"])
        h2_fc_h2_mw = float(network.links.loc[H2_TURBINE_NAME, "p_nom_opt"])
        h2_fc_elec_mw = h2.turbine_efficiency * h2_fc_h2_mw
        h2_store_mwh = float(network.stores.loc[H2_STORE_NAME, "e_nom_opt"])
        h2_store_mwh_e = h2.turbine_efficiency * h2_store_mwh

        h2_ch_e = network.links_t.p0[H2_ELECTROLYZER_NAME].astype(float).copy()
        h2_fc_in_h2 = network.links_t.p0[H2_TURBINE_NAME].astype(float).copy()
        h2_dis_e = (-network.links_t.p1[H2_TURBINE_NAME]).astype(float).copy()
        h2_soc = network.stores_t.e[H2_STORE_NAME].astype(float).copy()

        raw_lmp_h2 = network.buses_t.marginal_price[H2_BUS_NAME].astype(float)
        lmp_h2 = raw_lmp_h2.where(raw_lmp_h2.abs() > 1e-12, 0.0)

        if h2_store_mwh > 1e-6:
            h2_cycles = float(h2_fc_in_h2.sum() * dt * annual_weight) / h2_store_mwh
        else:
            h2_cycles = 0.0

        h2_prev = np.roll(h2_soc.to_numpy(dtype=float), 1)
        h2_expected = (
            (1.0 - h2.standing_loss) * h2_prev
            + h2.electrolyzer_efficiency * h2_ch_e.to_numpy(dtype=float) * dt
            - h2_fc_in_h2.to_numpy(dtype=float) * dt
        )
        cyclic_residuals.append(float(np.max(np.abs(h2_soc.to_numpy(dtype=float) - h2_expected))))

        h2_capex = (
            h2_el_mw * h2.electrolyzer_capital_cost_per_mw_e_yr
            + h2_fc_h2_mw * h2.turbine_link_capital_cost_per_mw_h2_yr
            + h2_store_mwh * h2.store_capital_cost_per_mwh_h2_yr
        )
        h2_rev = float(((h2_dis_e - h2_ch_e) * lmp_elec).sum() * dt * annual_weight)
        h2_net_profit = h2_rev - h2_capex
        if h2_store_mwh > 1e-6 or h2_el_mw > 1e-6:
            zero_profit_residuals.append(abs(h2_net_profit))
    else:
        h2_el_mw = 0.0
        h2_fc_elec_mw = 0.0
        h2_store_mwh = 0.0
        h2_store_mwh_e = 0.0
        h2_ch_e = zeros.copy()
        h2_dis_e = zeros.copy()
        h2_soc = zeros.copy()
        lmp_h2 = pd.Series(math.nan, index=idx, dtype=float)
        h2_cycles = 0.0
        h2_capex = 0.0
        h2_net_profit = 0.0

    gen_capex = float((gens["capital_cost"] * gen_p_nom).sum())
    gen_opex = float((gen_p.sum() * gens["marginal_cost"] * dt).sum()) * annual_weight
    capex_yr = gen_capex + bess_capex + h2_capex
    opex_yr = gen_opex + bess_opex
    total_cost_yr = capex_yr + opex_yr
    system_lcoe = total_cost_yr / annual_demand_mwh

    annual_demand_payment = float((load * lmp_elec).sum() * dt * annual_weight)
    strong_duality_gap = annual_demand_payment - total_cost_yr

    ac_net = gen_p.sum(axis=1) + bess_dis - bess_ch + h2_dis_e - h2_ch_e - load
    power_balance_res = float(ac_net.abs().max())

    return StorageScenarioResult(
        status=status,
        termination_condition=condition,
        n_snapshots=n_t,
        annual_weight=annual_weight,
        peak_demand_mw=float(load.max()),
        annual_demand_mwh=annual_demand_mwh,
        generator_p_nom_mw=gen_p_nom,
        generator_dispatch_mw=gen_p,
        bess_p_nom_mw=bess_p_nom,
        bess_e_nom_mwh=bess_e_nom,
        bess_charge_mw=bess_ch,
        bess_discharge_mw=bess_dis,
        bess_soc_mwh=bess_soc,
        bess_equiv_cycles_per_year=bess_cycles,
        h2_electrolyzer_mw_e=h2_el_mw,
        h2_turbine_mw_e=h2_fc_elec_mw,
        h2_store_mwh_h2=h2_store_mwh,
        h2_store_mwh_e_equiv=h2_store_mwh_e,
        h2_charge_mw_e=h2_ch_e,
        h2_discharge_mw_e=h2_dis_e,
        h2_soc_mwh_h2=h2_soc,
        h2_equiv_cycles_per_year=h2_cycles,
        curtailment_mw=curtailment_mw,
        annual_curtailment_mwh=annual_curtailment_mwh,
        lmp_elec_per_mwh=lmp_elec,
        lmp_h2_per_mwh=lmp_h2,
        capex_per_year=capex_yr,
        opex_per_year=opex_yr,
        total_cost_per_year=total_cost_yr,
        system_lcoe_per_mwh=system_lcoe,
        annual_demand_payment=annual_demand_payment,
        strong_duality_gap=strong_duality_gap,
        bess_net_profit_per_year=bess_net_profit,
        h2_net_profit_per_year=h2_net_profit,
        max_zero_profit_residual=max(zero_profit_residuals, default=0.0),
        power_balance_max_residual_mw=power_balance_res,
        soc_cyclic_residual_mwh=max(cyclic_residuals),
    )


@dataclass(frozen=True)
class StorageLinprogCrossCheck:
    """Independent SciPy HiGHS matrix LP primal + dual verification for storage."""

    objective_per_year: float
    rel_objective_diff: float
    max_capacity_diff: float
    max_lmp_elec_diff: float


def solve_storage_linprog_crosscheck(
    profiles: pd.DataFrame | None = None,
    *,
    include_bess: bool = True,
    include_h2: bool = True,
    bess_spec: BatteryStorageSpec | None = None,
    h2_spec: HydrogenP2XSpec | None = None,
    hours_per_year: float = HOURS_PER_YEAR,
) -> StorageLinprogCrossCheck:
    """Formulate and solve the multi-carrier storage LP directly in SciPy and compare."""
    prof = (
        generate_dunkelflaute_profiles(n_days=3, lull_start_hour=24, lull_duration_hours=24)
        if profiles is None
        else validate_profiles(profiles)
    )
    bess = BatteryStorageSpec() if bess_spec is None else bess_spec
    h2 = HydrogenP2XSpec() if h2_spec is None else h2_spec

    pypsa_res = run_storage_scenario(
        prof,
        include_bess=include_bess,
        include_h2=include_h2,
        max_gas_mw=0.0,
        bess_spec=bess,
        h2_spec=h2,
        hours_per_year=hours_per_year,
    )

    dt = SNAPSHOT_HOURS
    n_t = len(prof)
    w = float(hours_per_year) / (float(n_t) * dt)
    load = prof["load_mw"].to_numpy(dtype=float)
    solar_cf = prof["solar_cf"].to_numpy(dtype=float)
    wind_cf = prof["wind_cf"].to_numpy(dtype=float)

    # Variable blocks:
    # 0: solar_p(T), 1: wind_p(T), 2: bess_ch(T), 3: bess_dis(T), 4: bess_soc(T),
    # 5: h2_el(T), 6: h2_fc_h2(T), 7: h2_soc(T)
    # Capacities at 8*T + [0:P_solar, 1:P_wind, 2:P_bess, 3:P_el, 4:P_fc_h2, 5:E_h2]
    n_var = 8 * n_t + 6
    c = np.zeros(n_var, dtype=float)
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_var

    c[8 * n_t + 0] = float(DEFAULT_TECHNOLOGIES["Solar_PV"]["capital_cost"])
    c[8 * n_t + 1] = float(DEFAULT_TECHNOLOGIES["Wind_Turbine"]["capital_cost"])
    c[8 * n_t + 2] = bess.capital_cost_per_mw_yr
    c[3 * n_t : 4 * n_t] = w * bess.marginal_cost_per_mwh * dt
    c[8 * n_t + 3] = h2.electrolyzer_capital_cost_per_mw_e_yr
    c[8 * n_t + 4] = h2.turbine_link_capital_cost_per_mw_h2_yr
    c[8 * n_t + 5] = h2.store_capital_cost_per_mwh_h2_yr

    if not include_bess:
        bounds[8 * n_t + 2] = (0.0, 0.0)
        for idx_i in range(2 * n_t, 5 * n_t):
            bounds[idx_i] = (0.0, 0.0)
    if not include_h2:
        for cap_i in (3, 4, 5):
            bounds[8 * n_t + cap_i] = (0.0, 0.0)
        for idx_i in range(5 * n_t, 8 * n_t):
            bounds[idx_i] = (0.0, 0.0)

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []

    for t in range(n_t):
        r = np.zeros(n_var, dtype=float)
        r[0 * n_t + t] = 1.0
        r[1 * n_t + t] = 1.0
        r[2 * n_t + t] = -1.0
        r[3 * n_t + t] = 1.0
        r[5 * n_t + t] = -1.0
        r[6 * n_t + t] = h2.turbine_efficiency
        eq_rows.append(r)
        eq_rhs.append(float(load[t]))

    if include_bess:
        for t in range(n_t):
            r = np.zeros(n_var, dtype=float)
            prev_t = (t - 1) % n_t
            r[4 * n_t + t] = 1.0
            r[4 * n_t + prev_t] -= 1.0 - bess.standing_loss
            r[2 * n_t + t] = -bess.efficiency_store * dt
            r[3 * n_t + t] = (1.0 / bess.efficiency_dispatch) * dt
            eq_rows.append(r)
            eq_rhs.append(0.0)

    if include_h2:
        for t in range(n_t):
            r = np.zeros(n_var, dtype=float)
            prev_t = (t - 1) % n_t
            r[7 * n_t + t] = 1.0
            r[7 * n_t + prev_t] -= 1.0 - h2.standing_loss
            r[5 * n_t + t] = -h2.electrolyzer_efficiency * dt
            r[6 * n_t + t] = 1.0 * dt
            eq_rows.append(r)
            eq_rhs.append(0.0)

    ub_rows: list[np.ndarray] = []
    for t in range(n_t):
        r0 = np.zeros(n_var, dtype=float)
        r0[0 * n_t + t] = 1.0
        r0[8 * n_t + 0] = -solar_cf[t]
        ub_rows.append(r0)

        r1 = np.zeros(n_var, dtype=float)
        r1[1 * n_t + t] = 1.0
        r1[8 * n_t + 1] = -wind_cf[t]
        ub_rows.append(r1)

        if include_bess:
            rb_ch = np.zeros(n_var, dtype=float)
            rb_ch[2 * n_t + t] = 1.0
            rb_ch[8 * n_t + 2] = -1.0
            ub_rows.append(rb_ch)

            rb_dis = np.zeros(n_var, dtype=float)
            rb_dis[3 * n_t + t] = 1.0
            rb_dis[8 * n_t + 2] = -1.0
            ub_rows.append(rb_dis)

            rb_soc = np.zeros(n_var, dtype=float)
            rb_soc[4 * n_t + t] = 1.0
            rb_soc[8 * n_t + 2] = -bess.max_hours
            ub_rows.append(rb_soc)

        if include_h2:
            rh_el = np.zeros(n_var, dtype=float)
            rh_el[5 * n_t + t] = 1.0
            rh_el[8 * n_t + 3] = -1.0
            ub_rows.append(rh_el)

            rh_fc = np.zeros(n_var, dtype=float)
            rh_fc[6 * n_t + t] = 1.0
            rh_fc[8 * n_t + 4] = -1.0
            ub_rows.append(rh_fc)

            rh_e = np.zeros(n_var, dtype=float)
            rh_e[7 * n_t + t] = 1.0
            rh_e[8 * n_t + 5] = -1.0
            ub_rows.append(rh_e)

    lp = linprog(
        c,
        A_ub=np.asarray(ub_rows, dtype=float),
        b_ub=np.zeros(len(ub_rows), dtype=float),
        A_eq=np.asarray(eq_rows, dtype=float),
        b_eq=np.asarray(eq_rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if lp.status != 0:
        raise RuntimeError(f"SciPy storage LP cross-check failed: {lp.message}")

    scipy_caps = np.array(
        [
            lp.x[8 * n_t + 0],
            lp.x[8 * n_t + 1],
            lp.x[8 * n_t + 2],
            lp.x[8 * n_t + 3],
            h2.turbine_efficiency * lp.x[8 * n_t + 4],
            lp.x[8 * n_t + 5],
        ],
        dtype=float,
    )
    pypsa_caps = np.array(
        [
            float(pypsa_res.generator_p_nom_mw.get("Solar_PV", 0.0)),
            float(pypsa_res.generator_p_nom_mw.get("Wind_Turbine", 0.0)),
            pypsa_res.bess_p_nom_mw,
            pypsa_res.h2_electrolyzer_mw_e,
            pypsa_res.h2_turbine_mw_e,
            pypsa_res.h2_store_mwh_h2,
        ],
        dtype=float,
    )

    lmp_scipy_raw = np.asarray(lp.eqlin.marginals[:n_t], dtype=float) / (w * dt)
    lmp_scipy = np.where(np.abs(lmp_scipy_raw) > 1e-12, lmp_scipy_raw, 0.0)
    lmp_diff = float(np.max(np.abs(pypsa_res.lmp_elec_per_mwh.to_numpy(dtype=float) - lmp_scipy)))

    rel_obj = abs(pypsa_res.total_cost_per_year - float(lp.fun)) / max(1.0, abs(float(lp.fun)))
    return StorageLinprogCrossCheck(
        objective_per_year=float(lp.fun),
        rel_objective_diff=rel_obj,
        max_capacity_diff=float(np.max(np.abs(pypsa_caps - scipy_caps))),
        max_lmp_elec_diff=lmp_diff,
    )


def compare_storage_adequacy_regimes(
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare the 5 canonical zero-fossil and gas-firmed adequacy regimes on Dunkelflaute."""
    prof = generate_dunkelflaute_profiles() if profiles is None else validate_profiles(profiles)
    configs = [
        ("vre_only_zero_gas", "VRE Only (0% Gas)", False, False, 0.0),
        ("vre_bess_zero_gas", "VRE + 4h BESS (0% Gas)", True, False, 0.0),
        ("vre_h2_zero_gas", "VRE + Green H2 (0% Gas)", False, True, 0.0),
        ("vre_hybrid_bess_h2_zero_gas", "Hybrid BESS + H2 (0% Gas)", True, True, 0.0),
        ("vre_hybrid_with_unconstrained_gas", "Hybrid + Uncapped Gas", True, True, None),
    ]
    rows: list[dict[str, object]] = []
    for key, label, use_bess, use_h2, gas_cap in configs:
        res = run_storage_scenario(
            prof,
            include_bess=use_bess,
            include_h2=use_h2,
            max_gas_mw=gas_cap,
        )
        rows.append(
            {
                "regime": key,
                "label": label,
                "solar_mw": float(res.generator_p_nom_mw.get("Solar_PV", 0.0)),
                "wind_mw": float(res.generator_p_nom_mw.get("Wind_Turbine", 0.0)),
                "gas_mw": float(res.generator_p_nom_mw.get("Gas_Peaker", 0.0)),
                "bess_mw": res.bess_p_nom_mw,
                "bess_mwh": res.bess_e_nom_mwh,
                "h2_electrolyzer_mw_e": res.h2_electrolyzer_mw_e,
                "h2_turbine_mw_e": res.h2_turbine_mw_e,
                "h2_store_mwh_h2": res.h2_store_mwh_h2,
                "system_lcoe_per_mwh": res.system_lcoe_per_mwh,
                "total_cost_per_year": res.total_cost_per_year,
                "annual_curtailment_mwh": res.annual_curtailment_mwh,
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def sweep_dunkelflaute_duration(
    lull_durations: Sequence[int] = DEFAULT_LULL_DURATION_SWEEP,
    n_days: int = 7,
    lull_start_hour: int = 72,
) -> pd.DataFrame:
    """Sweep Dunkelflaute lull duration (hours) across BESS-only, H2-only & Hybrid."""
    if not lull_durations:
        raise ValueError("lull_durations must contain at least one duration")

    records: list[dict[str, float]] = []
    for dur in lull_durations:
        prof = generate_dunkelflaute_profiles(
            n_days=n_days,
            lull_start_hour=lull_start_hour,
            lull_duration_hours=int(dur),
        )
        res_bess = run_storage_scenario(prof, include_bess=True, include_h2=False, max_gas_mw=0.0)
        res_h2 = run_storage_scenario(prof, include_bess=False, include_h2=True, max_gas_mw=0.0)
        res_hyb = run_storage_scenario(prof, include_bess=True, include_h2=True, max_gas_mw=0.0)
        records.append(
            {
                "lull_hours": float(dur),
                "bess_only_lcoe": res_bess.system_lcoe_per_mwh,
                "h2_only_lcoe": res_h2.system_lcoe_per_mwh,
                "hybrid_lcoe": res_hyb.system_lcoe_per_mwh,
                "hybrid_bess_mwh": res_hyb.bess_e_nom_mwh,
                "hybrid_h2_mwh_e_equiv": res_hyb.h2_store_mwh_e_equiv,
            }
        )
    return pd.DataFrame(records)


def format_storage_report(result: StorageScenarioResult) -> str:
    """Human-readable summary of chronological multi-duration storage & KKT arbitrage."""
    lines = [
        "Multi-Duration Storage & Dunkelflaute Chronological Adequacy Report:",
        f"  Horizon: {result.n_snapshots} h chronological window (annual weight = "
        f"{result.annual_weight:.4f}x)",
        f"  Peak demand: {result.peak_demand_mw:,.2f} MW | Annual demand: "
        f"{result.annual_demand_mwh:,.1f} MWh/yr",
        f"  System LCOE: {result.system_lcoe_per_mwh:,.4f} currency/MWh | "
        f"Total annual cost: {result.total_cost_per_year:,.2f} currency/yr",
        "",
        "  Optimal Generation & Multi-Duration Storage Portfolio:",
    ]
    for g_name, val in result.generator_p_nom_mw.items():
        lines.append(f"    Generator {g_name:<15} {val:>10,.2f} MW")
    lines.extend(
        [
            (
                f"    Li-Ion BESS (4h coupled)  {result.bess_p_nom_mw:>10,.2f} MW | "
                f"{result.bess_e_nom_mwh:>10,.2f} MWh "
                f"({result.bess_equiv_cycles_per_year:>6.1f} cyc/yr)"
            ),
            f"    Green H2 Electrolyzer     {result.h2_electrolyzer_mw_e:>10,.2f} MW_e",
            f"    Green H2 Turbine          {result.h2_turbine_mw_e:>10,.2f} MW_e",
            f"    Green H2 Cavern Store     {result.h2_store_mwh_h2:>10,.2f} MWh_H2 | "
            f"{result.h2_store_mwh_e_equiv:>10,.2f} MWh_e "
            f"({result.h2_equiv_cycles_per_year:>6.1f} cyc/yr)",
            "",
            "  Physical Conservation & KKT Intertemporal Arbitrage Checks:",
            (
                "    Max AC power-balance residual (MW):   "
                f"{result.power_balance_max_residual_mw:>12.4e}"
            ),
            f"    Max cyclic SoC residual (MWh):        {result.soc_cyclic_residual_mwh:>12.4e}",
            f"    Strong-duality gap (currency/yr):     {result.strong_duality_gap:>12.4e}",
            f"    BESS net arbitrage profit (/yr):      {result.bess_net_profit_per_year:>12.4e}",
            f"    Green H2 net arbitrage profit (/yr):  {result.h2_net_profit_per_year:>12.4e}",
        ]
    )
    return "\n".join(lines)
