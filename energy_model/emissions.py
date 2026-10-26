"""CO2 cap-tax Lagrangian duality, multi-fuel emissions accounting & MACC analysis.

Mathematical identities verified by this module
-----------------------------------------------
1. **Primary-to-Electric Specific Emissions Accounting:**
   For generator ``g`` burning fuel carrier ``c(g)`` with primary thermal
   emission factor ``EF_th,c`` (tCO2 / MWh_th) and LHV-to-electric efficiency ``eta_g``
   in ``(0, 1]``, specific electric emissions intensity is

       e_g = EF_th,c(g) / eta_g   [tCO2 / MWh_e],
       E_CO2,yr = sum_{g,t} w_t * e_g * p_{g,t} * dt   [tCO2 / yr].

2. **Lagrangian Cap-Tax Duality & Saddle-Point Identity:**
   Consider the emissions-capped capacity-expansion LP with annual cap ``E_bar >= 0``
   and KKT shadow carbon price ``mu_CO2* = -dJ_cap*(E_bar)/dE_bar >= 0``:

       J_cap*(E_bar) = min_{x in X} { C_resource(x) : E(x) <= E_bar }.

   Relaxing ``E(x) <= E_bar`` with multiplier ``tau = mu_CO2*`` yields the
   equivalent Pigouvian carbon-tax LP ``J_tax*(tau) = min_{x in X} { C_resource(x) + tau * E(x) }``,
   satisfying strong Lagrangian saddle-point duality for every ``E_bar >= 0``:

       J_tax*(mu_CO2*) = J_cap*(E_bar) + mu_CO2* * E_bar.

   Conversely, for any non-degenerate carbon tax ``tau >= 0`` selecting a unique
   extreme-point optimum ``x*(tau)`` with realised emissions ``E_bar = E(x*(tau))``,
   solving the cap LP at ``E_bar`` recovers identical capacities ``K_g*``,
   dispatch ``p_{g,t}*``, and hourly nodal shadow prices ``lambda_t*``.

3. **Net-Zero Boundary Subdifferential Regularisation (E_bar = 0):**
   In the dual LP ``max_{y, mu >= 0} b^T y - E_bar * mu``, setting ``E_bar = 0``
   assigns zero objective weight to ``mu``, so the subdifferential ``-dJ_cap*(0)``
   is the unbounded half-ray ``[tau_net_zero_min, +inf)``. Evaluating the
   right-hand directional limit ``lim_{eps -> 0+} mu*(eps)`` uniquely recovers
   the minimum Pigouvian carbon price ``tau_net_zero_min`` achieving exact
   zero-emissions dispatch.
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

from .model import (
    BUS_CARRIER,
    BUS_NAME,
    DEFAULT_TECHNOLOGIES,
    LOAD_NAME,
    RENEWABLE_CARRIER,
    THERMAL_CARRIER,
    SolverError,
    solve,
)
from .profiles import CAPACITY_FACTOR_COLUMNS, SNAPSHOT_HOURS, validate_profiles

if hasattr(pypsa, "options") and hasattr(pypsa.options, "api"):
    pypsa.options.api.legacy_string_dtype = True

ANNUAL_HOURS_PER_YEAR = 8760.0
CO2_CONSTRAINT_NAME = "CO2_Annual_Cap"

DEFAULT_CARRIER_CO2_INTENSITY_T_PER_MWH_TH: dict[str, float] = {
    BUS_CARRIER: 0.0,
    RENEWABLE_CARRIER: 0.0,
    THERMAL_CARRIER: 0.202,
    "coal": 0.340,
    "gas_ccgt": 0.202,
    "gas_ocgt": 0.202,
    "gas_ccs": 0.0202,
    "battery": 0.0,
    "hydrogen": 0.0,
}

DEFAULT_ABATEMENT_FRACTIONS: tuple[float, ...] = (
    0.00,
    0.15,
    0.30,
    0.45,
    0.60,
    0.75,
    0.85,
    0.92,
    0.96,
    0.985,
    0.995,
    1.00,
)

DEFAULT_CARBON_TAX_SWEEP: tuple[float, ...] = (
    0.0,
    8.0,
    15.0,
    22.0,
    30.0,
    55.0,
    80.0,
    120.0,
    250.0,
    420.0,
)

# Storage parameters for the multi-fuel decarbonization portfolio
BESS_STORE_CAPITAL_COST_PER_MWH_YR = 12_000.0
BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR = 6_500.0
BESS_CHARGE_EFFICIENCY = 0.95
BESS_DISCHARGE_EFFICIENCY = 0.95

H2_STORE_CAPITAL_COST_PER_MWH_TH_YR = 950.0
H2_ELECTROLYZER_CAPITAL_COST_PER_MW_ELEC_YR = 32_000.0
H2_TURBINE_CAPITAL_COST_PER_MW_ELEC_YR = 24_000.0
H2_ELECTROLYZER_EFFICIENCY = 0.70
H2_TURBINE_EFFICIENCY = 0.52


def _validate_non_negative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return val


@dataclass(frozen=True)
class FuelTechEmissionSpec:
    """Generator cost, efficiency & primary-fuel CO2 specification."""

    name: str
    carrier: str
    efficiency: float
    primary_co2_t_per_mwh_th: float
    capital_cost_per_mw_yr: float
    base_marginal_cost_per_mwh_e: float
    profile: str | None = None
    p_nom_extendable: bool = True
    p_nom_mw: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.efficiency, bool) or not isinstance(self.efficiency, (int, float)):
            raise ValueError("efficiency must be a real number")
        if not math.isfinite(float(self.efficiency)) or not (0.0 < float(self.efficiency) <= 1.0):
            raise ValueError("efficiency must lie in (0.0, 1.0]")
        _validate_non_negative_float(self.primary_co2_t_per_mwh_th, "primary_co2_t_per_mwh_th")
        _validate_non_negative_float(self.capital_cost_per_mw_yr, "capital_cost_per_mw_yr")
        _validate_non_negative_float(
            self.base_marginal_cost_per_mwh_e, "base_marginal_cost_per_mwh_e"
        )
        _validate_non_negative_float(self.p_nom_mw, "p_nom_mw")
        if self.profile is not None and self.profile not in CAPACITY_FACTOR_COLUMNS:
            raise ValueError(
                f"unknown profile {self.profile!r}; expected one of {CAPACITY_FACTOR_COLUMNS}"
            )

    @property
    def specific_emissions_tco2_per_mwh_e(self) -> float:
        """Electric-output CO2 intensity ``EF_th / eta`` (tCO2 / MWh_e)."""
        return float(self.primary_co2_t_per_mwh_th) / float(self.efficiency)


def make_emissions_week_profiles() -> pd.DataFrame:
    """Return a deterministic 168 h representative week with a 36 h winter lull.

    Synthetic benchmark chronology designed to exercise all four electricity
    abatement stages (wind/solar expansion, coal-to-CCGT fuel switching,
    90%-capture post-combustion Gas CCS, and long-duration green H2 storage).
    """
    idx = pd.date_range("2026-01-01", periods=168, freq="h")
    h = np.arange(168, dtype=float)
    hod = h % 24.0
    dow = np.floor(h / 24.0)

    weekday_factor = np.where(dow < 5.0, 1.03, 0.93)
    morning_bump = 18.0 * np.exp(-0.5 * ((hod - 8.5) / 2.2) ** 2)
    evening_peak = 34.0 * np.exp(-0.5 * ((hod - 18.5) / 2.6) ** 2)
    night_trough = -22.0 * np.exp(-0.5 * ((hod - 3.0) / 2.8) ** 2)
    load_mw = (155.0 + morning_bump + evening_peak + night_trough) * weekday_factor

    day_solar_amp = np.array([0.74, 0.70, 0.60, 0.22, 0.28, 0.68, 0.75])[dow.astype(int)]
    raw_sin = np.sin(np.pi * (hod - 6.0) / 12.0)
    solar_cf = np.where(
        (hod >= 6.0) & (hod <= 18.0),
        np.maximum(0.0, raw_sin) * day_solar_amp,
        0.0,
    )

    synoptic = (
        0.38
        + 0.18 * np.sin(2.0 * np.pi * (h + 10.0) / 84.0)
        + 0.06 * np.cos(2.0 * np.pi * h / 24.0)
    )
    lull_mask = np.exp(-0.5 * ((h - 90.0) / 9.5) ** 4)
    wind_cf = np.clip(synoptic * (1.0 - 0.78 * lull_mask), 0.06, 0.84)

    df = pd.DataFrame(
        {"load_mw": load_mw, "solar_cf": solar_cf, "wind_cf": wind_cf},
        index=idx,
    )
    return validate_profiles(df)


def default_multifuel_specs() -> dict[str, FuelTechEmissionSpec]:
    """Return the 6-technology multi-fuel + VRE portfolio for MACC analysis."""
    items = (
        FuelTechEmissionSpec(
            name="Coal_Steam",
            carrier="coal",
            efficiency=0.40,
            primary_co2_t_per_mwh_th=0.340,
            capital_cost_per_mw_yr=62_000.0,
            base_marginal_cost_per_mwh_e=16.0,
            profile=None,
        ),
        FuelTechEmissionSpec(
            name="CCGT_Gas",
            carrier="gas_ccgt",
            efficiency=0.58,
            primary_co2_t_per_mwh_th=0.202,
            capital_cost_per_mw_yr=40_000.0,
            base_marginal_cost_per_mwh_e=35.0,
            profile=None,
        ),
        FuelTechEmissionSpec(
            name="OCGT_Peaker",
            carrier="gas_ocgt",
            efficiency=0.38,
            primary_co2_t_per_mwh_th=0.202,
            capital_cost_per_mw_yr=20_000.0,
            base_marginal_cost_per_mwh_e=65.0,
            profile=None,
        ),
        FuelTechEmissionSpec(
            name="Gas_CCS",
            carrier="gas_ccs",
            efficiency=0.50,
            primary_co2_t_per_mwh_th=0.0202,
            capital_cost_per_mw_yr=74_000.0,
            base_marginal_cost_per_mwh_e=42.0,
            profile=None,
        ),
        FuelTechEmissionSpec(
            name="Solar_PV",
            carrier=RENEWABLE_CARRIER,
            efficiency=1.00,
            primary_co2_t_per_mwh_th=0.0,
            capital_cost_per_mw_yr=45_000.0,
            base_marginal_cost_per_mwh_e=0.0,
            profile="solar_cf",
        ),
        FuelTechEmissionSpec(
            name="Wind_Turbine",
            carrier=RENEWABLE_CARRIER,
            efficiency=1.00,
            primary_co2_t_per_mwh_th=0.0,
            capital_cost_per_mw_yr=72_000.0,
            base_marginal_cost_per_mwh_e=0.0,
            profile="wind_cf",
        ),
    )
    return {sp.name: sp for sp in items}


def baseline_emission_specs(
    technologies: dict[str, dict] | None = None,
) -> dict[str, FuelTechEmissionSpec]:
    """Map DEFAULT_TECHNOLOGIES (or custom model.py dict) onto FuelTechEmissionSpec."""
    techs = DEFAULT_TECHNOLOGIES if technologies is None else technologies
    if not techs:
        raise ValueError("technologies must not be empty")
    out: dict[str, FuelTechEmissionSpec] = {}
    for name, raw in techs.items():
        profile = raw.get("profile")
        default_carrier = RENEWABLE_CARRIER if profile is not None else THERMAL_CARRIER
        carrier = str(raw.get("carrier", default_carrier))
        default_eff = 1.0 if profile is not None else 0.38
        eff = float(raw.get("efficiency", default_eff))
        default_ef = DEFAULT_CARRIER_CO2_INTENSITY_T_PER_MWH_TH.get(
            carrier, 0.0 if profile is not None else 0.202
        )
        ef_th = float(raw.get("primary_co2_t_per_mwh_th", default_ef))
        out[name] = FuelTechEmissionSpec(
            name=name,
            carrier=carrier,
            efficiency=eff,
            primary_co2_t_per_mwh_th=ef_th,
            capital_cost_per_mw_yr=float(raw["capital_cost"]),
            base_marginal_cost_per_mwh_e=float(raw["marginal_cost"]),
            profile=profile,
            p_nom_extendable=bool(raw.get("p_nom_extendable", True)),
            p_nom_mw=float(raw.get("p_nom", 0.0)),
        )
    return out


def analytical_fuel_switching_carbon_price(
    high_emitting: FuelTechEmissionSpec,
    low_emitting: FuelTechEmissionSpec,
    capacity_factor: float | None = None,
) -> float:
    """Compute exact short-run (dispatch) or long-run (greenfield) CO2 switching price.

    Parameters
    ----------
    high_emitting, low_emitting:
        Generator specifications satisfying ``e_high > e_low``.
    capacity_factor:
        If ``None``, returns the short-run switching price given by
        ratio ``(mc_low - mc_high) / (e_high - e_low)``. If provided in ``(0, 1]``,
        includes annualised capital cost per MWh at ``8760 * capacity_factor``.
    """
    delta_e = (
        high_emitting.specific_emissions_tco2_per_mwh_e
        - low_emitting.specific_emissions_tco2_per_mwh_e
    )
    if delta_e <= 1e-12:
        raise ValueError(
            "high_emitting technology must have strictly higher specific emissions "
            "than low_emitting technology"
        )
    delta_cost = (
        low_emitting.base_marginal_cost_per_mwh_e - high_emitting.base_marginal_cost_per_mwh_e
    )
    if capacity_factor is not None:
        if (
            isinstance(capacity_factor, bool)
            or not isinstance(capacity_factor, (int, float))
            or not math.isfinite(float(capacity_factor))
            or not (0.0 < float(capacity_factor) <= 1.0)
        ):
            raise ValueError("capacity_factor must lie in (0.0, 1.0]")
        cf = float(capacity_factor)
        delta_capex_per_mwh = (
            low_emitting.capital_cost_per_mw_yr - high_emitting.capital_cost_per_mw_yr
        ) / (ANNUAL_HOURS_PER_YEAR * cf)
        delta_cost += delta_capex_per_mwh
    return float(delta_cost / delta_e)


@dataclass
class EmissionsScenarioResult:
    """Complete primal, dual, emissions & fiscal cost-incidence summary."""

    policy_mode: str
    co2_cap_tco2_per_year: float | None
    co2_price_per_tco2: float
    shadow_co2_price_per_tco2: float
    raw_solver_mu_per_tco2: float
    effective_co2_price_per_tco2: float
    annual_emissions_tco2: float
    emissions_by_generator_tco2: pd.Series
    grid_intensity_kgco2_per_mwh: float
    annual_demand_mwh: float
    p_nom_mw: pd.Series
    storage_e_nom_mwh: pd.Series
    link_p_nom_elec_mw: pd.Series
    dispatch_mw: pd.DataFrame
    annual_generation_mwh: pd.Series
    annual_curtailment_mwh: float
    annual_gen_capex_cost: float
    annual_storage_capex_cost: float
    annual_capex_cost: float
    annual_fuel_opex_cost: float
    annual_carbon_tax_payment: float
    total_resource_cost_per_year: float
    total_private_cost_per_year: float
    resource_lcoe_per_mwh: float
    private_lcoe_per_mwh: float
    load_weighted_lmp_per_mwh: float
    lmp_per_mwh: pd.Series
    power_balance_max_residual_mw: float


def build_emissions_network(
    profiles: pd.DataFrame,
    *,
    specs: dict[str, FuelTechEmissionSpec] | None = None,
    co2_cap_tco2_per_year: float | None = None,
    co2_price_per_tco2: float = 0.0,
    include_storage: bool = True,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> pypsa.Network:
    """Construct a PyPSA capacity-expansion network with CO2 cap and/or Pigouvian tax."""
    prof = validate_profiles(profiles)
    gen_specs = default_multifuel_specs() if specs is None else specs
    if not gen_specs:
        raise ValueError("specs must contain at least one technology")

    tax = _validate_non_negative_float(co2_price_per_tco2, "co2_price_per_tco2")
    cap_val: float | None = None
    if co2_cap_tco2_per_year is not None:
        cap_val = _validate_non_negative_float(co2_cap_tco2_per_year, "co2_cap_tco2_per_year")
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

    network.add("Carrier", BUS_CARRIER, co2_emissions=0.0)
    added_carriers = {BUS_CARRIER}
    for sp in gen_specs.values():
        if sp.carrier not in added_carriers:
            network.add(
                "Carrier",
                sp.carrier,
                co2_emissions=float(sp.primary_co2_t_per_mwh_th),
            )
            added_carriers.add(sp.carrier)
        else:
            existing_ef = float(network.carriers.at[sp.carrier, "co2_emissions"])
            if abs(existing_ef - float(sp.primary_co2_t_per_mwh_th)) > 1e-12:
                raise ValueError(
                    f"conflicting primary_co2_t_per_mwh_th on shared carrier {sp.carrier!r}"
                )

    network.add("Bus", BUS_NAME, carrier=BUS_CARRIER)
    network.add("Load", LOAD_NAME, bus=BUS_NAME, p_set=prof["load_mw"])

    for name, sp in gen_specs.items():
        mc_effective = sp.base_marginal_cost_per_mwh_e + tax * sp.specific_emissions_tco2_per_mwh_e
        kwargs: dict[str, object] = {
            "bus": BUS_NAME,
            "carrier": sp.carrier,
            "efficiency": float(sp.efficiency),
            "capital_cost": float(sp.capital_cost_per_mw_yr),
            "marginal_cost": float(mc_effective),
            "p_nom_extendable": bool(sp.p_nom_extendable),
        }
        if not sp.p_nom_extendable:
            kwargs["p_nom"] = float(sp.p_nom_mw)
        network.add("Generator", name, **kwargs)
        if sp.profile is not None:
            network.generators_t.p_max_pu[name] = prof[sp.profile]

    if include_storage:
        for c_name in ("battery", "hydrogen"):
            if c_name not in added_carriers:
                network.add("Carrier", c_name, co2_emissions=0.0)
                added_carriers.add(c_name)

        network.add("Bus", "Battery_Bus", carrier="battery")
        network.add(
            "Store",
            "LiIon_BESS",
            bus="Battery_Bus",
            carrier="battery",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=BESS_STORE_CAPITAL_COST_PER_MWH_YR,
            marginal_cost=0.0,
        )
        network.add(
            "Link",
            "BESS_Charge",
            bus0=BUS_NAME,
            bus1="Battery_Bus",
            efficiency=BESS_CHARGE_EFFICIENCY,
            p_nom_extendable=True,
            capital_cost=BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR,
            marginal_cost=0.0,
        )
        network.add(
            "Link",
            "BESS_Discharge",
            bus0="Battery_Bus",
            bus1=BUS_NAME,
            efficiency=BESS_DISCHARGE_EFFICIENCY,
            p_nom_extendable=True,
            capital_cost=BESS_DISCHARGE_EFFICIENCY * BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR,
            marginal_cost=0.0,
        )

        network.add("Bus", "Hydrogen_Bus", carrier="hydrogen")
        network.add(
            "Store",
            "H2_Cavern",
            bus="Hydrogen_Bus",
            carrier="hydrogen",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=H2_STORE_CAPITAL_COST_PER_MWH_TH_YR,
            marginal_cost=0.0,
        )
        network.add(
            "Link",
            "H2_Electrolyzer",
            bus0=BUS_NAME,
            bus1="Hydrogen_Bus",
            efficiency=H2_ELECTROLYZER_EFFICIENCY,
            p_nom_extendable=True,
            capital_cost=H2_ELECTROLYZER_CAPITAL_COST_PER_MW_ELEC_YR,
            marginal_cost=0.0,
        )
        network.add(
            "Link",
            "H2_Turbine",
            bus0="Hydrogen_Bus",
            bus1=BUS_NAME,
            efficiency=H2_TURBINE_EFFICIENCY,
            p_nom_extendable=True,
            capital_cost=H2_TURBINE_EFFICIENCY * H2_TURBINE_CAPITAL_COST_PER_MW_ELEC_YR,
            marginal_cost=0.0,
        )

    if cap_val is not None:
        network.add(
            "GlobalConstraint",
            CO2_CONSTRAINT_NAME,
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=cap_val,
        )

    return network


def _solve_and_extract_emissions(
    prof: pd.DataFrame,
    gen_specs: dict[str, FuelTechEmissionSpec],
    *,
    co2_cap_tco2_per_year: float | None,
    co2_price_per_tco2: float,
    include_storage: bool,
    annual_hours: float,
    solver_name: str,
    regularise_zero_cap_dual: bool = True,
) -> EmissionsScenarioResult:
    network = build_emissions_network(
        prof,
        specs=gen_specs,
        co2_cap_tco2_per_year=co2_cap_tco2_per_year,
        co2_price_per_tco2=co2_price_per_tco2,
        include_storage=include_storage,
        annual_hours=annual_hours,
    )
    solve(network, solver_name=solver_name)

    n_t = len(prof.index)
    w = annual_hours / (float(n_t) * SNAPSHOT_HOURS)
    weight_mwh = w * SNAPSHOT_HOURS

    names = list(gen_specs)
    p = network.generators_t.p[names].astype(float).copy()
    p_nom = network.generators["p_nom_opt"].reindex(names).astype(float)
    annual_gen = p.sum(axis=0) * weight_mwh

    ef_elec = pd.Series(
        {g: gen_specs[g].specific_emissions_tco2_per_mwh_e for g in names},
        dtype=float,
    )
    emissions_by_gen = (annual_gen * ef_elec).clip(lower=0.0)
    emissions_by_gen = emissions_by_gen.where(emissions_by_gen.abs() > 1e-9, 0.0)
    annual_emissions = float(emissions_by_gen.sum())

    load_mw = prof["load_mw"].astype(float)
    annual_demand = float(load_mw.sum() * weight_mwh)
    grid_intensity = 1000.0 * annual_emissions / annual_demand if annual_demand > 0.0 else 0.0

    # Renewable curtailment
    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")[names]
    avail = p_max_pu * p_nom
    vre_cols = [g for g in names if gen_specs[g].profile is not None]
    if vre_cols:
        annual_curtailment = float(
            (avail[vre_cols] - p[vre_cols]).clip(lower=0.0).sum().sum() * weight_mwh
        )
    else:
        annual_curtailment = 0.0

    # Storage & conversion capacities + net AC injection for power balance check
    net_storage_ac_mw = pd.Series(0.0, index=prof.index, dtype=float)
    if include_storage and len(network.stores) > 0:
        storage_e_nom = network.stores["e_nom_opt"].astype(float).clip(lower=0.0)
        link_p0_nom = network.links["p_nom_opt"].astype(float).clip(lower=0.0)
        link_elec = pd.Series(
            {
                "BESS_Charge": float(link_p0_nom["BESS_Charge"]),
                "BESS_Discharge": BESS_DISCHARGE_EFFICIENCY * float(link_p0_nom["BESS_Discharge"]),
                "H2_Electrolyzer": float(link_p0_nom["H2_Electrolyzer"]),
                "H2_Turbine": H2_TURBINE_EFFICIENCY * float(link_p0_nom["H2_Turbine"]),
            },
            dtype=float,
        )
        # Net injection into Regional_Grid: -p0 for bus0==Regional_Grid, -p1 for bus1==Regional_Grid
        net_storage_ac_mw = (
            -network.links_t.p0["BESS_Charge"]
            - network.links_t.p1["BESS_Discharge"]
            - network.links_t.p0["H2_Electrolyzer"]
            - network.links_t.p1["H2_Turbine"]
        ).astype(float)
        storage_capex = float(
            (network.stores["capital_cost"] * network.stores["e_nom_opt"]).sum()
            + (network.links["capital_cost"] * network.links["p_nom_opt"]).sum()
        )
    else:
        storage_e_nom = pd.Series(dtype=float)
        link_elec = pd.Series(dtype=float)
        storage_capex = 0.0

    balance_res = (p.sum(axis=1) + net_storage_ac_mw - load_mw).abs()
    power_balance_max_res = float(balance_res.max())

    gen_capex = float(sum(gen_specs[g].capital_cost_per_mw_yr * float(p_nom[g]) for g in names))
    total_capex = gen_capex + storage_capex
    fuel_opex = float(
        sum(gen_specs[g].base_marginal_cost_per_mwh_e * float(annual_gen[g]) for g in names)
    )
    tax_payment = float(co2_price_per_tco2) * annual_emissions
    resource_cost = total_capex + fuel_opex
    private_cost = resource_cost + tax_payment

    raw_lmp = network.buses_t.marginal_price[BUS_NAME].astype(float).copy()
    lmp = raw_lmp.where(raw_lmp.abs() > 1e-12, 0.0)
    load_wtd_lmp = float((load_mw * lmp).sum() * weight_mwh / annual_demand)

    if co2_cap_tco2_per_year is not None:
        raw_mu = -float(network.global_constraints.at[CO2_CONSTRAINT_NAME, "mu"])
        if abs(raw_mu) < 1e-9:
            raw_mu = 0.0
        shadow_mu = raw_mu
        # Regularise LP dual degeneracy at exact E_bar = 0 (where -E_bar * mu == 0 in dual obj)
        if (
            regularise_zero_cap_dual
            and co2_cap_tco2_per_year <= 1e-7
            and any(sp.specific_emissions_tco2_per_mwh_e > 0.0 for sp in gen_specs.values())
        ):
            eps_cap = 1.0  # 1 tCO2/yr right-hand directional derivative
            eps_res = _solve_and_extract_emissions(
                prof,
                gen_specs,
                co2_cap_tco2_per_year=eps_cap,
                co2_price_per_tco2=co2_price_per_tco2,
                include_storage=include_storage,
                annual_hours=annual_hours,
                solver_name=solver_name,
                regularise_zero_cap_dual=False,
            )
            if eps_res.annual_emissions_tco2 >= 0.5 * eps_cap:
                shadow_mu = eps_res.raw_solver_mu_per_tco2
    else:
        raw_mu = 0.0
        shadow_mu = 0.0

    if co2_cap_tco2_per_year is None and co2_price_per_tco2 <= 0.0:
        mode = "unconstrained"
    elif co2_cap_tco2_per_year is not None and co2_price_per_tco2 <= 0.0:
        mode = "co2_cap"
    elif co2_cap_tco2_per_year is None and co2_price_per_tco2 > 0.0:
        mode = "co2_tax"
    else:
        mode = "hybrid_cap_and_tax"

    return EmissionsScenarioResult(
        policy_mode=mode,
        co2_cap_tco2_per_year=co2_cap_tco2_per_year,
        co2_price_per_tco2=float(co2_price_per_tco2),
        shadow_co2_price_per_tco2=shadow_mu,
        raw_solver_mu_per_tco2=raw_mu,
        effective_co2_price_per_tco2=float(co2_price_per_tco2) + shadow_mu,
        annual_emissions_tco2=annual_emissions,
        emissions_by_generator_tco2=emissions_by_gen,
        grid_intensity_kgco2_per_mwh=grid_intensity,
        annual_demand_mwh=annual_demand,
        p_nom_mw=p_nom,
        storage_e_nom_mwh=storage_e_nom,
        link_p_nom_elec_mw=link_elec,
        dispatch_mw=p,
        annual_generation_mwh=annual_gen,
        annual_curtailment_mwh=annual_curtailment,
        annual_gen_capex_cost=gen_capex,
        annual_storage_capex_cost=storage_capex,
        annual_capex_cost=total_capex,
        annual_fuel_opex_cost=fuel_opex,
        annual_carbon_tax_payment=tax_payment,
        total_resource_cost_per_year=resource_cost,
        total_private_cost_per_year=private_cost,
        resource_lcoe_per_mwh=resource_cost / annual_demand,
        private_lcoe_per_mwh=private_cost / annual_demand,
        load_weighted_lmp_per_mwh=load_wtd_lmp,
        lmp_per_mwh=lmp,
        power_balance_max_residual_mw=power_balance_max_res,
    )


def run_emissions_scenario(
    profiles: pd.DataFrame | None = None,
    *,
    specs: dict[str, FuelTechEmissionSpec] | None = None,
    co2_cap_tco2_per_year: float | None = None,
    co2_price_per_tco2: float = 0.0,
    include_storage: bool = True,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
    solver_name: str = "highs",
) -> EmissionsScenarioResult:
    """Solve a multi-fuel capacity-expansion & dispatch scenario under CO2 cap/tax."""
    prof = make_emissions_week_profiles() if profiles is None else validate_profiles(profiles)
    gen_specs = default_multifuel_specs() if specs is None else specs
    if not gen_specs:
        raise ValueError("specs must contain at least one technology")
    tax = _validate_non_negative_float(co2_price_per_tco2, "co2_price_per_tco2")
    cap_val: float | None = None
    if co2_cap_tco2_per_year is not None:
        cap_val = _validate_non_negative_float(co2_cap_tco2_per_year, "co2_cap_tco2_per_year")
    ann_h = _validate_non_negative_float(annual_hours, "annual_hours")
    if ann_h <= 0.0:
        raise ValueError("annual_hours must be strictly positive")

    return _solve_and_extract_emissions(
        prof,
        gen_specs,
        co2_cap_tco2_per_year=cap_val,
        co2_price_per_tco2=tax,
        include_storage=include_storage,
        annual_hours=ann_h,
        solver_name=solver_name,
    )


@dataclass(frozen=True)
class CapTaxDualityComparison:
    """Verification of Lagrangian saddle-point & primal/dual equivalence between Cap and Tax."""

    direction: str
    target_emissions_tco2: float
    shadow_co2_price_per_tco2: float
    cap_resource_cost_per_year: float
    tax_resource_cost_per_year: float
    tax_private_cost_per_year: float
    tax_revenue_per_year: float
    lagrangian_saddle_gap: float
    rel_saddle_gap: float
    emissions_diff_tco2: float
    max_p_nom_diff_mw: float
    max_lmp_diff_per_mwh: float


def verify_cap_tax_lagrangian_duality(
    profiles: pd.DataFrame | None = None,
    *,
    specs: dict[str, FuelTechEmissionSpec] | None = None,
    co2_cap_tco2_per_year: float | None = None,
    co2_price_per_tco2: float | None = None,
    include_storage: bool = True,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> CapTaxDualityComparison:
    """Cross-verify Cap-and-Trade vs Pigouvian Carbon Tax Lagrangian equivalence.

    Provide **either** ``co2_cap_tco2_per_year`` (Cap -> Tax direction: solves Cap LP
    at ``E_bar``, extracts KKT multiplier ``mu_CO2*``, then solves Tax LP at ``tau = mu_CO2*``)
    **or** ``co2_price_per_tco2`` (Tax -> Cap direction: solves Tax LP at ``tau``,
    records optimal emissions ``E_bar = E*(tau)``, then solves Cap LP at ``E_bar``).
    """
    if (co2_cap_tco2_per_year is None) == (co2_price_per_tco2 is None):
        raise ValueError("provide exactly one of co2_cap_tco2_per_year or co2_price_per_tco2")
    prof = make_emissions_week_profiles() if profiles is None else validate_profiles(profiles)
    gen_specs = default_multifuel_specs() if specs is None else specs

    if co2_cap_tco2_per_year is not None:
        cap_res = run_emissions_scenario(
            prof,
            specs=gen_specs,
            co2_cap_tco2_per_year=co2_cap_tco2_per_year,
            co2_price_per_tco2=0.0,
            include_storage=include_storage,
            annual_hours=annual_hours,
        )
        tau = cap_res.shadow_co2_price_per_tco2
        tax_res = run_emissions_scenario(
            prof,
            specs=gen_specs,
            co2_cap_tco2_per_year=None,
            co2_price_per_tco2=tau,
            include_storage=include_storage,
            annual_hours=annual_hours,
        )
        e_bar = float(co2_cap_tco2_per_year)
        direction = "cap_to_tax"
    else:
        tau = float(co2_price_per_tco2)
        tax_res = run_emissions_scenario(
            prof,
            specs=gen_specs,
            co2_cap_tco2_per_year=None,
            co2_price_per_tco2=tau,
            include_storage=include_storage,
            annual_hours=annual_hours,
        )
        e_bar = tax_res.annual_emissions_tco2
        cap_res = run_emissions_scenario(
            prof,
            specs=gen_specs,
            co2_cap_tco2_per_year=e_bar,
            co2_price_per_tco2=0.0,
            include_storage=include_storage,
            annual_hours=annual_hours,
        )
        direction = "tax_to_cap"

    # Strong Lagrangian Saddle-Point Identity:
    # J_tax*(tau) == J_cap*(E_bar) + tau * E_bar
    expected_tax_obj = cap_res.total_resource_cost_per_year + tau * e_bar
    saddle_gap = abs(tax_res.total_private_cost_per_year - expected_tax_obj)
    rel_gap = saddle_gap / max(1.0, abs(expected_tax_obj))

    return CapTaxDualityComparison(
        direction=direction,
        target_emissions_tco2=e_bar,
        shadow_co2_price_per_tco2=tau,
        cap_resource_cost_per_year=cap_res.total_resource_cost_per_year,
        tax_resource_cost_per_year=tax_res.total_resource_cost_per_year,
        tax_private_cost_per_year=tax_res.total_private_cost_per_year,
        tax_revenue_per_year=tax_res.annual_carbon_tax_payment,
        lagrangian_saddle_gap=saddle_gap,
        rel_saddle_gap=rel_gap,
        emissions_diff_tco2=abs(cap_res.annual_emissions_tco2 - tax_res.annual_emissions_tco2),
        max_p_nom_diff_mw=float((cap_res.p_nom_mw - tax_res.p_nom_mw).abs().max()),
        max_lmp_diff_per_mwh=float((cap_res.lmp_per_mwh - tax_res.lmp_per_mwh).abs().max()),
    )


def sweep_co2_pareto_and_macc(
    profiles: pd.DataFrame | None = None,
    *,
    abatement_fractions: Sequence[float] = DEFAULT_ABATEMENT_FRACTIONS,
    specs: dict[str, FuelTechEmissionSpec] | None = None,
    include_storage: bool = True,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> pd.DataFrame:
    """Trace the cost-emissions Pareto frontier and Marginal Abatement Cost Curve (MACC)."""
    if not abatement_fractions:
        raise ValueError("abatement_fractions must contain at least one value")
    fracs = []
    for f in abatement_fractions:
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(float(f)):
            raise ValueError("abatement_fractions must contain finite numbers in [0.0, 1.0]")
        fv = float(f)
        if fv < 0.0 or fv > 1.0:
            raise ValueError("abatement_fractions must lie within [0.0, 1.0]")
        fracs.append(fv)

    prof = make_emissions_week_profiles() if profiles is None else validate_profiles(profiles)
    gen_specs = default_multifuel_specs() if specs is None else specs

    bau = run_emissions_scenario(
        prof,
        specs=gen_specs,
        co2_cap_tco2_per_year=None,
        co2_price_per_tco2=0.0,
        include_storage=include_storage,
        annual_hours=annual_hours,
    )
    e_bau = bau.annual_emissions_tco2
    c_bau = bau.total_resource_cost_per_year

    rows: list[dict[str, float]] = []
    prev_e = e_bau
    prev_c = c_bau

    for frac in sorted(fracs):
        if frac <= 1e-12:
            res = bau
            cap_val = e_bau
            saddle_gap = 0.0
        else:
            cap_val = max(0.0, e_bau * (1.0 - frac))
            res = run_emissions_scenario(
                prof,
                specs=gen_specs,
                co2_cap_tco2_per_year=cap_val,
                co2_price_per_tco2=0.0,
                include_storage=include_storage,
                annual_hours=annual_hours,
            )
            tax_chk = run_emissions_scenario(
                prof,
                specs=gen_specs,
                co2_cap_tco2_per_year=None,
                co2_price_per_tco2=res.shadow_co2_price_per_tco2,
                include_storage=include_storage,
                annual_hours=annual_hours,
            )
            saddle_gap = abs(
                tax_chk.total_private_cost_per_year
                - (res.total_resource_cost_per_year + res.shadow_co2_price_per_tco2 * cap_val)
            )

        abated = max(0.0, e_bau - res.annual_emissions_tco2)
        cost_delta = max(0.0, res.total_resource_cost_per_year - c_bau)
        avg_mac = cost_delta / abated if abated > 1e-6 else 0.0
        step_de = prev_e - res.annual_emissions_tco2
        step_mac = (res.total_resource_cost_per_year - prev_c) / step_de if step_de > 1e-6 else 0.0
        prev_e = res.annual_emissions_tco2
        prev_c = res.total_resource_cost_per_year

        tot_gen = max(1e-9, float(res.annual_generation_mwh.sum()))
        coal_gen = float(res.annual_generation_mwh.get("Coal_Steam", 0.0))
        gas_unabated_gen = float(
            res.annual_generation_mwh.get("CCGT_Gas", 0.0)
            + res.annual_generation_mwh.get("OCGT_Peaker", 0.0)
            + res.annual_generation_mwh.get("Gas_Peaker", 0.0)
        )
        gas_ccs_gen = float(res.annual_generation_mwh.get("Gas_CCS", 0.0))
        solar_gen = float(res.annual_generation_mwh.get("Solar_PV", 0.0))
        wind_gen = float(res.annual_generation_mwh.get("Wind_Turbine", 0.0))

        rows.append(
            {
                "abatement_target_pct": 100.0 * frac,
                "abatement_realised_pct": 100.0 * abated / e_bau if e_bau > 0.0 else 0.0,
                "co2_cap_tco2": cap_val,
                "emissions_tco2": res.annual_emissions_tco2,
                "abated_tco2": abated,
                "grid_intensity_kgco2_per_mwh": res.grid_intensity_kgco2_per_mwh,
                "marginal_abatement_cost_per_tco2": res.shadow_co2_price_per_tco2,
                "average_abatement_cost_per_tco2": avg_mac,
                "incremental_step_mac_per_tco2": step_mac,
                "total_resource_cost_per_year": res.total_resource_cost_per_year,
                "abatement_cost_delta_per_year": cost_delta,
                "annual_gen_capex_cost": res.annual_gen_capex_cost,
                "annual_storage_capex_cost": res.annual_storage_capex_cost,
                "annual_fuel_opex_cost": res.annual_fuel_opex_cost,
                "equivalent_tax_revenue_per_year": (
                    res.shadow_co2_price_per_tco2 * res.annual_emissions_tco2
                ),
                "resource_lcoe_per_mwh": res.resource_lcoe_per_mwh,
                "load_weighted_lmp_per_mwh": res.load_weighted_lmp_per_mwh,
                "coal_mw": float(res.p_nom_mw.get("Coal_Steam", 0.0)),
                "ccgt_mw": float(res.p_nom_mw.get("CCGT_Gas", 0.0)),
                "ocgt_mw": float(res.p_nom_mw.get("OCGT_Peaker", 0.0)),
                "gas_ccs_mw": float(res.p_nom_mw.get("Gas_CCS", 0.0)),
                "solar_mw": float(res.p_nom_mw.get("Solar_PV", 0.0)),
                "wind_mw": float(res.p_nom_mw.get("Wind_Turbine", 0.0)),
                "bess_mwh": float(res.storage_e_nom_mwh.get("LiIon_BESS", 0.0)),
                "h2_mwh": float(res.storage_e_nom_mwh.get("H2_Cavern", 0.0)),
                "coal_share_pct": 100.0 * coal_gen / tot_gen,
                "gas_unabated_share_pct": 100.0 * gas_unabated_gen / tot_gen,
                "gas_ccs_share_pct": 100.0 * gas_ccs_gen / tot_gen,
                "solar_share_pct": 100.0 * solar_gen / tot_gen,
                "wind_share_pct": 100.0 * wind_gen / tot_gen,
                "vre_share_pct": 100.0 * (solar_gen + wind_gen) / tot_gen,
                "lagrangian_saddle_gap": saddle_gap,
            }
        )
    return pd.DataFrame(rows)


def sweep_carbon_tax_equivalents(
    profiles: pd.DataFrame | None = None,
    *,
    tax_values: Sequence[float] = DEFAULT_CARBON_TAX_SWEEP,
    specs: dict[str, FuelTechEmissionSpec] | None = None,
    include_storage: bool = True,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> pd.DataFrame:
    """Sweep Pigouvian carbon taxes and record realised emissions, resource cost & tax revenue."""
    if not tax_values:
        raise ValueError("tax_values must contain at least one carbon price")
    prof = make_emissions_week_profiles() if profiles is None else validate_profiles(profiles)
    gen_specs = default_multifuel_specs() if specs is None else specs

    bau = run_emissions_scenario(
        prof,
        specs=gen_specs,
        co2_cap_tco2_per_year=None,
        co2_price_per_tco2=0.0,
        include_storage=include_storage,
        annual_hours=annual_hours,
    )
    e_bau = bau.annual_emissions_tco2

    rows: list[dict[str, float]] = []
    for tau in tax_values:
        t_val = _validate_non_negative_float(tau, "tax_value")
        res = run_emissions_scenario(
            prof,
            specs=gen_specs,
            co2_cap_tco2_per_year=None,
            co2_price_per_tco2=t_val,
            include_storage=include_storage,
            annual_hours=annual_hours,
        )
        abated_pct = (
            100.0 * max(0.0, e_bau - res.annual_emissions_tco2) / e_bau if e_bau > 0.0 else 0.0
        )
        rows.append(
            {
                "co2_tax_per_tco2": t_val,
                "emissions_tco2": res.annual_emissions_tco2,
                "abatement_realised_pct": abated_pct,
                "grid_intensity_kgco2_per_mwh": res.grid_intensity_kgco2_per_mwh,
                "total_resource_cost_per_year": res.total_resource_cost_per_year,
                "carbon_tax_revenue_per_year": res.annual_carbon_tax_payment,
                "total_private_cost_per_year": res.total_private_cost_per_year,
                "resource_lcoe_per_mwh": res.resource_lcoe_per_mwh,
                "private_lcoe_per_mwh": res.private_lcoe_per_mwh,
                "load_weighted_lmp_per_mwh": res.load_weighted_lmp_per_mwh,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class EmissionsLinprogCrossCheck:
    """Direct sparse SciPy HiGHS cross-check of the multi-fuel + storage + CO2 LP."""

    pypsa_objective: float
    scipy_objective: float
    pypsa_shadow_co2_price: float
    scipy_shadow_co2_price: float
    rel_objective_diff: float
    shadow_co2_price_diff: float
    max_p_nom_diff_mw: float
    max_storage_e_nom_diff_mwh: float
    max_lmp_diff_per_mwh: float


def solve_emissions_linprog_crosscheck(
    profiles: pd.DataFrame | None = None,
    *,
    specs: dict[str, FuelTechEmissionSpec] | None = None,
    co2_cap_tco2_per_year: float | None = 300_000.0,
    co2_price_per_tco2: float = 0.0,
    include_storage: bool = True,
    annual_hours: float = ANNUAL_HOURS_PER_YEAR,
) -> EmissionsLinprogCrossCheck:
    """Formulate the complete multi-fuel + BESS/H2 + CO2 LP directly in SciPy and verify PyPSA."""
    prof = make_emissions_week_profiles() if profiles is None else validate_profiles(profiles)
    gen_specs = default_multifuel_specs() if specs is None else specs

    pypsa_res = run_emissions_scenario(
        prof,
        specs=gen_specs,
        co2_cap_tco2_per_year=co2_cap_tco2_per_year,
        co2_price_per_tco2=co2_price_per_tco2,
        include_storage=include_storage,
        annual_hours=annual_hours,
    )

    names = list(gen_specs)
    n_gen = len(names)
    n_t = len(prof.index)
    w = annual_hours / (float(n_t) * SNAPSHOT_HOURS)
    weight_mwh = w * SNAPSHOT_HOURS
    load = prof["load_mw"].to_numpy(dtype=float)

    # Variable layout:
    # [0 .. n_gen*n_t): p_{g,t}
    # [n_gen*n_t .. n_gen*n_t + n_gen): K_g
    offset_kg = n_gen * n_t
    n_var = offset_kg + n_gen

    # Optional 2 storage units (BESS=0, H2=1):
    # each has p_chg(n_t), p_dis(n_t), e_soc(n_t), K_chg(1), K_dis(1), E_nom(1) -> 3*n_t + 3 vars
    st_defs = []
    if include_storage:
        st_defs = [
            (
                "LiIon_BESS",
                BESS_CHARGE_EFFICIENCY,
                BESS_DISCHARGE_EFFICIENCY,
                BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR,
                BESS_DISCHARGE_EFFICIENCY * BESS_LINK_CAPITAL_COST_PER_MW_ELEC_YR,
                BESS_STORE_CAPITAL_COST_PER_MWH_YR,
            ),
            (
                "H2_Cavern",
                H2_ELECTROLYZER_EFFICIENCY,
                H2_TURBINE_EFFICIENCY,
                H2_ELECTROLYZER_CAPITAL_COST_PER_MW_ELEC_YR,
                H2_TURBINE_EFFICIENCY * H2_TURBINE_CAPITAL_COST_PER_MW_ELEC_YR,
                H2_STORE_CAPITAL_COST_PER_MWH_TH_YR,
            ),
        ]
    st_offsets: list[int] = []
    for _ in st_defs:
        st_offsets.append(n_var)
        n_var += 3 * n_t + 3

    c = np.zeros(n_var, dtype=float)
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_var

    for i, g in enumerate(names):
        sp = gen_specs[g]
        mc_eff = (
            sp.base_marginal_cost_per_mwh_e
            + float(co2_price_per_tco2) * sp.specific_emissions_tco2_per_mwh_e
        )
        c[i * n_t : (i + 1) * n_t] = weight_mwh * mc_eff
        c[offset_kg + i] = sp.capital_cost_per_mw_yr
        if not sp.p_nom_extendable:
            bounds[offset_kg + i] = (sp.p_nom_mw, sp.p_nom_mw)

    for s_idx, (_, _, _, cap_chg, cap_dis, cap_store) in enumerate(st_defs):
        base = st_offsets[s_idx]
        c[base + 3 * n_t] = cap_chg
        c[base + 3 * n_t + 1] = cap_dis
        c[base + 3 * n_t + 2] = cap_store

    # Equality constraints: n_t nodal AC balance + len(st_defs)*n_t cyclic SoC balance
    n_eq = n_t + len(st_defs) * n_t
    a_eq = lil_matrix((n_eq, n_var), dtype=float)
    b_eq = np.zeros(n_eq, dtype=float)
    b_eq[:n_t] = load

    for t in range(n_t):
        for i in range(n_gen):
            a_eq[t, i * n_t + t] = 1.0
        for s_idx, (_, eta_c, eta_d, _, _, _) in enumerate(st_defs):
            base = st_offsets[s_idx]
            a_eq[t, base + t] = -1.0  # -p_chg
            a_eq[t, base + n_t + t] = eta_d  # +eta_d * p_dis

            # Cyclic SoC: e_t - e_{t-1} - eta_c * p_chg * dt + p_dis * dt == 0
            eq_row = n_t + s_idx * n_t + t
            t_prev = (t - 1) % n_t
            a_eq[eq_row, base + 2 * n_t + t] = 1.0
            a_eq[eq_row, base + 2 * n_t + t_prev] = -1.0
            a_eq[eq_row, base + t] = -eta_c * SNAPSHOT_HOURS
            a_eq[eq_row, base + n_t + t] = SNAPSHOT_HOURS

    # Inequality constraints:
    # n_gen*n_t generator capacity + len(st_defs)*3*n_t link/store capacity + (1 if co2_cap)
    has_cap = co2_cap_tco2_per_year is not None
    n_ub = n_gen * n_t + len(st_defs) * 3 * n_t + (1 if has_cap else 0)
    a_ub = lil_matrix((n_ub, n_var), dtype=float)
    b_ub = np.zeros(n_ub, dtype=float)

    row_ptr = 0
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
                a_ub[row_ptr, i * n_t + t] = 1.0
                a_ub[row_ptr, offset_kg + i] = -cf[t]
            row_ptr += 1

    for s_idx in range(len(st_defs)):
        base = st_offsets[s_idx]
        for t in range(n_t):
            # p_chg - K_chg <= 0
            a_ub[row_ptr, base + t] = 1.0
            a_ub[row_ptr, base + 3 * n_t] = -1.0
            row_ptr += 1
            # p_dis - K_dis <= 0
            a_ub[row_ptr, base + n_t + t] = 1.0
            a_ub[row_ptr, base + 3 * n_t + 1] = -1.0
            row_ptr += 1
            # e_soc - E_nom <= 0
            a_ub[row_ptr, base + 2 * n_t + t] = 1.0
            a_ub[row_ptr, base + 3 * n_t + 2] = -1.0
            row_ptr += 1

    if has_cap:
        co2_row = row_ptr
        for i, g in enumerate(names):
            ef_e = gen_specs[g].specific_emissions_tco2_per_mwh_e
            if ef_e > 0.0:
                for t in range(n_t):
                    a_ub[co2_row, i * n_t + t] = weight_mwh * ef_e
        b_ub[co2_row] = float(co2_cap_tco2_per_year)

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
        raise SolverError(f"SciPy linprog emissions cross-check failed: {lp.message}")

    p_nom_sp = pd.Series(lp.x[offset_kg : offset_kg + n_gen], index=names, dtype=float)
    lmp_raw = np.asarray(lp.eqlin.marginals[:n_t], dtype=float) / weight_mwh
    lmp_sp = pd.Series(
        np.where(np.abs(lmp_raw) > 1e-12, lmp_raw, 0.0),
        index=prof.index,
        dtype=float,
    )

    if include_storage:
        st_e_sp = pd.Series(
            {
                st_defs[s_idx][0]: float(lp.x[st_offsets[s_idx] + 3 * n_t + 2])
                for s_idx in range(len(st_defs))
            },
            dtype=float,
        )
        max_st_diff = float((pypsa_res.storage_e_nom_mwh - st_e_sp).abs().max())
    else:
        max_st_diff = 0.0

    if has_cap:
        mu_sp = -float(lp.ineqlin.marginals[-1])
        if abs(mu_sp) < 1e-9:
            mu_sp = 0.0
        pypsa_mu_cmp = pypsa_res.raw_solver_mu_per_tco2
    else:
        mu_sp = 0.0
        pypsa_mu_cmp = 0.0

    scipy_obj = float(lp.fun)
    pypsa_obj = pypsa_res.total_private_cost_per_year
    rel_obj_diff = abs(pypsa_obj - scipy_obj) / max(1.0, abs(scipy_obj))

    return EmissionsLinprogCrossCheck(
        pypsa_objective=pypsa_obj,
        scipy_objective=scipy_obj,
        pypsa_shadow_co2_price=pypsa_mu_cmp,
        scipy_shadow_co2_price=mu_sp,
        rel_objective_diff=rel_obj_diff,
        shadow_co2_price_diff=abs(pypsa_mu_cmp - mu_sp),
        max_p_nom_diff_mw=float((pypsa_res.p_nom_mw[names] - p_nom_sp).abs().max()),
        max_storage_e_nom_diff_mwh=max_st_diff,
        max_lmp_diff_per_mwh=float((pypsa_res.lmp_per_mwh - lmp_sp).abs().max()),
    )


def format_emissions_report(result: EmissionsScenarioResult) -> str:
    """Format single-scenario CO2 emissions, shadow carbon price & fiscal incidence."""
    cap_str = (
        "unconstrained"
        if result.co2_cap_tco2_per_year is None
        else f"{result.co2_cap_tco2_per_year:,.2f} tCO2/yr"
    )
    lines = [
        "CO2 Policy & Lagrangian Shadow-Price Diagnostics:",
        f"  Policy mode:                          {result.policy_mode}",
        f"  Annual CO2 cap:                       {cap_str}",
        f"  Exogenous carbon tax (currency/tCO2): {result.co2_price_per_tco2:>12,.4f}",
        f"  KKT shadow CO2 price (currency/tCO2): {result.shadow_co2_price_per_tco2:>12,.4f}",
        f"  Effective CO2 price (currency/tCO2):  {result.effective_co2_price_per_tco2:>12,.4f}",
        f"  Realised annual emissions (tCO2/yr):  {result.annual_emissions_tco2:>12,.2f}",
        f"  Grid CO2 intensity (kgCO2/MWh):       {result.grid_intensity_kgco2_per_mwh:>12,.2f}",
        f"  Resource cost (CapEx+Fuel, /yr):      {result.total_resource_cost_per_year:>12,.2f}",
        f"  Carbon tax payment (/yr):             {result.annual_carbon_tax_payment:>12,.2f}",
        f"  Private market cost (/yr):            {result.total_private_cost_per_year:>12,.2f}",
        f"  Resource LCOE (currency/MWh):         {result.resource_lcoe_per_mwh:>12,.4f}",
        f"  Demand-weighted LMP (currency/MWh):   {result.load_weighted_lmp_per_mwh:>12,.4f}",
    ]
    return "\n".join(lines)


def format_macc_report(macc_df: pd.DataFrame) -> str:
    """Format multi-fuel CO2 Pareto frontier & Marginal Abatement Cost Curve table."""
    lines = [
        "Multi-Fuel CO2 Pareto Frontier & Marginal Abatement Cost Curve (MACC):",
        (
            "  Abate (%)   Emissions (tCO2/yr)   MAC (/tCO2)   Avg Cost (/tCO2)   "
            "Res LCOE (/MWh)   VRE (%)   CCS (%)   Saddle Gap"
        ),
    ]
    for _, row in macc_df.iterrows():
        lines.append(
            f"  {row['abatement_realised_pct']:>8.2f}%   "
            f"{row['emissions_tco2']:>19,.1f}   "
            f"{row['marginal_abatement_cost_per_tco2']:>11,.2f}   "
            f"{row['average_abatement_cost_per_tco2']:>16,.2f}   "
            f"{row['resource_lcoe_per_mwh']:>15,.2f}   "
            f"{row['vre_share_pct']:>7.1f}   "
            f"{row['gas_ccs_share_pct']:>7.1f}   "
            f"{row['lagrangian_saddle_gap']:>10.2e}"
        )
    return "\n".join(lines)
