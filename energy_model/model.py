"""Single-bus least-cost capacity expansion and dispatch model.

Formulation (as solved by PyPSA/linopy with HiGHS):

    minimise   sum_g capital_cost_g * p_nom_g                       (annual)
             + DAYS_PER_YEAR * sum_{g,t} marginal_cost_g * p_{g,t} * dt
    subject to sum_g p_{g,t} = load_t                  for every hour t
               0 <= p_{g,t} <= p_max_pu_{g,t} * p_nom_g
               p_nom_g fixed or 0 <= p_nom_g (extendable)

Units and definitions
---------------------
* Power ``p`` and capacity ``p_nom`` are in MW. Energy is MWh.
* Every snapshot is one hour (``dt = 1 h``).
* ``capital_cost`` is a cost per MW of installed capacity **per year**. The
  values in ``DEFAULT_TECHNOLOGIES`` are carried over from the original
  ``main.py``; they are assumed to be annualised and are not verified against a
  cost source. The currency is not specified in the original model.
* The input horizon is one *representative day* (the profile file). The model
  assumes that day repeats ``DAYS_PER_YEAR`` times per year. Operating costs
  and energy are therefore scaled by ``DAYS_PER_YEAR``; capital costs are
  already annual and are not scaled. Seasonality, leap years and
  multi-day storage are not represented.
* There is no discounting: the objective is an annual system cost, not a net
  present cost.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pypsa

from .profiles import CAPACITY_FACTOR_COLUMNS, SNAPSHOT_HOURS, validate_profiles

BUS_NAME = "Regional_Grid"
LOAD_NAME = "System_Demand"
BUS_CARRIER = "AC"
DAYS_PER_YEAR = 365
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "data" / "profiles_synthetic_24h.csv"

RENEWABLE_CARRIER = "renewable"
THERMAL_CARRIER = "thermal"

# Technology definitions. ``profile`` names the column of the profile table
# that gives the hourly availability (``p_max_pu``); ``None`` means the
# technology is dispatchable with availability 1.0 in every hour.
DEFAULT_TECHNOLOGIES: dict[str, dict] = {
    "Solar_PV": {
        "capital_cost": 45000.0,  # currency / MW / year (assumed annualised)
        "marginal_cost": 0.0,  # currency / MWh
        "p_nom_extendable": True,
        "profile": "solar_cf",
    },
    "Wind_Turbine": {
        "capital_cost": 70000.0,
        "marginal_cost": 0.0,
        "p_nom_extendable": True,
        "profile": "wind_cf",
    },
    "Gas_Peaker": {
        "capital_cost": 20000.0,
        "marginal_cost": 65.0,  # currency / MWh
        "p_nom_extendable": True,
        "profile": None,
    },
}


class SolverError(RuntimeError):
    """Raised when the optimiser does not return an optimal solution."""


@dataclass
class ScenarioResult:
    """Results of one solved scenario. All energies in MWh, powers in MW."""

    status: str
    termination_condition: str
    days_per_year: int
    snapshot_hours: float
    n_snapshots: int
    peak_demand_mw: float
    demand_mwh_per_day: float
    p_nom_mw: pd.Series
    dispatch_mw: pd.DataFrame  # snapshots x generators
    energy_mwh_per_day: pd.Series
    energy_mwh_per_year: pd.Series
    curtailment_mwh_per_day: pd.Series  # renewables only
    capacity_factor: pd.Series  # realised, dispatch / (p_nom * horizon hours)
    energy_balance_residual_mwh: float  # generation - curtailed-free demand
    power_balance_max_residual_mw: float  # max_t |sum_g p - load_t|
    capacity_violation_max_mw: float  # max over g,t of p - p_max_pu * p_nom (<= 0 if ok)
    capex_per_year: float
    opex_per_year: float
    total_cost_per_year: float
    objective_reported: float
    objective_difference: float  # objective_reported - total_cost_per_year
    notes: list[str] = field(default_factory=list)


def _technology_spec(spec: dict) -> dict:
    required = {"capital_cost", "marginal_cost"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"technology is missing keys {sorted(missing)}")
    return spec


def build_network(
    profiles: pd.DataFrame,
    technologies: dict[str, dict] | None = None,
    days_per_year: int = DAYS_PER_YEAR,
) -> pypsa.Network:
    """Create a PyPSA network whose time series come from ``profiles``.

    ``profiles`` must already be validated (see :func:`validate_profiles`); it is
    validated again here so that callers cannot bypass the checks.
    """
    profiles = validate_profiles(profiles)
    if isinstance(days_per_year, bool) or not isinstance(days_per_year, int) or days_per_year < 1:
        raise ValueError("days_per_year must be a positive integer")

    techs = copy.deepcopy(DEFAULT_TECHNOLOGIES if technologies is None else technologies)
    if not techs:
        raise ValueError("at least one technology is required")

    network = pypsa.Network()
    network.set_snapshots(profiles.index)
    # Each snapshot represents one hour on each of the `days_per_year` days.
    network.snapshot_weightings.loc[:, "objective"] = float(days_per_year)
    network.snapshot_weightings.loc[:, "generators"] = float(days_per_year)

    network.add("Carrier", BUS_CARRIER)
    network.add("Carrier", RENEWABLE_CARRIER)
    network.add("Carrier", THERMAL_CARRIER)
    network.add("Bus", BUS_NAME, carrier=BUS_CARRIER)
    network.add("Load", LOAD_NAME, bus=BUS_NAME, p_set=profiles["load_mw"])

    for name, raw_spec in techs.items():
        spec = _technology_spec(raw_spec)
        profile = spec.get("profile")
        if profile is not None and profile not in CAPACITY_FACTOR_COLUMNS:
            raise ValueError(
                f"technology '{name}' references unknown profile '{profile}'; "
                f"expected one of {CAPACITY_FACTOR_COLUMNS}"
            )
        extendable = bool(spec.get("p_nom_extendable", True))
        kwargs = {
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
        network.add("Generator", name, **kwargs)

        if profile is not None:
            network.generators_t.p_max_pu[name] = profiles[profile]

    return network


def solve(network: pypsa.Network, solver_name: str = "highs") -> tuple[str, str]:
    """Solve the network. Raise :class:`SolverError` unless the solution is optimal."""
    status, condition = network.optimize(
        solver_name=solver_name,
        log_to_console=False,
        include_objective_constant=False,
    )
    if status != "ok" or condition != "optimal":
        raise SolverError(
            f"optimisation did not return an optimal solution "
            f"(status={status!r}, termination={condition!r})"
        )
    objective = network.objective
    if objective is None or not math.isfinite(float(objective)):
        raise SolverError("solver reported optimal status but no finite objective value")
    return status, condition


def extract_results(
    network: pypsa.Network,
    *,
    status: str,
    termination_condition: str,
    days_per_year: int = DAYS_PER_YEAR,
) -> ScenarioResult:
    """Compute reported quantities directly from the solved time series."""
    dt = SNAPSHOT_HOURS
    gens = network.generators
    p = network.generators_t.p[gens.index].copy()  # MW, snapshots x generators
    p_nom = gens["p_nom_opt"].astype(float)
    load = network.loads_t.p_set[network.loads.index].sum(axis=1).astype(float)
    n_snapshots = len(p.index)
    horizon_hours = n_snapshots * dt

    energy_per_day = p.sum() * dt  # MWh per representative day
    energy_per_year = energy_per_day * days_per_year

    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")[gens.index]
    available = p_max_pu * p_nom  # MW available each snapshot
    headroom = available - p

    renewable = gens.index[gens["carrier"] == RENEWABLE_CARRIER]
    curtailment_per_day = headroom[renewable].sum() * dt

    with_capacity = p_nom > 0
    capacity_factor = pd.Series(math.nan, index=gens.index)
    capacity_factor[with_capacity] = energy_per_day[with_capacity] / (
        p_nom[with_capacity] * horizon_hours
    )

    balance_residual = p.sum(axis=1) - load
    capex = float((gens["capital_cost"] * p_nom).sum())
    opex_per_day = float((p.sum() * gens["marginal_cost"] * dt).sum())
    opex = opex_per_day * days_per_year
    total = capex + opex

    notes: list[str] = []
    if (headroom[renewable] < -1e-6).any().any():
        notes.append("renewable dispatch exceeds available output")

    return ScenarioResult(
        status=status,
        termination_condition=termination_condition,
        days_per_year=days_per_year,
        snapshot_hours=dt,
        n_snapshots=n_snapshots,
        peak_demand_mw=float(load.max()),
        demand_mwh_per_day=float(load.sum() * dt),
        p_nom_mw=p_nom,
        dispatch_mw=p,
        energy_mwh_per_day=energy_per_day,
        energy_mwh_per_year=energy_per_year,
        curtailment_mwh_per_day=curtailment_per_day,
        capacity_factor=capacity_factor,
        energy_balance_residual_mwh=float(energy_per_day.sum() - load.sum() * dt),
        power_balance_max_residual_mw=float(balance_residual.abs().max()),
        capacity_violation_max_mw=float((p - available).max().max()),
        capex_per_year=capex,
        opex_per_year=opex,
        total_cost_per_year=total,
        objective_reported=float(network.objective),
        objective_difference=float(network.objective) - total,
        notes=notes,
    )


def run_scenario(
    profiles: pd.DataFrame | None = None,
    technologies: dict[str, dict] | None = None,
    days_per_year: int = DAYS_PER_YEAR,
    solver_name: str = "highs",
) -> ScenarioResult:
    """Build, solve and summarise one scenario. Defaults to the shipped profiles."""
    if profiles is None:
        from .profiles import load_profiles

        profiles = load_profiles(DEFAULT_PROFILE_PATH)
    network = build_network(profiles, technologies=technologies, days_per_year=days_per_year)
    status, condition = solve(network, solver_name=solver_name)
    return extract_results(
        network,
        status=status,
        termination_condition=condition,
        days_per_year=days_per_year,
    )


def format_report(result: ScenarioResult) -> str:
    """Human-readable report with explicit units for every quantity."""
    lines = [
        f"Solver status: {result.status} / {result.termination_condition}",
        f"Horizon: {result.n_snapshots} hourly snapshots = one representative day "
        f"(x {result.days_per_year} days/year)",
        f"Peak demand (MW, instantaneous): {result.peak_demand_mw:,.2f}",
        f"Demand energy (MWh per representative day): {result.demand_mwh_per_day:,.2f}",
        "",
        "Optimal installed capacity (MW):",
    ]
    lines += [f"  {name:<14} {value:>12,.2f}" for name, value in result.p_nom_mw.items()]
    lines += ["", "Generation energy (MWh per representative day | per year):"]
    for name in result.energy_mwh_per_day.index:
        lines.append(
            f"  {name:<14} {result.energy_mwh_per_day[name]:>12,.2f} | "
            f"{result.energy_mwh_per_year[name]:>16,.2f}"
        )
    if len(result.curtailment_mwh_per_day):
        lines += ["", "Renewable curtailment (MWh per representative day):"]
        lines += [
            f"  {name:<14} {value:>12,.2f}"
            for name, value in result.curtailment_mwh_per_day.items()
        ]
    lines += [
        "",
        "Costs (currency per year):",
        f"  capital (annualised)  {result.capex_per_year:>18,.2f}",
        f"  operating             {result.opex_per_year:>18,.2f}",
        f"  total                 {result.total_cost_per_year:>18,.2f}",
        f"  solver objective diff {result.objective_difference:>18.6e}",
        "",
        f"Max power-balance residual (MW): {result.power_balance_max_residual_mw:.3e}",
    ]
    lines += [f"NOTE: {note}" for note in result.notes]
    return "\n".join(lines)
