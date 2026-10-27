"""Interactive Energy Transition Workbench: dataset builder, renderer & mean-CVaR kernel.

What this module adds (Module 7, ``v1.0.0``)
------------------------------------------
Modules 1-6 expose their physics and economics through Python APIs and a CLI.  This
module turns the complete model family into a *browser-ready, zero-dependency*
artifact and adds two pieces of quantitative machinery that only make sense once
every module can be evaluated repeatedly and cheaply:

1. **A reduced-order surrogate of the (WACC, VRE-CapEx) cost surface.**
   The exact surface requires one linear program per grid node.  The surrogate
   interpolates the *exact* solved nodes bilinearly, so the page can sweep the
   two financial drivers continuously.  Holdout nodes (never used for
   interpolation) quantify the interpolation error in currency/MWh, so the
   approximation is reported, not assumed.

2. **An explicit mean-CVaR stochastic program with a Rockafellar-Uryasev epigraph.**
   Module 6 dualises a risk-averse capacity-expansion model through PyPSA; this
   module keeps a *fully explicit* LP (variable layout, sparse matrices, duals)
   so that the risk-aversion weight can be swept in a few milliseconds and the
   dual risk kernel ``q*`` can be read directly from the shadow prices.  It is an
   independent re-derivation on a self-declared 8-regime scenario set, not a
   numerical replica of the Module 6 run; see ``README.md`` section 7.

Deterministic data contract
---------------------------
``build_dataset()`` returns a JSON-serialisable ``dict`` with schema
``etm-workbench/1``.  Everything the page displays is generated here from solved
models - no figure in the HTML is hand-typed.  ``render_page()`` injects that
dict into ``energy_model/data/workbench_template.html``; the result is a single
self-contained file with no network dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import pypsa
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from . import __version__
from .economics import (
    DEFAULT_WACC_SWEEP,
    REFERENCE_WACC,
    analyze_market_duality_and_lcoe,
    build_wacc_technologies,
    default_financial_specs,
    run_scenario,
)
from .emissions import sweep_co2_pareto_and_macc
from .flexibility import (
    SpinningReservePolicy,
    default_flexible_fleet_specs,
    make_duck_curve_profiles,
    run_flexibility_scenario,
    verify_ramp_induced_negative_lmp,
)
from .model import DEFAULT_PROFILE_PATH
from .profiles import load_profiles
from .storage import (
    compare_storage_adequacy_regimes,
    generate_dunkelflaute_profiles,
    run_storage_scenario,
    sweep_dunkelflaute_duration,
)
from .transmission import (
    ZONES_4BUS,
    compute_ptdf_matrix,
    default_transmission_corridors,
    make_four_zone_profiles,
    run_transmission_scenario,
    sweep_transmission_capacity_scale,
    verify_braess_paradox_dcopf,
)

if hasattr(pypsa, "options") and hasattr(pypsa.options, "api"):
    pypsa.options.api.legacy_string_dtype = True

SCHEMA_VERSION = "etm-workbench/1"
TEMPLATE_DATA_TOKEN = "__ETM_WORKBENCH_DATA__"
TEMPLATE_SHA_TOKEN = "__ETM_WORKBENCH_DATA_SHA__"

WACC_GRID: tuple[float, ...] = (0.01, 0.04, 0.07, 0.11, 0.15)
VRE_CAPEX_GRID: tuple[float, ...] = (0.55, 0.75, 1.00, 1.35, 1.75)
STORAGE_DURATIONS: tuple[int, ...] = (0, 3, 6, 8, 9, 12, 18, 24, 48, 72)

# ---------------------------------------------------------------------------
# Mean-CVaR kernel constants
# ---------------------------------------------------------------------------
RISK_HOURS = 24
RISK_DAYS_PER_YEAR = 365.0
RISK_BASE_DEMAND_MW = 150.0
RISK_ALPHA = 0.95
RISK_BETA_SWEEP: tuple[float, ...] = (
    0.0,
    0.10,
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    0.90,
    1.00,
)
RISK_SOLAR_CAPEX_PER_MW_YR = 45_000.0
RISK_WIND_CAPEX_PER_MW_YR = 70_000.0
RISK_GAS_CAPEX_PER_MW_YR = 20_000.0
RISK_BESS_CAPEX_PER_MW_YR = 30_000.0
RISK_GAS_MARGINAL_COST_PER_MWH = 65.0
RISK_VOLL_PER_MWH = 1_800.0
RISK_BESS_ROUND_TRIP_EFFICIENCY = 0.95
RISK_BESS_DURATION_H = 4.0
RISK_TECH_NAMES: tuple[str, ...] = ("solar", "wind", "gas", "bess")


_SOLVE_TALLY: dict[str, int] = {"lp": 0, "pypsa": 0}


def reset_solve_tally() -> None:
    """Reset the optimisation counters used for provenance reporting."""
    _SOLVE_TALLY["lp"] = 0
    _SOLVE_TALLY["pypsa"] = 0


def solve_tally() -> dict[str, int]:
    """Return the number of optimisations executed since the last reset."""
    return dict(_SOLVE_TALLY)


class _CountPypsaSolves:
    """Context manager that counts every PyPSA/linopy optimisation."""

    def __enter__(self) -> _CountPypsaSolves:
        import linopy

        self._original = linopy.Model.solve

        def counted(model: Any, *args: Any, **kwargs: Any) -> Any:
            _SOLVE_TALLY["pypsa"] += 1
            return self._original(model, *args, **kwargs)

        linopy.Model.solve = counted  # type: ignore[method-assign]
        self._linopy = linopy
        return self

    def __exit__(self, *exc: object) -> None:
        self._linopy.Model.solve = self._original  # type: ignore[method-assign]


def _repo_root() -> Path:
    """Return the repository root when running from a source checkout."""
    candidate = Path(__file__).resolve().parent.parent
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


def template_path() -> Path:
    """Path of the packaged single-file HTML template."""
    return Path(__file__).resolve().parent / "data" / "workbench_template.html"


def artifact_path(root: Path | None = None) -> Path:
    """Path of the rendered, committed workbench page."""
    return (root or _repo_root()) / "workbench" / "index.html"


def data_path(root: Path | None = None) -> Path:
    """Path of the committed workbench dataset (JSON)."""
    return (root or _repo_root()) / "workbench" / "data" / "etm_workbench_data.json"


# ---------------------------------------------------------------------------
# Mean-CVaR stochastic kernel
# ---------------------------------------------------------------------------
class RiskScenario(Protocol):
    """Structural type required by the mean-CVaR kernel."""

    name: str
    probability: float

    def profiles(self) -> dict[str, np.ndarray]:
        """Return hourly ``load`` (MW), ``solar_cf`` and ``wind_cf`` (p.u.)."""
        ...


@dataclass(frozen=True)
class RiskRegime:
    """One weather / demand regime of the workbench scenario set."""

    name: str
    probability: float
    load_adder_mw: float
    solar_scale: float
    wind_base_cf: float
    wind_cap_cf: float
    wind_floor_cf: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("regime name must be non-empty")
        if not 0.0 < self.probability <= 1.0:
            raise ValueError(f"regime {self.name!r}: probability must lie in (0, 1]")
        if self.load_adder_mw < 0.0:
            raise ValueError(f"regime {self.name!r}: load adder must be non-negative")
        if not 0.0 <= self.wind_floor_cf <= self.wind_cap_cf <= 1.0:
            raise ValueError(
                f"regime {self.name!r}: wind CF bounds must satisfy 0 <= floor <= cap <= 1"
            )

    def profiles(self) -> dict[str, np.ndarray]:
        """Return hourly load (MW) and VRE capacity factors (p.u.) for this regime."""
        hours = np.arange(RISK_HOURS, dtype=float)
        shape = _diurnal_shape(hours)
        load = (RISK_BASE_DEMAND_MW + self.load_adder_mw) * shape
        solar = np.clip(self.solar_scale * _solar_shape(hours), 0.0, 1.0)
        wind = np.clip(
            self.wind_base_cf
            + 0.10 * np.cos(2.0 * np.pi * (hours - 4.0) / 24.0)
            + 0.05 * np.sin(2.0 * np.pi * hours / 48.0),
            self.wind_floor_cf,
            self.wind_cap_cf,
        )
        return {"load": load, "solar_cf": solar, "wind_cf": wind}


def _diurnal_shape(hours: np.ndarray) -> np.ndarray:
    """Mean-one diurnal demand shape with an evening peak and a night trough."""
    evening = np.exp(-0.5 * ((hours - 19.0) / 1.10) ** 2)
    night = np.exp(-0.5 * ((hours - 4.0) / 2.50) ** 2)
    shape = 1.0 + 0.42 * evening - 0.18 * night
    return shape / float(shape.mean())


def _solar_shape(hours: np.ndarray) -> np.ndarray:
    """Clear-sky solar availability shape (p.u. of installed capacity)."""
    raw = np.sin(np.pi * (hours - 6.0) / 13.0)
    return np.clip(np.maximum(0.0, raw), 0.0, 1.0) ** 1.10


DEFAULT_RISK_REGIMES: tuple[RiskRegime, ...] = (
    RiskRegime("WY1_Windy_Equinox", 0.180, 0.0, 1.00, 0.62, 0.92, 0.12),
    RiskRegime("WY2_Sunny_Anticyclone", 0.170, 4.0, 1.12, 0.34, 0.80, 0.05),
    RiskRegime("WY3_Temperate_Normal", 0.250, 0.0, 0.86, 0.48, 0.88, 0.08),
    RiskRegime("WY4_Cloudy_Maritime", 0.160, 10.0, 0.46, 0.55, 0.90, 0.10),
    RiskRegime("WY5_Summer_Heat_Wave", 0.140, 22.0, 1.05, 0.30, 0.75, 0.04),
    RiskRegime("WY6_Winter_Cloud_Trough", 0.060, 28.0, 0.34, 0.42, 0.70, 0.06),
    RiskRegime("WY7_Polar_Vortex_Freeze", 0.025, 38.0, 0.26, 0.30, 0.24, 0.06),
    RiskRegime("WY8_Severe_Dunkelflaute", 0.015, 78.0, 0.18, 0.26, 0.10, 0.04),
)


def validate_risk_regimes(regimes: Sequence[RiskScenario]) -> tuple[RiskScenario, ...]:
    """Validate a regime set: probabilities must sum to one and names be unique."""
    if not regimes:
        raise ValueError("at least one risk regime is required")
    total = float(sum(r.probability for r in regimes))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"regime probabilities must sum to 1.0, got {total!r}")
    names = [r.name for r in regimes]
    if len(set(names)) != len(names):
        raise ValueError("regime names must be unique")
    return tuple(regimes)


@dataclass
class RiskKernelSolution:
    """Optimal mean-CVaR plan and its explicit primal/dual diagnostics."""

    beta: float
    alpha: float
    objective: float
    expected_cost_per_year: float
    cvar_per_year: float
    var_per_year: float
    risk_adjusted_cost_per_year: float
    expected_demand_mwh_per_year: float
    lcoe_per_mwh: float
    cvar_lcoe_per_mwh: float
    expected_unserved_mwh_per_year: float
    capacities_mw: dict[str, float]
    scenario_cost_per_year: np.ndarray
    scenario_energy_mwh_per_year: np.ndarray
    risk_weights: np.ndarray
    total_risk_weight: float
    lmp_per_mwh: np.ndarray
    shed_mw: np.ndarray
    dispatch_mw: dict[str, np.ndarray]
    n_variables: int
    n_constraints: int
    duality_gap: float
    weight_envelope_excess: float

    def summary(self) -> str:
        caps = " | ".join(f"{k} {v:,.2f} MW" for k, v in self.capacities_mw.items())
        return (
            f"beta={self.beta:.2f} E[Z]={self.expected_cost_per_year:,.0f} "
            f"CVaR={self.cvar_per_year:,.0f} LCOE={self.lcoe_per_mwh:.3f} "
            f"EENS={self.expected_unserved_mwh_per_year:,.2f} MWh/yr | {caps}"
        )


def discrete_cvar(
    costs: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Return ``(VaR, CVaR)`` of a discrete distribution at confidence level ``alpha``.

    The worst ``1 - alpha`` probability mass is integrated with exact atom
    splitting, so a single scenario can be partially counted when the tail
    boundary falls inside its probability atom (this happens for the 6% ``WY6``
    atom at ``alpha = 0.95``).  Evaluating the tail directly keeps the reported
    CVaR well defined even at ``beta = 0``, where the epigraph variable ``eta``
    carries no objective weight.
    """
    c = np.asarray(costs, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if c.shape != p.shape or c.size == 0:
        raise ValueError("costs and probabilities must be non-empty and equally sized")
    if abs(float(p.sum()) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1.0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie within (0, 1)")
    order = np.argsort(c)
    c_sorted, p_sorted = c[order], p[order]
    remaining = 1.0 - alpha
    total = 0.0
    var = float(c_sorted[-1])
    for i in range(c_sorted.size - 1, -1, -1):
        take = min(float(p_sorted[i]), remaining)
        total += take * float(c_sorted[i])
        remaining -= take
        if remaining <= 1e-12:
            var = float(c_sorted[i])
            break
    return var, total / (1.0 - alpha)


def solve_mean_cvar(
    beta: float,
    regimes: Sequence[RiskScenario] | None = None,
    *,
    alpha: float = RISK_ALPHA,
    fixed_capacities_mw: dict[str, float] | None = None,
) -> RiskKernelSolution:
    """Solve the explicit two-stage mean-CVaR capacity-expansion LP.

    The risk-averse objective is the Rockafellar-Uryasev convex combination

        minimise  (1 - beta) * E[Z] + beta * CVaR_alpha[Z]

    evaluated through its epigraph ``eta + (1 / (1 - alpha)) * E[(Z - eta)+]``,
    where ``Z`` is the annual cost of one future (capacity CapEx + fuel + VoLL of
    unserved energy).  ``beta = 0`` is risk-neutral expected cost and ``beta = 1``
    is the pure worst-tail CVaR objective.

    Parameters
    ----------
    beta:
        Risk-aversion weight in ``[0, 1]``.
    regimes:
        Scenario set; defaults to :data:`DEFAULT_RISK_REGIMES` (probabilities
        must sum to one).
    alpha:
        CVaR confidence level in ``(0, 1)``; ``0.95`` targets the worst 5% tail.
    fixed_capacities_mw:
        Optional capacity vector (``{"solar", "wind", "gas", "bess"}``) used to
        *evaluate* a first-stage plan under the full scenario set, e.g. for the
        expectation-of-expected-value (EEV) bound.
    """
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("beta must be a real number")
    beta = float(beta)
    if not 0.0 <= beta <= 1.0 or not math.isfinite(beta):
        raise ValueError("beta must lie within [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie within (0, 1)")

    scens = validate_risk_regimes(DEFAULT_RISK_REGIMES if regimes is None else regimes)
    n_s = len(scens)
    per = 7 * RISK_HOURS  # solar, wind, gas, charge, discharge, SoC, unserved
    i_eta = 4 + n_s * per
    i_z = i_eta + 1
    n_var = i_z + n_s

    cost = np.zeros(n_var)
    cost[:4] = np.array(
        [
            RISK_SOLAR_CAPEX_PER_MW_YR,
            RISK_WIND_CAPEX_PER_MW_YR,
            RISK_GAS_CAPEX_PER_MW_YR,
            RISK_BESS_CAPEX_PER_MW_YR,
        ]
    ) * (1.0 - beta)
    cost[i_eta] = beta
    for s, sc in enumerate(scens):
        w_expect = (1.0 - beta) * sc.probability
        cost[i_z + s] = beta * sc.probability / (1.0 - alpha)
        o = 4 + s * per
        cost[o + 2 * RISK_HOURS : o + 3 * RISK_HOURS] = (
            RISK_GAS_MARGINAL_COST_PER_MWH * RISK_DAYS_PER_YEAR * w_expect
        )
        cost[o + 6 * RISK_HOURS : o + 7 * RISK_HOURS] = (
            RISK_VOLL_PER_MWH * RISK_DAYS_PER_YEAR * w_expect
        )

    rows_i: list[int] = []
    cols_i: list[int] = []
    vals_i: list[float] = []
    rhs_i: list[float] = []
    rows_e: list[int] = []
    cols_e: list[int] = []
    vals_e: list[float] = []
    rhs_e: list[float] = []
    epigraph_row = np.zeros(n_s, dtype=int)
    balance_row = np.zeros((n_s, RISK_HOURS), dtype=int)
    n_row = 0
    n_eq = 0
    eta, dur = RISK_BESS_ROUND_TRIP_EFFICIENCY, RISK_BESS_DURATION_H

    for s, sc in enumerate(scens):
        prof = sc.profiles()
        o = 4 + s * per
        idx = {k: o + k * RISK_HOURS for k in range(7)}

        # z_s >= capex + fuel + VoLL - eta  ->  capex + fuel + VoLL - eta - z_s <= 0
        epigraph_row[s] = n_row
        for j, coef in enumerate(
            [
                RISK_SOLAR_CAPEX_PER_MW_YR,
                RISK_WIND_CAPEX_PER_MW_YR,
                RISK_GAS_CAPEX_PER_MW_YR,
                RISK_BESS_CAPEX_PER_MW_YR,
            ]
        ):
            rows_i.append(n_row)
            cols_i.append(j)
            vals_i.append(coef)
        for t in range(RISK_HOURS):
            rows_i.append(n_row)
            cols_i.append(idx[2] + t)
            vals_i.append(RISK_GAS_MARGINAL_COST_PER_MWH * RISK_DAYS_PER_YEAR)
            rows_i.append(n_row)
            cols_i.append(idx[6] + t)
            vals_i.append(RISK_VOLL_PER_MWH * RISK_DAYS_PER_YEAR)
        for col, val in ((i_eta, -1.0), (i_z + s, -1.0)):
            rows_i.append(n_row)
            cols_i.append(col)
            vals_i.append(val)
        rhs_i.append(0.0)
        n_row += 1

        for t in range(RISK_HOURS):
            # Power balance (load must be met exactly): generation + discharge - charge + shed.
            balance_row[s, t] = n_eq
            for k, coef in ((0, 1.0), (1, 1.0), (2, 1.0), (4, 1.0), (3, -1.0), (6, 1.0)):
                rows_e.append(n_eq)
                cols_e.append(idx[k] + t)
                vals_e.append(coef)
            rhs_e.append(float(prof["load"][t]))
            n_eq += 1

            # Cyclic state of charge: e_t - e_{t-1} - eta * charge_t + discharge_t / eta = 0.
            for col, val in (
                (idx[5] + t, 1.0),
                (idx[5] + (t - 1) % RISK_HOURS, -1.0),
                (idx[3] + t, -eta),
                (idx[4] + t, 1.0 / eta),
            ):
                rows_i.append(n_row)
                cols_i.append(col)
                vals_i.append(val)
            rhs_i.append(0.0)
            n_row += 1

            # discharge_t <= P_bess and e_t <= duration * P_bess.
            for col, val in ((idx[4] + t, 1.0), (3, -1.0)):
                rows_i.append(n_row)
                cols_i.append(col)
                vals_i.append(val)
            rhs_i.append(0.0)
            n_row += 1
            for col, val in ((idx[5] + t, 1.0), (3, -dur)):
                rows_i.append(n_row)
                cols_i.append(col)
                vals_i.append(val)
            rhs_i.append(0.0)
            n_row += 1

            # Available energy: solar and wind dispatch cannot exceed capacity-factor output.
            for k, cf in ((0, prof["solar_cf"][t]), (1, prof["wind_cf"][t])):
                for col, val in ((idx[k] + t, 1.0), (k, -float(cf))):
                    rows_i.append(n_row)
                    cols_i.append(col)
                    vals_i.append(val)
                rhs_i.append(0.0)
                n_row += 1

            # Thermal capacity limit on gas dispatch.
            for col, val in ((idx[2] + t, 1.0), (2, -1.0)):
                rows_i.append(n_row)
                cols_i.append(col)
                vals_i.append(val)
            rhs_i.append(0.0)
            n_row += 1

    a_ub = coo_matrix((vals_i, (rows_i, cols_i)), shape=(n_row, n_var)).tocsr()
    a_eq = coo_matrix((vals_e, (rows_e, cols_e)), shape=(n_eq, n_var)).tocsr()
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_var
    if fixed_capacities_mw is not None:
        for j, name in enumerate(RISK_TECH_NAMES):
            value = float(fixed_capacities_mw.get(name, 0.0))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"fixed capacity {name!r} must be finite and non-negative")
            bounds[j] = (value, value)

    _SOLVE_TALLY["lp"] += 1
    res = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.asarray(rhs_i, dtype=float),
        A_eq=a_eq,
        b_eq=np.asarray(rhs_e, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"mean-CVaR kernel did not solve to optimality: {res.message}")

    x = res.x
    caps = {name: float(x[j]) for j, name in enumerate(RISK_TECH_NAMES)}
    scen_cost = np.zeros(n_s)
    scen_energy = np.zeros(n_s)
    shed = np.zeros((n_s, RISK_HOURS))
    gas = np.zeros((n_s, RISK_HOURS))
    solar = np.zeros((n_s, RISK_HOURS))
    wind = np.zeros((n_s, RISK_HOURS))
    discharge = np.zeros((n_s, RISK_HOURS))
    charge = np.zeros((n_s, RISK_HOURS))
    soc = np.zeros((n_s, RISK_HOURS))
    lmp = np.zeros((n_s, RISK_HOURS))
    for s, sc in enumerate(scens):
        o = 4 + s * per
        solar[s] = x[o : o + RISK_HOURS]
        wind[s] = x[o + RISK_HOURS : o + 2 * RISK_HOURS]
        gas[s] = x[o + 2 * RISK_HOURS : o + 3 * RISK_HOURS]
        discharge[s] = x[o + 4 * RISK_HOURS : o + 5 * RISK_HOURS]
        charge[s] = x[o + 3 * RISK_HOURS : o + 4 * RISK_HOURS]
        soc[s] = x[o + 5 * RISK_HOURS : o + 6 * RISK_HOURS]
        shed[s] = x[o + 6 * RISK_HOURS : o + 7 * RISK_HOURS]
        scen_cost[s] = sum(caps[n] * c for n, c in _capex_map().items()) + RISK_DAYS_PER_YEAR * (
            RISK_GAS_MARGINAL_COST_PER_MWH * float(gas[s].sum())
            + RISK_VOLL_PER_MWH * float(shed[s].sum())
        )
        scen_energy[s] = float(sc.profiles()["load"].sum()) * RISK_DAYS_PER_YEAR
        if res.eqlin is not None and getattr(res.eqlin, "marginals", None) is not None:
            lmp[s] = np.asarray(res.eqlin.marginals, dtype=float)[balance_row[s]]

    eta_star = float(x[i_eta])
    probs = np.array([sc.probability for sc in scens])
    risk_weights = (1.0 - beta) * probs
    if res.ineqlin is not None and getattr(res.ineqlin, "marginals", None) is not None:
        mu = np.abs(np.asarray(res.ineqlin.marginals, dtype=float)[epigraph_row])
        risk_weights = risk_weights + mu

    expected_cost = float(np.dot(probs, scen_cost))
    var, cvar = discrete_cvar(scen_cost, probs, alpha)
    total_demand = float(np.dot(probs, scen_energy))
    total_shed = float(np.dot(probs, shed.sum(axis=1))) * RISK_DAYS_PER_YEAR
    envelope = (1.0 - beta) * probs + beta * probs / (1.0 - alpha)
    envelope_excess = float(np.max(risk_weights - envelope))

    return RiskKernelSolution(
        beta=beta,
        alpha=alpha,
        objective=float(res.fun),
        expected_cost_per_year=expected_cost,
        cvar_per_year=cvar,
        var_per_year=eta_star,
        risk_adjusted_cost_per_year=float(np.dot(risk_weights, scen_cost)),
        expected_demand_mwh_per_year=total_demand,
        lcoe_per_mwh=expected_cost / total_demand,
        cvar_lcoe_per_mwh=cvar / total_demand,
        expected_unserved_mwh_per_year=total_shed,
        capacities_mw=caps,
        scenario_cost_per_year=scen_cost,
        scenario_energy_mwh_per_year=scen_energy,
        risk_weights=risk_weights,
        total_risk_weight=float(risk_weights.sum()),
        lmp_per_mwh=lmp,
        shed_mw=shed,
        dispatch_mw={
            "solar": solar,
            "wind": wind,
            "gas": gas,
            "charge": charge,
            "discharge": discharge,
            "soc": soc,
            "shed": shed,
        },
        n_variables=n_var,
        n_constraints=n_row + n_eq,
        duality_gap=abs(float(res.fun) - float(np.dot(risk_weights, scen_cost))),
        weight_envelope_excess=envelope_excess,
    )


def _capex_map() -> dict[str, float]:
    return {
        "solar": RISK_SOLAR_CAPEX_PER_MW_YR,
        "wind": RISK_WIND_CAPEX_PER_MW_YR,
        "gas": RISK_GAS_CAPEX_PER_MW_YR,
        "bess": RISK_BESS_CAPEX_PER_MW_YR,
    }


def risk_frontier(
    betas: Sequence[float] = RISK_BETA_SWEEP,
    regimes: Sequence[RiskScenario] | None = None,
) -> pd.DataFrame:
    """Sweep the risk-aversion weight and tabulate the risk/return frontier."""
    scens = validate_risk_regimes(DEFAULT_RISK_REGIMES if regimes is None else regimes)
    rows: list[dict[str, float]] = []
    for beta in betas:
        sol = solve_mean_cvar(beta, scens)
        rows.append(
            {
                "beta": beta,
                "expected_cost_per_year": sol.expected_cost_per_year,
                "cvar_per_year": sol.cvar_per_year,
                "var_per_year": sol.var_per_year,
                "risk_adjusted_cost_per_year": sol.risk_adjusted_cost_per_year,
                "lcoe_per_mwh": sol.lcoe_per_mwh,
                "cvar_lcoe_per_mwh": sol.cvar_lcoe_per_mwh,
                "risk_premium_per_mwh": sol.lcoe_per_mwh,
                "expected_unserved_mwh_per_year": sol.expected_unserved_mwh_per_year,
                "solar_mw": sol.capacities_mw["solar"],
                "wind_mw": sol.capacities_mw["wind"],
                "gas_mw": sol.capacities_mw["gas"],
                "bess_mw": sol.capacities_mw["bess"],
                "duality_gap": sol.duality_gap,
                "risk_weight_sum": sol.total_risk_weight,
            }
        )
    frame = pd.DataFrame(rows)
    frame["risk_premium_per_mwh"] = frame["lcoe_per_mwh"] - frame.loc[0, "lcoe_per_mwh"]
    frame["cvar_reduction_pct"] = 100.0 * (
        1.0 - frame["cvar_per_year"] / frame.loc[0, "cvar_per_year"]
    )
    return frame


@dataclass
class BirgeLouveauxBounds:
    """Wait-and-see / recourse / expected-EV bounds for the 8-regime problem."""

    wait_and_see_per_year: float
    recourse_per_year: float
    expected_ev_per_year: float
    evpi_per_year: float
    vss_per_year: float
    expected_ev_plan_mw: dict[str, float]
    recourse_plan_mw: dict[str, float]
    expected_value_plan_mw: dict[str, float]

    def ordering_ok(self) -> bool:
        """``WS <= RP <= EEV`` must hold for a consistent stochastic program."""
        tol = 1e-6 * max(1.0, abs(self.recourse_per_year))
        return (
            self.wait_and_see_per_year <= self.recourse_per_year + tol
            and self.recourse_per_year <= self.expected_ev_per_year + tol
            and self.evpi_per_year >= -tol
            and self.vss_per_year >= -tol
        )


def birge_louveaux_bounds(
    regimes: Sequence[RiskScenario] | None = None,
) -> BirgeLouveauxBounds:
    """Compute the classic stochastic-programming bounds for the risk-neutral plan.

    * *Wait-and-see* (WS): each future is solved with its own bespoke capacity plan.
    * *Recourse problem* (RP): one first-stage plan hedged across all futures.
    * *Expected result of the expected-value problem* (EEV): the plan built for the
      probability-weighted mean future, then evaluated against every future.
    """
    scens = validate_risk_regimes(DEFAULT_RISK_REGIMES if regimes is None else regimes)
    ws = 0.0
    for sc in scens:
        prof = sc.profiles()
        single = solve_mean_cvar(
            0.0, (_MeanFuture(prof["load"], prof["solar_cf"], prof["wind_cf"]),)
        )
        ws += sc.probability * single.expected_cost_per_year
    rp = solve_mean_cvar(0.0, scens)

    mean_load = np.zeros(RISK_HOURS)
    mean_solar = np.zeros(RISK_HOURS)
    mean_wind = np.zeros(RISK_HOURS)
    for sc in scens:
        prof = sc.profiles()
        mean_load += sc.probability * prof["load"]
        mean_solar += sc.probability * prof["solar_cf"]
        mean_wind += sc.probability * prof["wind_cf"]
    ev_plan = _solve_mean_future(mean_load, mean_solar, mean_wind)
    eev = 0.0
    for sc in scens:
        prof = sc.profiles()
        evaluated = solve_mean_cvar(
            0.0,
            (_MeanFuture(prof["load"], prof["solar_cf"], prof["wind_cf"]),),
            fixed_capacities_mw=ev_plan,
        )
        eev += sc.probability * evaluated.expected_cost_per_year

    return BirgeLouveauxBounds(
        wait_and_see_per_year=ws,
        recourse_per_year=rp.expected_cost_per_year,
        expected_ev_per_year=eev,
        evpi_per_year=rp.expected_cost_per_year - ws,
        vss_per_year=eev - rp.expected_cost_per_year,
        expected_ev_plan_mw=ev_plan,
        recourse_plan_mw=dict(rp.capacities_mw),
        expected_value_plan_mw=dict(ev_plan),
    )


def _solve_mean_future(
    mean_load: np.ndarray,
    mean_solar: np.ndarray,
    mean_wind: np.ndarray,
) -> dict[str, float]:
    """Solve the expected-value problem for an explicit mean load / VRE future."""
    regime = _MeanFuture(mean_load, mean_solar, mean_wind)
    solution = solve_mean_cvar(0.0, (regime,))
    return dict(solution.capacities_mw)


class _MeanFuture:
    """Scenario whose hourly profiles are the probability-weighted expected future."""

    name = "Mean_Future"
    probability = 1.0

    def __init__(self, load: np.ndarray, solar_cf: np.ndarray, wind_cf: np.ndarray) -> None:
        self._load = np.asarray(load, dtype=float)
        self._solar = np.asarray(solar_cf, dtype=float)
        self._wind = np.asarray(wind_cf, dtype=float)

    def profiles(self) -> dict[str, np.ndarray]:
        """Return the mean load and mean capacity-factor profiles."""
        return {"load": self._load, "solar_cf": self._solar, "wind_cf": self._wind}


# ---------------------------------------------------------------------------
# Reduced-order surrogate of the (WACC, VRE-CapEx) surface
# ---------------------------------------------------------------------------
def _bracket_axis(grid: Sequence[float], value: float) -> tuple[int, int, float]:
    """Return ``(i0, i1, weight)`` for linear interpolation with edge clamping."""
    axis = np.asarray(grid, dtype=float)
    if axis.size < 2:
        raise ValueError("interpolation axis needs at least two nodes")
    if value <= axis[0]:
        return 0, 1, 0.0
    if value >= axis[-1]:
        return axis.size - 2, axis.size - 1, 1.0
    i1 = int(np.searchsorted(axis, value))
    i0 = i1 - 1
    span = axis[i1] - axis[i0]
    return i0, i1, float((value - axis[i0]) / span)


def bilinear_interpolate(
    wacc_grid: Sequence[float],
    capex_grid: Sequence[float],
    values: np.ndarray,
    wacc: float,
    capex_scale: float,
) -> float:
    """Bilinearly interpolate a 2-D response surface (clamped at the edges)."""
    grid = np.asarray(values, dtype=float)
    if grid.shape != (len(wacc_grid), len(capex_grid)):
        raise ValueError("values must have shape (len(wacc_grid), len(capex_grid))")
    i0, i1, tx = _bracket_axis(wacc_grid, wacc)
    j0, j1, ty = _bracket_axis(capex_grid, capex_scale)
    low = (1.0 - ty) * grid[i0, j0] + ty * grid[i0, j1]
    high = (1.0 - ty) * grid[i1, j0] + ty * grid[i1, j1]
    return float((1.0 - tx) * low + tx * high)


def interpolate_surface(
    surface: dict[str, Any],
    section: str,
    wacc: float,
    capex_scale: float,
) -> float:
    """Interpolate a named 2-D field of a workbench surface payload."""
    return bilinear_interpolate(
        surface["wacc_grid"],
        surface["capex_grid"],
        np.asarray(surface[section], dtype=float),
        wacc,
        capex_scale,
    )


def validate_surrogate(
    *,
    holdout_wacc: Sequence[float] = (0.025, 0.055, 0.090, 0.130),
    holdout_capex: Sequence[float] = (0.65, 0.875, 1.175, 1.550),
    surface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the bilinear surrogate against exact solves at unseen grid nodes."""
    payload = build_surface() if surface is None else surface
    profiles = load_profiles(DEFAULT_PROFILE_PATH)
    specs = default_financial_specs()
    rows: list[dict[str, float]] = []
    for wacc in holdout_wacc:
        for capex in holdout_capex:
            techs = build_wacc_technologies(wacc, specs)
            for name in ("Solar_PV", "Wind_Turbine"):
                techs[name]["capital_cost"] = float(techs[name]["capital_cost"]) * capex
            result = run_scenario(profiles, techs)
            dual = analyze_market_duality_and_lcoe(result, profiles=profiles, technologies=techs)
            rows.append(
                {
                    "wacc": float(wacc),
                    "capex_scale": float(capex),
                    "exact_lcoe": float(dual.system_lcoe_per_mwh),
                    "surrogate_lcoe": interpolate_surface(payload, "lcoe", wacc, capex),
                }
            )
    frame = pd.DataFrame(rows)
    residual = frame["surrogate_lcoe"] - frame["exact_lcoe"]
    ss_res = float((residual**2).sum())
    ss_tot = float(((frame["exact_lcoe"] - frame["exact_lcoe"].mean()) ** 2).sum())
    return {
        "holdout_wacc": [float(v) for v in holdout_wacc],
        "holdout_capex_scale": [float(v) for v in holdout_capex],
        "samples": frame.to_dict(orient="records"),
        "max_abs_error_per_mwh": float(residual.abs().max()),
        "rmse_per_mwh": float(np.sqrt(ss_res / len(frame))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "max_rel_error_pct": float((residual.abs() / frame["exact_lcoe"]).max() * 100.0),
        "n_solves": int(len(frame)),
    }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def _r(value: float, digits: int = 4) -> float:
    """Round a float for compact, stable JSON output."""
    return float(round(float(value), digits))


def _series(values: Sequence[float] | np.ndarray, digits: int = 3) -> list[float]:
    """Round an iterable of floats to a JSON-friendly list."""
    return [round(float(v), digits) for v in np.asarray(values, dtype=float).ravel()]


def _column(frame: pd.DataFrame, name: str, length: int) -> np.ndarray:
    """Return a dispatch column as a float array, zero-filled when absent."""
    if name in frame.columns:
        return np.asarray(frame[name], dtype=float)
    return np.zeros(length, dtype=float)


def _collect_silently(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with solver chatter suppressed (HiGHS writes to stdout)."""
    import contextlib
    import io
    import logging

    buffer = io.StringIO()
    previous = logging.getLogger("linopy").level
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            return fn(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)
        logging.getLogger("linopy").setLevel(previous)


def build_surface() -> dict[str, Any]:
    """Solve the exact 5x5 (WACC, VRE-CapEx) surface and capture hourly dispatch."""
    profiles = load_profiles(DEFAULT_PROFILE_PATH)
    specs = default_financial_specs()
    lcoe: list[list[float]] = []
    vre_share: list[list[float]] = []
    curtail: list[list[float]] = []
    peak_lmp: list[list[float]] = []
    capex_cost: list[list[float]] = []
    opex_cost: list[list[float]] = []
    cases: list[dict[str, Any]] = []
    duality_gap = 0.0
    zeros = pd.Series(np.zeros(len(profiles), dtype=float))
    for wacc in WACC_GRID:
        row_lcoe, row_vre, row_curt, row_peak, row_cx, row_ox = [], [], [], [], [], []
        for capex in VRE_CAPEX_GRID:
            techs = build_wacc_technologies(wacc, specs)
            for name in ("Solar_PV", "Wind_Turbine"):
                techs[name]["capital_cost"] = float(techs[name]["capital_cost"]) * capex
            result = _collect_silently(run_scenario, profiles, techs)
            dm = result.dispatch_mw
            dual = _collect_silently(analyze_market_duality_and_lcoe, result, profiles, techs)
            vre_energy = float(
                result.energy_mwh_per_day.get("Solar_PV", 0.0)
                + result.energy_mwh_per_day.get("Wind_Turbine", 0.0)
            )
            curtailed = float(result.curtailment_mwh_per_day.sum())
            available = vre_energy + curtailed
            row_lcoe.append(_r(dual.system_lcoe_per_mwh))
            row_vre.append(_r(100.0 * vre_energy / result.demand_mwh_per_day, 3))
            row_curt.append(_r(100.0 * curtailed / available if available > 0 else 0.0, 3))
            row_peak.append(_r(dual.peak_lmp_per_mwh, 3))
            row_cx.append(_r(result.capex_per_year, 1))
            row_ox.append(_r(result.opex_per_year, 1))
            duality_gap = max(duality_gap, abs(dual.strong_duality_gap))
            cases.append(
                {
                    "wacc": _r(wacc),
                    "capex_scale": _r(capex, 3),
                    "lcoe": _r(dual.system_lcoe_per_mwh, 3),
                    "caps": {
                        "solar": _r(float(result.p_nom_mw.get("Solar_PV", 0.0)), 2),
                        "wind": _r(float(result.p_nom_mw.get("Wind_Turbine", 0.0)), 2),
                        "gas": _r(float(result.p_nom_mw.get("Gas_Peaker", 0.0)), 2),
                    },
                    "capex_per_year": _r(result.capex_per_year, 1),
                    "opex_per_year": _r(result.opex_per_year, 1),
                    "curtailment_pct": _r(
                        100.0 * curtailed / available if available > 0 else 0.0, 3
                    ),
                    "load": _series(profiles["load_mw"]),
                    "solar": _series(dm.select_dtypes("number").get("Solar_PV", zeros)),
                    "wind": _series(dm.select_dtypes("number").get("Wind_Turbine", zeros)),
                    "gas": _series(dm.select_dtypes("number").get("Gas_Peaker", zeros)),
                    "lmp": _series(result.lmp_per_mwh.fillna(0.0), 3),
                }
            )
        lcoe.append(row_lcoe)
        vre_share.append(row_vre)
        curtail.append(row_curt)
        peak_lmp.append(row_peak)
        capex_cost.append(row_cx)
        opex_cost.append(row_ox)
    return {
        "wacc_grid": [float(v) for v in WACC_GRID],
        "capex_grid": [float(v) for v in VRE_CAPEX_GRID],
        "lcoe": lcoe,
        "vre_share_pct": vre_share,
        "curtailment_pct": curtail,
        "peak_lmp_per_mwh": peak_lmp,
        "capex_per_year": capex_cost,
        "opex_per_year": opex_cost,
        "hours": list(range(len(profiles))),
        "cases": cases,
        "duality_gap": duality_gap,
    }


def build_storage() -> dict[str, Any]:
    """Sweep Dunkelflaute lull duration and capture one chronological dispatch."""
    durations = STORAGE_DURATIONS
    sweep = _collect_silently(sweep_dunkelflaute_duration, durations)
    regimes = _collect_silently(compare_storage_adequacy_regimes)
    profiles = generate_dunkelflaute_profiles(lull_duration_hours=48)
    dispatch = _collect_silently(
        run_storage_scenario, profiles, include_bess=True, include_h2=True, max_gas_mw=0.0
    )
    zeros_n = pd.Series(np.zeros(dispatch.n_snapshots, dtype=float))
    gm = dispatch.generator_dispatch_mw
    bess_lcoe = sweep["bess_only_lcoe"].to_numpy(dtype=float)
    h2_lcoe = sweep["h2_only_lcoe"].to_numpy(dtype=float)
    hours = [float(v) for v in sweep["lull_hours"]]
    crossover = float("nan")
    for k in range(1, len(hours)):
        delta_prev = bess_lcoe[k - 1] - h2_lcoe[k - 1]
        delta_next = bess_lcoe[k] - h2_lcoe[k]
        if delta_prev < 0.0 <= delta_next:
            frac = -delta_prev / (delta_next - delta_prev)
            crossover = hours[k - 1] + frac * (hours[k] - hours[k - 1])
            break
    return {
        "durations": hours,
        "bess_only_lcoe": [_r(v, 4) for v in bess_lcoe],
        "h2_only_lcoe": [_r(v, 4) for v in h2_lcoe],
        "hybrid_lcoe": [_r(v, 4) for v in sweep["hybrid_lcoe"]],
        "hybrid_bess_mwh": [_r(v, 2) for v in sweep["hybrid_bess_mwh"]],
        "hybrid_h2_mwh_e_equiv": [_r(v, 2) for v in sweep["hybrid_h2_mwh_e_equiv"]],
        "crossover_hours": _r(crossover, 2) if math.isfinite(crossover) else None,
        "regimes": [
            {
                "id": idx,
                "label": str(row["label"]),
                "solar_mw": _r(row["solar_mw"], 2),
                "wind_mw": _r(row["wind_mw"], 2),
                "gas_mw": _r(row["gas_mw"], 2),
                "bess_mw": _r(row["bess_mw"], 2),
                "bess_mwh": _r(row["bess_mwh"], 2),
                "h2_electrolyzer_mw_e": _r(row["h2_electrolyzer_mw_e"], 2),
                "h2_turbine_mw_e": _r(row["h2_turbine_mw_e"], 2),
                "h2_store_mwh_h2": _r(row["h2_store_mwh_h2"], 2),
                "system_lcoe_per_mwh": _r(row["system_lcoe_per_mwh"], 4),
                "annual_curtailment_mwh": _r(row["annual_curtailment_mwh"], 1),
            }
            for idx, row in regimes.reset_index().iterrows()
        ],
        "dispatch": {
            "hours": list(range(dispatch.n_snapshots)),
            "load": _series(profiles["load_mw"]),
            "solar": _series(gm.select_dtypes("number").get("Solar_PV", zeros_n)),
            "wind": _series(gm.select_dtypes("number").get("Wind_Turbine", zeros_n)),
            "gas": _series(gm.select_dtypes("number").get("Gas_Peaker", zeros_n)),
            "bess_charge": _series(dispatch.bess_charge_mw, 2),
            "bess_discharge": _series(dispatch.bess_discharge_mw, 2),
            "bess_soc": _series(dispatch.bess_soc_mwh, 2),
            "h2_charge": _series(dispatch.h2_charge_mw_e, 2),
            "h2_discharge": _series(dispatch.h2_discharge_mw_e, 2),
            "h2_soc": _series(dispatch.h2_soc_mwh_h2, 2),
            "curtailment": _series(dispatch.curtailment_mw, 2),
            "lmp": _series(dispatch.lmp_elec_per_mwh, 3),
        },
        "duality_gap": float(abs(dispatch.strong_duality_gap)),
        "bess_cycles_per_year": _r(dispatch.bess_equiv_cycles_per_year, 2),
        "h2_cycles_per_year": _r(dispatch.h2_equiv_cycles_per_year, 2),
    }


def build_carbon() -> dict[str, Any]:
    """Evaluate the CO2 cap/tax Pareto frontier, MACC and fuel-switching mix."""
    frame = _collect_silently(sweep_co2_pareto_and_macc)
    net_zero = frame.loc[frame["abatement_target_pct"].idxmax()]
    bau = frame.loc[frame["abatement_target_pct"].idxmin()]
    return {
        "targets_pct": [_r(v, 3) for v in frame["abatement_target_pct"]],
        "emissions_tco2": [_r(v, 1) for v in frame["emissions_tco2"]],
        "cap_tco2": [_r(v, 1) for v in frame["co2_cap_tco2"]],
        "resource_lcoe_per_mwh": [_r(v, 4) for v in frame["resource_lcoe_per_mwh"]],
        "shadow_price_per_tco2": [_r(v, 4) for v in frame["marginal_abatement_cost_per_tco2"]],
        "average_cost_per_tco2": [_r(v, 4) for v in frame["average_abatement_cost_per_tco2"]],
        "grid_intensity_kgco2_per_mwh": [_r(v, 3) for v in frame["grid_intensity_kgco2_per_mwh"]],
        "total_cost_per_year": [_r(v, 1) for v in frame["total_resource_cost_per_year"]],
        "mix": {
            "coal_mw": [_r(v, 2) for v in frame["coal_mw"]],
            "ccgt_mw": [_r(v, 2) for v in frame["ccgt_mw"]],
            "ocgt_mw": [_r(v, 2) for v in frame["ocgt_mw"]],
            "gas_ccs_mw": [_r(v, 2) for v in frame["gas_ccs_mw"]],
            "solar_mw": [_r(v, 2) for v in frame["solar_mw"]],
            "wind_mw": [_r(v, 2) for v in frame["wind_mw"]],
            "bess_mwh": [_r(v, 2) for v in frame["bess_mwh"]],
            "h2_mwh": [_r(v, 2) for v in frame["h2_mwh"]],
        },
        "bau_emissions_tco2": _r(bau["emissions_tco2"], 1),
        "bau_lcoe_per_mwh": _r(bau["resource_lcoe_per_mwh"], 4),
        "net_zero_shadow_price_per_tco2": _r(net_zero["marginal_abatement_cost_per_tco2"], 4),
        "net_zero_lcoe_per_mwh": _r(net_zero["resource_lcoe_per_mwh"], 4),
        "max_saddle_gap": float(np.max(np.abs(frame["lagrangian_saddle_gap"]))),
        "saddle_identity_gap": float(np.max(np.abs(frame["lagrangian_saddle_gap"]))),
    }


FLEX_CONFIGS: tuple[tuple[str, bool, bool, bool, bool], ...] = (
    ("1_Relaxed_NoReserves", False, False, False, False),
    ("2_MustRun_And_Ramping", True, True, False, False),
    ("3_Inflexible_With_Reserves", True, True, True, False),
    ("4_Flexible_BESS_CoOptimized", True, True, True, True),
)


def build_flexibility() -> dict[str, Any]:
    """Re-solve the four duck-curve regimes and keep their hourly price series."""
    profiles = make_duck_curve_profiles()
    specs = default_flexible_fleet_specs()
    policy_on = SpinningReservePolicy()
    policy_off = SpinningReservePolicy(enabled=False)
    rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    for name, ramp, must_run, reserve, bess in FLEX_CONFIGS:
        result = _collect_silently(
            run_flexibility_scenario,
            profiles,
            specs=specs,
            enforce_ramping=ramp,
            enforce_must_run=must_run,
            include_bess=bess,
            reserve_policy=policy_on if reserve else policy_off,
        )
        rows.append(
            {
                "regime": name,
                "system_lcoe_per_mwh": _r(result.system_lcoe_per_mwh, 4),
                "total_annual_cost": _r(result.total_annual_cost, 1),
                "load_weighted_lmp_per_mwh": _r(result.load_weighted_lmp_per_mwh, 4),
                "load_weighted_reserve_price_per_mw_h": _r(
                    result.load_weighted_reserve_price_per_mw_h, 4
                ),
                "max_lmp_per_mwh": _r(float(result.lmp_per_mwh.max()), 3),
                "max_reserve_price_per_mw_h": _r(float(result.reserve_price_per_mw_h.max()), 3),
                "baseload_mw": _r(float(result.p_nom_mw.get("Baseload_Thermal", 0.0)), 2),
                "ccgt_mw": _r(float(result.p_nom_mw.get("CCGT_MidMerit", 0.0)), 2),
                "ocgt_mw": _r(float(result.p_nom_mw.get("OCGT_FastPeaker", 0.0)), 2),
                "solar_mw": _r(float(result.p_nom_mw.get("Solar_PV", 0.0)), 2),
                "wind_mw": _r(float(result.p_nom_mw.get("Wind_Turbine", 0.0)), 2),
                "vre_total_mw": _r(
                    float(result.p_nom_mw.get("Solar_PV", 0.0))
                    + float(result.p_nom_mw.get("Wind_Turbine", 0.0)),
                    2,
                ),
                "bess_e_nom_mwh": _r(result.bess_e_nom_mwh, 2),
                "vre_curtailment_pct": _r(result.vre_curtailment_pct, 3),
                "max_evening_net_load_ramp_mw_per_h": _r(
                    result.max_evening_net_load_ramp_mw_per_h, 2
                ),
                "strong_duality_revenue_gap": float(result.strong_duality_revenue_gap),
            }
        )
        series.append(
            {
                "regime": name,
                "net_load": _series(result.net_load_mw, 2),
                "lmp": _series(result.lmp_per_mwh, 3),
                "reserve_price": _series(result.reserve_price_per_mw_h, 3),
                "bess_discharge": _series(result.bess_discharge_mw, 2),
                "reserve_requirement": _series(result.reserve_requirement_mw, 2),
            }
        )
    proof = _collect_silently(verify_ramp_induced_negative_lmp)
    return {
        "hours": list(range(len(profiles))),
        "regimes": rows,
        "series": series,
        "negative_lmp_proof": {
            "min_generator_marginal_cost": _r(proof.min_generator_marginal_cost, 4),
            "trough_lmp_per_mwh": _r(proof.trough_lmp_per_mwh, 4),
            "analytical_trough_lmp_per_mwh": _r(proof.analytical_trough_lmp_per_mwh, 4),
            "formula_error": float(proof.abs_lmp_formula_error),
            "lmp_series": _series(proof.lmp_series_per_mwh, 3),
        },
        "max_duality_gap": max(float(r["strong_duality_revenue_gap"]) for r in rows),
    }


GRID_CONFIGS: tuple[tuple[str, str, bool], ...] = (
    ("1_CopperPlate_SingleBus", "copper_plate", False),
    ("2_Congested_AC_KVL_FixedGrid", "ac_kvl", False),
    ("3_Controllable_HVDC_Transport", "hvdc_transport", False),
    ("4_CoOptimized_AC_TEP", "ac_kvl", True),
)


def build_grid() -> dict[str, Any]:
    """Re-solve the four meshed-network regimes with hourly flows and zonal prices."""
    profiles = make_four_zone_profiles()
    corridors = default_transmission_corridors()
    rows: list[dict[str, Any]] = []
    flows: list[dict[str, Any]] = []
    prices: list[dict[str, Any]] = []
    for label, mode, extendable in GRID_CONFIGS:
        result = _collect_silently(
            run_transmission_scenario, profiles, mode=mode, extendable_lines=extendable
        )
        rows.append(
            {
                "regime": label,
                "system_lcoe_per_mwh": _r(result.system_lcoe_per_mwh, 4),
                "mean_lmp_spread_per_mwh": _r(result.mean_spatial_lmp_spread_per_mwh, 4),
                "max_lmp_spread_per_mwh": _r(result.max_spatial_lmp_spread_per_mwh, 3),
                "north_wind_mw": _r(float(result.gen_p_nom_mw["North_Offshore_Wind"]), 2),
                "west_solar_mw": _r(float(result.gen_p_nom_mw["West_Utility_Solar"]), 2),
                "east_ocgt_mw": _r(float(result.gen_p_nom_mw["East_Metro_OCGT"]), 2),
                "total_line_mw": _r(float(result.line_s_nom_opt_mw.sum()), 2),
                "added_line_mw": _r(float(result.line_expansion_mw.sum()), 2),
                "annual_congestion_rent": _r(result.annual_congestion_rent, 1),
                "annual_line_capex_cost": _r(result.annual_line_capex_cost, 1),
                "vre_curtailment_pct": _r(result.vre_curtailment_pct, 3),
                "gen_revenue_adequacy_gap": float(result.generator_revenue_adequacy_gap),
                "merchandising_surplus_gap": float(result.merchandising_surplus_identity_max_gap),
            }
        )
        flows.append(
            {
                "regime": label,
                "corridor": list(result.line_flows_mw.columns),
                "mw": [_series(result.line_flows_mw[c], 2) for c in result.line_flows_mw.columns],
            }
        )
        prices.append(
            {
                "regime": label,
                "zone": list(result.zonal_lmp_per_mwh.columns),
                "per_mwh": [
                    _series(result.zonal_lmp_per_mwh[c], 3)
                    for c in result.zonal_lmp_per_mwh.columns
                ],
            }
        )
    sweep = _collect_silently(sweep_transmission_capacity_scale)
    braess = _collect_silently(verify_braess_paradox_dcopf)
    ptdf = compute_ptdf_matrix()
    return {
        "zones": list(ZONES_4BUS),
        "hours": list(range(len(profiles))),
        "corridors": [
            {
                "name": spec.name,
                "bus0": spec.bus0,
                "bus1": spec.bus1,
                "length_km": _r(spec.length_km, 1),
                "x_ohm": _r(spec.x_ohm, 1),
                "s_nom_mw": _r(spec.s_nom_mw, 1),
                "capital_cost_per_mw_yr": _r(spec.capital_cost_per_mw_yr, 1),
            }
            for spec in corridors.values()
        ],
        "ptdf": [[_r(v, 4) for v in row] for row in ptdf.to_numpy(dtype=float)],
        "regimes": rows,
        "flows": flows,
        "prices": prices,
        "scale_sweep": {
            "scale": [_r(v, 3) for v in sweep["capacity_scale"]],
            "total_corridor_mw": [_r(v, 1) for v in sweep["total_corridor_mw"]],
            "lcoe_per_mwh": [_r(v, 4) for v in sweep["system_lcoe_per_mwh"]],
            "mean_spread_per_mwh": [_r(v, 4) for v in sweep["mean_lmp_spread_per_mwh"]],
            "congestion_rent": [_r(v, 1) for v in sweep["annual_congestion_rent"]],
            "north_wind_mw": [_r(v, 2) for v in sweep["north_wind_mw"]],
            "east_ocgt_mw": [_r(v, 2) for v in sweep["east_ocgt_mw"]],
        },
        "braess": {
            "base_cost_per_h": _r(braess.base_4line_cost_per_h, 2),
            "ac_bridge_cost_per_h": _r(braess.ac_5line_cost_per_h, 2),
            "hvdc_cost_per_h": _r(braess.hvdc_5line_cost_per_h, 2),
            "base_transfer_mw": _r(braess.base_transfer_mw, 2),
            "ac_bridge_transfer_mw": _r(braess.ac_5line_transfer_mw, 2),
            "hvdc_transfer_mw": _r(braess.hvdc_5line_transfer_mw, 2),
            "cost_increase_pct": _r(braess.braess_cost_increase_pct, 3),
        },
        "max_merchandising_gap": max(float(r["merchandising_surplus_gap"]) for r in rows),
    }


def build_risk() -> dict[str, Any]:
    """Sweep the mean-CVaR frontier, capture tail dispatch and stochastic bounds."""
    frontier = risk_frontier(RISK_BETA_SWEEP)
    bounds = birge_louveaux_bounds()
    solutions = {beta: solve_mean_cvar(beta) for beta in (0.0, 0.5, 1.0)}
    showcase = solve_mean_cvar(0.35)
    envelope_cap = [_r(r.probability / (1.0 - RISK_ALPHA), 4) for r in DEFAULT_RISK_REGIMES]
    dispatch = {
        "beta": showcase.beta,
        "scenario": [r.name for r in DEFAULT_RISK_REGIMES],
        "solar": [_series(v, 2) for v in showcase.dispatch_mw["solar"]],
        "wind": [_series(v, 2) for v in showcase.dispatch_mw["wind"]],
        "gas": [_series(v, 2) for v in showcase.dispatch_mw["gas"]],
        "discharge": [_series(v, 2) for v in showcase.dispatch_mw["discharge"]],
        "shed": [_series(v, 3) for v in showcase.dispatch_mw["shed"]],
        "lmp": [_series(v, 3) for v in showcase.lmp_per_mwh],
    }
    return {
        "alpha": RISK_ALPHA,
        "betas": [float(v) for v in RISK_BETA_SWEEP],
        "hours": list(range(RISK_HOURS)),
        "expected_cost_per_year": [_r(v, 1) for v in frontier["expected_cost_per_year"]],
        "cvar_per_year": [_r(v, 1) for v in frontier["cvar_per_year"]],
        "var_per_year": [_r(v, 1) for v in frontier["var_per_year"]],
        "risk_adjusted_cost_per_year": [_r(v, 1) for v in frontier["risk_adjusted_cost_per_year"]],
        "lcoe_per_mwh": [_r(v, 4) for v in frontier["lcoe_per_mwh"]],
        "cvar_lcoe_per_mwh": [_r(v, 4) for v in frontier["cvar_lcoe_per_mwh"]],
        "risk_premium_per_mwh": [_r(v, 4) for v in frontier["risk_premium_per_mwh"]],
        "cvar_reduction_pct": [_r(v, 3) for v in frontier["cvar_reduction_pct"]],
        "eens_mwh_per_year": [_r(v, 2) for v in frontier["expected_unserved_mwh_per_year"]],
        "capacities_mw": {
            "solar": [_r(v, 2) for v in frontier["solar_mw"]],
            "wind": [_r(v, 2) for v in frontier["wind_mw"]],
            "gas": [_r(v, 2) for v in frontier["gas_mw"]],
            "bess": [_r(v, 2) for v in frontier["bess_mw"]],
        },
        "duality_gap_max": float(np.max(np.abs(frontier["duality_gap"]))),
        "risk_weight_sum_error_max": float(np.max(np.abs(frontier["risk_weight_sum"] - 1.0))),
        "regimes": [
            {
                "name": r.name,
                "probability": _r(r.probability, 4),
                "load_adder_mw": _r(r.load_adder_mw, 2),
                "solar_scale": _r(r.solar_scale, 3),
                "wind_cap_cf": _r(r.wind_cap_cf, 3),
                "wind_floor_cf": _r(r.wind_floor_cf, 3),
                "envelope_cap": envelope_cap[i],
                "profiles": {
                    "load": _series(r.profiles()["load"], 2),
                    "solar_cf": _series(r.profiles()["solar_cf"], 3),
                    "wind_cf": _series(r.profiles()["wind_cf"], 3),
                },
            }
            for i, r in enumerate(DEFAULT_RISK_REGIMES)
        ],
        "risk_weights": [
            {
                "beta": _r(beta, 2),
                "weights": [_r(v, 4) for v in solutions[beta].risk_weights],
                "var_per_year": _r(solutions[beta].var_per_year, 1),
            }
            for beta in (0.0, 0.5, 1.0)
        ],
        "dispatch": dispatch,
        "birge_louveaux": {
            "wait_and_see_per_year": _r(bounds.wait_and_see_per_year, 1),
            "recourse_per_year": _r(bounds.recourse_per_year, 1),
            "expected_ev_per_year": _r(bounds.expected_ev_per_year, 1),
            "evpi_per_year": _r(bounds.evpi_per_year, 1),
            "vss_per_year": _r(bounds.vss_per_year, 1),
            "ordering_ok": bool(bounds.ordering_ok()),
            "expected_value_plan_mw": {
                k: _r(v, 2) for k, v in bounds.expected_value_plan_mw.items()
            },
            "recourse_plan_mw": {k: _r(v, 2) for k, v in bounds.recourse_plan_mw.items()},
        },
        "kernel_size": {
            "variables": showcase.n_variables,
            "constraints": showcase.n_constraints,
        },
        "weight_envelope_excess_max": float(
            max(s.weight_envelope_excess for s in solutions.values())
        ),
    }


def build_verification(
    surface: dict[str, Any],
    storage: dict[str, Any],
    carbon: dict[str, Any],
    flexibility: dict[str, Any],
    grid: dict[str, Any],
    risk: dict[str, Any],
) -> list[dict[str, Any]]:
    """Assemble the cross-module invariant table displayed in the workbench."""
    return [
        {
            "module": "M1 KKT duality",
            "identity": "System LCOE = load-weighted LMP",
            "residual": float(surface["duality_gap"]),
            "unit": "currency/MWh",
            "tolerance": 1e-6,
        },
        {
            "module": "M2 storage duality",
            "identity": "Capital + fuel = energy market revenue",
            "residual": float(storage["duality_gap"]),
            "unit": "currency/yr",
            "tolerance": 1e-4,
        },
        {
            "module": "M3 carbon saddle point",
            "identity": "J_tax(mu*) = J_cap(E) + mu* E",
            "residual": float(carbon["saddle_identity_gap"]),
            "unit": "currency/yr",
            "tolerance": 1e-3,
        },
        {
            "module": "M4 reserve adequacy",
            "identity": "LCOE = energy + reserve revenue",
            "residual": float(flexibility["max_duality_gap"]),
            "unit": "currency/yr",
            "tolerance": 1e-4,
        },
        {
            "module": "M5 merchandising surplus",
            "identity": "Congestion rent = line CapEx",
            "residual": float(grid["max_merchandising_gap"]),
            "unit": "currency/yr",
            "tolerance": 1e-4,
        },
        {
            "module": "M7 CVaR duality",
            "identity": "objective = sum(q* Z)",
            "residual": float(risk["duality_gap_max"]),
            "unit": "currency/yr",
            "tolerance": 1e-3,
        },
        {
            "module": "M7 risk kernel",
            "identity": "sum(q*) = 1 and q* <= pi/(1-alpha)",
            "residual": float(
                max(risk["risk_weight_sum_error_max"], risk["weight_envelope_excess_max"])
            ),
            "unit": "probability",
            "tolerance": 1e-9,
        },
    ]


def _test_inventory(root: Path) -> list[dict[str, Any]]:
    """Count ``def test_*`` functions per test module (static, no collection)."""
    inventory: list[dict[str, Any]] = []
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return inventory
    pattern = re.compile(r"^\s*def test_")
    for path in sorted(tests_dir.glob("test_*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        count = len([line for line in lines if pattern.match(line)])
        inventory.append({"file": path.name, "tests": count})
    return inventory


def build_dataset(*, include_surrogate: bool = True) -> dict[str, Any]:
    """Run every module and assemble the ``etm-workbench/1`` dataset."""
    started = time.time()
    root = _repo_root()
    reset_solve_tally()
    with _CountPypsaSolves():
        surface = build_surface()
        storage = build_storage()
        carbon = build_carbon()
        flexibility = build_flexibility()
        grid = build_grid()
        risk = build_risk()
        surrogate = (
            validate_surrogate(surface=surface)
            if include_surrogate
            else {
                "holdout_wacc": [],
                "holdout_capex_scale": [],
                "samples": [],
                "max_abs_error_per_mwh": None,
                "rmse_per_mwh": None,
                "r2": None,
                "max_rel_error_pct": None,
                "n_solves": 0,
            }
        )
    tally = solve_tally()
    reference = interpolate_surface(surface, "lcoe", REFERENCE_WACC, 1.0)
    vre_at_reference = interpolate_surface(surface, "vre_share_pct", REFERENCE_WACC, 1.0)
    verification = build_verification(surface, storage, carbon, flexibility, grid, risk)
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "meta": {
            "package_version": __version__,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "pypsa": pypsa.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "runtime_seconds": round(time.time() - started, 2),
            "n_solves": int(tally["pypsa"] + tally["lp"]),
            "n_pypsa_solves": int(tally["pypsa"]),
            "n_explicit_lp_solves": int(tally["lp"]),
            "currency": "unspecified currency units (currency/yr, currency/MWh)",
            "reference_wacc": REFERENCE_WACC,
            "wacc_sweep_reference": [float(v) for v in DEFAULT_WACC_SWEEP],
        },
        "kpi": {
            "lcoe_at_reference_per_mwh": _r(reference, 4),
            "vre_share_at_reference_pct": _r(vre_at_reference, 2),
            "net_zero_carbon_price_per_tco2": carbon["net_zero_shadow_price_per_tco2"],
            "congestion_rent_per_year": grid["regimes"][3]["annual_congestion_rent"],
            "risk_premium_at_beta_half": _r(
                float(np.interp(0.5, risk["betas"], risk["risk_premium_per_mwh"])), 3
            ),
            "surrogate_r2": _r(surrogate["r2"], 6) if surrogate["r2"] is not None else None,
            "max_abs_surrogate_error_per_mwh": (
                _r(surrogate["max_abs_error_per_mwh"], 4)
                if surrogate["max_abs_error_per_mwh"] is not None
                else None
            ),
        },
        "surface": surface,
        "storage": storage,
        "carbon": carbon,
        "flexibility": flexibility,
        "grid": grid,
        "risk": risk,
        "surrogate": surrogate,
        "verification": verification,
        "tests": _test_inventory(root),
    }
    return payload


def dataset_digest(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON encoding of the dataset."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_page(template: str, payload: dict[str, Any]) -> str:
    """Inject the dataset into the HTML template and return the single-file page."""
    if payload.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"dataset schema {payload.get('schema')!r} does not match {SCHEMA_VERSION!r}"
        )
    if TEMPLATE_DATA_TOKEN not in template or TEMPLATE_SHA_TOKEN not in template:
        raise ValueError("template is missing the data or digest placeholder token")
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
    digest = dataset_digest(payload)
    page = template.replace(TEMPLATE_SHA_TOKEN, digest).replace(TEMPLATE_DATA_TOKEN, blob)
    if TEMPLATE_DATA_TOKEN in page or TEMPLATE_SHA_TOKEN in page:
        raise ValueError("placeholder token replacement was incomplete")
    return page


def write_workbench(
    root: Path | None = None,
    *,
    include_surrogate: bool = True,
    regenerate_data: bool = True,
) -> dict[str, Any]:
    """Build (optionally) and render the workbench artifact; returns a summary."""
    root = root or _repo_root()
    html_path = artifact_path(root)
    json_path = data_path(root)
    if regenerate_data or not json_path.is_file():
        payload = build_dataset(include_surrogate=include_surrogate)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    else:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    page = render_page(template_path().read_text(encoding="utf-8"), payload)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    return {
        "html": str(html_path),
        "json": str(json_path),
        "html_bytes": len(page.encode("utf-8")),
        "digest": dataset_digest(payload),
        "runtime_seconds": payload["meta"]["runtime_seconds"],
    }


def check_workbench(root: Path | None = None) -> list[str]:
    """Verify the committed page matches the committed dataset; return problems."""
    root = root or _repo_root()
    problems: list[str] = []
    html_path = artifact_path(root)
    json_path = data_path(root)
    if not json_path.is_file():
        return [f"missing dataset: {json_path}"]
    if not html_path.is_file():
        return [f"missing rendered page: {html_path}"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    expected = render_page(template_path().read_text(encoding="utf-8"), payload)
    actual = html_path.read_text(encoding="utf-8")
    if actual != expected:
        problems.append("workbench/index.html is stale: re-run `etm-workbench render`")
    if f'data-digest" content="{dataset_digest(payload)}"' not in actual:
        problems.append("page digest does not match the dataset digest")
    for needle in ("http://", "https://", "<script src", "cdn."):
        hits = [
            line
            for line in actual.splitlines()
            if needle in line and "http://www.w3.org" not in line
        ]
        if hits:
            problems.append(f"page contains external reference {needle!r} ({len(hits)} lines)")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``etm-workbench`` and ``main.py --workbench``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="etm-workbench",
        description="Build, render or verify the interactive Energy Transition Workbench.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="build",
        choices=("build", "render", "check", "rebuild"),
        help="build: data + page, render: page only, check: verify, rebuild: alias of build",
    )
    parser.add_argument("--root", default=None, help="repository root (defaults to auto-detect)")
    parser.add_argument(
        "--no-surrogate",
        action="store_true",
        help="skip the 16 holdout solves used to score the reduced-order surrogate",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.root).resolve() if args.root else None

    if args.command == "check":
        problems = check_workbench(root)
        if problems:
            for problem in problems:
                print(f"FAIL: {problem}")
            return 1
        print("workbench artifact is up to date with the committed dataset")
        return 0
    if args.command in ("build", "rebuild"):
        summary = write_workbench(
            root,
            include_surrogate=not args.no_surrogate,
            regenerate_data=True,
        )
    else:
        summary = write_workbench(root, regenerate_data=False)
    print("Energy Transition Workbench written")
    for key in ("html", "json", "html_bytes", "digest", "runtime_seconds"):
        print(f"  {key:<16} {summary[key]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Static figure for the README (same solved data, print-quality)
# ---------------------------------------------------------------------------
def plot_risk_frontier_and_kernel(
    output_path: str | Path = "results/risk_frontier_and_cvar_kernel.png",
    *,
    frontier: pd.DataFrame | None = None,
    solutions: dict[float, RiskKernelSolution] | None = None,
) -> Path:
    """Render a 4-panel figure of the mean-CVaR frontier, risk kernel and bounds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = risk_frontier(RISK_BETA_SWEEP) if frontier is None else frontier
    sols = (
        {beta: solve_mean_cvar(beta) for beta in (0.0, 0.5, 1.0)}
        if solutions is None
        else solutions
    )
    bounds = birge_louveaux_bounds()
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
        "Module 7 - Mean-CVaR dispatch, dual risk kernel & stochastic bounds "
        f"({len(DEFAULT_RISK_REGIMES)} regimes, alpha = {RISK_ALPHA:.0%})",
        fontsize=12.6,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(
        frame["expected_cost_per_year"] / 1e6,
        frame["cvar_per_year"] / 1e6,
        "-o",
        color="#0f766e",
        linewidth=2.1,
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="mean-CVaR efficient set",
    )
    for _, row in frame.iterrows():
        if row["beta"] in (0.0, 0.35, 1.0):
            ax.annotate(
                rf"$\beta$={row['beta']:.2f}",
                (row["expected_cost_per_year"] / 1e6, row["cvar_per_year"] / 1e6),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=8.4,
                color="#0f766e",
            )
    ax.set_xlabel("Expected annual cost  E[Z]  (million currency/yr)")
    ax.set_ylabel("Tail cost  CVaR$_{95\\%}$[Z]  (million currency/yr)")
    ax.set_title("(a) Efficient frontier: expected cost vs tail cost")
    ax.legend(loc="upper right", frameon=False, fontsize=8.6)

    ax = axes[0, 1]
    names = [r.name.replace("_", "\n") for r in DEFAULT_RISK_REGIMES]
    pos = np.arange(len(DEFAULT_RISK_REGIMES))
    mid = sols[0.5].risk_weights
    prior = np.array([r.probability for r in DEFAULT_RISK_REGIMES])
    ceiling = prior / (1.0 - RISK_ALPHA)
    ax.bar(pos - 0.19, prior, width=0.36, color="#94a3b8", label=r"prior $\pi_\omega$")
    ax.bar(pos + 0.19, mid, width=0.36, color="#7c3aed", label=r"dual kernel $q^*$ ($\beta$=0.5)")
    ax.scatter(
        pos,
        ceiling,
        marker="_",
        s=210,
        linewidths=2.0,
        color="#dc2626",
        label=r"ceiling $\pi_\omega/(1-\alpha)$",
        zorder=5,
    )
    ax.set_xticks(pos, names, fontsize=6.7)
    ax.set_ylabel("Probability weight")
    ax.set_title(r"(b) Rockafellar-Uryasev dual risk kernel $q^*$")
    ax.legend(loc="upper left", frameon=False, fontsize=8.2)

    ax = axes[1, 0]
    labels = ["Wait-and-see\n(WS)", "Recourse\n(RP)", "Expected-value plan\n(EEV)"]
    vals = [
        bounds.wait_and_see_per_year / 1e6,
        bounds.recourse_per_year / 1e6,
        bounds.expected_ev_per_year / 1e6,
    ]
    bars = ax.bar(labels, vals, color=["#16a34a", "#0891b2", "#dc2626"], width=0.55)
    for bar, val in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.012,
            f"{val:,.1f}M",
            ha="center",
            fontsize=8.6,
        )
    ax.set_ylabel("Annual cost (million currency/yr)")
    ax.set_title(
        f"(c) Birge-Louveaux bounds: EVPI {bounds.evpi_per_year / 1e6:,.1f}M | "
        f"VSS {bounds.vss_per_year / 1e6:,.1f}M"
    )
    ax.set_ylim(0, max(vals) * 1.14)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    bottom = np.zeros(len(frame))
    palette = {
        "solar": ("#f59e0b", "Solar PV"),
        "wind": ("#0ea5e9", "Wind"),
        "gas": ("#64748b", "Gas peaker"),
        "bess": ("#7c3aed", "Li-ion BESS"),
    }
    for key, (color, label) in palette.items():
        vals = frame[f"{key}_mw"].to_numpy(dtype=float)
        ax.bar(
            frame["beta"],
            vals,
            bottom=bottom,
            width=0.085,
            color=color,
            label=label,
        )
        bottom = bottom + vals
    ax2.plot(
        frame["beta"],
        frame["lcoe_per_mwh"],
        "-o",
        color="#b91c1c",
        linewidth=2.0,
        markersize=4.2,
        label="System LCOE (right)",
    )
    ax.set_xlabel(r"Risk-aversion weight $\beta$")
    ax.set_ylabel("First-stage capacity (MW)")
    ax2.set_ylabel("System LCOE (currency/MWh)", color="#b91c1c")
    ax.set_title("(d) Hedge composition: the mix rotates from solar to wind as beta rises")
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="upper center", frameon=False, fontsize=7.6, ncol=3)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
