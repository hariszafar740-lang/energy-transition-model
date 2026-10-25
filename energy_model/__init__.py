"""Energy transition capacity-expansion, dispatch, WACC/LCOE & KKT duality model."""

from .economics import (
    DEFAULT_WACC_SWEEP,
    REFERENCE_WACC,
    MarketDualitySummary,
    PrimalDualCrossCheck,
    TechnologyFinancialSpec,
    analyze_market_duality_and_lcoe,
    annualize_overnight_capex,
    build_wacc_technologies,
    capital_recovery_factor,
    default_financial_specs,
    format_duality_report,
    solve_primal_dual_linprog_crosscheck,
    sweep_wacc_lcoe_and_mix,
)
from .model import (
    DAYS_PER_YEAR,
    DEFAULT_PROFILE_PATH,
    DEFAULT_TECHNOLOGIES,
    ScenarioResult,
    SolverError,
    build_network,
    extract_results,
    format_report,
    run_scenario,
    solve,
)
from .plotting import plot_baseline_dispatch_and_lmp_duality
from .profiles import load_profiles, validate_profiles

__version__ = "0.3.0"

__all__ = [
    "DAYS_PER_YEAR",
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_TECHNOLOGIES",
    "DEFAULT_WACC_SWEEP",
    "MarketDualitySummary",
    "PrimalDualCrossCheck",
    "REFERENCE_WACC",
    "ScenarioResult",
    "SolverError",
    "TechnologyFinancialSpec",
    "__version__",
    "analyze_market_duality_and_lcoe",
    "annualize_overnight_capex",
    "build_network",
    "build_wacc_technologies",
    "capital_recovery_factor",
    "default_financial_specs",
    "extract_results",
    "format_duality_report",
    "format_report",
    "load_profiles",
    "plot_baseline_dispatch_and_lmp_duality",
    "run_scenario",
    "solve",
    "solve_primal_dual_linprog_crosscheck",
    "sweep_wacc_lcoe_and_mix",
    "validate_profiles",
]
