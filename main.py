"""Command-line entry point: solve the shipped representative-day scenario.

Usage:
    python3 main.py [--profiles PATH] [--days-per-year N] [--duality-report] [--save-plot PATH]

Exits with status 1 and an ERROR message on stderr if the input is invalid or the
optimisation is not solved to optimality, so a solver failure is never reported
as a dispatch.
"""

from __future__ import annotations

import argparse
import sys

from energy_model import (
    DAYS_PER_YEAR,
    DEFAULT_PROFILE_PATH,
    SolverError,
    analyze_market_duality_and_lcoe,
    format_duality_report,
    format_report,
    load_profiles,
    plot_baseline_dispatch_and_lmp_duality,
    run_scenario,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profiles",
        default=str(DEFAULT_PROFILE_PATH),
        help="hourly profile CSV (default: shipped synthetic 24 h profiles)",
    )
    parser.add_argument(
        "--days-per-year",
        type=int,
        default=DAYS_PER_YEAR,
        help="number of days per year the representative day stands for",
    )
    parser.add_argument(
        "--duality-report",
        action="store_true",
        help="append KKT nodal shadow-price (LMP), LCOE, and zero-profit diagnostics",
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        help="optional PNG path to render the 4-panel dispatch, LMP duality & WACC figure",
    )
    args = parser.parse_args(argv)

    try:
        profiles = load_profiles(args.profiles)
        result = run_scenario(profiles=profiles, days_per_year=args.days_per_year)
    except (ValueError, FileNotFoundError, SolverError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("--- Energy transition model: representative-day capacity expansion ---")
    print(f"Profiles: {args.profiles}")
    print(format_report(result))

    if args.duality_report:
        print("")
        print(format_duality_report(analyze_market_duality_and_lcoe(result, profiles=profiles)))

    if args.save_plot:
        saved = plot_baseline_dispatch_and_lmp_duality(
            output_path=args.save_plot,
            profiles=profiles,
            result=result,
        )
        print(f"\nSaved figure: {saved}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
