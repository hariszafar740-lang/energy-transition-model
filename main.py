"""Command-line entry point: solve the shipped representative-day scenario.

Usage:
    python3 main.py [--profiles PATH] [--days-per-year N]

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
    format_report,
    load_profiles,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
