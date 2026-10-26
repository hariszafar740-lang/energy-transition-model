"""Command-line entry point: solve representative-day & chronological storage scenarios.

Usage:
    python3 main.py [--profiles PATH] [--days-per-year N]
                    [--duality-report] [--save-plot PATH]
                    [--dunkelflaute-report] [--save-storage-plot PATH]

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
    format_storage_report,
    generate_dunkelflaute_profiles,
    load_profiles,
    plot_baseline_dispatch_and_lmp_duality,
    plot_multiduration_storage_dunkelflaute,
    run_scenario,
    run_storage_scenario,
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
    parser.add_argument(
        "--dunkelflaute-report",
        action="store_true",
        help="solve and print the 168 h multi-duration storage (BESS + Green H2) report",
    )
    parser.add_argument(
        "--save-storage-plot",
        default=None,
        help="optional PNG path to render the 4-panel Dunkelflaute multi-duration storage figure",
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

    if args.dunkelflaute_report or args.save_storage_plot:
        dunk_prof = generate_dunkelflaute_profiles()
        storage_res = run_storage_scenario(
            dunk_prof, include_bess=True, include_h2=True, max_gas_mw=0.0
        )
        if args.dunkelflaute_report:
            print("")
            print(format_storage_report(storage_res))
        if args.save_storage_plot:
            saved_st = plot_multiduration_storage_dunkelflaute(
                output_path=args.save_storage_plot,
                profiles=dunk_prof,
                hybrid_result=storage_res,
            )
            print(f"\nSaved storage figure: {saved_st}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
