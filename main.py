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
    SpinningReservePolicy,
    analyze_market_duality_and_lcoe,
    baseline_emission_specs,
    baseline_flexibility_specs,
    compare_flexibility_regimes,
    compare_transmission_regimes,
    format_duality_report,
    format_emissions_report,
    format_flexibility_report,
    format_macc_report,
    format_report,
    format_single_flexibility_report,
    format_single_transmission_report,
    format_storage_report,
    format_transmission_report,
    generate_dunkelflaute_profiles,
    load_profiles,
    plot_baseline_dispatch_and_lmp_duality,
    plot_co2_pareto_frontier_and_macc,
    plot_duck_curve_ramping_and_reserves,
    plot_multiduration_storage_dunkelflaute,
    plot_zonal_dcopf_congestion_and_tep,
    run_emissions_scenario,
    run_flexibility_scenario,
    run_scenario,
    run_storage_scenario,
    run_transmission_scenario,
    sweep_co2_pareto_and_macc,
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
    parser.add_argument(
        "--co2-cap",
        type=float,
        default=None,
        help="optional annual CO2 emissions cap in tCO2/yr on loaded profiles",
    )
    parser.add_argument(
        "--co2-price",
        type=float,
        default=None,
        help="optional Pigouvian carbon tax in currency/tCO2 on loaded profiles",
    )
    parser.add_argument(
        "--macc-report",
        action="store_true",
        help="append multi-fuel CO2 Pareto frontier & Marginal Abatement Cost Curve table",
    )
    parser.add_argument(
        "--save-macc-plot",
        default=None,
        help="optional PNG path to render the 4-panel CO2 Pareto & MACC figure",
    )
    parser.add_argument(
        "--ramp-limit",
        type=float,
        default=None,
        help="optional thermal hourly ramp limit in (0, 1] on loaded profiles",
    )
    parser.add_argument(
        "--must-run",
        type=float,
        default=None,
        help="optional thermal minimum stable output p_min_pu in [0, 1] on loaded profiles",
    )
    parser.add_argument(
        "--reserve-margin",
        type=float,
        default=None,
        help="optional load-proportional upward spinning reserve fraction in [0, 1]",
    )
    parser.add_argument(
        "--flexibility-report",
        action="store_true",
        help="append 4-regime Duck-Curve ramping, must-run & spinning-reserve comparison",
    )
    parser.add_argument(
        "--save-flexibility-plot",
        default=None,
        help="optional PNG path to render the 4-panel Duck-Curve ramping & reserve figure",
    )
    parser.add_argument(
        "--grid-scale",
        type=float,
        default=None,
        help="optional 4-zone 380 kV AC corridor capacity scale factor (> 0)",
    )
    parser.add_argument(
        "--cooptimize-tep",
        action="store_true",
        help="solve co-optimized 4-zone generation + AC transmission expansion planning",
    )
    parser.add_argument(
        "--transmission-report",
        action="store_true",
        help="append 4-regime 4-zone DC-OPF, HVDC & AC-TEP comparison table",
    )
    parser.add_argument(
        "--save-transmission-plot",
        default=None,
        help="optional PNG path to render the 4-panel 4-zone DC-OPF & TEP figure",
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

    if args.co2_cap is not None or args.co2_price is not None:
        try:
            em_res = run_emissions_scenario(
                profiles=profiles,
                specs=baseline_emission_specs(),
                co2_cap_tco2_per_year=args.co2_cap,
                co2_price_per_tco2=0.0 if args.co2_price is None else args.co2_price,
                include_storage=False,
                annual_hours=float(args.days_per_year) * float(len(profiles)),
            )
        except (ValueError, SolverError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print("")
        print(format_emissions_report(em_res))

    macc_df = None
    if args.macc_report or args.save_macc_plot:
        try:
            macc_df = sweep_co2_pareto_and_macc()
        except (ValueError, SolverError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.macc_report and macc_df is not None:
        print("")
        print(format_macc_report(macc_df))

    if args.save_macc_plot and macc_df is not None:
        saved_macc = plot_co2_pareto_frontier_and_macc(
            output_path=args.save_macc_plot,
            macc_df=macc_df,
        )
        print(f"\nSaved MACC figure: {saved_macc}")

    if args.ramp_limit is not None or args.must_run is not None or args.reserve_margin is not None:
        try:
            r_lim = 0.40 if args.ramp_limit is None else args.ramp_limit
            m_run = 0.00 if args.must_run is None else args.must_run
            r_pol = SpinningReservePolicy(
                enabled=args.reserve_margin is not None and args.reserve_margin > 0.0,
                load_reserve_fraction=0.0 if args.reserve_margin is None else args.reserve_margin,
                vre_reserve_fraction=0.0,
            )
            flx_res = run_flexibility_scenario(
                profiles=profiles,
                specs=baseline_flexibility_specs(
                    thermal_p_min_pu=m_run,
                    thermal_ramp_limit_pu=r_lim,
                ),
                enforce_ramping=args.ramp_limit is not None,
                enforce_must_run=args.must_run is not None,
                include_bess=False,
                reserve_policy=r_pol,
                annual_hours=float(args.days_per_year) * float(len(profiles)),
            )
        except (ValueError, SolverError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print("")
        print(format_single_flexibility_report(flx_res))

    flex_df = None
    if args.flexibility_report or args.save_flexibility_plot:
        try:
            flex_df = compare_flexibility_regimes()
        except (ValueError, SolverError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.flexibility_report and flex_df is not None:
        print("")
        print(format_flexibility_report(flex_df))

    if args.save_flexibility_plot and flex_df is not None:
        saved_flex = plot_duck_curve_ramping_and_reserves(
            output_path=args.save_flexibility_plot,
            regime_df=flex_df,
        )
        print(f"\nSaved flexibility figure: {saved_flex}")

    if args.grid_scale is not None or args.cooptimize_tep:
        try:
            g_scale = 1.0 if args.grid_scale is None else args.grid_scale
            tr_res = run_transmission_scenario(
                mode="ac_kvl",
                extendable_lines=bool(args.cooptimize_tep),
                capacity_scale=g_scale,
            )
        except (ValueError, SolverError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print("")
        print(format_single_transmission_report(tr_res))

    trans_df = None
    if args.transmission_report or args.save_transmission_plot:
        try:
            trans_df = compare_transmission_regimes()
        except (ValueError, SolverError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.transmission_report and trans_df is not None:
        print("")
        print(format_transmission_report(trans_df))

    if args.save_transmission_plot and trans_df is not None:
        saved_tr = plot_zonal_dcopf_congestion_and_tep(
            output_path=args.save_transmission_plot,
            regime_df=trans_df,
        )
        print(f"\nSaved transmission figure: {saved_tr}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
