"""Command-line entry point: solve representative-day, DC-OPF & stochastic CVaR scenarios.

Usage:
    python3 main.py [--profiles PATH] [--days-per-year N]
                    [--wacc FLOAT] [--vre-capex-scale FLOAT]
                    [--duality-report] [--save-plot PATH]
                    [--dunkelflaute-report] [--save-storage-plot PATH]
                    [--sensitivity-report] [--cvar-report] [--save-risk-plot PATH]

Also installed as the packaged ``etm-sim`` console command.
"""

from __future__ import annotations

import sys

from energy_model.cli import build_parser, cli_entrypoint, main

__all__ = ["build_parser", "cli_entrypoint", "main"]


if __name__ == "__main__":
    sys.exit(main())
