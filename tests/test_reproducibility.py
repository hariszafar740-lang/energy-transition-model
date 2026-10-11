"""Reproducibility of solved runs and the command-line entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from energy_model import run_scenario

from .conftest import make_profiles, technologies

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_identical_runs_give_identical_results(shipped_profiles):
    first = run_scenario(shipped_profiles)
    second = run_scenario(shipped_profiles)
    assert np.allclose(first.p_nom_mw.to_numpy(), second.p_nom_mw.to_numpy(), atol=1e-9)
    assert np.allclose(first.dispatch_mw.to_numpy(), second.dispatch_mw.to_numpy(), atol=1e-9)
    assert first.total_cost_per_year == second.total_cost_per_year


def test_run_does_not_mutate_input_profiles(shipped_profiles):
    before = shipped_profiles.copy(deep=True)
    run_scenario(shipped_profiles)
    assert shipped_profiles.equals(before)


def test_run_does_not_mutate_default_technologies():
    from energy_model import DEFAULT_TECHNOLOGIES

    before = {k: dict(v) for k, v in DEFAULT_TECHNOLOGIES.items()}
    run_scenario(make_profiles(solar=0.5), technologies("Solar_PV", "Gas_Peaker"))
    assert DEFAULT_TECHNOLOGIES == before


def test_cli_runs_end_to_end_with_shipped_profiles():
    completed = subprocess.run(
        [sys.executable, "main.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Solver status: ok / optimal" in completed.stdout
    assert "Demand energy (MWh per representative day): 3,600.00" in completed.stdout
    assert "Optimal installed capacity (MW)" in completed.stdout


def test_cli_exits_nonzero_on_invalid_profiles(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "timestamp,load_mw,solar_cf,wind_cf\n"
        "2026-01-01 00:00:00,150,,0.2\n"
        "2026-01-01 01:00:00,150,0.1,0.2\n"
    )
    completed = subprocess.run(
        [sys.executable, "main.py", "--profiles", str(bad)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 1
    assert "ERROR" in completed.stderr
    assert "Optimal installed capacity" not in completed.stdout
