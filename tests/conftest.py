"""Shared helpers for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_model import DEFAULT_TECHNOLOGIES

HOURS = pd.date_range("2026-01-01 00:00", periods=24, freq="h")

# Tolerances (documented in README). HiGHS primal feasibility is 1e-7 in
# scaled units; for MW-scale quantities 1e-5 MW is a conservative bound that is
# still far below any engineering significance.
POWER_TOL_MW = 1e-5
ENERGY_TOL_MWH = 1e-4
COST_REL_TOL = 1e-7


def make_profiles(load=150.0, solar=0.0, wind=0.0) -> pd.DataFrame:
    """24-hour profile table. Scalars are broadcast; sequences must have 24 values."""

    def as_array(value):
        arr = np.full(24, float(value)) if np.isscalar(value) else np.asarray(value, float)
        assert arr.shape == (24,), "profiles must have 24 hourly values"
        return arr

    return pd.DataFrame(
        {"load_mw": as_array(load), "solar_cf": as_array(solar), "wind_cf": as_array(wind)},
        index=HOURS,
    )


def technologies(*names: str, **overrides) -> dict:
    """Subset of DEFAULT_TECHNOLOGIES, with optional per-technology overrides.

    Example: technologies("Gas_Peaker", Solar_PV={"p_nom_extendable": False, "p_nom": 300.0})
    """
    selected = {name: dict(DEFAULT_TECHNOLOGIES[name]) for name in names}
    for name, spec in overrides.items():
        selected.setdefault(name, dict(DEFAULT_TECHNOLOGIES[name]))
        selected[name].update(spec)
    return selected


@pytest.fixture
def shipped_profiles():
    from energy_model import DEFAULT_PROFILE_PATH, load_profiles

    return load_profiles(DEFAULT_PROFILE_PATH)
