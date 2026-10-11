"""Loading and validation of hourly input profiles.

A profile file is a CSV with a timestamp column (the index) and three columns:

* ``load_mw``  -- system electricity demand in MW (mean power over the hour).
* ``solar_cf`` -- solar PV availability, i.e. the fraction of installed
  capacity that is available in the hour (dimensionless, 0..1).
* ``wind_cf``  -- wind availability, same convention (dimensionless, 0..1).

Each row is one snapshot of duration one hour. Missing, non-numeric,
non-finite, or out-of-range values are rejected: they are never silently
filled, because a silent fill would turn invalid input into a seemingly valid
dispatch result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("load_mw", "solar_cf", "wind_cf")
CAPACITY_FACTOR_COLUMNS = ("solar_cf", "wind_cf")
SNAPSHOT_HOURS = 1.0  # every snapshot in a profile represents one hour


def validate_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Validate a profile table and return a sorted copy of it.

    Raises ``ValueError`` describing the first problem found.
    """
    if not isinstance(profiles, pd.DataFrame):
        raise ValueError("profiles must be a pandas DataFrame")

    missing = [c for c in REQUIRED_COLUMNS if c not in profiles.columns]
    if missing:
        raise ValueError(f"profiles are missing required column(s): {missing}")

    if len(profiles) == 0:
        raise ValueError("profiles contain no rows")

    index = profiles.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("profiles must be indexed by a DatetimeIndex")
    if index.hasnans:
        raise ValueError("profile timestamps contain missing values")
    if index.has_duplicates:
        raise ValueError("profile timestamps contain duplicates")

    profiles = profiles.sort_index()
    if len(profiles) > 1:
        steps = pd.Series(profiles.index).diff().dropna()
        if not (steps == pd.Timedelta(hours=SNAPSHOT_HOURS)).all():
            raise ValueError(
                "profile timestamps must be spaced exactly one hour apart; "
                f"found steps {sorted(set(steps.astype(str)))}"
            )

    table = profiles[list(REQUIRED_COLUMNS)]
    for column in REQUIRED_COLUMNS:
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any():
            bad = table.index[values.isna()]
            raise ValueError(
                f"column '{column}' has missing or non-numeric values (first at {bad[0]})"
            )
        values = values.astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"column '{column}' contains non-finite values")
        if column in CAPACITY_FACTOR_COLUMNS and ((values < 0) | (values > 1)).any():
            raise ValueError(f"column '{column}' must lie within [0, 1]")
        if column == "load_mw" and (values < 0).any():
            raise ValueError("column 'load_mw' must be non-negative")
        table[column] = values

    return table


def load_profiles(path: str | Path) -> pd.DataFrame:
    """Read and validate an hourly profile CSV.

    The first column is interpreted as the timestamp index.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"profile file not found: {path}")
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    return validate_profiles(raw)
