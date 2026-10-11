"""Input validation and the shipped synthetic profile file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_model import DEFAULT_PROFILE_PATH, load_profiles, validate_profiles

from .conftest import HOURS, make_profiles


def test_shipped_profile_file_is_hourly_24h_and_valid(shipped_profiles):
    assert len(shipped_profiles) == 24
    assert shipped_profiles.index.equals(HOURS)
    assert set(shipped_profiles.columns) == {"load_mw", "solar_cf", "wind_cf"}


def test_shipped_solar_profile_is_time_varying_and_zero_at_night(shipped_profiles):
    solar = shipped_profiles["solar_cf"]
    night = solar.index.hour.isin(list(range(0, 6)) + list(range(19, 24)))
    assert (solar[night] == 0).all()
    assert solar.max() > 0.5
    assert solar.idxmax().hour == 12
    assert solar.nunique() > 5, "solar availability must vary through the day"


def test_shipped_wind_profile_is_time_varying(shipped_profiles):
    wind = shipped_profiles["wind_cf"]
    assert wind.nunique() > 5
    assert 0.0 < wind.min() < wind.max() < 1.0


def test_shipped_load_is_flat_150mw_as_documented(shipped_profiles):
    assert np.allclose(shipped_profiles["load_mw"], 150.0)


def test_validate_returns_sorted_copy_and_does_not_mutate_input():
    profiles = make_profiles(solar=0.5)
    reversed_profiles = profiles.iloc[::-1]
    validated = validate_profiles(reversed_profiles)
    assert validated.index.is_monotonic_increasing
    assert list(reversed_profiles.index) == list(HOURS[::-1])


@pytest.mark.parametrize("missing", ["load_mw", "solar_cf", "wind_cf"])
def test_missing_required_column_is_rejected(missing):
    profiles = make_profiles().drop(columns=[missing])
    with pytest.raises(ValueError, match="missing required column"):
        validate_profiles(profiles)


def test_empty_profiles_rejected():
    empty = make_profiles().iloc[:0]
    with pytest.raises(ValueError, match="no rows"):
        validate_profiles(empty)


def test_nan_is_rejected_not_silently_filled():
    profiles = make_profiles()
    profiles.loc[HOURS[5], "solar_cf"] = np.nan
    with pytest.raises(ValueError, match="missing or non-numeric"):
        validate_profiles(profiles)


def test_non_numeric_text_is_rejected():
    profiles = make_profiles().astype({"wind_cf": object})
    profiles.loc[HOURS[3], "wind_cf"] = "windy"
    with pytest.raises(ValueError, match="missing or non-numeric"):
        validate_profiles(profiles)


def test_infinite_value_is_rejected():
    profiles = make_profiles()
    profiles.loc[HOURS[0], "load_mw"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_profiles(profiles)


@pytest.mark.parametrize("column, value", [("solar_cf", 1.2), ("wind_cf", -0.01)])
def test_capacity_factor_out_of_unit_range_is_rejected(column, value):
    profiles = make_profiles()
    profiles.loc[HOURS[10], column] = value
    with pytest.raises(ValueError, match=r"must lie within \[0, 1\]"):
        validate_profiles(profiles)


def test_negative_load_is_rejected():
    profiles = make_profiles()
    profiles.loc[HOURS[2], "load_mw"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        validate_profiles(profiles)


def test_duplicate_timestamps_rejected():
    profiles = make_profiles()
    profiles.index = pd.DatetimeIndex([HOURS[0]] * 24)
    with pytest.raises(ValueError, match="duplicates"):
        validate_profiles(profiles)


def test_non_hourly_spacing_rejected():
    profiles = make_profiles()
    profiles.index = pd.date_range("2026-01-01", periods=24, freq="30min")
    with pytest.raises(ValueError, match="exactly one hour apart"):
        validate_profiles(profiles)


def test_gap_in_timestamps_rejected():
    profiles = make_profiles().drop(HOURS[7])
    with pytest.raises(ValueError, match="exactly one hour apart"):
        validate_profiles(profiles)


def test_non_datetime_index_rejected():
    profiles = make_profiles().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_profiles(profiles)


def test_load_profiles_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profiles(tmp_path / "does_not_exist.csv")


def test_load_profiles_round_trip(tmp_path):
    path = tmp_path / "profiles.csv"
    make_profiles(load=120.0, solar=[0.0] * 12 + [0.5] * 12, wind=0.3).to_csv(path)
    loaded = load_profiles(path)
    assert np.allclose(loaded["load_mw"], 120.0)
    assert np.allclose(loaded["solar_cf"].iloc[12:], 0.5)


def test_load_profiles_rejects_invalid_csv_content(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,load_mw,solar_cf,wind_cf\n"
        "2026-01-01 00:00:00,150,0.1,0.2\n"
        "2026-01-01 01:00:00,150,1.5,0.2\n"
    )
    with pytest.raises(ValueError, match=r"must lie within"):
        load_profiles(path)


def test_default_profile_path_points_at_packaged_data():
    assert DEFAULT_PROFILE_PATH.is_file()
    assert DEFAULT_PROFILE_PATH.parent.name == "data"
