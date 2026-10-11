"""Generate the SYNTHETIC 24-hour input profiles shipped in ``energy_model/data``.

These are NOT measured data and NOT a real site. They are deterministic,
analytic shapes that exist so the model has a time-varying renewable resource
to respond to in tests and the demo run. Replace the CSV with traceable data
(e.g. measured or reanalysis capacity factors and metered demand) for any real
study.

Generated columns (hourly, timestamp = start of hour, 2026-01-01 00:00..23:00):

* solar_cf: clear-sky-like half-sine, zero from sunset to sunrise, peak 0.75 at
  12:00. Dimensionless availability, 0..1. Not derived from irradiance data.
* wind_cf:  0.35 mean with a +/-0.12 diurnal cycle peaking near 03:00. Dimensionless.
* load_mw:  flat 150.0 MW, the demand assumption carried over from the original
  ``main.py``. It is not a measured load shape.

Usage::

    python3 scripts/generate_synthetic_profiles.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

OUTPUT = (
    Path(__file__).resolve().parents[1] / "energy_model" / "data" / "profiles_synthetic_24h.csv"
)


def build_profiles() -> pd.DataFrame:
    hours = pd.date_range("2026-01-01 00:00", "2026-01-01 23:00", freq="h")
    solar = [round(0.75 * max(0.0, math.sin(math.pi * (h - 6) / 12)), 3) for h in range(24)]
    wind = [round(0.35 + 0.12 * math.cos(2 * math.pi * (h - 3) / 24), 3) for h in range(24)]
    return pd.DataFrame(
        {"load_mw": 150.0, "solar_cf": solar, "wind_cf": wind},
        index=pd.DatetimeIndex(hours, name="timestamp"),
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table = build_profiles()
    table.to_csv(OUTPUT, float_format="%.3f")
    print(f"wrote {OUTPUT} ({len(table)} rows)")


if __name__ == "__main__":
    main()
