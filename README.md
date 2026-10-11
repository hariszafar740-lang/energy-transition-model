# Energy Transition Optimization Model

An open-source energy-system capacity expansion and dispatch model built with **PyPSA** (Python for Power System Analysis) and the **HiGHS** linear-programming solver.

## Academic Credential & Course Context
This repository serves as the practical engineering implementation for the official microcredential:
* **Course & Badge Title:** Modelling the Energy Transition: the revolution of open-source software
* **Issuing Institution:** Politecnico di Milano
* **Qualification Level:** EQF Level 7

## Model purpose and scope

The model answers a narrow question: for a single electricity bus, which mix of solar PV, wind and gas peaker capacity meets a given hourly demand at the lowest annual system cost, given the hourly availability of the renewable resources?

It is a **single-bus, single-representative-day** linear program:

* One bus (`Regional_Grid`), one load (`System_Demand`).
* Three candidate generators: `Solar_PV`, `Wind_Turbine`, `Gas_Peaker`. All are extendable (capacity is a decision variable).
* No storage, no transmission, no unit commitment, no minimum-stable-output or ramp limits, no reserves, no emissions accounting, no discounting.

## Formulation

Minimise annual system cost

    sum_g capital_cost_g * p_nom_g  +  DAYS_PER_YEAR * sum_{g,t} marginal_cost_g * p_{g,t} * 1 h

subject to, for every hour `t`:

* power balance: `sum_g p_{g,t} = load_t` (equality; surplus renewables are curtailed by the solver choosing `p` below availability);
* availability: `0 <= p_{g,t} <= p_max_pu_{g,t} * p_nom_g`, where `p_max_pu` is the hourly capacity factor from the profile file (1.0 for gas);
* `p_nom_g >= 0`.

The implementation is in `energy_model/model.py`; PyPSA builds the LP and HiGHS solves it.

## Inputs and provenance

### Hourly profiles: `energy_model/data/profiles_synthetic_24h.csv`

| column | meaning | units | provenance |
|---|---|---|---|
| `load_mw` | system demand, mean power over the hour | MW | **Flat 150 MW. This is the demand assumption carried over from the original `main.py`, not a measured load.** |
| `solar_cf` | solar PV availability (fraction of installed capacity) | dimensionless, 0–1 | **Synthetic.** Half-sine between 06:00 and 18:00, peak 0.75 at 12:00. Not derived from irradiance data. |
| `wind_cf` | wind availability | dimensionless, 0–1 | **Synthetic.** 0.35 mean with ±0.12 diurnal cycle. Not derived from wind-speed data. |

The file is generated deterministically by `scripts/generate_synthetic_profiles.py`. **These profiles are synthetic and exist to exercise the time-series logic. They are not measured data and do not describe any real site or system.** For a real study, replace the CSV with traceable data (metered demand, and renewable capacity factors from measured or reanalysis data) with the same columns and an hourly `DatetimeIndex`. The loader validates the file (see below), so an invalid replacement fails loudly.

### Technology costs: `DEFAULT_TECHNOLOGIES` in `energy_model/model.py`

| technology | capital_cost | marginal_cost |
|---|---|---|
| Solar_PV | 45 000 per MW per year | 0 per MWh |
| Wind_Turbine | 70 000 per MW per year | 0 per MWh |
| Gas_Peaker | 20 000 per MW per year | 65 per MWh |

These values are carried over unchanged from the original `main.py`. **They are unverified:** no cost source is cited, the currency is not specified, and the capital costs are *assumed* to be annualised. Treat any cost result as a consequence of these inputs, not as an estimate of real project economics.

## Time resolution, horizon and annualisation

* Each snapshot is **one hour**. The profile file must be hourly; any other spacing is rejected.
* The horizon is **one representative day (24 hourly snapshots)**. Each snapshot is weighted by `DAYS_PER_YEAR = 365` in the objective and in the energy accounting. The assumption is that the same 24-hour profile repeats on every day of the year.
* Consequently:
  * **Operating energy and operating cost** are *per representative day* × 365 for the annual figures.
  * **Capital cost** is already annual and is *not* multiplied by anything.
  * There is no seasonality, no leap-year handling, and no multi-day storage. A single synthetic day cannot represent seasonal resource variation.
* The `days_per_year` argument can be changed (`python3 main.py --days-per-year N`). It is validated as a positive integer.
* There is no discounting. The objective is an annual cost, not a net present cost. (The original README described a "Net Present Cost" objective; the code never implemented one, so that description has been corrected.)

### Why the original model chose gas for all demand

The original code used annualised capital costs with a **single** hour-weighted operating period, so the operating cost of one day was compared with a full year of capital cost. Operating costs were under-counted by a factor of 365. With constant renewable availability, the same inputs give solar-only supply when the operating cost is annualised. So the gas-only result came from the horizon mismatch, not from the inputs. It is fixed by the weighting above and is covered by regression tests.

## Renewable availability

Solar and wind availability is assigned as an **hourly time series** (`p_max_pu`) per generator. The original model left `p_max_pu` at its static default of 1.0, which made solar and wind available at full capacity at all hours, including midnight. That is fixed.

## Dispatch and feasibility

* `solve()` raises `SolverError` unless PyPSA/HiGHS returns status `ok` with termination `optimal` and a finite objective. An infeasible model is therefore reported as an error and never as a dispatch.
* The power balance is an equality constraint, so there is no unserved-energy variable. A scenario that cannot meet demand (for example, no thermal capacity and no renewable resource, with fixed capacities) is infeasible and is rejected.
* Curtailment is not a separate variable. It is the difference between available output (`p_max_pu × p_nom`) and dispatched output, summed over the representative day, for renewable generators only.

## Energy, power and cost definitions

| quantity | definition | unit |
|---|---|---|
| `p_nom` | installed capacity (decision variable) | MW |
| dispatch `p_{g,t}` | mean power of generator `g` in hour `t` | MW |
| energy per representative day | `sum_t p_{g,t} × 1 h` | MWh/day |
| energy per year | energy per day × `DAYS_PER_YEAR` | MWh/yr |
| curtailment | `sum_t (p_max_pu_{g,t} × p_nom_g − p_{g,t}) × 1 h` (renewables) | MWh/day |
| capacity factor | energy per day ÷ (`p_nom` × 24 h) | dimensionless |
| capital cost | `sum_g capital_cost_g × p_nom_g` | currency/yr |
| operating cost | `DAYS_PER_YEAR × sum_{g,t} marginal_cost_g × p_{g,t} × 1 h` | currency/yr |

The independent verifications reported by `run_scenario` are: the maximum power-balance residual over all hours, the maximum capacity violation over all generators and hours, and the difference between the solver's objective and the recomputed annual cost. All three should be at solver tolerance (about 1e-7 relative).

### What "150 MW / 3600 MWh" means

With a flat 150 MW load:

* **150 MW** is the *instantaneous* demand in every hour, and the installed gas capacity that serves it. It is a power, not an energy.
* **3600 MWh** is the *energy of one representative day*: 150 MW × 24 h × 1 h per snapshot. It is the total demand per day, and the total gas generation per day in the gas-only case.
* The **annual** energy for the same flat load is 3600 × 365 = 1 314 000 MWh.

These are verified by `tests/test_regressions.py`.

## Storage

**Not applicable.** The model does not include batteries, pumped hydro or any other storage, so storage validation (state of charge, efficiencies, end-of-horizon assumptions) does not apply. A test asserts that no storage components are present, so that adding storage later forces the validation to be written.

## Installation

Requires Python 3.11 or newer (PyPSA 1.4 requires 3.11+).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies are declared in `pyproject.toml`: `pypsa>=1.4,<2`, `highspy>=1.15`, `pandas>=2.2`. The development extra adds `pytest`, `ruff` and `scipy` (used by one independent-LP cross-check test).

## Running the model

```bash
python3 main.py                       # shipped synthetic representative day
python3 main.py --profiles my.csv     # your own hourly profile file
python3 main.py --days-per-year 365   # default
```

The command exits with status 0 on an optimal solution and with status 1, printing `ERROR:` to stderr, on invalid input or a non-optimal solve.

## Testing and verification

```bash
pytest                 # full test suite
ruff check .           # lint
ruff format --check .  # formatting
```

The suite (`tests/`) calls the production code paths (`build_network`, `solve`, `run_scenario`, `main.py`). It covers:

* **Input validation:** missing columns, NaN and non-numeric values, out-of-range capacity factors, negative load, duplicate, missing or non-hourly timestamps, and missing files.
* **Time series:** the solar and wind profiles reach the solved network hour by hour, a time-varying load is met hour by hour, and a shifted profile moves the output.
* **Dispatch:** power balance at every hour, generation within availability and capacity, renewable-rich, low-renewable and thermal-required scenarios, curtailment accounting, and infeasibility reporting.
* **Costs and annualisation:** capital and operating cost by hand for small scenarios, linear scaling with `days_per_year`, and an independent LP reformulation (`scipy.optimize.linprog`) that must give the same optimal cost on three scenarios.
* **Regression:** one test per defect in the original model (see the module docstring of `tests/test_regressions.py`), including reproduction of the original 3 234 000 gas-only objective with the original one-day weighting.
* **Reproducibility:** repeated runs give identical results; inputs and defaults are not mutated; the CLI runs end-to-end and fails correctly on bad input.

Numerical tolerances: power and energy checks use 1e-5 MW / 1e-4 MWh, which is far below any engineering significance and above HiGHS's primal feasibility tolerance. Cost checks use relative tolerance 1e-7 (or 1e-6 where the solver's objective is compared with an independently computed value).

## Validation status

* **Software tests:** the suite above. Passing means the code behaves as specified in the tests.
* **Mathematical consistency checks:** power balance, capacity limits, energy and cost recomputation, and the independent LP cross-check. These are in the test suite and in the verification fields of each result.
* **Validation against external reference data:** **none.** No external reference dataset or published model result has been compared against.
* **Calibration against real-world measurements:** **none.** The demand, renewable profiles and cost inputs are not calibrated.
* **Validation of an actual power system:** **none.** This is an educational single-bus model.

## Known limitations and unverified assumptions

* One synthetic representative day, repeated 365 times. No seasonality, no year-round variability, no weather correlation.
* Synthetic renewable profiles and flat demand. Results depend strongly on these inputs.
* Capital costs are assumed to be annual figures; their source and currency are unknown.
* No discounting, no financing, no lifetimes, no decommissioning.
* No storage, transmission, reserves, unit commitment, minimum output, ramp limits, startup costs or emissions.
* Gas peaker is modelled as a linear, unconstrained generator (no fuel-price, efficiency or emissions detail).
* The model is linear, so results are a single optimal vertex; ties between equal-cost technologies can be resolved differently by different solver versions. The objective value is the reproducible quantity.
* Tested with Python 3.11.2, PyPSA 1.4.0, highspy 1.15.1, pandas 3.0.6, numpy 2.4.6 (Linux). Other versions are not verified.
* PyPSA currently emits a pandas `str`-dtype `FutureWarning` on import with pandas 3. It does not affect results.
