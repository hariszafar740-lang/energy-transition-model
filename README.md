# Energy Transition Optimization Model

[![CI](https://github.com/hariszafar740-lang/energy-transition-model/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/energy-transition-model/actions/workflows/ci.yml)
![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source energy-system capacity-expansion, economic-dispatch, discounted cash-flow (`WACC` / `CRF` / `LCOE`), and KKT nodal shadow-price (`LMP`) duality model built with **PyPSA** (Python for Power System Analysis) and the **HiGHS** linear-programming solver.

![Representative-Day Dispatch, KKT Nodal LMP Duality & WACC Sensitivity](results/baseline_dispatch_and_lmp_duality.png)

## Academic Credential & Course Context

This repository serves as the practical engineering implementation for the official microcredential:
* **Course & Badge Title:** Modelling the Energy Transition: the revolution of open-source software
* **Issuing Institution:** Politecnico di Milano
* **Qualification Level:** EQF Level 7

## Model Purpose and Scope (`v0.3.0`)

The baseline model solves and verifies least-cost electricity capacity expansion and hourly dispatch for a single AC bus (`Regional_Grid`), alongside discounted cash-flow annualization and KKT market-equilibrium duality:

* **Single-bus LP baseline (`energy_model/model.py`):** One bus (`Regional_Grid`), one hourly load (`System_Demand`), and three extendable candidate generators (`Solar_PV`, `Wind_Turbine`, `Gas_Peaker`).
* **Discounted cash-flow & KKT nodal price duality (`energy_model/economics.py`):**
  * Exact Capital Recovery Factor $\text{CRF}(r, N)$ annualization of overnight capital cost $I_{0,g}$ across real discount rates (`WACC` $r \in [0, 1]$).
  * Extraction of hourly KKT power-balance dual multipliers $\lambda_t$ (Locational Marginal Prices, `LMP`) and verification of **LP Strong Duality** ($\sum_t w_t \lambda_t d_t \Delta t = C_{\text{total,yr}}$), **Boiteux-Turvey-Stoft Zero-Profit Equilibrium** ($R_g = \text{CapEx}_g + \text{OpEx}_g$), and **Brown-Hirth Market-Value Parity** ($\text{LCOE}_g = \text{MV}_g$ for every built generator).
  * Independent matrix-level primal + dual cross-check via `scipy.optimize.linprog(..., method="highs")` (`< 1e-12` primal and dual agreement).
* **Baseline boundaries:** Default `build_network()` models a single representative day (`24` hourly snapshots weighted by `DAYS_PER_YEAR = 365`) with no storage, transmission lines, unit commitment, reserves, or emissions cap (extended in subsequent modules).

## Mathematical Formulation

### 1. Primal Capacity-Expansion Linear Program

Minimise annual system cost:

$$\min_{p_{\text{nom},g},\, p_{g,t}} \; \sum_{g \in \mathcal{G}} c_g^{\text{cap}} p_{\text{nom},g} + D_{\text{yr}} \sum_{g \in \mathcal{G}} \sum_{t=1}^{T} c_g^{\text{marg}} p_{g,t} \Delta t$$

subject to, for every hour $t \in \{1,\dots,T\}$:

* **Nodal power balance** (with annualised dual multiplier $w_t \lambda_t$, $w_t = D_{\text{yr}}$):
  $$\sum_{g \in \mathcal{G}} p_{g,t} = d_t$$
* **Hourly resource availability & capacity envelope:**
  $$0 \le p_{g,t} \le \bar{a}_{g,t} p_{\text{nom},g}, \qquad p_{\text{nom},g} \ge 0$$

### 2. Discounted Cash-Flow Annualisation (`WACC` / `CRF`) & KKT Equilibrium Identities

For real cost of capital $r \ge 0$ and economic lifetime $N_g > 0$ years, overnight capital expenditure $I_{0,g}$ (currency/MW) and fixed O&M $f_g$ annualise via:

$$\text{CRF}(r, N_g) = \begin{cases} \dfrac{r(1+r)^{N_g}}{(1+r)^{N_g} - 1}, & r > 0 \\[6pt] \dfrac{1}{N_g}, & r = 0 \end{cases}, \qquad c_g^{\text{cap}}(r) = I_{0,g}\,\text{CRF}(r, N_g) + f_g$$

At the optimal primal-dual solution $(p_{\text{nom},g}^*, p_{g,t}^*, \lambda_t^*)$:

1. **System Cost & Demand-Weighted `LMP` Identity (Strong Duality):**
   $$C_{\text{total,yr}}^* = \sum_{t=1}^{T} w_t \lambda_t^* d_t \Delta t \implies \text{System LCOE} = \frac{C_{\text{total,yr}}^*}{\sum_t w_t d_t \Delta t} = \bar{\lambda}_{\text{load}}^* = 28.0555\text{ currency/MWh}$$
2. **Long-Run Zero-Profit & Market-Value Identity (`p_nom,g* > 0`):**
   $$R_g = \sum_{t=1}^{T} w_t \lambda_t^* p_{g,t}^* \Delta t = c_g^{\text{cap}} p_{\text{nom},g}^* + c_g^{\text{marg}} E_{g,\text{yr}} \implies \text{MV}_g = \frac{R_g}{E_{g,\text{yr}}} = \text{LCOE}_g$$
3. **Scarcity Pricing (`18:00` Evening Net-Load Peak):** When `Gas_Peaker` binds at full capacity ($p_{\text{gas},18}^* = p_{\text{nom,gas}}^* = 53.05\text{ MW}$) for a single hour per representative day, its entire annualised capital cost is recovered through that scarcity hour:
   $$\lambda_{18}^* = c_{\text{gas}}^{\text{marg}} + \frac{c_{\text{gas}}^{\text{cap}}}{D_{\text{yr}}} = 65.0 + \frac{20{,}000}{365} = 119.7945\text{ currency/MWh}$$

## Inputs and Provenance

### Hourly Profiles: `energy_model/data/profiles_synthetic_24h.csv`

| Column | Meaning | Units | Provenance |
| :--- | :--- | :--- | :--- |
| `load_mw` | System demand, mean power over the hour | MW | **Flat 150 MW** demand assumption carried over from the original `main.py`, not metered load. |
| `solar_cf` | Solar PV availability fraction | dimensionless (`0..1`) | **Synthetic.** Half-sine between `06:00` and `18:00`, peak `0.75` at `12:00`. Not derived from irradiance measurements. |
| `wind_cf` | Wind availability fraction | dimensionless (`0..1`) | **Synthetic.** `0.35` mean with `±0.12` diurnal cycle. Not derived from wind-speed reanalysis. |

The file is generated deterministically by `scripts/generate_synthetic_profiles.py`. **These profiles are synthetic and exist to verify the optimisation and time-series logic; they do not describe any real site or power system.**

### Technology Costs: `DEFAULT_TECHNOLOGIES` in `energy_model/model.py`

| Technology | `capital_cost` (annualised) | `marginal_cost` | Baseline `p_nom*` | Realised `LCOE == MV` | Value Factor `VF` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Solar_PV` | `45,000` / MW / yr | `0` / MWh | `82.87 MW` | `23.47` / MWh | `0.84x` |
| `Wind_Turbine` | `70,000` / MW / yr | `0` / MWh | `365.85 MW` | `24.26` / MWh | `0.86x` |
| `Gas_Peaker` | `20,000` / MW / yr | `65` / MWh | `53.05 MW` | `75.67` / MWh | `2.70x` |

The baseline `capital_cost` and `marginal_cost` values are carried over unchanged from the original `main.py` (**unverified** illustrative inputs without a specified currency). For `WACC` comparative-statics sweeps (`energy_model/economics.py`), overnight capital costs are calibrated so that $\text{CRF}(r_{\text{ref}}=7\%, N_g)$ reproduces `DEFAULT_TECHNOLOGIES` identically (`N = 25 yr` for `Solar_PV` and `Wind_Turbine`, `N = 20 yr` for `Gas_Peaker`). Treat cost outputs as mathematical consequences of these stated inputs, not real-world project estimates or engineering certification.

## Quickstart & Verification

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip "setuptools>=77.0" wheel build
.venv/bin/python -m pip install --no-build-isolation -e ".[dev]"

# Run baseline representative-day solver + KKT duality report + 4-panel figure
.venv/bin/python main.py --duality-report --save-plot results/baseline_dispatch_and_lmp_duality.png

# Run lint, formatting, and full pytest verification suite
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -v
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Haris Zafar.
