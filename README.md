# Energy Transition Optimization Model

[![CI](https://github.com/hariszafar740-lang/energy-transition-model/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/energy-transition-model/actions/workflows/ci.yml)
![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source energy-system capacity-expansion, chronological multi-duration energy storage (`Li-Ion BESS` + `Green H2` Power-to-X), discounted cash-flow (`WACC` / `CRF` / `LCOE`), and KKT nodal shadow-price (`LMP`) duality model built with **PyPSA** (Python for Power System Analysis) and the **HiGHS** linear-programming solver.

![Representative-Day Dispatch, KKT Nodal LMP Duality & WACC Sensitivity](results/baseline_dispatch_and_lmp_duality.png)

## Academic Credential & Course Context

This repository serves as the practical engineering implementation for the official microcredential:
* **Course & Badge Title:** Modelling the Energy Transition: the revolution of open-source software
* **Issuing Institution:** Politecnico di Milano
* **Qualification Level:** EQF Level 7

## Architecture & Module Capabilities (`v0.8.0`)

* **Single-bus LP baseline (`energy_model/model.py`):** Representative-day capacity expansion (`Regional_Grid`, `System_Demand`, `Solar_PV`, `Wind_Turbine`, `Gas_Peaker`) with strict hourly time-series validation (`energy_model/profiles.py`).
* **Discounted cash-flow & KKT nodal price duality (`energy_model/economics.py`):**
  * Exact Capital Recovery Factor $\text{CRF}(r, N)$ annualisation across real discount rates (`WACC` $r \in [0, 1]$).
  * Extraction of hourly KKT power-balance dual multipliers $\lambda_t$ (`LMP`), verification of **LP Strong Duality** ($\text{System LCOE} = \bar{\lambda}_{\text{load}}^* = 28.0555\text{ /MWh}$), **Boiteux-Turvey-Stoft Zero-Profit Equilibrium**, and **Brown-Hirth Market-Value Parity** ($\text{LCOE}_g = \text{MV}_g$).
* **Multi-duration storage & `Dunkelflaute` chronological adequacy (`energy_model/storage.py`):**
  * **Short-duration power-coupled battery (`Li_Ion_BESS` via PyPSA `StorageUnit`):** `4 h` duration, $\eta_{\text{ch}} = \eta_{\text{dis}} = 95\%$ ($\eta_{\text{RTE}} = 90.25\%$), cyclic State-of-Charge (`SoC`), optimal for daily midday-solar-to-evening shifting (`278.7` eq. cycles/yr).
  * **Long-duration decoupled Power-to-X hydrogen (`Green_H2` via PyPSA `Bus` + `Store` + `Link`s):** Independently sized `H2_Electrolyzer` ($\eta_{\text{el}} = 68\%$), underground `H2_Cavern` (`350` currency/MWh$_{\text{H2}}$/yr, cyclic `SoC`), and `H2_Turbine` ($\eta_{\text{fc}} = 50\%$, $\eta_{\text{RTE}} = 34.0\%$), optimal for bridging multi-day **`Dunkelflaute`** wind/solar lulls (`52.1` eq. cycles/yr).
  * **Independent matrix LP cross-checks:** Both `economics.py` and `storage.py` construct standalone `scipy.optimize.linprog(..., method="highs")` primal + dual matrices confirming `< 1e-11` agreement with PyPSA/linopy across capacities, objective values, and nodal shadow prices.

## Multi-Duration Storage & `Dunkelflaute` Adequacy (`v0.4.0`)

![Multi-Duration Storage Across a 168 h Chronological Dunkelflaute](results/multiduration_storage_dunkelflaute_dispatch.png)

### Coupled AC + Hydrogen Intertemporal Formulation

Over a chronological horizon $t \in \{0,\dots,T-1\}$ ($T = 168\text{ h}$, annual weight $w = 8760 / T$):

$$\sum_{g \in \mathcal{G}} p_{g,t} + p_{\text{bess,dis},t} - p_{\text{bess,ch},t} + p_{\text{H2,dis},t}^{\text{elec}} - p_{\text{H2,ch},t}^{\text{elec}} = d_t \qquad [\text{dual } w \lambda_t^{\text{elec}}]$$

$$e_{\text{bess},t} = (1 - \sigma_{\text{bess}}) e_{\text{bess},(t-1)\bmod T} + \eta_{\text{ch}} p_{\text{bess,ch},t} \Delta t - \frac{1}{\eta_{\text{dis}}} p_{\text{bess,dis},t} \Delta t, \quad 0 \le e_{\text{bess},t} \le \tau_{\text{bess}} P_{\text{bess}}$$

$$e_{\text{H2},t} = (1 - \sigma_{\text{H2}}) e_{\text{H2},(t-1)\bmod T} + \eta_{\text{el}} p_{\text{H2,ch},t}^{\text{elec}} \Delta t - \frac{1}{\eta_{\text{fc}}} p_{\text{H2,dis},t}^{\text{elec}} \Delta t, \quad 0 \le e_{\text{H2},t} \le E_{\text{H2}} \quad [\text{dual } w \lambda_t^{\text{H2}}]$$

### Verified 168-Hour `Dunkelflaute` Adequacy Regimes (`48 h` Cold Wind/Solar Lull)

| Adequacy Regime | `Solar_PV` | `Wind_Turbine` | `Li_Ion_BESS` (`4 h`) | `H2` Electrolyzer / Turbine | `H2_Cavern` Store | System `LCOE` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. VRE Only (`0%` Gas)** | `0.0 MW` | `5,657.3 MW` | `—` | `—` | `—` | `294.70` / MWh |
| **2. VRE + `4 h` BESS (`0%` Gas)** | `0.0 MW` | `599.5 MW` | `1,570.0 MW` (`6,280 MWh`) | `—` | `—` | `66.28` / MWh |
| **3. VRE + Green `H2` (`0%` Gas)** | `148.1 MW` | `707.9 MW` | `—` | `176.3` / `173.2 MW_e` | `10,745 MWh_H2` | `51.92` / MWh |
| **4. Hybrid `BESS + H2` (`0%` Gas)** | `180.2 MW` | `679.6 MW` | `51.8 MW` (`207 MWh`) | `153.3` / `132.8 MW_e` | `10,466 MWh_H2` | **`51.25` / MWh** |
| **5. Hybrid + Uncapped Gas Ref.** | `72.2 MW` | `375.8 MW` | `58.5 MW` (`234 MWh`) | `5.1` / `1.5 MW_e` | `215 MWh_H2` | `43.49` / MWh |

* **Efficiency vs. Energy-CapEx Crossover (`~6.9 h`):** For short lulls (`0–6 h`), `Li-Ion BESS` (`90.25%` RTE) achieves lower zero-fossil `LCOE` than `Green H2` (`24.01` vs. `26.53` / MWh at `0 h`). Beyond `~6.9 h` (`12–72 h` `Dunkelflaute`), decoupled `H2` cavern storage (`350` / MWh$_{\text{H2}}$ / yr) decisively beats `4 h` battery packs (`66.76` vs. `85.31` / MWh at `72 h`), while the **Hybrid `BESS + H2` portfolio forms the Pareto-optimal lower envelope across every duration**.

## CO2 Cap-Tax Lagrangian Duality & Marginal Abatement Cost Curve (v0.5.0)

Module 3 (`energy_model/emissions.py`) extends the single-bus capacity-expansion engine with primary-to-electric multi-fuel emissions accounting ($e_g = \text{EF}_{\text{th},c(g)} / \eta_g$), PyPSA `GlobalConstraint` annual $\text{CO}_2$ caps ($\bar{E}_{\text{CO2}}$), Pigouvian carbon taxes ($\tau$), and an independent sparse `scipy.optimize.linprog` (`HiGHS`) primal/dual cross-check:

1. **Lagrangian Cap-Tax Saddle-Point Equivalence:** Relaxing $\sum_{g,t} w_t e_g p_{g,t} \Delta t \le \bar{E}$ with KKT shadow carbon price $\mu_{\text{CO2}}^* = -\partial J_{\text{cap}}^*(\bar{E})/\partial \bar{E} \ge 0$ yields the exact Pigouvian carbon-tax objective identity across every point on the convex Pareto frontier:
   $$J_{\text{tax}}^*(\mu_{\text{CO2}}^*) = J_{\text{cap}}^*(\bar{E}) + \mu_{\text{CO2}}^* \cdot \bar{E}$$
   verified numerically to relative saddle-point residual $\le 1.6 \times 10^{-15}$.
2. **Net-Zero Boundary Subdifferential Regularisation ($\bar{E} = 0$):** Because the dual objective $\max b^\top y - \bar{E}\mu$ assigns zero weight to $\mu$ at $\bar{E} = 0$, the subdifferential $-\partial J_{\text{cap}}^*(0)$ is the half-ray $[\tau_{\min}^{\text{net-zero}}, +\infty)$. Evaluating the right-hand limit $\lim_{\epsilon \downarrow 0^+} \mu^*(\epsilon)$ uniquely extracts the minimum Net-Zero carbon price (`418.25` currency / $\text{tCO}_2$).

![CO2 Pareto Frontier, MACC & Fuel Switching](results/co2_pareto_frontier_and_macc.png)

### Verified Multi-Fuel Decarbonization Milestones (`168 h` Representative Week, Annualised)

| Abatement Target | Realised $\text{CO}_2$ (`tCO2/yr`) | Shadow Price $\mu_{\text{CO2}}^*$ (`/tCO2`) | Average Cost (`/tCO2`) | Resource `LCOE` (`/MWh`) | Dominant Portfolio Regime |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `0%` (BAU) | `1,206,958` | `0.00` | `0.00` | `24.34` | `Coal_Steam` (`181.8 MW`) + `CCGT_Gas` (`7.5 MW`) |
| `45%` Cap | `663,827` | `6.99` | `6.15` | `26.69` | Stage I: `Wind_Turbine` (`220.0 MW`) + `OCGT` |
| `75%` Cap | `301,740` | `18.61` | `8.30` | `29.61` | Stage II: `Coal -> CCGT` (`36.2 MW`) + `Solar_PV` (`61.6 MW`) |
| `96%` Cap | `48,278` | `52.77` | `12.84` | `34.79` | Stage III: `90% Gas_CCS` (`14.5 MW`) + `LiIon_BESS` (`40.5 MWh`) |
| `100%` Net-Zero | `0` | `418.25` | `15.89` | `37.81` | Stage IV: `Solar` + `Wind` + `Green H2` (`3,851.9 MWh_H2`) |

```bash
# Run multi-fuel CO2 Pareto frontier & MACC report + 4-panel figure
.venv/bin/python main.py --macc-report --save-macc-plot results/co2_pareto_frontier_and_macc.png

# Apply an explicit Pigouvian carbon tax or annual CO2 cap on the baseline 24 h scenario
.venv/bin/python main.py --co2-price 50 --co2-cap 20000
```

## Thermal Ramp-Rate Envelopes, Must-Run Floors & Spinning Reserves (v0.6.0)

Module 4 (`energy_model/flexibility.py`) introduces intertemporal thermal ramp-rate envelopes ($-r^{\text{dn}}_g K_g \le p_{g,t} - p_{g,t-1} \le r^{\text{up}}_g K_g$), minimum stable `must-run` loading ($p_{g,t} \ge p^{\min,\text{pu}}_g K_g$), co-optimized upward spinning operating reserves ($R^{\text{req}}_t = \alpha_{\text{load}} L_t + \alpha_{\text{vre}} P_{\text{VRE},t}$) coupled to battery state-of-charge (`SoC`) sustainment, and a direct sparse `scipy.optimize.linprog` (`HiGHS`) primal/dual cross-check:

1. **Co-Optimized Energy + Ancillary-Services Revenue Adequacy (Strong LP Duality):** Across all greenfield regimes with homogeneous operational, ramping, must-run, and cyclic `SoC` constraints, total annualised capital + fuel cost equals nodal energy market revenue plus exogenous load-reserve market revenue to machine precision ($\le 7.5 \times 10^{-9}$):
   $$\text{LCOE}_{\text{sys}} = \overline{\lambda}_{\text{load-wtd}} + \alpha_{\text{load}} \cdot \overline{\mu}^{\text{res}}_{\text{load-wtd}}$$
2. **Intertemporal Ramp-Induced Negative Nodal `LMP` Theorem (Zero Subsidies):** Even when every generator has strictly non-negative fuel cost ($c_g \ge 0$), binding downward and upward ramp limits across a deep midday net-load trough telescope multi-period fuel substitutions into an exact negative nodal shadow price ($\lambda_{t_0} = -138.00$ currency/MWh at $\min_g c_g = 0$).

![Duck-Curve Ramping, Must-Run Floors & Spinning Reserves](results/duck_curve_ramping_and_reserves.png)

### Verified 48 h Duck-Curve Operational Flexibility Regimes (Annualised)

| Regime | System `LCOE` (`/MWh`) | Peak `LMP` (`/MWh`) | Peak Reserve Price (`/MW-h`) | `VRE` Capacity (`MW`) | `OCGT` Peaker (`MW`) | `LiIon_BESS` (`MWh`) | Strong Duality Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_Relaxed_NoReserves` | `31.56` | `187.59` | `0.00` | `298.58` | `17.05` | `0.0` | `0.00e+00` |
| `2_MustRun_And_Ramping` | `33.61` | `187.59` | `0.00` | `67.32` | `24.96` | `0.0` | `0.00e+00` |
| `3_Inflexible_With_Reserves` | `33.80` | `187.59` | `109.59` | `59.70` | `36.46` | `0.0` | `7.45e-09` |
| `4_Flexible_BESS_CoOptimized` | `32.96` | `122.87` | `46.52` | `188.92` | `5.87` | `254.3` | `7.45e-09` |

```bash
# Run 48 h Duck-Curve flexibility & spinning-reserve comparison + 4-panel figure
.venv/bin/python main.py --flexibility-report --save-flexibility-plot results/duck_curve_ramping_and_reserves.png

# Evaluate explicit thermal ramp, must-run & spinning-reserve constraints on baseline 24 h profiles
.venv/bin/python main.py --ramp-limit 0.35 --must-run 0.25 --reserve-margin 0.08
```

## 4-Zone Meshed DC-OPF, KVL Loop Flows & Transmission Expansion (v0.7.0)

Module 5 (`energy_model/transmission.py`) models a 4-zone, 5-corridor 380 kV meshed AC transmission grid (`North_Wind`, `West_Solar`, `South_Hub`, `East_Metro` with two fundamental Kirchhoff Voltage Law cycles), analytical `PTDF` matrices, ISO Merchandising Surplus (congestion rent), co-optimized Transmission Expansion Planning (`AC-TEP`), and an independent sparse `scipy.optimize.linprog` (`HiGHS`) nodal phase-angle $B\theta$ cross-check:

1. **Cycle-Basis `KVL` vs. Nodal Voltage-Angle $B\theta$ / `PTDF` Duality:** Passive AC flows satisfy $f_{\ell,t} = b_\ell(\theta_{i,t} - \theta_{j,t}) = [\mathbf{\Phi}(\mathbf{g}_t - \mathbf{d}_t)]_\ell$ and nodal `LMP` decomposition $\boldsymbol{\lambda}_t = \lambda_{\text{ref},t}\mathbf{1} - \mathbf{\Phi}^\top(\boldsymbol{\mu}^+_t - \boldsymbol{\mu}^-_t)$ to $\le 1.5 \times 10^{-13}$.
2. **Triple Merchandising Surplus & Joskow–Tirole Cost Recovery:** ISO congestion rent satisfies $\sum_z \lambda_{z,t}(d_{z,t} - g_{z,t}) = \sum_{\ell=(i\to j)}(\lambda_{j,t} - \lambda_{i,t})f_{\ell,t} = \sum_\ell F^*_\ell(\mu^+_{\ell,t} + \mu^-_{\ell,t}) \ge 0$ at every hour, and recovers `100.00%` of optimal `AC-TEP` annualised line `CapEx` (`7,088,134` currency/yr, gap `0.00e+00`).
3. **Wheatstone-Bridge Braess's Paradox in Passive AC `DC-OPF`:** Adding a zero-cost `500 MW` passive AC bridge across a symmetric congested network reduces cheap power transfer (`120.0 -> 82.5 MW`) and raises dispatch cost by **`+62.50%`** (`4,800 -> 7,800` currency/h) via `KVL` loop binding, whereas a controllable `HVDC` link unlocks `160.0 MW` and cuts cost by **`-66.67%`** (`1,600` currency/h).

![4-Zone Meshed DC-OPF, Congestion Rent & Co-Optimized TEP](results/zonal_dcopf_congestion_and_tep.png)

### Verified 4-Zone 380 kV Transmission & `AC-TEP` Regimes (48 h Annualised)

| Regime | System `LCOE` (`/MWh`) | Mean `LMP` Spread (`/MWh`) | `North_Wind` (`MW`) | `East_OCGT` (`MW`) | Total Grid (`MW`) | Congestion Rent (`/yr`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_CopperPlate_SingleBus` | `19.88` | `0.00` | `836.91` | `0.00` | `0.0` | `0` |
| `2_Congested_AC_KVL_FixedGrid` | `30.74` | `52.90` | `385.51` | `154.47` | `400.0` | `67,585,265` |
| `3_Controllable_HVDC_Transport` | `28.44` | `50.90` | `489.75` | `131.40` | `400.0` | `81,148,386` |
| `4_CoOptimized_AC_TEP` | `21.59` | `2.70` | `835.53` | `7.98` | `737.1` | `7,088,134` |

```bash
# Run 4-zone DC-OPF, HVDC & co-optimized AC-TEP comparison + 4-panel figure
.venv/bin/python main.py --transmission-report --save-transmission-plot results/zonal_dcopf_congestion_and_tep.png

# Evaluate co-optimized 4-zone AC transmission expansion or corridor scaling
.venv/bin/python main.py --cooptimize-tep
.venv/bin/python main.py --grid-scale 1.5
```

## 2D WACC x CapEx Sensitivity Surface & Weather CVaR_95 Adequacy (v0.8.0)

Module 6 (`energy_model/risk_sensitivity.py` + `energy_model/cli.py`) unifies parametric 2D financing/learning-curve sensitivity analysis with two-stage multi-weather-year Rockafellar-Uryasev `CVaR_95` stochastic resource adequacy and ships the packaged `etm-sim` console entry point:

1. **Hotelling-Shephard Envelope Theorem & Global Cost Concavity:** Across real discount rates $r \in [1\%, 15\%]$ and VRE overnight `CapEx` multipliers $\kappa \in [0.55\times, 1.75\times]$, optimal greenfield cost $J^*(r, \kappa)$ is globally concave in annualised capital costs (midpoint concavity surplus `+5.82M` currency/yr; portfolio substitution dividend up to **`+22.02%`** vs. a frozen reference mix) and satisfies exact analytical partial derivatives $\partial J^*/\partial r = \sum_g K_g^* \kappa_g I_{0,g} \text{CRF}'(r, N_g)$ to $\le 5.4 \times 10^{-10}$.
2. **Rockafellar-Uryasev Two-Stage Stochastic `Mean-CVaR_95` & Risk-Adjusted Complete-Market Duality:** Across an 8-regime meteorological ensemble (`WY1_Windy_Equinox` $\to$ `WY8_Severe_Dunkelflaute`, $V_{\text{VoLL}} = 1{,}800$ /MWh), KKT epigraph multipliers $\rho_\omega^*$ induce an exact risk-adjusted probability measure $q_\omega^* = (1-\beta)\pi_\omega + \rho_\omega^*$ ($\sum_\omega q_\omega^* = 1$, amplifying tail `Dunkelflaute` weights by **`7.65x`** at $\beta = 0.35$) under which Strong LP Duality and 4-technology zero-profit revenue adequacy hold to $\le 1.5 \times 10^{-8}$, cross-checked against an independent sparse `scipy.optimize.linprog` (`HiGHS`) matrix to $\le 1.5 \times 10^{-15}$.
3. **Birge-Louveaux Stochastic Value Sandwich & Tail Blackout Elimination:** Deterministic mean-weather planning suffers **`194,479.6 MWh/yr`** out-of-sample `EENS` (`EEV = 385.13M` /yr vs. `RP = 49.37M` /yr, `VSS = 335.76M` /yr; `WS = 39.21M` /yr, `EVPI = 10.16M` /yr). Risk-neutral `RP` ($\beta = 0$) leaves `29.3 MWh/yr` residual Stoft scarcity shedding in `WY8`, whereas modest `CVaR_95` hedging ($\beta = 0.15$) eliminates **`100%` of load shedding (`0.0 MWh/yr`)** for **`+0.09%`** expected `LCOE` (`36.39 -> 36.42` /MWh).

![2D WACC x CapEx Sensitivity Surface & Weather CVaR_95 Adequacy](results/wacc_capex_surface_and_cvar_adequacy.png)

### Verified Multi-Year Weather `Mean-CVaR_95` Risk Frontier (`VoLL = 1,800` /MWh)

| Risk Weight $\beta$ | Expected Cost `E[Z]` (`/yr`) | Tail `CVaR_95` (`/yr`) | `E[LCOE]` (`/MWh`) | `EENS` (`MWh/yr`) | `Solar_PV` (`MW`) | `Wind` (`MW`) | `Li_Ion_BESS` (`MW`) | `Gas_Peaker` (`MW`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.00` (`RP`) | `49,367,339` | `123,698,409` | `36.39` | `29.3` | `144.9` | `320.9` | `75.8` | `184.4` |
| `0.15` | `49,409,902` | `122,790,530` | `36.42` | `0.0` | `149.4` | `318.9` | `81.1` | `184.5` |
| `0.35` (Hedged) | `49,452,092` | `122,643,766` | `36.45` | `0.0` | `159.5` | `313.1` | `81.6` | `184.1` |
| `0.60` | `49,925,568` | `122,289,968` | `36.80` | `0.0` | `192.5` | `299.8` | `81.9` | `184.1` |
| `1.00` (`CVaR`) | `83,937,794` | `113,396,667` | `61.87` | `0.0` | `491.7` | `0.0` | `0.0` | `273.0` |

```bash
# Run packaged etm-sim CLI: 2D WACC x CapEx surface + CVaR_95 frontier + 4-panel figure
.venv/bin/etm-sim --sensitivity-report --cvar-report \
  --save-risk-plot results/wacc_capex_surface_and_cvar_adequacy.png

# Evaluate custom financing/CapEx scenario + stochastic CVaR run + JSON telemetry export
.venv/bin/etm-sim --wacc 0.04 --vre-capex-scale 0.75 --cvar-beta 0.35 --json-summary /tmp/etm_summary.json
```

## Inputs, Assumptions & Provenance

* **Shipped 24 h baseline (`energy_model/data/profiles_synthetic_24h.csv`)** and **168 h `Dunkelflaute` sequence (`generate_dunkelflaute_profiles`)** are **deterministic synthetic benchmarks** designed to verify time-series constraints, KKT shadow-price duality, and chronological storage arbitrage; they are not metered SCADA or meteorological reanalysis records.
* **Cost inputs** (`DEFAULT_TECHNOLOGIES`, `BatteryStorageSpec`, `HydrogenP2XSpec`) are illustrative annualised parameters without a specified currency. Outputs represent exact mathematical consequences of the stated linear programs and are **not** real-world investment quotes or engineering certification.

## Quickstart & Verification

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip "setuptools>=77.0" wheel build
.venv/bin/python -m pip install --no-build-isolation -e ".[dev]"

# Representative-day baseline + KKT LMP duality + 168 h Dunkelflaute multi-duration storage
.venv/bin/python main.py --duality-report --dunkelflaute-report   --save-plot results/baseline_dispatch_and_lmp_duality.png   --save-storage-plot results/multiduration_storage_dunkelflaute_dispatch.png

# Lint, formatting check, and full pytest suite
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -v
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Haris Zafar.
