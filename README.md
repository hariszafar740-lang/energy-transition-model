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

## Architecture & Module Capabilities (`v0.4.0`)

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
