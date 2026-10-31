# Energy Transition Optimization Model

[![CI](https://github.com/hariszafar740-lang/energy-transition-model/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/energy-transition-model/actions/workflows/ci.yml)
[![Pages](https://github.com/hariszafar740-lang/energy-transition-model/actions/workflows/deploy_pages.yml/badge.svg)](https://hariszafar740-lang.github.io/energy-transition-model/workbench/)
![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)
[![PyPI](https://img.shields.io/pypi/v/energy-transition-model?color=blue)](https://pypi.org/project/energy-transition-model/)
[![Tests: 285](https://img.shields.io/badge/tests-285%20passing-2ea043)](#quickstart--verification)
[![Offline workbench](https://img.shields.io/badge/workbench-offline%20%C2%B7%20zero%20requests-0e7490)](https://hariszafar740-lang.github.io/energy-transition-model/workbench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Capacity expansion, multi-duration storage and CO₂ pricing for a 100 % renewable grid, built with PyPSA and HiGHS, with every headline number cross-checked against a solver that shares no code with it.**

I built this as the practical half of the Energy Transition microcredential, and I keep the numbers reproducible because a capacity-expansion result is easy to produce and hard to trust.
I would read the LCOE as a relative comparison rather than a forecast: the WACC and CapEx surface is probably where most of the uncertainty sits, and the weather year is likely the second.
The abatement curve looks roughly steadier, though I would not quote it without the cross-check beside it.

285 tests · Python 3.11–3.13 · MIT · no data download, no API key, no network at run time.

**[▸ Open the workbench](https://hariszafar740-lang.github.io/energy-transition-model/workbench/)**, the whole model solved and in one self-contained HTML file.
Offline · [Quickstart](#quickstart--verification) · [Validation](#validation-six-independent-cross-checks-not-claims)

I would rather ship a narrower model I can verify than a broader one I cannot.
If a figure here ever stops matching the code, that is a bug in my README, and I want the issue.

## Validation: six independent cross-checks, not claims

| Check | This model | Independent reference | Residual | Reproduce with |
|:--- |:--- |:--- |:---: |:--- |
| LP strong duality | system LCOE = demand-weighted LMP = 28.0555 /MWh | KKT dual of the same LP | 7.45e-09 /yr | `test_kkt_strong_duality_and_zero_profit_equilibrium` |
| Nodal LMP | hourly λ(t) and capacities from PyPSA/linopy | standalone sparse `scipy.optimize.linprog` (HiGHS) primal + dual | rel. objective `< 1e-10` | `test_primal_dual_crosscheck_matches_independent_scipy_linprog` |
| CO₂ cap ↔ tax | `J*_tax(μ*) = J*_cap(Ē) + μ*Ē` | Lagrangian saddle point | rel. gap `< 1e-12` | `test_tax_to_cap_primal_and_saddle_point_equivalence` |
| Meshed DC-OPF | congestion rents from PyPSA PTDF flows | cycle-KVL / B-θ OPF on the same network | rel. objective `< 1e-11` | `test_scipy_btheta_matches_pypsa_cycle_kvl_to_machine_precision` |
| Mean-CVaR plan | risk-averse objective, CVaR, first-stage capacities | independent 1,357-variable matrix LP | rel. CVaR `< 1e-11` | `test_scipy_matrix_matches_pypsa_stochastic_cvar_to_machine_precision` |
| Offline workbench | `workbench/index.html` served by Pages | committed `workbench/data/etm_workbench_data.json` | byte-identical | `etm-workbench check`, re-run by CI on every push |

Every figure further down is asserted against the code that produces it: `tests/test_workbench_and_cvar.py` fails the build if a README number drifts from the committed dataset.

## Quickstart & verification

Four commands.
The first one downloads PyPSA, HiGHS, SciPy and Matplotlib and builds the package.
After that the full duality report below runs in about 10 s on a laptop.
No data download, no API key, nothing to configure:

~~~bash
git clone --depth 1 https://github.com/hariszafar740-lang/energy-transition-model.git
cd energy-transition-model
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/etm-sim --duality-report
~~~

~~~
System LCOE (currency/MWh): 28.0555
Demand-weighted LMP (currency/MWh): 28.0555
Strong-duality gap (currency/yr): -7.4506e-09
Max built zero-profit residual: 4.6566e-10
~~~

That is the solver's price of energy for a representative day, the demand-weighted dual price, and the gap between them, the identity asserted in the first row of the table above. Add `--dunkelflaute-report` for the 168 h storage study, `--macc-report` for the CO₂ abatement curve, `--cvar-report` for the risk-averse plan.

Full suite: `
.venv/bin/pip install -e ".[dev]" && .venv/bin/pytest -v`, 285 tests, usually several minutes on a laptop.

## Deep dive

Formulations, assumptions, sensitivities and figures, module by module.

An open-source energy-system model built on PyPSA (Python for Power System Analysis) and the HiGHS linear-programming solver.
It covers capacity expansion with chronological multi-duration storage (`Li-Ion BESS` + `Green H2` Power-to-X).
Discounted cash-flow (`WACC` / `CRF` / `LCOE`), KKT nodal shadow-price (`LMP`) duality, CO2 cap-and-trade and ramp/reserve adequacy.
On top of that sits meshed `DC-OPF`/`AC-TEP` transmission with mean-CVaR stochastic planning.
It also ships a **single-file, offline, zero-dependency interactive workbench** that renders every solved module in the browser.

![Representative-Day Dispatch, KKT Nodal LMP Duality & WACC Sensitivity](results/baseline_dispatch_and_lmp_duality.png)

## Academic Credential & Course Context

This repository serves as the practical engineering implementation for the official microcredential:
* Course & Badge Title: Modelling the Energy Transition, the revolution of open-source software.
* Issuing Institution: Politecnico di Milano.
* Qualification Level: EQF Level 7.

## Architecture and modules

* Single-bus LP baseline (`energy_model/model.py`): Representative-day capacity expansion (`Regional_Grid`, `System_Demand`, `Solar_PV`, `Wind_Turbine`, `Gas_Peaker`) with strict hourly time-series validation (`energy_model/profiles.py`).
* Discounted cash-flow & KKT nodal price duality (`energy_model/economics.py`).
 * Exact Capital Recovery Factor CRF(r, N) annualisation across real discount rates (WACC, r from 0 to 1).
 * Extraction of hourly KKT power-balance dual multipliers λ(t) (`LMP`), verification of LP Strong Duality (system LCOE = load-weighted λ = 28.0555 /MWh).
 * Boiteux-Turvey-Stoft Zero-Profit Equilibrium, and Brown-Hirth Market-Value Parity (LCOE_g = MV_g).
* Multi-duration storage & `Dunkelflaute` chronological adequacy (`energy_model/storage.py`).
 * Short-duration power-coupled battery (`Li_Ion_BESS` via PyPSA `StorageUnit`): 4 h duration, η(ch) = η(dis) = 95 % (η(RTE) = 90.25 %).
 * Cyclic State-of-Charge (`SoC`), optimal for daily midday-solar-to-evening shifting (278.7 eq. cycles/yr).
 * Long-duration decoupled Power-to-X hydrogen (`Green_H2` via PyPSA `Bus` + `Store` + `Link`s).
 * The three components are sized independently: `H2_Electrolyzer` (η(el) = 68 %), `H2_Cavern` (350 currency/MWh_H2/yr, cyclic `SoC`), `H2_Turbine` (η(fc) = 50 %, η(RTE) = 34.0 %).
 * Together they bridge multi-day `Dunkelflaute` wind/solar lulls (52.1 eq. cycles/yr).
 * Independent matrix LP cross-checks: Both `economics.py` and `storage.py` construct standalone `scipy.optimize.linprog(..., method="highs")` primal + dual matrices confirming `< 1e-11` agreement with PyPSA/linopy across capacities.
 * Objective values, and nodal shadow prices.
* CO2 cap–tax Lagrangian duality & MACC (`energy_model/emissions.py`): Partial-equilibrium multi-fuel emissions accounting, annual `CO2` caps.
* Pigouvian taxes, and the saddle-point identity J(tax)(μ) = J(cap)(Ē) + μ·Ē to 2.2e-08 currency/yr.
* Operational flexibility (`energy_model/flexibility.py`): Intertemporal ramp envelopes, must-run floors, co-optimised upward spinning reserves.
* And a proof that binding ramps can produce strictly negative nodal prices with non-negative fuel costs (-138.00 currency/MWh).
* Meshed transmission (`energy_model/transmission.py`): 4-zone / 5-corridor `DC-OPF` with `PTDF`/`KVL` equivalence.
* ISO merchandising surplus, `AC-TEP` co-optimisation, and a Wheatstone-bridge Braess paradox (`+62.5%` dispatch cost for a zero-cost passive line).
* Risk-aware planning (`energy_model/workbench.py`): An explicit 1,357-variable mean-CVaR linear program over 8 weather-and-demand regimes with a Rockafellar–Uryasev epigraph.
* Dual risk kernel q, and Birge–Louveaux bounds (`WS ≤ RP ≤ EEV`).
* Offline workbench (`workbench/`): `etm-workbench build` re-solves every module.
* It writes `workbench/data/etm_workbench_data.json`, and renders a single self-contained `workbench/index.html` (no CDN, no fonts, no network requests) that CI re-verifies byte-for-byte on every push.

## Multi-duration storage and `Dunkelflaute` adequacy

![Multi-Duration Storage Across a 168 h Chronological Dunkelflaute](results/multiduration_storage_dunkelflaute_dispatch.png)

### Coupled AC + Hydrogen Intertemporal Formulation

Over a chronological horizon t = 0 … T−1 (T = 168 h, annual weight w = 8760 / T):

~~~text
Σ_g p[g,t] + p[bess,dis,t] − p[bess,ch,t] + p[H2,dis,t] − p[H2,ch,t] = d[t] dual: w · λ[elec,t]
~~~

~~~text
e[bess,t] = (1 − σ[bess]) e[bess,(t−1) mod T] + η[ch] p[bess,ch,t] Δt − (1/η[dis]) p[bess,dis,t] Δt
 0 ≤ e[bess,t] ≤ τ[bess] P[bess]
~~~

~~~text
e[H2,t] = (1 − σ[H2]) e[H2,(t−1) mod T] + η[el] p[H2,ch,t] Δt − (1/η[fc]) p[H2,dis,t] Δt
 0 ≤ e[H2,t] ≤ E[H2] dual: w · λ[H2,t]
~~~

### 168-hour `Dunkelflaute` adequacy regimes (48 h cold wind and solar lull)

| Adequacy Regime | `Solar_PV` | `Wind_Turbine` | `Li_Ion_BESS` (4 h) | `H2` Electrolyzer / Turbine | `H2_Cavern` Store | System `LCOE` |
|:--- |:---: |:---: |:---: |:---: |:---: |:---: |
| 1. VRE Only (0% Gas) | 0.0 MW | 5,657.3 MW | n/a | n/a | n/a | 294.70 / MWh |
| 2. VRE + 4 h BESS (0% Gas) | 0.0 MW | 599.5 MW | 1,570.0 MW (6,280 MWh) | n/a | n/a | 66.28 / MWh |
| 3. VRE + Green `H2` (0% Gas) | 148.1 MW | 707.9 MW | n/a | 176.3 / 173.2 MW_e | `10,745 MWh_H2` | 51.92 / MWh |
| 4. Hybrid `BESS + H2` (0% Gas) | 180.2 MW | 679.6 MW | 51.8 MW (207 MWh) | 153.3 / 132.8 MW_e | `10,466 MWh_H2` | 51.25 / MWh |
| 5. Hybrid + Uncapped Gas Ref. | 72.2 MW | 375.8 MW | 58.5 MW (234 MWh) | 5.1 / 1.5 MW_e | `215 MWh_H2` | 43.49 / MWh |

* Efficiency vs.
* Energy-CapEx Crossover (`~6.9 h`): For short lulls (`0–6 h`), `Li-Ion BESS` (90.25% RTE) achieves lower zero-fossil `LCOE` than `Green H2` (24.01 vs.
* 26.53 / MWh at 0 h).
* Beyond `~6.9 h` (`12–72 h` `Dunkelflaute`), decoupled `H2` cavern storage (350 / MWh_H2 / yr) decisively beats 4 h battery packs (66.76 vs.
* 85.31 / MWh at 72 h), while the Hybrid `BESS + H2` portfolio forms the Pareto-optimal lower envelope across every duration.

## CO2 cap-tax duality and the abatement curve

Module 3 (`energy_model/emissions.py`) extends the single-bus capacity-expansion engine with primary-to-electric multi-fuel emissions accounting (e(g) = EF_th,c(g) / η(g)) and PyPSA `GlobalConstraint` annual CO₂ caps (Ē_CO2).
Pigouvian carbon taxes (τ) come with an independent sparse `scipy.optimize.linprog` (`HiGHS`) primal/dual cross-check:

1. Lagrangian cap-tax saddle-point equivalence. Relaxing Σ w(t) e(g) p(g),t Δt ≤ Ē with KKT shadow carbon price μ(CO2) = −dJ_cap(Ē)/dĒ ≥ 0 yields an exact Pigouvian carbon-tax objective. It holds at every point on the convex Pareto frontier:
J(tax)(μ(CO2)) = J(cap)(Ē) + μ(CO2) · Ē verified numerically to relative saddle-point residual ≤ 1.6e-15.
2. Net-Zero Boundary Subdifferential Regularisation (Ē = 0): Because the dual objective max bᵀy − Ēμ assigns zero weight to μ at Ē = 0, the subdifferential −dJ_cap(0) is the half-ray [τ(min)_net-zero, +∞). Evaluating the right-hand limit lim ε↓0⁺ μ(ε) uniquely extracts the minimum Net-Zero carbon price (418.25 currency / tCO₂).

![CO2 Pareto Frontier, MACC & Fuel Switching](results/co2_pareto_frontier_and_macc.png)

### Multi-fuel decarbonization milestones (168 h representative week, annualised)

| Abatement Target | Realised CO₂ (`tCO2/yr`) | Shadow Price μ(CO2) (`/tCO2`) | Average Cost (`/tCO2`) | Resource `LCOE` (`/MWh`) | Dominant Portfolio Regime |
|:---: |:---: |:---: |:---: |:---: |:--- |
| 0% (BAU) | 1,206,958 | 0.00 | 0.00 | 24.34 | `Coal_Steam` (181.8 MW) + `CCGT_Gas` (7.5 MW) |
| 45% Cap | 663,827 | 6.99 | 6.15 | 26.69 | Stage I: `Wind_Turbine` (220.0 MW) + `OCGT` |
| 75% Cap | 301,740 | 18.61 | 8.30 | 29.61 | Stage II: `Coal -> CCGT` (36.2 MW) + `Solar_PV` (61.6 MW) |
| 96% Cap | 48,278 | 52.77 | 12.84 | 34.79 | Stage III: 90% Gas_CCS (14.5 MW) + `LiIon_BESS` (40.5 MWh) |
| 100% Net-Zero | 0 | 418.25 | 15.89 | 37.81 | Stage IV: `Solar` + `Wind` + `Green H2` (`3,851.9 MWh_H2`) |

```bash
# Run multi-fuel CO2 Pareto frontier & MACC report + 4-panel figure
.venv/bin/python main.py --macc-report --save-macc-plot results/co2_pareto_frontier_and_macc.png

# Apply an explicit Pigouvian carbon tax or annual CO2 cap on the baseline 24 h scenario
.venv/bin/python main.py --co2-price 50 --co2-cap 20000
```

## Ramp-rate envelopes, must-run floors and spinning reserves

Module 4 (`energy_model/flexibility.py`) introduces intertemporal thermal ramp-rate envelopes (−r(dn),g K(g) ≤ p(g),t − p(g),t−1 ≤ r(up),g K(g)) and minimum stable `must-run` loading (p(g),t ≥ p(min),pu,g K(g)).
It also adds co-optimized upward spinning operating reserves (R(req),t = α_load L(t) + α_vre P(VRE),t) coupled to battery state-of-charge (`SoC`) sustainment.
And a direct sparse `scipy.optimize.linprog` (`HiGHS`) primal/dual cross-check:

1. Co-optimized energy and ancillary-services revenue adequacy (strong LP duality). Across all greenfield regimes with homogeneous operational, ramping, must-run and cyclic `SoC` constraints, total annualised capital + fuel cost equals nodal energy market revenue plus exogenous load-reserve market revenue to machine precision (≤ 7.5e-09):
LCOE_sys = λ̄[load-wtd] + α[load] · μ̄[res, load-wtd]
2. Intertemporal ramp-induced negative nodal `LMP` theorem (zero subsidies). Even when every generator has strictly non-negative fuel cost (c(g) ≥ 0), binding downward and upward ramp limits across a deep midday net-load trough telescope multi-period fuel substitutions into an exact negative nodal shadow price. At min c(g) = 0 that price is λ(t0) = −138.00 currency/MWh.

![Duck-Curve Ramping, Must-Run Floors & Spinning Reserves](results/duck_curve_ramping_and_reserves.png)
### Duck-curve flexibility regimes over 48 h (annualised)

| Regime | System `LCOE` (`/MWh`) | Peak `LMP` (`/MWh`) | Peak Reserve Price (`/MW-h`) | `VRE` Capacity (`MW`) | `OCGT` Peaker (`MW`) | `LiIon_BESS` (`MWh`) | Strong Duality Gap |
|:--- |:---: |:---: |:---: |:---: |:---: |:---: |:---: |
| 1_Relaxed_NoReserves | 31.56 | 187.59 | 0.00 | 298.58 | 17.05 | 0.0 | 0.00e+00 |
| 2_MustRun_And_Ramping | 33.61 | 187.59 | 0.00 | 67.32 | 24.96 | 0.0 | 0.00e+00 |
| 3_Inflexible_With_Reserves | 33.80 | 187.59 | 109.59 | 59.70 | 36.46 | 0.0 | 7.45e-09 |
| 4_Flexible_BESS_CoOptimized | 32.96 | 122.87 | 46.52 | 188.92 | 5.87 | 254.3 | 7.45e-09 |

```bash
# Run 48 h Duck-Curve flexibility & spinning-reserve comparison + 4-panel figure
.venv/bin/python main.py --flexibility-report --save-flexibility-plot results/duck_curve_ramping_and_reserves.png

# Evaluate explicit thermal ramp, must-run & spinning-reserve constraints on baseline 24 h profiles
.venv/bin/python main.py --ramp-limit 0.35 --must-run 0.25 --reserve-margin 0.08
```

## 4-zone DC-OPF, loop flows and transmission expansion

Module 5 (`energy_model/transmission.py`) models a 4-zone.
5-corridor 380 kV meshed AC transmission grid (`North_Wind`, `West_Solar`, `South_Hub`, `East_Metro`) with two fundamental Kirchhoff Voltage Law cycles and analytical `PTDF` matrices.
It reports ISO Merchandising Surplus (congestion rent) and co-optimized Transmission Expansion Planning (`AC-TEP`), and carries an independent sparse `scipy.optimize.linprog` (`HiGHS`) nodal phase-angle Bθ cross-check:

1. Cycle-basis `KVL` vs. nodal voltage-angle Bθ / `PTDF` duality. Passive AC flows satisfy f_ℓ,t = b_ℓ(θ_i,t − θ_j,t) = [Φ(g(t) − d(t))]_ℓ, and nodal `LMP` decomposition λ(t) = λ(ref),t·1 − Φᵀ(μ⁺_t − μ⁻_t) holds to ≤ 1.5e-13.
2. Triple merchandising surplus and Joskow–Tirole cost recovery. ISO congestion rent satisfies Σ λ(z),t(d(z),t − g(z),t) = Σ over each corridor of (λ(j),t − λ(i),t)·f_ℓ,t = Σ F_ℓ(μ⁺_ℓ,t + μ⁻_ℓ,t) ≥ 0 at every hour. It recovers 100.00% of optimal `AC-TEP` annualised line `CapEx` (7,088,134 currency/yr, gap 0.00e+00).
3. Wheatstone-bridge Braess's paradox in passive AC `DC-OPF`. Adding a zero-cost 500 MW passive AC bridge across a symmetric congested network reduces cheap power transfer (`120.0 -> 82.5 MW`) and raises dispatch cost by **`+62.50%`** (`4,800 -> 7,800` currency/h) via `KVL` loop binding. A controllable `HVDC` link, by contrast, unlocks 160.0 MW and cuts cost by -66.67% (1,600 currency/h).

![4-Zone Meshed DC-OPF, Congestion Rent & Co-Optimized TEP](results/zonal_dcopf_congestion_and_tep.png)
### 4-zone 380 kV transmission and `AC-TEP` regimes (48 h, annualised)

| Regime | System `LCOE` (`/MWh`) | Mean `LMP` Spread (`/MWh`) | `North_Wind` (`MW`) | `East_OCGT` (`MW`) | Total Grid (`MW`) | Congestion Rent (`/yr`) |
|:--- |:---: |:---: |:---: |:---: |:---: |:---: |
| 1_CopperPlate_SingleBus | 19.88 | 0.00 | 836.91 | 0.00 | 0.0 | 0 |
| 2_Congested_AC_KVL_FixedGrid | 30.74 | 52.90 | 385.51 | 154.47 | 400.0 | 67,585,265 |
| 3_Controllable_HVDC_Transport | 28.44 | 50.90 | 489.75 | 131.40 | 400.0 | 81,148,386 |
| 4_CoOptimized_AC_TEP | 21.59 | 2.70 | 835.53 | 7.98 | 737.1 | 7,088,134 |

```bash
# Run 4-zone DC-OPF, HVDC & co-optimized AC-TEP comparison + 4-panel figure
.venv/bin/python main.py --transmission-report --save-transmission-plot results/zonal_dcopf_congestion_and_tep.png

# Evaluate co-optimized 4-zone AC transmission expansion or corridor scaling
.venv/bin/python main.py --cooptimize-tep
.venv/bin/python main.py --grid-scale 1.5
```

## WACC and CapEx sensitivity surface, and weather CVaR_95 adequacy

Module 6 (`energy_model/risk_sensitivity.py` + `energy_model/cli.py`) unifies parametric 2D financing/learning-curve sensitivity analysis with two-stage multi-weather-year Rockafellar-Uryasev `CVaR_95` stochastic resource adequacy and ships the packaged `etm-sim` console entry point:

1. Hotelling-Shephard envelope theorem and global cost concavity. Across real discount rates r from 1 % to 15 % and VRE overnight `CapEx` multipliers κ from 0.55× to 1.75×, optimal greenfield cost J(r, κ) is globally concave in annualised capital costs. Midpoint concavity surplus is `+5.82M` currency/yr and the portfolio substitution dividend reaches **`+22.02%`** against a frozen reference mix. The cost also satisfies exact analytical partial derivatives dJ/dr = Σ K(g) κ_g I(0),g CRF′(r, N(g)) to ≤ 5.4e-10.
2. Rockafellar-Uryasev two-stage stochastic `Mean-CVaR_95` and risk-adjusted complete-market duality. Across an 8-regime meteorological ensemble (`WY1_Windy_Equinox` → `WY8_Severe_Dunkelflaute`, V(VoLL) = 1,800 /MWh), KKT epigraph multipliers ρ_ω induce an exact risk-adjusted probability measure q_ω = (1 − β)π_ω + ρ_ω. That measure sums to Σ q_ω = 1 and amplifies tail `Dunkelflaute` weights by **7.65x** at β = 0.35. Under it, strong LP duality and 4-technology zero-profit revenue adequacy hold to ≤ 1.5e-8, cross-checked against an independent sparse `scipy.optimize.linprog` (`HiGHS`) matrix to ≤ 1.5e-15.
3. Birge-Louveaux Stochastic Value Sandwich & Tail Blackout Elimination: Deterministic mean-weather planning suffers **194,479.6 MWh/yr** out-of-sample `EENS` (`EEV = 385.13M` /yr vs. `RP = 49.37M` /yr, `VSS = 335.76M` /yr; `WS = 39.21M` /yr, `EVPI = 10.16M` /yr). Risk-neutral `RP` (β = 0) leaves 29.3 MWh/yr residual Stoft scarcity shedding in `WY8`, whereas modest `CVaR_95` hedging (β = 0.15) eliminates 100% of load shedding (0.0 MWh/yr) for `+0.09%` expected `LCOE` (`36.39 -> 36.42` /MWh).

![2D WACC x CapEx Sensitivity Surface & Weather CVaR_95 Adequacy](results/wacc_capex_surface_and_cvar_adequacy.png)
### Multi-year weather `Mean-CVaR_95` risk frontier (VoLL = 1,800 /MWh)

| Risk Weight β | Expected Cost `E[Z]` (`/yr`) | Tail `CVaR_95` (`/yr`) | `E[LCOE]` (`/MWh`) | `EENS` (`MWh/yr`) | `Solar_PV` (`MW`) | `Wind` (`MW`) | `Li_Ion_BESS` (`MW`) | `Gas_Peaker` (`MW`) |
|:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |
| 0.00 (`RP`) | 49,367,339 | 123,698,409 | 36.39 | 29.3 | 144.9 | 320.9 | 75.8 | 184.4 |
| 0.15 | 49,409,902 | 122,790,530 | 36.42 | 0.0 | 149.4 | 318.9 | 81.1 | 184.5 |
| 0.35 (Hedged) | 49,452,092 | 122,643,766 | 36.45 | 0.0 | 159.5 | 313.1 | 81.6 | 184.1 |
| 0.60 | 49,925,568 | 122,289,968 | 36.80 | 0.0 | 192.5 | 299.8 | 81.9 | 184.1 |
| 1.00 (`CVaR`) | 83,937,794 | 113,396,667 | 61.87 | 0.0 | 491.7 | 0.0 | 0.0 | 273.0 |

```bash
# Run packaged etm-sim CLI: 2D WACC x CapEx surface + CVaR_95 frontier + 4-panel figure
.venv/bin/etm-sim --sensitivity-report --cvar-report \
--save-risk-plot results/wacc_capex_surface_and_cvar_adequacy.png

# Evaluate custom financing/CapEx scenario + stochastic CVaR run + JSON telemetry export
.venv/bin/etm-sim --wacc 0.04 --vre-capex-scale 0.75 --cvar-beta 0.35 --json-summary /tmp/etm_summary.json
```

## Interactive Energy-Transition Workbench
Module 7 turns the model family into an artifact you can open: a single ~170 kB HTML file with no CDN.
No web fonts, no `<script src>`, no `fetch()`, and no build step required to read it.
Open `workbench/index.html` from a local clone (or the live Pages deployment) and every panel (dispatch, storage, carbon.
Reserves, meshed grid and risk) is driven by numbers that were produced by solving the linear programs in this repository.

[▶ Open the live workbench](https://hariszafar740-lang.github.io/energy-transition-model/workbench/) · `workbench/index.html` · `workbench/data/etm_workbench_data.json`
### Build → render → verify contract

```
solved models ──► workbench/data/etm_workbench_data.json ──► workbench/index.html
 (PyPSA/HiGHS) (single source of truth) (single-file UI)
```

* `etm-workbench build` re-solves all six model families (149 optimisations, 118 PyPSA/linopy solves + 31 explicit mean-CVaR LPs.
* ≈120 s), embeds the results as a JSON dataset with an `etm-workbench/1` schema, and renders the page.
* `etm-workbench render` re-renders the page from the committed dataset only (≈2 s, no solver).
* `etm-workbench check` re-renders in memory and compares byte-for-byte with the committed file.
* CI fails on any drift, so a hand-edited figure cannot survive a push.
* Every page embeds `sha256` of its dataset in `<meta name="etm-data-digest">`.
* The footer prints the same digest, so a downloaded page can be traced back to the exact dataset that produced it.

### The page itself

| View | What it shows | Interaction |
|:--- |:--- |:--- |
| 00 Overview | 5×5 exact `LCOE` surface, module map, cross-module invariant table | click any cell → loads that case |
| 01 Dispatch & LCOE | stacked hourly dispatch, annual cost stack, nodal price | `WACC` and VRE-CapEx sliders (bilinear interpolation between 25 solved nodes) |
| 02 Storage & lulls | `BESS`-only / `H₂`-only / hybrid `LCOE` vs. lull length, 168 h SoC and dispatch | lull-duration scrubber |
| 03 Carbon & MACC | Pareto frontier, dual carbon price, step `MACC`, fuel-switching mix | abatement-target scrubber |
| 04 Flexibility | net load vs. nodal price, reserve requirement vs. reserve price | 4-regime switch |
| 05 Meshed grid | live 4-zone network diagram, corridor flows, zonal `LMP`, `PTDF` matrix | 4-regime switch + hour slider |
| 07 Risk & CVaR | mean–CVaR frontier, dual risk kernel q, Birge–Louveaux bounds | β scrubber |
| 99 Method & checks | surrogate holdout parity, build provenance, test inventory, explicit limits | n/a |

Keyboard: `1…8` switch views, `[` / `]` step through them.
Each view has a deep-link hash (`#/risk?beta=5`), CSV export of its table, SVG export of its active chart, and a light/dark theme toggle.
The page works offline, from `file://`, in print, and under `prefers-reduced-motion`.
The workbench is mostly a renderer: it solves nothing when you open it.

### The explicit mean-CVaR program

Eight weather-and-demand regimes ω (windy equinox → severe `Dunkelflaute`, π_ω summing to 1.000) are solved jointly as one linear program in 1,357 variables and 1,352 constraints.
The constraint matrices are explicit and sparse:

~~~text
min over K, p, η, z: (1 − β) Σ_ω π_ω Z_ω + β [ η + (1/(1 − α)) Σ_ω π_ω z_ω ]
subject to: Z_ω ≤ η + z_ω, z_ω ≥ 0
~~~

where Z_ω is the annual cost of future ω (annualised CapEx + gas fuel + `VoLL` × unserved energy, 1,800 currency/MWh) and K is the shared first-stage capacity vector. The hourly recourse dispatch p_ω carries the cyclic battery state of charge. The epigraph is the Rockafellar–Uryasev representation of CVaR_α, so η = VaR_α, and the duals of the epigraph rows are the risk kernel q.

![Mean-CVaR Frontier, Dual Risk Kernel, Birge–Louveaux Bounds & Hedge Composition](results/risk_frontier_and_cvar_kernel.png)

#### Verified risk/return frontier (α = 95 %, annualised, cost in `currency/yr`)

| β | E[Z] (M) | CVaR_95 (M) | System `LCOE` | Risk premium | Tail reduction | `EENS` (MWh/yr) | `Solar` | `Wind` | `Gas` | `BESS` |
|:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |
| 0.00 | 40.14 | 105.24 | 28.79 / MWh | `+0.00` | 0.0% | 167.5 | 79 MW | 309 MW | 192 MW | 69 MW |
| 0.10 | 40.41 | 99.67 | 28.98 / MWh | `+0.19` | 5.3% | 0.0 | 91 MW | 315 MW | 192 MW | 99 MW |
| 0.20 | 40.78 | 97.57 | 29.25 / MWh | `+0.46` | 7.3% | 0.0 | 51 MW | 374 MW | 189 MW | 97 MW |
| 0.35 | 42.08 | 94.86 | 30.18 / MWh | `+1.39` | 9.9% | 0.0 | 61 MW | 413 MW | 215 MW | 67 MW |
| 0.50 | 42.41 | 94.38 | 30.42 / MWh | `+1.63` | 10.3% | 0.0 | 30 MW | 437 MW | 215 MW | 65 MW |
| 0.65 | 48.55 | 90.31 | 34.82 / MWh | `+6.03` | 14.2% | 0.0 | 0 MW | 571 MW | 172 MW | 94 MW |
| 0.80 | 59.41 | 86.40 | 42.61 / MWh | `+13.82` | 17.9% | 0.0 | 0 MW | 752 MW | 216 MW | 32 MW |
| 0.90 | 60.75 | 86.17 | 43.57 / MWh | `+14.78` | 18.1% | 0.0 | 0 MW | 772 MW | 203 MW | 43 MW |
| 1.00 | 61.23 | 86.12 | 43.92 / MWh | `+15.13` | 18.2% | 0.0 | 0 MW | 779 MW | 198 MW | 47 MW |

Reading the frontier: 1.39 currency/MWh of insurance buys 9.9% off the worst-5% tail and eliminates every expected blackout hour (β = 0.35).
Beyond β ≈ 0.65 the hedge rotates out of solar entirely into wind plus storage, and the marginal price of tail reduction rises steeply.
The classic convexity of the mean–CVaR frontier.
Objective-to-primal duality gap `≤ 7.4e-08` currency/yr against qᵀZ at every β.

#### Dual risk kernel q (Rockafellar–Uryasev)

At β = 0.5 the epigraph duals concentrate probability exactly where the tail lives:

| Regime | π_ω | q_ω (β=0.5) | Ceiling π_ω/(1 − α) |
|:--- |:---: |:---: |:---: |
| `WY1_Windy_Equinox` | 0.180 | 0.090 | 3.60 |
| `WY2_Sunny_Anticyclone` | 0.170 | 0.085 | 3.40 |
| `WY3_Temperate_Normal` | 0.250 | 0.125 | 5.00 |
| `WY4_Cloudy_Maritime` | 0.160 | 0.080 | 3.20 |
| `WY5_Summer_Heat_Wave` | 0.140 | 0.170 | 2.80 |
| `WY6_Winter_Cloud_Trough` | 0.060 | 0.030 | 1.20 |
| `WY7_Polar_Vortex_Freeze` | 0.025 | 0.2625 | 0.50 |
| `WY8_Severe_Dunkelflaute` | 0.015 | 0.1575 | 0.30 |

The kernel satisfies Σ q_ω = 1 and 0 ≤ q_ω ≤ π_ω/(1 − α) to 1e-9.
At β = 1 it degenerates onto the three worst futures only (`WY8` 0.30, `WY7` 0.50, `WY2` 0.20) each sitting exactly on its ceiling.
This is what "pure worst-tail optimisation with £-per-tail-mass pricing" means numerically.

#### Birge–Louveaux stochastic bounds (risk-neutral, β = 0)

| Bound | Annual cost | Construction |
|:--- |:---: |:--- |
| Wait-and-see `WS` | 28.62 M | bespoke capacity plan per future (8 separate LPs) |
| Recourse `RP` | 40.14 M | one hedged plan across all 8 futures |
| Expected-value plan `EEV` | 345.08 M | plan built for the mean future, then evaluated on all 8 |

`EVPI = 11.52 M/yr`, `VSS = 304.94 M/yr`, ordering `WS ≤ RP ≤ EEV` holds with slack.
The expected-value plan builds 24 MW solar, 320 MW wind, 93 MW battery and 0 MW gas.
Averaged over the eight futures it costs `8.6×` the hedged plan, because a single mean future never contains the `+78 MW` `Dunkelflaute` night that dominates the tail.

### The reduced-order surrogate, and its measured error

The overview and dispatch views let the reader sweep `WACC in [1%, 15%]` and VRE CapEx ` in [0.55, 1.75]×` continuously.
A continuous sweep is not free: every grid node is one LP.
The workbench therefore ships the exact 5×5 solved surface and bilinearly interpolates between nodes.
It then measures the interpolation error on 16 node positions that were never used to build it.

| Metric | Value |
|:--- |:---: |
| Holdout solves | 16 (4 × 4 mid-quadrant nodes) |
| `R²` (surrogate vs. exact) | 0.999705 |
| `RMSE` | 0.206 currency/MWh |
| Max absolute error | 0.80 currency/MWh |
| Max relative error | 1.37% |

Where a slider sits between solved nodes, the page labels every KPI as *interpolated* and draws the hourly chart from the nearest exactly-solved node.
Interpolation never masquerades as a solve.

### Reproduce the workbench

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip "setuptools>=77.0" wheel build
.venv/bin/python -m pip install --no-build-isolation -e ".[dev]"

# Re-solve every module (149 optimisations) and re-render the single-file page
.venv/bin/etm-workbench build

# Cheap re-render from the committed dataset, and the drift gate used by CI
.venv/bin/etm-workbench render
.venv/bin/etm-workbench check

# Static 4-panel figure for this section
.venv/bin/python -c "from energy_model.workbench import plot_risk_frontier_and_kernel; \
plot_risk_frontier_and_kernel('results/risk_frontier_and_cvar_kernel.png')"
```

### Limits of the workbench (read before citing)

* The `CVaR` panel is an independent explicit-LP re-derivation on its own declared 8-regime scenario set (own demand shape, own VRE availability, own `VoLL`).
* It is *not* a numerical replica of the Module 6 risk model, and it deliberately does not call it.
* Agreement between two independent formulations would be meaningless if one called the other.
* The 8 regimes are synthetic stress shapes, not reanalysis weather years; π_ω is a declared prior, not an estimated frequency.
* All costs are illustrative and currency-agnostic.
* The surrogate is an interpolation of solved LPs, bounded by the holdout error above.
* It is not a fitted model and carries no extrapolation guarantee outside the plotted grid (sliders clamp at the edges).
* In practice the binding limit is the input data, not the solver: synthetic profiles and illustrative costs set the ceiling on what any number here can mean.
* `DC` power flow, lossless corridors, continuous investment, no unit commitment; see the shared boundaries in *Inputs, Assumptions & Provenance*.
## Inputs, Assumptions & Provenance

* Shipped 24 h baseline (`energy_model/data/profiles_synthetic_24h.csv`) and 168 h `Dunkelflaute` sequence (`generate_dunkelflaute_profiles`) are deterministic synthetic benchmarks designed to verify time-series constraints.
* KKT shadow-price duality, and chronological storage arbitrage.
* They are not metered SCADA or meteorological reanalysis records.
* Cost inputs (`DEFAULT_TECHNOLOGIES`, `BatteryStorageSpec`, `HydrogenP2XSpec`) are illustrative annualised parameters without a specified currency.
* Outputs represent exact mathematical consequences of the stated linear programs and are not real-world investment quotes or engineering certification.
* I chose synthetic profiles over a data download, so the whole repository runs offline in one command.
* Workbench dataset (`workbench/data/etm_workbench_data.json`) is generated, never hand-edited: it is the serialised output of `etm-workbench build`.
* Its `sha256` is embedded in `workbench/index.html`, and CI fails if the two ever disagree.
* The 8 Module-7 regimes are declared synthetic stress cases with an assumed prior, not estimated weather frequencies.

## Quickstart & Verification

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip "setuptools>=77.0" wheel build
.venv/bin/python -m pip install --no-build-isolation -e ".[dev]"

# Representative-day baseline + KKT LMP duality + 168 h Dunkelflaute multi-duration storage
.venv/bin/python main.py --duality-report --dunkelflaute-report --save-plot results/baseline_dispatch_and_lmp_duality.png --save-storage-plot results/multiduration_storage_dunkelflaute_dispatch.png

# Lint, formatting check, artifact freshness gate, and full pytest suite
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/etm-workbench check
.venv/bin/pytest -v
```

> **Workbench in one line:** `etm-workbench build` re-solves all six model families and rewrites `workbench/index.html`; `etm-workbench check` proves the committed page still matches the committed dataset. The published copy lives at <https://hariszafar740-lang.github.io/energy-transition-model/workbench/>.

## License

Released under the [MIT License](LICENSE).
Copyright (c) 2026 Haris Zafar.
