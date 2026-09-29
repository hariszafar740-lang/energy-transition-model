# Energy Transition Optimization Model

An open-source energy system capacity expansion model built using **PyPSA** (Python for Power System Analysis) and the **HiGHS** linear programming solver engine.

## Academic Credential & Course Context
This repository serves as the practical engineering implementation for the official microcredential:
* **Course & Badge Title:** Modelling the Energy Transition: the revolution of open-source software
* **Issuing Institution:** Politecnico di Milano 
* **Qualification Level:** EQF Level 7

## Technical Overview
Formulates a least-cost capacity expansion and multi-period dispatch optimization model for renewable grid integration (Solar PV, Wind, Thermal Backup), mathematically minimizing Net Present Cost (NPC) under technical linear constraints.

## Quickstart
```bash
python3 -m venv venv
source venv/bin/activate
pip install pypsa highspy pandas matplotlib
python3 main.py
