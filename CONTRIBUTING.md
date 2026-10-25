# Contributing Guide

Thank you for your interest in `energy-transition-model`.

## Development Environment

1. Use Python `3.11`, `3.12`, or `3.13`.
2. Create an isolated virtual environment and install editable dependencies:
   - `python -m venv .venv`
   - `.venv/bin/python -m pip install --upgrade pip "setuptools>=77.0" wheel build`
   - `.venv/bin/python -m pip install --no-build-isolation -e ".[dev]"`

## Verification Gates

Every change must pass all local quality gates before opening a pull request:

1. Static lint check: `ruff check .`
2. Formatting check: `ruff format --check .`
3. Full test suite: `pytest -v`
4. CLI smoke run: `python main.py`
5. Package wheel/sdist build: `python -m build --no-isolation`

## Mathematical & Data Integrity Standards

- Never silently fill missing, non-numeric, or out-of-range time-series inputs; fail loudly with explicit `ValueError` messages.
- Verify new optimization formulations against analytical KKT optimality identities or independent matrix `scipy.optimize.linprog` primal/dual checks.
- Document synthetic profiles and cost assumptions transparently without overstating real-world project economics.
