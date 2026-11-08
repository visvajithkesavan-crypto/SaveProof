---
name: architecture-guard
description: Use when designing a new module or feature, deciding where code belongs, adding a data source, refactoring across files, or making any change that touches more than 3 files. Keeps SaveProof's layered architecture intact and records decisions.
---

# Architecture guard

## The architecture — dependencies point one way
```
 sources: bills, meters, simulator
        │  ingest/   adapters → contract table   (messy formats live ONLY here)
        ▼
 contract.validate()                              (the single interface between layers)
        │
        ├─► quality.py   coverage, gaps, outliers → score + flags
        ▼
 engine/   PURE: no file I/O, no Streamlit, no global state
    baseline.py     fit model on the baseline period
    stats.py        CV(RMSE), NMBE, R², autocorrelation, FSU
    savings.py      adjusted baseline, savings, band, ADEETIE verdict
    comparators.py  naive bill diff, BEE SEC-ratio
    drift.py        CUSUM on post-retrofit residuals
        │  returns frozen dataclasses (JSON-serialisable)
        ▼
 report/ (charts, PDF, CO₂)      experiments/ (scripts → results/)
        ▼
 streamlit_app.py   thin: calls functions, renders results
```

## Allowed imports
| module | may import | must never import |
|---|---|---|
| config.py, contract.py | pandas, numpy | other saveproof modules |
| ingest/* | contract, config | engine, report, streamlit |
| quality.py | contract, config | engine, report, streamlit |
| engine/* | contract, config, numpy, pandas, statsmodels, scipy | ingest, report, streamlit; no file I/O |
| report/* | engine result types, config, matplotlib, reportlab | ingest, streamlit |
| experiments/* | any saveproof module | streamlit |
| streamlit_app.py | any saveproof module | — but holds no calculations |

If a change needs a forbidden import, the design is wrong: move the code, don't break the rule.

## Where does it go?
| you need to… | put it in |
|---|---|
| read a new file format | `ingest/<source>.py`, function `load_<source>()` |
| a threshold, emission factor, tariff number | `config.py`, with a source comment |
| a new statistic | `engine/stats.py` + a test against a hand-computed value |
| another way to estimate savings | `engine/comparators.py` (alternatives) or `engine/savings.py` (ours) |
| a chart used in both app and PDF | `report/charts.py`, returns a matplotlib Figure |
| a one-off analysis | an `experiments/` script; outputs to `results/` |

## Before implementing anything non-trivial, answer these in the chat
1. Which layer does it live in, and does it respect the import table?
2. Which submission requirement or judging criterion does it serve? If none, don't build it.
3. Can it be a plain function over the contract table?
4. What test proves it works?
5. Does it add a dependency, a file format or a config option? Justify each.

## Adding a data source
- [ ] `ingest/<source>.py` with `load_<source>(path) -> pd.DataFrame`
- [ ] output passes `contract.validate()`; `source` and `dataset` set
- [ ] units converted to kWh and timezone resolved at the edge
- [ ] contract test in `tests/ingest/test_<source>.py` using a small fixture, not the full dataset
- [ ] provenance and licence noted in `DATA_SOURCES.md`

## Result objects
The engine returns frozen dataclasses — `BaselineModel`, `SavingsResult`, `DriftResult` — holding numbers and metadata only (no raw DataFrames, no figures). The app and the PDF render the *same* object, so their numbers can't disagree.

## Decisions
Record anything costly to reverse in `docs/decisions.md`:
`## YYYY-MM-DD — <decision>` then Context / Decision / Alternatives rejected / Consequences. Five lines is enough.

## Scale story vs build
For judges: the engine is stateless and per-plant, so it scales horizontally (batch per cluster or per auditor). Say this in the deck; don't build infrastructure for it.
