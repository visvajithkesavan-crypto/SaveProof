---
name: debug-energy-data
description: Use when something is wrong — a failing test, implausible savings, numbers that differ between app and report, a badly fitting model, a crash, or results that changed unexpectedly. Systematic debugging plus the failure modes specific to meter, bill and production data.
---

# Debugging, energy-data edition

## Method — no guessing
1. **Reproduce** with the smallest input (one dataset, a few days). Write down the command, expected and actual.
2. **Look at the data before the code:** `df.describe()`, `df.isna().sum()`, rows per day, and a plot of kWh and production over time. Most "model bugs" are data bugs.
3. **One hypothesis at a time.** State it, design a check that could prove it wrong, run it. Never change two things at once.
4. **Fix the cause, not the symptom.** No widened tolerances, no try/except to hide errors, no special case for one dataset.
5. **Lock it in:** the minimal failing input becomes a regression test that fails before the fix and passes after.
6. **Explain it** in the commit message: what was wrong and why.

## Failure catalogue — check these first
| symptom | likely cause | check |
|---|---|---|
| totals off by ×1000 or ×4 | W vs kW, kWh vs MWh; power averaged where energy should be summed (15-min kW × 0.25 = kWh) | units in names; recompute one day by hand |
| daily totals shifted by one interval | interval-end vs interval-start timestamps (UCI marks end, Irish marks start) | first and last timestamp of one day |
| UCI day starts with a midnight spike / order broken | UCI's `00:00` row is 24:00 of the labelled day | see energy-data-ingestion |
| days with 92 or 100 intervals | DST transitions (Ireland) | rows per local day |
| huge or negative jumps | cumulative register treated as interval energy; rollover; meter replaced | is the raw series monotonic? |
| savings with no retrofit | production rose and the model lacks a base-load term, or SEC-ratio logic crept into our engine | null test on that window; inspect intercept |
| band too narrow, null test fails | autocorrelated residuals ignored | lag-1 ρ; use n′ or aggregate further |
| great baseline fit, absurd savings | reporting production outside the baseline range | coverage check 90–110% |
| savings vanish or flip sign | leakage: model fitted or scaled with reporting data | which rows went into `fit()`? |
| bill model is noisy | billing cycles of different lengths treated as equal months | per-day normalisation |
| dates swapped (Mar ↔ Dec) | pandas guessed day/month order | explicit `format=` |
| first column is `﻿date` | UTF-8 BOM | `encoding="utf-8-sig"` |
| site total too high | summing hierarchical sub-meters | use the incomer only |
| grid import dips on sunny days | on-site PV | model consumption, or include PV |
| outliers around festivals | Diwali or holiday shutdowns | `event="holiday"`; never delete silently |
| app and PDF disagree | each recomputes separately | both render the same result dataclass |

## Statistical sanity checks
- Production coefficient positive and close to the sector's marginal SEC (kWh per unit); intercept ≈ daily base load, ≥ 0.
- Residuals: mean ≈ 0, no trend over time, no pattern against production. Heteroscedasticity → discuss a per-unit or log model before changing anything.
- Production and operating days highly correlated → coefficients swing; check correlation, consider dropping one.
- Push a known-answer synthetic case through the same code path. If that fails, it's the code, not the data.

## Stuck for 30 minutes?
Write down the exact symptom, what you've ruled out and how, and the smallest reproducing input. Then ask your teammate or Claude with that note.
