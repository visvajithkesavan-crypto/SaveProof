---
name: energy-data-ingestion
description: Use when writing or fixing an adapter in saveproof/ingest, the data contract (contract.py), the data quality score (quality.py), the simulator's output format, or Indian utility-bill handling (billing cycles, kVAh, max demand). Turns messy meter and bill data into the contract table safely.
---

# Energy data ingestion

## Principle
Messy formats stop at the edge. Each adapter turns one source into the contract table (see `00-saveproof-context`). Nothing downstream knows where data came from except the `source` and `dataset` columns.

## Adapter template
```python
def load_<source>(path: Path) -> pd.DataFrame:
    """Read <source>; return the contract table (daily rows, kWh, plant-local dates)."""
    raw = pd.read_csv(path, encoding="utf-8-sig")   # strips a BOM if present
    raw.columns = raw.columns.str.strip()           # some headers have trailing spaces
    # 1. parse timestamps explicitly (format=...), resolve timezone and interval start/end
    # 2. convert to interval kWh (never cumulative registers)
    # 3. aggregate to the period: kwh summed; coverage_pct = readings present / expected
    # 4. build contract columns; set source, dataset, production_unit
    return contract.validate(df)
```
Always pass an explicit `format=` when parsing dates; never let pandas guess day/month order. Raw files are read-only; processed outputs go to `data/processed/`.

## Known sources and their traps
| dataset | what to know |
|---|---|
| UCI Steel `Steel_industry_data.csv` | BOM in header. `date` is `dd/mm/yyyy HH:MM`, timestamps mark **interval end** (first row 00:15). **Each day's `00:00` row is midnight at the END of that labelled day** (NSM = 0, it follows 23:45). Fix: if time is 00:00, add one day → true interval end; interval start = end − 15 min. Naive parsing sorts it to the start of the day and breaks the time order. 15-min `Usage_kWh`. No production → proxy `proxy:active_intervals` = count of Medium_Load + Maximum_Load intervals per day. Year 2018 only. |
| Irish site `extracted/*_15min_ALLphase.csv` | `window_start_utc` = UTC **interval start**; `WindowStartLocal`/`DateLocal` are Europe/Dublin (DST: 92/100-interval days). Use `mi_a` (grid incomer) for whole-site. Meters are hierarchical — never sum them. PV (`mi_b`) makes grid import < consumption on sunny days. `IsReliableWindow`, `DataCoveragePct` and `metadata/meter_data_quality_log.csv` are ground truth for our quality score. Grid meter starts 22 May 2025. |
| Laayati mine sample | `Time` is `dd/mm/yy H:MM`, hourly, ~32 days; header `HM ` has a trailing space. Demo use only. |
| Simulator | Already contract-shaped; `source="simulated"`. True savings stay in plant metadata, never in the table. |
| Indian bills (synthetic) | One row per billing cycle (often 28–35 days, not calendar months). HT bills may carry kVAh, max demand (kVA) and power factor. |

## Indian bill specifics
- Normalise to per-day values before modelling (`kwh / days`, `production / days`).
- kVAh billing: kWh = kVAh × power factor. Model kWh (physical energy); keep kVAh for cost only. If only kVAh is given, convert with the billed PF and flag it.
- The meter-reading date isn't the month end. Production must be cut to the same dates or the regression is misaligned.
- Tariff numbers (energy charge, demand charge, ToD, PF incentive) come only from the state tariff order, cited in `config.py`. Cost is outside savings verification.

## Data quality score (0–100)
Show the components, not just the total:
- completeness: readings present / expected (coverage_pct)
- continuity: number of gaps, longest gap
- plausibility: negatives, zeros on operating days, spikes beyond k·MAD
- consistency: duplicate timestamps, non-monotonic cumulative registers, kWh-per-unit outliers
Flag, don't fix. The engine decides what to exclude based on the flags.

## contract.validate()
Raise `ValueError` naming the column and the first bad row when: a required column is missing, a dtype is wrong, kwh/production is negative, period_end < period_start, periods overlap, or `source` isn't one of the allowed values. Return the frame sorted by `period_start`.

## Licensing
Never commit anything in `data/`. UCI and Irish data are CC BY 4.0 (attribution in DATA_SOURCES.md). BEE/UNIDO/CEA PDFs and the Schneider register map: cite, never redistribute. ASI microdata: aggregates only.
