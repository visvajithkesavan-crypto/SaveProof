# SaveProof build plan (8–20 Nov 2026)

Each task says what it **receives** and what it **returns**. These signatures are the agreement between Visva (engine, experiments, report) and Rohith (data, simulator, app). Change a signature only by editing this file and telling the other person.
All signatures are proposals — review them together before Task 0 is committed.

## Task 0 — the shared contract (both, together, 1 hour, 8–9 Nov)
`saveproof/contract.py`
```python
SOURCES = {"real_meter", "utility_bill", "simulated"}
REQUIRED = ["period_start", "period_end", "kwh", "production", "production_unit",
            "operating_days", "source", "dataset", "coverage_pct", "event"]

@dataclass(frozen=True)
class PlantMeta:
    plant_id: str
    sector: str
    timezone: str
    baseline_start: pd.Timestamp   # all dates: tz-naive, midnight, so they compare with datetime64 columns
    baseline_end: pd.Timestamp
    reporting_start: pd.Timestamp
    reporting_end: pd.Timestamp
    retrofit_date: pd.Timestamp | None

def validate(df: pd.DataFrame) -> pd.DataFrame: ...   # raises ValueError; returns sorted copy
```
`saveproof/config.py` — every value with its source comment:
`CONFIDENCE_REPORT = 0.90`, `CONFIDENCE_ASHRAE = 0.68`, `FSU_MAX_ASHRAE = 0.50`, `CVRMSE_GUIDE = 0.25`, `NMBE_GUIDE = 0.005`, `R2_GUIDE = 0.70`, `PROD_RANGE_LOW_FACTOR = 0.90`, `PROD_RANGE_HIGH_FACTOR = 1.10`, `MAX_EXTRAPOLATED_SHARE = 0.25` (added in V3), `MAX_EXCLUDED_SHARE = 0.25`, `ADEETIE_THRESHOLD = 0.10`, `GRID_FACTOR_TCO2_PER_MWH = 0.675`.
Test: `tests/test_contract.py` — a valid frame passes; each rule raises with the column name.
Also write `docs/validation_plan.md` (pass criteria from the validation-testing skill) and commit it **before** any experiment runs.

## Visva — engine, experiments, report
| # | dates | task | receives | returns | proving test |
|---|---|---|---|---|---|
| V1 | 9 Nov | `engine/stats.py` | arrays | `cvrmse(y, yhat, p) -> float`, `nmbe(y, yhat, p) -> float`, `lag1_autocorr(resid) -> float`, `n_effective(n, rho) -> float`, `savings_uncertainty_kwh(cv, n, p, rho, m, adjusted_baseline_kwh, confidence, monthly: bool) -> float` | hand-computed 6-row example; uncertainty independent of savings size |
| V2 | 9–10 Nov | `engine/baseline.py` | contract df (baseline rows) | `fit_baseline(df, formula="kwh ~ production + operating_days") -> BaselineModel`; `predict(model, df) -> pd.Series` (`kwh_pred`); `add_calendar_terms(df) -> df` (adds `day_type` = weekday/weekend from `period_start`, and `period_days` = calendar days, end inclusive; neither is a contract column; `fit_baseline` raises if monthly data uses `day_type`); `usable_rows(df, formula) -> pd.Series[bool]` (V3 reuses it); `InsufficientDataError(ValueError)` | known answer: `kwh = 180 + 0.4·production + noise` recovers 180 and 0.4 |
| V3 | 10 Nov | `engine/savings.py` | `BaselineModel`, contract df (reporting rows) | `verify_savings(model, df, confidence=CONFIDENCE_REPORT) -> SavingsResult`; `decide_verdict(...) -> str` | identical periods → savings ≈ 0; kWh × k → savings % unchanged; band width independent of the saving |
| V4 | 11 Nov | `engine/comparators.py` | baseline df, reporting df | `naive_savings(baseline_df, reporting_df, formula="kwh ~ 1") -> ComparatorResult`, `bee_sec_savings(baseline_df, reporting_df) -> ComparatorResult` | production +20%, no retrofit → SEC-ratio > 0, SaveProof ≈ 0 |
| V5 | 11–12 Nov | `experiments/null_test.py`, `experiments/injected_saving.py` | Rohith's UCI + Irish loaders | CSV + PNG in `results/` | criteria in `docs/validation_plan.md` |
| V6 | 13 Nov | `engine/drift.py` | `BaselineModel`, reporting df | `cusum(model, df) -> DriftResult` | planted 0.5%/week degradation triggers alarm; untouched data → false-alarm rate reported |
| V7 | 14–15 Nov | `report/co2.py`, `report/mrv_pdf.py` | result dataclasses + `PlantMeta` + `QualityReport` | `tco2_avoided(savings_kwh) -> float`; `build_mrv_pdf(...) -> Path` | PDF opens; numbers equal `SavingsResult` fields |

V2 note: monthly formula for the blind test is chosen on the dev set and written into `validation_plan.md` §6 before the blind run.

Result types (frozen dataclasses in the engine module that creates them):
- `BaselineModel`: formula, params, n, p, r2, cvrmse, nmbe, rho, n_eff, production_min, production_max (NaN if the formula has no production), monthly (bool, baseline periods longer than one day), n_excluded (baseline periods dropped by `usable_rows`), plus the fitted statsmodels result for `predict` (excluded from `to_dict()`, which is JSON-serialisable). `fit_baseline` raises `InsufficientDataError` when more than `MAX_EXCLUDED_SHARE` of periods are excluded or n − p < `MIN_BASELINE_DOF`.
- `SavingsResult` (frozen; `to_dict()` is JSON-safe): adjusted_baseline_kwh, actual_kwh, savings_kwh, savings_pct, band_low_pct, band_high_pct, confidence, fsu, n, m, m_excluded, extrapolated_share, ashrae_ok, verdict ∈ {`verified`, `likely_not_proven`, `below`}, gate_failures (tuple of human-readable reasons, e.g. `"CV(RMSE) 0.44 > 0.25"`). All `*_pct` fields are **percent (0–100)**, compare with `100 * ADEETIE_THRESHOLD`; `fsu` and `extrapolated_share` are fractions. `m` = usable reporting rows, `m_excluded` = rows dropped by `usable_rows`; `fsu` = half-width at `CONFIDENCE_ASHRAE` ÷ (`ADEETIE_THRESHOLD` × adjusted baseline), i.e. evaluated at a 10% saving. `extrapolated_share` = share of usable reporting rows with production outside [`PROD_RANGE_LOW_FACTOR` × baseline min, `PROD_RANGE_HIGH_FACTOR` × baseline max] (0.0 if the formula has no production). Raises `ValueError` if the reporting frame's granularity differs from `model.monthly`, `InsufficientDataError` if more than `MAX_EXCLUDED_SHARE` of reporting rows are unusable.
- `decide_verdict(band_low_pct, band_high_pct, ashrae_ok, extrapolated_share) -> str` (pure; `docs/validation_plan.md` §2 steps 3–4): `likely_not_proven` if `not ashrae_ok` or `extrapolated_share > MAX_EXTRAPOLATED_SHARE`; else `verified` if `band_low_pct >= 10`, `below` if `band_high_pct < 10`, else `likely_not_proven`.
- `ComparatorResult` (frozen; `to_dict()` is JSON-safe): method ∈ {`naive`, `bee_sec`}, savings_kwh, savings_pct (**percent 0–100**), verdict ∈ {`verified`, `below`} (`verified` if savings_pct ≥ 100 × `ADEETIE_THRESHOLD`; comparators have no band, `docs/validation_plan.md` §2). Rows are dropped with `contract.validate` + `usable_rows`, as in V3. `naive_savings(baseline_df, reporting_df, formula="kwh ~ 1")`: expected = baseline kWh per calendar day × reporting calendar days (`period_days`), usable rows via `usable_rows(df, formula)`; experiments pass the SaveProof model's formula so all three methods see identical rows. `bee_sec_savings`: SEC = Σ kWh / Σ production over usable rows (`"kwh ~ production"`), savings_kwh = (SEC_base − SEC_report) × reporting production, savings_pct = 100 × (1 − SEC_report / SEC_base); raises `ValueError` if `production_unit` differs between the frames or total production is 0.
- `DriftResult`: alarm (bool), alarm_period_start (date | None), cusum_values (list[float]), threshold.

## Rohith — data, simulator, app
| # | dates | task | receives | returns | proving test |
|---|---|---|---|---|---|
| R1 | 9–10 Nov | `ingest/uci.py`, `ingest/irish.py` | file path | `load_uci_steel(path) -> df` (daily; `production` = daily lagging kVArh, `production_unit="proxy:kvarh_lagging"`; NOT `Load_Type`, which is a tariff clock); `load_irish(path) -> df` (grid incomer, daily in Europe/Dublin dates rebuilt from 15-min; `production` = same-day PV kWh from mi_b, `production_unit="proxy:pv_kwh"`) | outputs pass `validate()`; UCI midnight-row trap handled (96 intervals per day) |
| R2 | 10 Nov | `quality.py` | contract df | `score(df) -> QualityReport` (total, completeness, continuity, plausibility, consistency, flags) | gaps and negatives in a fixture are flagged; Irish reliability flags agree |
| R3 | 11–12 Nov | `ingest/simulator.py` | `SimParams`, `rng` | `simulate_plant(params, rng) -> tuple[df, PlantMeta, SimTruth]` | zero saving → engine reports ≈ 0; known saving → recovered within band |
| R4 | 12–13 Nov | `experiments/make_blind_set.py`, `experiments/score_blind.py` | simulator | ≥ 200 plants in `data/processed/blind/`, truth in `data/blind_truth.csv` (engine code never reads it); scoring table in `results/` | scoring script reproduces a hand-checked plant |
| R5 | 14–16 Nov | `report/charts.py`, `streamlit_app.py` | result dataclasses | `savings_chart(...) -> Figure`, 5-page app (Data, Baseline, Savings, Drift, Report) | app opens on demo defaults with no errors |
| R6 | 17 Nov | deploy + screenshots | app | live link, full-HD screenshots | link works logged out |

`SimParams`: sector, days, base_load_kwh, variable_sec_kwh_per_unit, production_mean, production_swing_pct, ar1_phi, noise_cv, retrofit_day, saving_base_pct, saving_variable_pct, shutdown_days. Every default cites its source (BEE DPR / UNIDO page).
`SimTruth`: true_savings_kwh, true_savings_pct.

## Both
- 17–18 Nov: architecture diagram, form answers, deck with real numbers only.
- 19 Nov: 3–5 min video, README (setup, data sources, AI use).
- 20 Nov: submit. 21–22 Nov: buffer.

## Integration checkpoints
- **10 Nov evening:** R1 loaders + V2 baseline run end-to-end on UCI → first null-test number.
- **13 Nov evening:** blind set scored for all three methods → headline table exists.
- **16 Nov evening:** app shows the same numbers as `results/` and the PDF.

## If late, cut in this order
Modbus simulator → bill parser → drift page → dashboard polish. Never cut V1–V5, R1, R3–R4 or the MRV report.
