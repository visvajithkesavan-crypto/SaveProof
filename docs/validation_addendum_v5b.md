# Validation addendum V5b: real production data (Laayati mine)

Status: **SIGNED OFF 2026-11-12 by Visva and Rohith. Binding from the commit that adds this signed version.** Until V5b runs, nobody computes anything on **days 21–32 (2019-01-20 to 2019-01-31)** of any station. That commit's hash is recorded in every V5b results file.

What has already been looked at, so it can be disclosed in full:
- **Whole-file completeness counts.** Rows, missing hours, duplicates, NaN, negatives.
- **Daily fits of `kwh ~ production` on days 1–20 and 1–14 only**, with the FSU that those fits imply (§10).

Nothing about days 21–32 has been computed.

## 1. Relationship to V5
- V5 (results `de4c899`) is **permanently recorded as FAILED** on N1 and I1. V5b does not amend, replace or re-score V5.
- V5b is a **separate experiment**. It is reported separately and in full, whatever its outcome.
- Nothing in this addendum changes any V5 criterion or result.

## 2. What the V5 failure does and does not establish
Computed from the V5 implementation and its results file (`results/null_test_de4c899.csv`) together with the raw UCI file.

**Established**
- **The UCI-B savings estimates are the shift in kWh per lagging kVArh between baseline and reporting.** Across the 15 UCI-B null windows, the correlation between `savings_pct` and 100 × (1 − ratio_reporting / ratio_baseline) is **0.997**, with a mean absolute gap of **0.48 pp**.
- **In all 8 verified windows, the mean lagging power factor fell from baseline to reporting**, by 5.6–7.0 points (for example, 82.4% → 76.8% in rolling_00).
- **UCI-A**, which uses no proxy, failed the adequacy gate in all 15 windows and never verified.

**Not established**
- The physical cause of the shift: capacitor or power-factor equipment changes, load-mix changes, or a genuine change in energy use. The daily correlation between power factor and the share of idle intervals is weak (−0.13), so the partial-load explanation is **not** supported.
- That "nothing changed" at the UCI plant. Consumption also fell in the calendar-only spec, so the null assumption cannot be verified for this dataset.

**Wording we may use:** *"The V5 false verifications are fully accounted for by a shift in the kWh-to-kVArh relationship of the proxy, coinciding with a fall in power factor. The cause of that shift is unknown."*

## 3. Dataset (condition 1)
| Item | Value |
|---|---|
| Source | Laayati O., Bouzi M., Chebak A. (2022), *Applied System Innovation* 5(1), 18, doi:10.3390/asi5010018. Data also on IEEE DataPort, doi:10.21227/fw19-7f85 |
| Licence | CC BY 4.0 (stated in the paper and on DataPort). Redistribution with attribution is permitted |
| Our copy | `SaveProof_Datasets/01_real_energy_data/Laayati_Morocco_Mine_sample/` |
| Integrity | The three CSVs are byte-identical to the MDPI supplement zip. SHA-256 prefixes: Destoning `f02ae5b31d356fef`, Loading `5919a9fb00b232a5`, Screening `3217602d6eb7fef7` |
| Site | Experimental open-pit mine, Benguerir, Morocco. Three process stations: destoning, screening, train loading |
| Period, resolution | 2018-12-31 00:00 to 2019-01-31 23:00, hourly. 768 rows per station, which is 32 full days |
| `KWh` | Energy per hour from **Schneider PM8000 power meters** (paper §3) |
| `THE` | The production KPI per hour, taken from the process SCADA production database (paper §6.2: "a database that contains functioning hours and production"). **The unit is not documented** in the paper or the CSVs |
| `HM` | Functioning time per hour, as a fraction from 0 to 1. **Not used** in V5b |
| Completeness | 0 missing hours, 0 duplicate timestamps, 0 negative values. Destoning has 3 NaN in `HM` (2019-01-05, 08:00–10:00); `KWh` and `THE` are complete |
| Time zone | Not stated. Treated as local time; Morocco has no DST change in this period. Timestamps are treated as the **start** of each hour (assumption: the last row, 31 Jan 23:00, then closes day 32) |
| Availability | The MDPI supplement is public. DataPort requires a login |

## 4. Data preparation (frozen)
- New loader `saveproof/ingest/laayati.py`, one contract frame per station, built test-first from fixtures. **The unit tests must not load the real files.**
- `period_start = period_end` = the calendar date of the hourly timestamp. Each day needs 24 hours; otherwise the loader raises.
- `kwh` = Σ `KWh`. `production` = Σ `THE`. `production_unit = "process_kpi:THE"`. `operating_days = 1.0`.
- `coverage_pct` = 100 × (hours with `KWh` present) / 24.
- `source = "real_meter"`. `dataset` = `laayati_destoning`, `laayati_screening` or `laayati_loading`. `event = None`.

## 5. Model specification (frozen)
- One formula for all three stations: **`kwh ~ production`**.
- No day-type term, because a weekly pattern at the mine is not established, and with 14 baseline days every extra parameter costs degrees of freedom.
- All three stations are run. None is dropped after the fact.

## 6. Windows (frozen)
- **Fixed split (primary):** baseline days 1–20 (2018-12-31 to 2019-01-19), reporting days 21–32 (2019-01-20 to 2019-01-31).
- **Rolling (secondary):** 14-day baseline and 7-day reporting window, step 1 day, starting 2018-12-31. That gives 12 windows; the last has baseline 2019-01-11 to 2019-01-24 and reporting 2019-01-25 to 2019-01-31.
- The rolling windows overlap heavily, leaving about 2 independent trials per station. They are reported, but they are not grounds for any claim on their own.
- Per station that is 13 windows, so **39 null runs** in total.

## 7. Null and injection (frozen)
- Injected saving s ∈ {0, 5, 7, 10, 12, 15}%, applied to every window.
- **Uniform shape only:** reporting `kwh` × (1 − s). `production` is untouched. The weekday-only shape is not used, because no weekly cycle is established.
- That gives 234 SaveProof runs. `naive_savings(baseline, reporting, formula)` and `bee_sec_savings` are scored on exactly the same frames. Comparator verdict: "verified" if savings_pct ≥ 10.

## 8. Criteria (frozen)
"Gate" means the adequacy gate plus the extrapolation gate of validation plan §2, and the proxy gate of §9 (this one never fires on `process_kpi:THE`). Windows with insufficient data are excluded and counted.

| ID | Type | Criterion |
|---|---|---|
| B1 | hard | SaveProof gives **zero "verified"** in all null runs (s = 0) |
| B2 | hard | SaveProof gives **zero "verified"** at s ∈ {5, 7}% |
| B3 | hard | No "verified" in any run where the gate failed |
| B4 | hard (informativeness) | At least one station passes the gate in the **fixed split**. If none does, V5b is declared **uninformative**, which is not a pass |
| B5 | headline comparison (can fail) | Over null runs where SaveProof's gate passed, SaveProof's mean \|savings_pct\| is **lower than both** SEC-ratio's and naive's, in the fixed split and separately across the rolling windows |
| B6 | reported | False "verified" counts for SEC-ratio and naive at s ∈ {0, 5, 7}; the "verified" rate of all three methods at s ∈ {10, 12, 15}; the share of runs passing the gate, per station; the stability flag of §10 |

- **V5b passes** if B1 ∧ B2 ∧ B3 ∧ B4.
- B5 is reported as pass or fail separately, and it is the only basis for a comparative claim against SEC-ratio.
- B5 assumes nothing changed at the mine in January. That cannot be verified, and the result will say so.

## 9. Proxy gate (condition 4): product rule, adopted before V5b and the blind test
**Measured production** is an output quantity per period, recorded by the plant's production or process system **independently of any energy measurement**. Examples: weighbridge, dispatch records, a production register, a process-SCADA throughput counter. It is in physical units or a documented process KPI.

**Not measured (a proxy):**
- (a) anything derived from electrical measurements: kWh, kVArh, kVA, power factor, demand, current, or run-hours inferred from an energy signal;
- (b) values that are estimated, allocated or interpolated, such as a monthly total spread across days, or filled gaps;
- (c) calendar or schedule variables;
- (d) modelled outputs.

**Rule**
- Loaders must label (a)–(d) with `production_unit = "proxy:<name>"`.
- **"Verified" requires a `production` term in the formula and a production_unit that does not start with `proxy:`.** Otherwise the gate fails, with the gate_failure "production variable is not measured output (proxy or absent)", and the verdict is `likely_not_proven`.

**Scope**
- Applies to V5b, the blind test (R4) and the product.
- **Not applied retroactively to V5.** A labelled secondary table may show what V5 would have given under this gate. That is true by construction and is **not evidence**.
- Logged in validation plan §8.

## 10. Stability check (condition 5): method, threshold, role
**Method.** Take the residuals e_t of the fitted baseline in time order, with t = days since the baseline start. Regress e_t on t. Test the slope with **Newey–West (HAC) standard errors**, lag L = ⌊4 · (n/100)^(2/9)⌋, two-sided.

**Flag "baseline unstable" only if both hold:**
- **Significance:** p < **0.05**. This is the statistical-evidence condition.
- **Magnitude:** D = |slope| × (n − 1) / mean(baseline kWh) ≥ **0.05**, meaning the fitted drift across the baseline is at least 5% of mean baseline kWh. This is the materiality condition. 0.05 is half the 10% ADEETIE decision threshold, so a drift of that size can move a verdict.

These are two different thresholds. A tiny drift can be significant, and a large one can be insignificant; the flag requires both.

**Role**
- In V5b and the blind test it is a **reported flag, not a verdict gate**. Making it a gate would need a further signed §8 entry before those runs.
- It is applied to V5 as a **labelled post-hoc secondary analysis** only, with the threshold fixed here *before* it is computed on V5.

**Disclosed feasibility (baseline days only).** These come from fits on days 1–20, with FSU evaluated for a 12-day reporting period. The 14-day figures use 7-day reporting.

| Station | 20-day CV(RMSE) | 20-day ρ | FSU at 10%, 12-day reporting | 14-day FSU, 7-day reporting |
|---|---|---|---|---|
| Destoning | 0.058 | −0.08 | 0.18 | 0.29 |
| Screening | 0.089 | 0.23 | 0.39 | 0.55 |
| Loading | 0.167 | 0.55 | 1.40 | 0.65 |

So Destoning is expected to pass the gate, Screening is borderline (expected to fail in many rolling windows), and Loading is expected to fail.

## 11. Known caveats, stated before the run
- One mine and 32 days. A Moroccan open-pit mine, **not an Indian MSME**.
- The reporting periods are 7–12 days. The daily uncertainty factor (nmecr/ECAM polynomial) was built for **1–12 month** periods, so here x = 0.23–0.39 months, below its range. The band may be mis-calibrated.
- The unit of `THE` is undocumented.
- "Nothing changed" during January cannot be verified.
- The rolling windows overlap heavily.

## 12. Procedure and sign-off
1. Sign this addendum, then commit it together with the §8 entry.
2. Build test-first, using fixtures only:
   - the `laayati.py` loader;
   - the proxy gate in `savings.py`;
   - the stability check;
   - `experiments/v5b.py`, which writes `results/v5b_<hash>*`;
   - V6's post-hoc V5 analysis as a separate script.

   Commit.
3. Run once from a clean commit. Commit the outputs unchanged, pass or fail.
4. Report V5b in its own section, next to V5 and never instead of it.

| Name | Date | Signature (initials) |
|---|---|---|
| Visva | 2026-11-12 | visvajith |
| Rohith | 2026-11-12 | rohith |
