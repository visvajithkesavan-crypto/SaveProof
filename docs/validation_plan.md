# SaveProof validation plan (pre-registered)

Status: **SIGNED OFF 2026-11-09 by Visva and Rohith. Binding from the commit that adds this file.**
No experiment script may run on the real or blind data before that commit. Its hash is recorded in every results file.

## 1. What each experiment proves and what it doesn't

| Experiment | Data | Proves | Does not prove |
|---|---|---|---|
| Null test | real (UCI Steel, Irish grid incomer) | Bands are honest on real, autocorrelated noise; the verdict gate refuses to verify when nothing changed | Production normalisation (neither dataset has production records) |
| Injected saving | real noise, synthetic saving | How the verdict behaves around the 10% line: no false verification, plus honest power | Real retrofit effects (the injection touches kWh only) |
| Blind simulator | simulated, India-calibrated | Production normalisation beats naive and BEE SEC-ratio; band calibration over ≥ 200 independent plants | Performance on real Indian plants |

Production normalisation on real Indian plant data stays **unproven** until an auditor shares a plant's bills plus production records. Every results slide says so.

## 2. Verdict rule (fixed here; V3 implements exactly this)

All bands are two-sided at `CONFIDENCE_REPORT = 0.90`, so the lower bound is a one-sided 95% bound.

1. **Model adequacy gate (`ashrae_ok`)** needs both:
   - CV(RMSE) of the baseline fit ≤ `CVRMSE_GUIDE` (0.25), and
   - fractional savings uncertainty at 68% confidence ≤ `FSU_MAX_ASHRAE` (0.50), **evaluated at a 10% saving** (`ADEETIE_THRESHOLD`). This asks whether the model could verify a 10% saving at all, independent of what was measured.

   R² and NMBE are reported only. They don't gate the verdict: R² is low whenever production barely varies, and NMBE is about 0 by construction for OLS.
2. **Extrapolation gate:** if more than 25% of reporting periods have production outside [`PROD_RANGE_LOW_FACTOR` × baseline min, `PROD_RANGE_HIGH_FACTOR` × baseline max], the verdict is not proven. V3 adds `MAX_EXTRAPOLATED_SHARE = 0.25` to config.
3. If both gates pass:
   - `verified` if band_low_pct ≥ 10
   - `below` if band_high_pct < 10
   - otherwise `likely_not_proven`
4. If either gate fails, the verdict is **`likely_not_proven`**. A bad model can never produce `verified` or `below`.

Comparators have no band. Naive and BEE SEC-ratio give "verified" if their point estimate ≥ 10%, and "below" otherwise.

**Wrong side of 10%** means the verdict is `verified` while true saving < 10%, or the verdict is `below` while true saving ≥ 10%. `likely_not_proven` is an abstention, not a wrong-side error. Its rate is reported in every table so the method can't win by abstaining.

## 3. Real-data model specifications (fixed)

Day type is derived in the engine from `period_start`: weekday vs Saturday/Sunday. It is not a contract column.

| Spec | Dataset | `production` column | Formula | Purpose |
|---|---|---|---|---|
| UCI-A | UCI Steel, daily, 2018-01-01 to 2018-12-31 | none (NaN) | `kwh ~ C(day_type)` | Calendar only. Expected to **fail** the adequacy gate; tests that the gate stops a false "verified" |
| UCI-B | same | daily lagging kVArh, `proxy:kvarh_lagging` | `kwh ~ production + C(day_type)` | Electrical activity proxy. Tests band calibration on real noise |
| IR-A | Irish grid incomer, daily in Europe/Dublin local dates, from the first full day (23 May 2025) | none (NaN) | `kwh ~ C(day_type)` | As UCI-A |
| IR-B | same | same-day PV generation kWh (mi_b), `proxy:pv_kwh` | `kwh ~ production + C(day_type)` | PV drives grid import the way weather drives buildings |

Known weaknesses, stated up front:
- `Load_Type` in UCI is a time-of-day tariff class, not activity, so it is not used.
- kVArh comes from the same meter and is physically coupled to load. It would be an invalid proxy for a power-factor-correction or VFD retrofit, which change kVArh. Our injection leaves it unchanged, so UCI-B is optimistic for those measures.

Days with `coverage_pct` < `MIN_COVERAGE_PCT` are excluded. If more than `MAX_EXCLUDED_SHARE` of baseline days are excluded, that window is reported as "insufficient data" and not scored.

## 4. Null test (real data, nothing changed)

**Windows**
- **Fixed split.** UCI: baseline Jan–Aug, reporting Sep–Dec. Irish: first 60% of days as baseline, the rest as reporting.
- **Rolling.** UCI: 180-day baseline and 90-day reporting window, start shifted 7 days each time. Irish: 90-day baseline and 45-day reporting window, step 7 days.
- Rolling windows overlap heavily, leaving only about 2–3 effectively independent trials per dataset. Coverage from these windows is therefore **reported, not used as a pass/fail threshold**. The calibration claim rests on the simulator (§6).

**Pass criteria**
- **N1 (hard):** zero `verified` verdicts in every null window for every spec.
- **N2 (reported):** the share of windows whose band contains 0, per spec. If an adequate spec (UCI-B or IR-B) is below 50%, that goes on the results slide as a limitation, with the investigation. It is not fixed by tuning.
- **N3 (reported):** CV(RMSE), FSU at 10%, R², NMBE, lag-1 ρ and n′ for every window.

## 5. Injected saving (real noise, known answer)

Run on every null window (§4) for every spec. Reporting-period kWh is reduced by s ∈ {0, 5, 7, 10, 12, 15}%. `production` and day type stay untouched, which models an efficiency retrofit at unchanged output.

There are two shapes:
- **uniform:** every reporting day multiplied by (1 − s).
- **weekday-only:** only weekdays are cut, scaled so the reporting-period total still drops by exactly s.

Uniform injection is close to algebra on top of the null result (estimated saving ≈ s + (1 − s) × null estimate). The value of this test is the **verdict behaviour** near 10%, not the recovery error. The skill's earlier "mean error < 1 pp" criterion is dropped for that reason.

**Pass criteria**
- **I1 (hard):** zero `verified` at s ∈ {0, 5, 7}%, for both shapes and every spec.
- **I2 (hard):** any spec that fails the adequacy gate gives zero `verified` at every s (amended 2026-11-10, see §8).
- **I3 (reported):** the `verified` rate at s = 10, 12, 15% for UCI-B and IR-B, which is the power curve. If it's 0 even at 15%, we say so.

## 6. Blind simulator test (simulated, India-calibrated)

**Sets**
- **Dev set:** 50 plants, seed 1. Visva may use it freely while building.
- **Blind set:** 250 plants, seed chosen by Rohith and not shared until scoring. Truth goes to `data/blind_truth.csv`. Engine code never opens it, and `make_blind_set.py` writes it separately.

**Composition** (fixed shares of the 250):
| Stratum | Share | Production change, baseline → reporting |
|---|---|---|
| S1 stable | 25% | within ±5% |
| S2 up | 25% | +10% to +30% |
| S3 down | 25% | −10% to −30% |
| S4 misspecified | 25% | ±10–30% plus a product-mix shift or non-linear SEC that our linear formula does not capture |

**Granularity:** half the plants are monthly bills (12 baseline + 12 reporting months) and half are daily meter data (365 + 365 days).

**True saving:** uniform 0–20%, split into base-load and variable parts. AR(1) φ ∈ [0.2, 0.7], noise CV 3–10%, shutdown days in about 20% of plants. Every default cites a BEE DPR or UNIDO source (`SimParams`).

**Procedure**
1. Freeze the engine: record its commit hash.
2. Run all three methods once on the blind set.
3. Score with `score_blind.py`, the only code that reads the truth file.

Any rerun after a code change is logged in §8 with its reason, and **both** runs are reported.

**Metrics** for each method, overall, per stratum and per granularity:
- mean absolute error (pp)
- bias (pp)
- wrong-side-of-10% rate
- abstention rate
- band coverage (SaveProof only)

**Pass criteria**
- **P1 (headline):** SaveProof's wrong-side rate is lower than both naive and SEC-ratio, overall and within S2 ∪ S3.
- **P2 (hard):** false-verified rate (`verified` with true saving < 10%) ≤ 5%. This is the lender's and BEE's protection.
- **P3:** empirical 90% band coverage is within [0.85, 0.97] over the non-abstained plants. The lower bound is 2 standard errors below nominal at n ≈ 200. Above 0.97 means the bands are too wide, which costs honest MSMEs their subvention.
- **P4:** SaveProof's MAE is lower than SEC-ratio's in S2 ∪ S3, and no more than 1 pp worse than SEC-ratio in S1, where SEC-ratio is valid.
- **P5:** |bias| < 1 pp overall, excluding S4.

S4 is reported separately with only P2 applied. We expect worse error there, and we show it.

## 7. Drift alarm (cuttable; reported, not a headline)

On daily blind plants with a planted 0.5%/week degradation after reporting start, report the detection delay (weeks) and detection rate within 8 weeks. On the null windows (§4), report the false-alarm rate. Targets: detection ≥ 80%, false alarms ≤ 10%.

## 8. Honesty rules and deviation log

- Every result is labelled **real**, **injected (real noise, synthetic saving)** or **simulated**.
- Failed criteria are reported, never hidden. A criterion changes only by an entry below, written **before** the rerun.
- One command per experiment, fixed seeds, outputs in `results/` named with the commit hash.

| Date | Section | Change | Reason | Agreed by |
|---|---|---|---|---|
| 2026-11-10 | §5 I2 | I2 now reads: "any spec that FAILS the adequacy gate gives zero `verified` at every s" (was: "UCI-A and IR-A give zero `verified` at every s"). Specs that pass the gate are judged by I1 and I3. | A baseline-only fit (Irish first 60%, no reporting data examined) gives IR-A CV(RMSE) 0.218, below the 0.25 gate, so the §3 expectation that IR-A fails adequacy may not hold. I2 was meant to test that the gate blocks inadequate models, not to penalise a correct verification by an adequate calendar model. Made before any experiment ran. | VK; Rohith (pending) |

## 9. Sign-off

| Name | Date | Signature (initials) |
|---|---|---|
| Visva | 2026-11-09 | VK |
| Rohith | 2026-11-09 | Rohith |
