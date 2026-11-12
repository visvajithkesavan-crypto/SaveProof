# V5 results: null test and injected saving (run 2026-11-11, code de4c899)

Files: `results/null_test_de4c899*` and `results/injected_saving_de4c899*`. Plan: `docs/validation_plan.md` (signed off 920fbcd, amended 74bb2e9). The run was not repeated or changed after the results were seen.

## Pre-registered outcome

| Criterion | Hard? | Result |
|---|---|---|
| N1: zero "verified" in null windows | yes | **FAILED**: 8 of 58 windows verified, all UCI-B |
| N2: share of bands containing 0 | reported | UCI-A 0.60, **UCI-B 0.13 (flagged)**, IR-A 0.00, **IR-B 0.36 (flagged)** |
| I1: zero "verified" at injected 0/5/7% | yes | **FAILED**: 62 of 348 runs verified, all UCI-B |
| I2 (amended): no "verified" where the adequacy gate failed | yes | passed (0 of 264) |
| I2 (original wording): UCI-A and IR-A never verify | yes | passed (0 of 348) |
| I3: power at 10/12/15% | reported | UCI-B 0.87/0.93/1.00 (inflated by the same drift); UCI-A, IR-A and IR-B 0.00 |

## Diagnosis (evidence from the raw UCI file, monthly)

| Month 2018 | kWh/day | kWh per lagging kVArh | mean lagging power factor (%) |
|---|---|---|---|
| Jan | 4072 | 2.32 | 86.4 |
| Apr | 2626 | 2.27 | 81.5 |
| Jun | 2180 | 1.99 | 77.3 |
| Aug | 2212 | 1.80 | 75.4 |
| Oct | 2731 | 1.71 | 73.7 |
| Dec | 1917 | 2.40 | 88.8 |

- UCI-B uses lagging kVArh as its activity proxy, because UCI has no production records (plan §3).
- The plant's power factor fell from 86% to 74% between January and October and recovered to 89% in December, when leading kVArh also jumps. That is consistent with capacitor-bank or load-mix changes.
- As a result, the plant drew more kVArh for each kWh. A model fitted on January–June therefore over-predicts July–October kWh by about 18–22%, and SaveProof read that gap as savings. All 8 null-test failures and all 62 I1 failures come from this.
- The plan's §3 named this risk in advance: kVArh "would be an invalid proxy for a power-factor-correction or VFD retrofit".
- The statistics behaved as designed:
  - UCI-A (calendar only) failed the adequacy gate in every window and never verified.
  - The UCI-B bands are not unusually narrow. The error is a systematic shift, not noise.

**Irish site.** Grid consumption rises about 30% from summer into autumn and winter. Neither the calendar model nor the PV model captures that, because weather is out of scope (rules). The result is large negative "savings", verdicts of "below" or "not proven", and **zero power** even at an injected 15% saving. It produced no false "verified".

## What this means
1. SaveProof is only as good as its independent variable. An electrical proxy measured on the same meter can move with the plant's own electrical changes, such as power-factor correction, and fake a saving. **ADEETIE plants record production in physical units, which is the variable SaveProof is designed for.** Production normalisation on real data remains unproven (plan §1).
2. The real-data null test cannot be a positive headline. It is reported as a failure, with this diagnosis.
3. Product changes, with no change to the V5 verdicts and no rerun:
   - The report prints a prominent warning whenever the production variable is a `proxy:` unit.
   - The R2 quality score adds a consistency flag when kWh per proxy unit or power factor shifts materially between baseline and reporting.
4. Weather and seasonal load is a known limitation, listed on the roadmap.
