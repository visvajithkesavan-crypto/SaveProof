# V5b results: Laayati mine, real production records (run 2026-11-12, code ba61231)

Files: `results/v5b_ba61231*`. Protocol: `docs/validation_addendum_v5b.md`, signed in d02cce3 before any code was written. The run was not repeated or changed after the results were seen. V5 (de4c899) remains recorded as **FAILED**. V5b is a separate experiment and does not replace it.

## Pre-registered outcome
| Criterion | Result |
|---|---|
| B1: SaveProof never verifies in null runs | **passed**: 0 of 39 |
| B2: SaveProof never verifies at injected 5% or 7% | **passed**: 0 of 78 |
| B3: no "verified" where the gate failed | **passed**: 0 of 156 |
| B4: at least one station passes the gate in the fixed split | **passed** (informative): Destoning and Screening |
| **V5b overall (B1–B4)** | **PASSED** |
| B5: SaveProof's mean \|null error\| is lower than both SEC-ratio's and naive's | **passed**: fixed split 2.1 vs 30.3 vs 6.5 pp (2 runs); rolling windows 1.3 vs 17.4 vs 4.8 pp (11 runs) |

**False "verified" verdicts.** These are runs where nothing changed, or where the injected saving is below 10%. 39 runs per row.

| | s = 0 | s = 5% | s = 7% |
|---|---|---|---|
| SaveProof | 0 | 0 | 0 |
| BEE SEC-ratio | 13 | 16 | 18 |
| Naive | 10 | 18 | 21 |

## The clearest examples (fixed split, nothing changed)
- **Screening:** SEC-ratio reports an **18.5% saving and "verified"**. SaveProof gives 0.3%, with a band of −6.4% to +7.0% and the verdict "below".
- **Destoning:** SEC-ratio reports **−42%**. That is the underpayment direction: a genuine saver would be denied the subsidy. The measured fixed share here is about 69%.
- **Loading:** SaveProof's gate blocks it (FSU at 10% is 1.40), so the verdict is "likely, not proven". SEC-ratio and naive both "verify" about 12%.

## Limits (stated before the run in addendum §11; observed values added)
- **The band was too narrow here.** Of the 13 null runs that passed the gate, 10 bands contain 0 (77%, against a nominal 90%). All 3 misses are Destoning, with consumption about 3–4.5% above prediction. Possible causes: a real change at the mine, or bands that are too narrow for 7–12-day reporting periods, which are below the 1–12 month range of the daily uncertainty factor. This did not produce any false "verified".
- **Power is low.** SaveProof's "verified" rate at an injected 15% saving is 23% (Destoning) and 31% (Screening). At 10% it is 0% for both. The reasons are short reporting periods and a gate that Screening fails in about half its windows.
- **The scope is small.** One mine, 32 days, overlapping rolling windows, an undocumented production unit, and a "nothing changed" assumption that cannot be verified. This is **not an Indian MSME**.
- **The stability flag** fired in 54% of Loading runs, 8% of Destoning runs and 0% of Screening runs. It is reported only and does not affect verdicts.

## What can be claimed
*"In a second pre-registered test, on a plant with real production records, SaveProof issued no false verifications in 117 runs at true savings of 0–7%. Over the same runs, BEE's SEC-ratio falsely verified 47 times and naive bill comparison 49 times. One site, one month; SaveProof is conservative and verified few genuine savings at these short periods."*
