---
name: validation-testing
description: Use when writing tests, adding a feature that needs one, or building the validation experiments (null test, injected saving, blind simulator test). Defines SaveProof's test conventions and the pre-registered pass criteria behind the evidence shown to judges.
---

# Testing and validation

Two kinds of proof: **tests** show the code does what we meant; **experiments** show the method is right. Judges see the experiments. Tests stop us breaking things on day 11.

## Test conventions
- pytest; `tests/` mirrors `saveproof/` (e.g. `tests/engine/test_stats.py`). Whole suite < 30 s; mark slow ones `@pytest.mark.slow`.
- Fixtures are small: a few dozen rows built in the test or saved under `tests/fixtures/`. Unit tests never read the full datasets.
- Randomness: `rng = np.random.default_rng(42)`, passed into functions.
- Floats: `pytest.approx(x, rel=…, abs=…)` with the tolerance justified in a comment. Never widen a tolerance to pass a test you don't understand.
- Every bug fix starts with a failing test.

## What to test
| type | example |
|---|---|
| known answer | synthetic `kwh = 180 + 0.4·production + noise` (n = 365, small noise) → fitted intercept ≈ 180, slope ≈ 0.4 |
| hand-computed | CV(RMSE), NMBE, n′ on a 6-row example worked by hand in the test docstring |
| invariants | identical baseline and reporting data → savings ≈ 0; scaling kWh by k scales savings by k and leaves savings % unchanged; shuffling rows changes nothing |
| contract | each adapter's output passes `contract.validate()`; bad input raises a clear error |
| core claim | production +20% with no retrofit → BEE SEC-ratio reports positive savings, SaveProof reports ≈ 0 within its band |
| regression | the minimal input that triggered a past bug |

## Validation experiments (scripts in experiments/, outputs in results/)
**`docs/validation_plan.md` is the authority.** Its specs, windows, verdict rule and pass criteria override the summary below. Read it before writing any experiment or verdict code. Change a criterion only through its deviation log, before rerunning.

1. **Null test — real data, nothing changed.** UCI Steel: fit Jan–Aug, report Sep–Dec; plus rolling windows for many trials. Irish grid incomer: fit on the first 60%, report on the rest.
   Pass: zero "verified" verdicts. Coverage is reported, not gated: rolling windows overlap, so there are few independent trials (see the plan, §4).
2. **Injected saving — real noise, known answer.** Multiply reporting kWh by (1 − s), s ∈ {5, 7, 10, 12, 15}%; plus a load-shape version that cuts only high-load intervals.
   Pass: zero "verified" at s ≤ 7%; calendar-only specs never verify; power at 10/12/15% reported (see the plan, §5). Uniform injection is algebra on the null result, so recovery error is not a criterion.
3. **Blind simulator test.** Rohith generates ≥ 200 plants with hidden true savings (0–20%), production swings (±30%), product-mix shifts, shutdowns and AR(1) noise. The engine sees only the contract tables; the truth is read only by the scoring script.
   Report per method (SaveProof, naive, BEE SEC-ratio): mean absolute error (pp), bias, band coverage (ours), and **wrong-side-of-10% rate**. This table is the headline slide.

## Honesty rules
- Label every result real, simulated or injected. Never present simulated numbers as plant results.
- Keep failed runs and report them. Honestly uncertain beats confidently wrong.
- One command per experiment (`python experiments/null_test.py`), fixed seeds, outputs written to `results/` with the git commit hash.
