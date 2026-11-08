---
name: mv-engine
description: Use when building or changing anything in saveproof/engine — baseline regression, ASHRAE Guideline 14 statistics, fractional savings uncertainty, savings bands, the ADEETIE verdict, comparators (naive, BEE SEC-ratio), drift detection, CO₂ conversion, or the physics of the plant simulator. Encodes M&V practice for Indian MSME plants.
---

# M&V engine (IPMVP Option C, production-normalised)

## Core idea
Fit a model on the **baseline period only**. Use it to predict what the plant would have used in the **reporting period** at that period's actual production and operating days — the *adjusted baseline*. Savings = adjusted baseline − actual. Always report savings with a band, a confidence level and n — never a bare number.

## Baseline model — start simple, earn complexity
1. **Granularity: daily by default.** Sum 15-min data to daily kWh. Hourly models understate uncertainty, and ASHRAE's uncertainty formula was built for monthly models. Bill data: model per day (`kwh/day ~ production/day`) because billing cycles differ in length.
2. **Form:** OLS `kwh ~ production + operating_days`; for daily data `kwh ~ production + C(day_type)` (weekday / weekend / holiday-shutdown). Add product-mix shares only if they exist and improve baseline CV(RMSE).
3. **Choose the form using baseline data only** (e.g. compare candidates by leave-one-month-out error inside the baseline). Never look at reporting-period results to choose.
4. **Check before use** (baseline residuals): CV(RMSE), NMBE, R² against `config.py` guidance; lag-1 autocorrelation ρ; production coefficient > 0 and intercept ≥ 0 (base load); residuals show no trend or seasonality.
5. **Coverage check:** reporting-period production must lie within 90%–110% of the baseline range. Outside it, set `extrapolation=True` in the result and say so in the report.
6. **Non-routine events** (shutdowns, meter faults, Diwali week): exclude via the `event` column; excluded data ≤ 25% of the baseline; list exclusions in the report.

## Statistics — exact definitions in references/ashrae14-formulas.md
- CV(RMSE) and NMBE use (n − p) in the denominator.
- Fractional savings uncertainty (FSU): ASHRAE G14 formula with autocorrelation-adjusted n′. The 1.26 factor is for monthly models; daily models need the different factor described in the reference — don't apply 1.26 to daily data blindly.
- ASHRAE G14 requires FSU ≤ 50% at ≥ 68% confidence. We show the band at 90% and run the ASHRAE check at 68%. Confidence is a parameter with its default in `config.py`.
- Use the absolute-uncertainty form for the null test: FSU is undefined when savings ≈ 0.

## ADEETIE verdict — SaveProof's proposal (label it as ours in UI and report)
BEE requires ≥ 10% savings, sustained, but doesn't say how to treat uncertainty. SaveProof reports three states:
- **Verified ≥ 10%** — the lower bound of the band is ≥ 10%
- **Likely ≥ 10%, not proven** — point estimate ≥ 10%, lower bound < 10%
- **Below 10%** — point estimate < 10%
A model that fails the ASHRAE checks can never produce "Verified".

## Comparators — they exist to show why SaveProof is needed
- **Naive bill comparison:** `(kwh_per_day_base − kwh_per_day_report) / kwh_per_day_base`. Ignores production.
- **BEE SEC-ratio** (used in past BEE MSME programmes): `savings_kwh = (sec_base − sec_report) × production_report`, SEC = kWh / production. It assumes all energy scales with output. When production rises, fixed loads (lighting, idling compressors, furnace holding) spread over more units, SEC falls with **no** retrofit, and the formula reports false savings; when production falls it hides real ones. The blind test must show this.
- Comparators take the same inputs and return the same result shape as ours.

## Drift — "sustained" savings
One-sided CUSUM on post-retrofit residuals (actual − adjusted baseline). Set the threshold from the baseline residual standard deviation and report the false-alarm rate on untouched baseline data.

## CO₂
`tco2 = kwh_saved / 1000 × grid_factor`. Default: CEA CO₂ Baseline Database v22.0, FY 2025-26 weighted average **0.675 tCO₂/MWh** (`config.py`). Always print version and year next to the number.

## Simulator physics (code lives in ingest/simulator.py)
Daily kWh = base_load_kwh × running_flag + variable_sec × production + noise, with AR(1) noise. A retrofit cuts variable SEC and/or base load by a known amount from a known date; the truth is stored in plant metadata, never in the table. Calibrate only from documented sources with the source in a comment — e.g. BJL foundry melting 580 → 560 kWh/t, Bhimavaram tube ice 120.71 kWh/t, Bhubaneswar seafood 1,100–2,850 kWh/t seasonal. Never invent calibration numbers.

## Never
- Fit or scale features using reporting-period data.
- Call a production proxy (e.g. UCI load-type counts) "production" — keep `proxy:` in the unit and the label.
- Claim BEE or ASHRAE endorses SaveProof's verdict rules.
