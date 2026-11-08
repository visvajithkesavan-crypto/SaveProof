---
name: streamlit-mrv-report
description: Use when building streamlit_app.py, charts, the MRV report PDF, screenshots, the demo video, or anything a judge or energy auditor will see. Keeps the UI thin and the report credible to a BEE-certified energy auditor.
---

# Demo app and MRV report

## App (streamlit_app.py)
- Thin: every number comes from a saveproof function. The app collects inputs and renders results; no calculations beyond display formatting.
- One page per step, sidebar navigation:
  1. **Data** — pick UCI Steel, Irish site or a simulated plant, or upload a contract CSV; show quality score and flags.
  2. **Baseline** — baseline and reporting periods, retrofit date; model equation, R², CV(RMSE), NMBE, n, ρ, residual plot.
  3. **Savings** — adjusted baseline vs actual with a shaded band; savings kWh and %, band, confidence, FSU; ADEETIE verdict badge; naive and SEC-ratio side by side.
  4. **Drift** — CUSUM chart with the alarm threshold.
  5. **Report** — download the MRV PDF.
- `@st.cache_data` on loaders. Demo defaults so the app opens straight onto a meaningful result.
- Every view carries a source badge: Real meter data / Utility bill / Simulated / Injected saving.

## Charts (report/charts.py — matplotlib, shared by app and PDF)
- Units on every axis; dates on x. Fixed colours: actual = dark grey, adjusted baseline = blue, band = light-blue fill, retrofit date = dashed vertical line.
- The title states the finding ("Savings 11.8% ± 2.1%, 90% confidence"), not the topic.
- No 3D, no pie charts, no dual axes unless unavoidable.

## MRV report PDF (report/mrv_pdf.py — reportlab, 1–2 pages)
What a BEE-certified auditor expects, in order:
1. **Header:** plant name/ID, sector, cluster; retrofit description and date; report date; SaveProof version.
2. **Periods and data:** baseline and reporting dates, data source, quality score, excluded periods and why.
3. **Baseline model:** equation with coefficients and units; n, R², CV(RMSE), NMBE, ρ, n′; ASHRAE check results.
4. **Results:** adjusted baseline, actual, savings (kWh and %), band and confidence level, FSU; ADEETIE verdict with the three-state rule written out and marked as SaveProof's method.
5. **CO₂ avoided:** tCO₂ with factor and source (CEA CO₂ Baseline Database v22.0, FY 2025-26, 0.675 tCO₂/MWh).
6. **Comparators:** naive and BEE SEC-ratio results, one line on why they differ.
7. **Assumptions and limitations:** production proxies, extrapolation flags, simulated data.
8. **Sign-off:** Certified Energy Auditor name, BEE registration no., signature, date. Footer: "Prepared with SaveProof. Verification requires review and signature by a BEE-certified energy auditor."

## Submission assets
- Full-HD screenshots of each app step, using real (non-simulated) data where possible.
- Demo video, 3–5 min, scripted before recording: the problem (15 s) → one plant where naive, SEC-ratio and SaveProof disagree → the validation table → the auditor report → Schneider fit (EM6400 meter to verified savings).
- No placeholder or made-up numbers in any screenshot, slide or video.
