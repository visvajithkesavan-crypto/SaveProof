# Formulas and acceptance values used by the engine

Sources: ASHRAE Guideline 14 as reproduced in TAMU ESL-IC-09-11-23 (n′ definition, 1.26 constant, FSU ≤ 50% at 68%); LBNL "Acceptance criteria and documentation" (2017) for guidance values. **Verify the FSU expression against the Guideline text before quoting it in the deck.**

Notation: y_i measured, ŷ_i predicted, ȳ mean of measured baseline values, n baseline points, p model parameters (including intercept), m reporting-period points, e_t residuals.

```
CV(RMSE) = sqrt( Σ (y_i − ŷ_i)² / (n − p) ) / ȳ
NMBE     = Σ (y_i − ŷ_i) / ( (n − p) · ȳ )

ρ  = corr(e_t, e_{t−1})                 # lag-1 autocorrelation of baseline residuals
n′ = n · (1 − ρ) / (1 + ρ)              # effective independent points

FSU = ΔE_save / E_save
    = 1.26 · t · CV · sqrt( (n / n′) · (1 + 2/n) · (1 / m) ) / F

absolute form (use when savings ≈ 0, e.g. null test):
ΔE_save = 1.26 · t · CV · sqrt( (n / n′) · (1 + 2/n) · (1 / m) ) · E_adjusted_baseline
```
- t: two-sided Student-t at the chosen confidence, (n − p) degrees of freedom
- CV: CV(RMSE) as a fraction
- F: savings fraction = E_save / E_adjusted_baseline over the reporting period
- 1.26: empirical constant (Reddy & Claridge 2000) for **monthly** models only. For daily/hourly models the open-source nmecr package (kW-Labs, MIT licence) replaces 1.26 with a factor that depends on the number of observations, and uses n′ in both the t-statistic's degrees of freedom (n′ − p) and the MSE. Cross-check our implementation against `R/normalized_savings_and_uncertainty.R` in https://github.com/kW-Labs/nmecr before finalising daily-model bands.
- Band: E_save ± ΔE_save

## Acceptance values for config.py
| check | value | status |
|---|---|---|
| FSU at 68% confidence | ≤ 50% | ASHRAE G14 requirement |
| FSU | ≤ 25% good; 25–50% acceptable | LBNL guidance |
| CV(RMSE) | < 25% | LBNL guidance, not pass/fail |
| NMBE | < 0.5% | LBNL guidance; OLS gives ≈ 0 by construction, so a nonzero value signals a bug or excluded rows |
| R² | > 0.7 | LBNL guidance |
| reporting-period independent variables | 90%–110% of baseline range | LBNL guidance |
| non-routine data removed from baseline | ≤ 25% | ASHRAE G14 via LBNL |

## Cross-check
Reference implementations: nmecr (R, ASHRAE G14 uncertainty incl. daily/hourly handling) and OpenDSM (Python, formerly OpenEEmeter, CalTRACK methods). Port logic in our own code with an attribution comment; don't copy files.

statsmodels `get_prediction().summary_frame(alpha=…)` gives OLS prediction intervals. With autocorrelated daily residuals these are too narrow unless corrected (HAC standard errors or the n′ correction). Compare both in the null test and report which one achieves its nominal coverage.
