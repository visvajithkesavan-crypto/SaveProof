"""Constants from docs/build_plan.md. Each comment gives meaning, unit and source."""

# Confidence level of the reported savings band, fraction.
# Source: LBNL 2017 suggests 80-90%
CONFIDENCE_REPORT = 0.90
# Confidence level for fractional savings uncertainty, fraction.
# Source: ASHRAE Guideline 14, TODO(verify clause)
CONFIDENCE_ASHRAE = 0.68
# Maximum acceptable fractional savings uncertainty, fraction of savings.
# Source: ASHRAE Guideline 14, TODO(verify clause)
FSU_MAX_ASHRAE = 0.50
# Guidance CV(RMSE) of the baseline model, fraction of mean kWh.
# Source: LBNL, Acceptance Criteria and Documentation (2017), guidance values not pass/fail
CVRMSE_GUIDE = 0.25
# Guidance absolute NMBE of the baseline model, fraction of mean kWh.
# Source: LBNL, Acceptance Criteria and Documentation (2017), guidance values not pass/fail
NMBE_GUIDE = 0.005
# Guidance R-squared of the baseline model, dimensionless.
# Source: LBNL, Acceptance Criteria and Documentation (2017), guidance values not pass/fail
R2_GUIDE = 0.70
# A reporting period is flagged as extrapolation if production < this x baseline production_min.
# Unit: dimensionless factor. Source: SaveProof rule
PROD_RANGE_LOW_FACTOR = 0.90
# A reporting period is flagged as extrapolation if production > this x baseline production_max.
# Unit: dimensionless factor. Source: SaveProof rule
PROD_RANGE_HIGH_FACTOR = 1.10
# Max share of usable reporting periods whose production may lie outside the baseline range
# [PROD_RANGE_LOW_FACTOR x min, PROD_RANGE_HIGH_FACTOR x max] before the verdict is "not proven".
# Unit: fraction. Source: SaveProof rule, docs/validation_plan.md §2
MAX_EXTRAPOLATED_SHARE = 0.25
# Rows with coverage_pct below this are flagged by the quality score.
# Unit: percent 0-100, same as the coverage_pct column. Source: SaveProof rule
MIN_COVERAGE_PCT = 90.0
# Max share of baseline periods that may be excluded for events before the baseline is rejected.
# Unit: fraction. Source: SaveProof rule
MAX_EXCLUDED_SHARE = 0.25
# Minimum residual degrees of freedom (n - p) of a baseline fit, below which it is rejected.
# Unit: dimensionless count. Source: SaveProof rule
MIN_BASELINE_DOF = 5
# Empirical factor in the fractional savings uncertainty of MONTHLY models, dimensionless.
# Source: Reddy & Claridge 2000, as reproduced in ASHRAE Guideline 14 / TAMU ESL-IC-09-11-23
FSU_MONTHLY_FACTOR = 1.26
# Replaces FSU_MONTHLY_FACTOR for DAILY models: alpha = a*x^2 + b*x + c with x = m / days per month,
# m = reporting-period points. Dimensionless. Coefficients (a, b, c) ported from the daily branch of
# calculate_savings_and_uncertainty.R in https://github.com/kW-Labs/nmecr (kW-Labs, MIT licence).
# Same coefficients as the ECAM v6r5 'FSU factor'. nmecr/ECAM use (-0.00022, 0.03306, 0.94054)
# for monthly billing data; we keep 1.26 for monthly (ASHRAE G14). At m=12 the two differ by ~4%.
# TODO(verify): range of m for which the quadratic was fitted; not stated in nmecr.
FSU_DAILY_ALPHA_COEFFS = (-0.00024, 0.03535, 1.00286)
# Days per month used by nmecr to turn m daily points into x above: mean(30, 31). Unit: days.
# Source: same nmecr file as FSU_DAILY_ALPHA_COEFFS
FSU_DAILY_DAYS_PER_MONTH = 30.5
# Minimum sustained energy savings for ADEETIE interest subvention, fraction of baseline.
# Source: ADEETIE Operational Guidelines, TODO(verify page)
ADEETIE_THRESHOLD = 0.10
# Grid emission factor, tCO2 per MWh.
# Source: CEA CO2 Baseline Database v22.0, FY2025-26 weighted average
GRID_FACTOR_TCO2_PER_MWH = 0.675
