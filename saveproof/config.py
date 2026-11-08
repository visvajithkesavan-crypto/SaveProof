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
# Rows with coverage_pct below this are flagged by the quality score.
# Unit: percent 0-100, same as the coverage_pct column. Source: SaveProof rule
MIN_COVERAGE_PCT = 90.0
# Max share of baseline periods that may be excluded for events before the baseline is rejected.
# Unit: fraction. Source: SaveProof rule
MAX_EXCLUDED_SHARE = 0.25
# Minimum sustained energy savings for ADEETIE interest subvention, fraction of baseline.
# Source: ADEETIE Operational Guidelines, TODO(verify page)
ADEETIE_THRESHOLD = 0.10
# Grid emission factor, tCO2 per MWh.
# Source: CEA CO2 Baseline Database v22.0, FY2025-26 weighted average
GRID_FACTOR_TCO2_PER_MWH = 0.675
