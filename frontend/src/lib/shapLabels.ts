// Friendly labels for the real static feature names used in training
// (see backend/maagap/feature_engineering.py's STATIC_NUMERIC/STATIC_CATEGORICAL
// and engineered columns). Anything not listed here falls back to a
// snake_case -> Title Case formatter, so a changed feature set never crashes.
const LABELS: Record<string, string> = {
  approved_budget: "Approved budget",
  planned_duration_months: "Planned duration",
  start_month: "Start month",
  has_contractor: "Has contractor assigned",
  contractor_reliability: "Contractor reliability",
  agency_capacity: "Implementing agency capacity",
  typhoon_exposure: "Typhoon exposure",
  cpi_at_start: "CPI at project start",
  cmrpi_at_start: "CMRPI at project start",
  cpi_change: "CPI change",
  cmrpi_change: "CMRPI change",
  budget_log: "Budget size",
  is_infrastructure: "Infrastructure project",
  is_typhoon_start: "Started in typhoon season",
  infra_x_typhoon: "Infrastructure × typhoon exposure",
  infra_x_budget: "Infrastructure × budget size",
  contractor_x_typhoon: "Contractor reliability × typhoon exposure",
  budget_x_cpi_change: "Budget × CPI change",
  low_contractor_flag: "Low contractor reliability flag",
  high_budget_flag: "High budget flag",
  agency_risk: "Implementing agency risk",
  contractor_x_agency: "Contractor reliability × agency risk",
  infra_x_low_contractor: "Infrastructure × low contractor reliability",
  typhoon_x_budget: "Typhoon exposure × budget size",
  econ_pressure: "Economic pressure (CPI/CMRPI)",
  composite_risk_features: "Composite risk index",
  project_type_enc: "Project type",
  implementing_agency_enc: "Implementing agency",
  procurement_mode_enc: "Procurement mode",
  funding_source_enc: "Funding source",
};

export function friendlyFeatureLabel(feature: string): string {
  if (LABELS[feature]) return LABELS[feature];
  return feature
    .replace(/_enc$/, "")
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
