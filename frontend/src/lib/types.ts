// Shared shapes for API responses consumed by more than one client component
// (previously redefined slightly differently in each consumer -- Dashboard
// and TopRight both had their own near-duplicate Alert interface).
export interface Alert {
  id: string;
  type: "TIER_ESCALATION" | "CRITICAL_RISK" | "REPORT_NEEDS_REVISION";
  projectId: string;
  fromTier: string | null;
  toTier: string | null;
  riskScore: number | null;
  message: string;
  date: string;
}
