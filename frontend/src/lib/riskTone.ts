import type { BadgeTone } from "@/components/ui/Badge";

// Shared risk-tier -> Badge tone mapping, previously redefined locally in
// Dashboard's page.tsx and about to be needed identically elsewhere.
export const RISK_TONE: Record<string, BadgeTone> = {
  Low: "good",
  Medium: "warning",
  High: "serious",
  Critical: "critical",
  Pending: "neutral",
};
