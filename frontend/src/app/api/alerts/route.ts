import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";
import { pickLatestByKey } from "@/lib/inspectionReports";

interface RiskAlertRow {
  id: string;
  type: string;
  project_id: string;
  from_tier: string | null;
  to_tier: string;
  risk_score: number;
  message: string;
  alert_date: string;
}

interface ReportRow {
  report_id: string;
  project_id: string;
  submitted_at: string;
  review_status: "pending" | "approved" | "needs_revision";
  review_comment: string | null;
  reviewed_at: string | null;
}

// Serves risk-tier escalation and critical-risk alerts persisted in
// Supabase (computed by the backend pipeline diffing successive runs).
// Inspectors only see alerts for projects assigned to them.
export async function GET() {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json([]);
  }

  try {
    const supabase = getSupabaseServerClient();

    let ownProjectIds: Set<string> | null = null;
    if (profile.role === "inspector") {
      const { data: asg, error: asgErr } = await supabase
        .from("assignments")
        .select("project_id")
        .eq("inspector_id", profile.inspectorId ?? "__none__");
      if (asgErr) throw asgErr;
      const asgRows = (asg ?? []) as unknown as { project_id: string }[];
      ownProjectIds = new Set(asgRows.map((a) => a.project_id));
    }

    const { data, error } = await supabase
      .from("risk_alerts")
      .select("*")
      .order("risk_score", { ascending: false });
    if (error) throw error;

    let rows = (data ?? []) as unknown as RiskAlertRow[];
    if (ownProjectIds) rows = rows.filter((a) => ownProjectIds!.has(a.project_id));

    const alerts: {
      id: string;
      type: string;
      projectId: string;
      fromTier: string | null;
      toTier: string | null;
      riskScore: number | null;
      message: string;
      date: string;
    }[] = rows.map((a) => ({
      id: a.id,
      type: a.type,
      projectId: a.project_id,
      fromTier: a.from_tier,
      toTier: a.to_tier,
      riskScore: a.risk_score,
      message: a.message,
      date: a.alert_date,
    }));

    // Derived (not persisted) "needs revision" alerts for the Inspector's own
    // reports. risk_alerts is fully overwritten by every pipeline run, so a
    // review-requested notification is computed here on read instead of
    // being stored there -- see knowledge-base/04-Workflows-and-Gotchas.md.
    if (profile.role === "inspector" && ownProjectIds && ownProjectIds.size > 0) {
      const { data: reportData, error: reportErr } = await supabase
        .from("inspection_reports")
        .select("report_id, project_id, submitted_at, review_status, review_comment, reviewed_at")
        .in("project_id", Array.from(ownProjectIds));
      if (reportErr) throw reportErr;

      const reports = (reportData ?? []) as unknown as ReportRow[];
      const latestByProject = pickLatestByKey(reports, (r) => r.project_id, (r) => r.submitted_at);

      for (const report of latestByProject.values()) {
        if (report.review_status !== "needs_revision") continue;
        alerts.push({
          id: `report-revision-${report.report_id}`,
          type: "REPORT_NEEDS_REVISION",
          projectId: report.project_id,
          fromTier: null,
          toTier: null,
          riskScore: null,
          message: `Report for ${report.project_id} needs revision: ${report.review_comment ?? "See Reports page for details."}`,
          date: (report.reviewed_at ?? report.submitted_at).slice(0, 10),
        });
      }
    }

    return NextResponse.json(alerts);
  } catch (error) {
    // No alerts table/data yet is not an error condition for the bell icon — just empty.
    console.error("Failed to load alerts from Supabase:", error);
    return NextResponse.json([]);
  }
}
