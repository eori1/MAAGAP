import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";
import { fetchAllRowsIn } from "@/lib/supabasePaging";
import { pickLatestByKey } from "@/lib/inspectionReports";

interface PredictionRow {
  project_id: string;
  risk_tier: string;
}

interface InspectionLogRow {
  project_id: string;
  quarter: number;
  total_quarters: number;
  target_physical_pct: number;
  actual_physical_pct: number;
  slippage_pct: number;
  issues_noted: number;
  report_date: string;
}

interface InspectionReportRow {
  report_id: string;
  project_id: string;
  inspector_id: string;
  physical_accomplishment_pct: number | null;
  financial_accomplishment_pct: number | null;
  issues_noted: string | null;
  notes: string | null;
  photo_urls: string[] | null;
  submitted_at: string;
  review_status: "pending" | "approved" | "needs_revision";
  review_comment: string | null;
}

interface ProjectRow {
  project_id: string;
  start_date: string;
  planned_duration_months: number;
}

interface AssignmentRow {
  project_id: string;
  inspector_id: string;
}

interface InspectorNameRow {
  inspector_id: string;
  inspector_name: string;
}

function statusFromSlippage(slippage: number): "Validated" | "Pending Review" | "Flagged" | "Submitted" {
  if (slippage > 20) return "Flagged";
  if (slippage > 5) return "Pending Review";
  return "Validated";
}

// Serves one report per monitored project: a real, inspector-submitted
// report (from inspection_reports) when one exists, falling back to the
// latest synthetic quarterly inspection_logs entry otherwise. Inspectors
// only see reports for projects assigned to them.
export async function GET() {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
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

    const { data: predData, error: predErr } = await supabase
      .from("predictions")
      .select("project_id, risk_tier");
    if (predErr) throw predErr;

    let predictions = (predData ?? []) as unknown as PredictionRow[];
    if (ownProjectIds) predictions = predictions.filter((p) => ownProjectIds!.has(p.project_id));

    const projectIds = predictions.map((p) => p.project_id);
    const tierByProject = new Map(predictions.map((p) => [p.project_id, p.risk_tier]));
    if (projectIds.length === 0) return NextResponse.json([]);

    const [logs, realReports, projData, { data: asgData, error: asgErr2 }, { data: insData, error: insErr }] =
      await Promise.all([
        fetchAllRowsIn<InspectionLogRow>(supabase, "inspection_logs", "*", "project_id", projectIds),
        fetchAllRowsIn<InspectionReportRow>(supabase, "inspection_reports", "*", "project_id", projectIds),
        fetchAllRowsIn<ProjectRow>(supabase, "projects", "project_id, start_date, planned_duration_months", "project_id", projectIds),
        supabase.from("assignments").select("project_id, inspector_id").in("project_id", projectIds),
        supabase.from("inspectors").select("inspector_id, inspector_name"),
      ]);
    if (asgErr2) throw asgErr2;
    if (insErr) throw insErr;

    const inspectors = (insData ?? []) as unknown as InspectorNameRow[];
    const nameByInspector = new Map(inspectors.map((i) => [i.inspector_id, i.inspector_name]));

    // The officially assigned inspector (LP output) per project -- distinct
    // from inspection_logs, whose inspector_id is a synthetic round-robin
    // "who logged this report" placeholder unrelated to today's assignment.
    const assignments = (asgData ?? []) as unknown as AssignmentRow[];
    const assignedInspectorByProject = new Map(assignments.map((a) => [a.project_id, a.inspector_id]));

    const projectById = new Map((projData as unknown as ProjectRow[]).map((p) => [p.project_id, p]));

    // Latest synthetic quarter per project (pipeline baseline).
    const latestLogByProject = new Map<string, InspectionLogRow>();
    for (const log of logs) {
      const existing = latestLogByProject.get(log.project_id);
      if (!existing || log.quarter > existing.quarter) latestLogByProject.set(log.project_id, log);
    }

    // Latest real inspector-submitted report per project, if any (a project
    // can have more than one row over time -- resubmissions after a "needs
    // revision" review).
    const latestRealByProject = pickLatestByKey(
      realReports as unknown as InspectionReportRow[],
      (r) => r.project_id,
      (r) => r.submitted_at,
    );

    function expectedProgressPct(projectId: string, asOf: Date): number | null {
      const p = projectById.get(projectId);
      if (!p || !p.start_date || !p.planned_duration_months) return null;
      const start = new Date(p.start_date);
      const elapsedMonths = (asOf.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 30);
      return Math.max(0, Math.min(100, (elapsedMonths / p.planned_duration_months) * 100));
    }

    const reports = projectIds.map((projectId) => {
      const real = latestRealByProject.get(projectId);
      const assignedInspectorId = assignedInspectorByProject.get(projectId) ?? null;
      const inspectorName = assignedInspectorId ? (nameByInspector.get(assignedInspectorId) ?? assignedInspectorId) : "Unassigned";
      const riskTier = tierByProject.get(projectId) ?? "Low";

      if (real) {
        const submittedAt = new Date(real.submitted_at);
        const expected = expectedProgressPct(projectId, submittedAt);
        // physical_accomplishment_pct is an optional field on the report
        // form -- null means "not reported", not "reported as zero". Only
        // compute a slippage/status when the inspector actually gave a
        // number; otherwise there's nothing to compare against.
        const actual = real.physical_accomplishment_pct;
        const slippage = actual !== null && expected !== null ? expected - actual : null;
        return {
          projectId,
          source: "inspector" as const,
          quarter: null,
          totalQuarters: null,
          plannedProgress: expected,
          actualProgress: actual,
          slippage,
          issuesSummary: real.issues_noted?.trim() ? real.issues_noted : "No issues noted",
          notes: real.notes ?? "",
          photoUrls: real.photo_urls ?? [],
          date: real.submitted_at.slice(0, 10),
          status: slippage !== null ? statusFromSlippage(slippage) : "Submitted",
          inspectorId: real.inspector_id,
          inspectorName: nameByInspector.get(real.inspector_id) ?? real.inspector_id,
          riskTier,
          reportId: real.report_id,
          reviewStatus: real.review_status,
          reviewComment: real.review_comment,
        };
      }

      const log = latestLogByProject.get(projectId);
      if (!log) return null;
      const slippage = Number(log.slippage_pct);
      return {
        projectId,
        source: "pipeline" as const,
        quarter: log.quarter,
        totalQuarters: log.total_quarters,
        plannedProgress: log.target_physical_pct,
        actualProgress: log.actual_physical_pct,
        slippage,
        issuesSummary: `${log.issues_noted} issue${log.issues_noted === 1 ? "" : "s"} noted`,
        notes: "",
        photoUrls: [],
        date: log.report_date,
        status: statusFromSlippage(slippage),
        inspectorId: assignedInspectorId,
        inspectorName,
        riskTier,
        reportId: null,
        reviewStatus: null,
        reviewComment: null,
      };
    }).filter((r): r is NonNullable<typeof r> => r !== null);

    reports.sort((a, b) => (a.date < b.date ? 1 : -1));
    return NextResponse.json(reports);
  } catch (error) {
    console.error("Failed to load reports from Supabase:", error);
    return NextResponse.json(
      { error: "Report data not available. Run the backend pipeline (python main.py) with Supabase configured." },
      { status: 404 },
    );
  }
}
