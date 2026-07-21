import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";
import { fetchAllRowsIn } from "@/lib/supabasePaging";

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

interface AssignmentRow {
  project_id: string;
  inspector_id: string;
}

interface InspectorNameRow {
  inspector_id: string;
  inspector_name: string;
}

// Serves the latest quarterly inspection log entry per monitored project,
// joining inspection_logs with predictions (risk tier) and inspectors (name).
// Inspectors only see reports for projects assigned to them.
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

    const [logs, { data: asgData, error: asgErr2 }, { data: insData, error: insErr }] = await Promise.all([
      fetchAllRowsIn<InspectionLogRow>(supabase, "inspection_logs", "*", "project_id", projectIds),
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

    // Keep only the latest quarter per project (mirrors the backend's
    // sort-by-quarter + groupby-tail(1) used when these were JSON files).
    const latestByProject = new Map<string, InspectionLogRow>();
    for (const log of logs) {
      const existing = latestByProject.get(log.project_id);
      if (!existing || log.quarter > existing.quarter) latestByProject.set(log.project_id, log);
    }

    const reports = Array.from(latestByProject.values()).map((q) => {
      const slippage = Number(q.slippage_pct);
      const status = slippage > 20 ? "Flagged" : slippage > 5 ? "Pending Review" : "Validated";
      const assignedInspectorId = assignedInspectorByProject.get(q.project_id) ?? null;
      return {
        projectId: q.project_id,
        quarter: q.quarter,
        totalQuarters: q.total_quarters,
        plannedProgress: q.target_physical_pct,
        actualProgress: q.actual_physical_pct,
        slippage,
        issues: q.issues_noted,
        date: q.report_date,
        status,
        inspectorId: assignedInspectorId,
        inspectorName: assignedInspectorId ? (nameByInspector.get(assignedInspectorId) ?? assignedInspectorId) : "Unassigned",
        riskTier: tierByProject.get(q.project_id) ?? "Low",
      };
    });

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
