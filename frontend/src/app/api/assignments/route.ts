import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";

interface InspectorRow {
  inspector_id: string;
  inspector_name: string;
  availability_status: string;
  current_workload: number;
  vehicle_access: boolean;
  capacity: number;
}

interface AssignmentRow {
  assignment_id: string;
  project_id: string;
  inspector_id: string;
  project_type: string;
  location: string;
  risk_score: number;
  risk_tier: string;
  priority: string;
  urgency: string;
  status: string;
}

interface ReportRow {
  assignment_id: string;
  submitted_at: string;
}

// Serves the LP-optimized inspector deployment schedule from Supabase
// (populated by backend/main.py -> maagap.database.sync_all). Inspectors are
// scoped to their own schedule only; Manager/Admin see the full roster.
export async function GET() {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  try {
    const supabase = getSupabaseServerClient();

    const [{ data: insData, error: insErr }, { data: asgData, error: asgErr }, { count: totalProjects, error: cntErr }] =
      await Promise.all([
        supabase.from("inspectors").select("*"),
        supabase.from("assignments").select("*"),
        supabase.from("predictions").select("*", { count: "exact", head: true }),
      ]);

    if (insErr) throw insErr;
    if (asgErr) throw asgErr;
    if (cntErr) throw cntErr;

    let inspectors = (insData ?? []) as unknown as InspectorRow[];
    let assignments = (asgData ?? []) as unknown as AssignmentRow[];

    if (profile.role === "inspector") {
      inspectors = inspectors.filter((i) => i.inspector_id === profile.inspectorId);
      assignments = assignments.filter((a) => a.inspector_id === profile.inspectorId);
    }

    const assignmentIds = assignments.map((a) => a.assignment_id);
    let reportedAssignmentIds = new Set<string>();
    if (assignmentIds.length > 0) {
      const { data: reportData, error: reportErr } = await supabase
        .from("inspection_reports")
        .select("assignment_id, submitted_at")
        .in("assignment_id", assignmentIds);
      if (reportErr) throw reportErr;
      const reports = (reportData ?? []) as unknown as ReportRow[];
      reportedAssignmentIds = new Set(reports.map((r) => r.assignment_id));
    }

    const byInspector = new Map<string, AssignmentRow[]>();
    for (const a of assignments) {
      const list = byInspector.get(a.inspector_id) ?? [];
      list.push(a);
      byInspector.set(a.inspector_id, list);
    }

    const inspectorPayload = inspectors.map((insp) => {
      const rows = (byInspector.get(insp.inspector_id) ?? []).sort((a, b) => b.risk_score - a.risk_score);
      return {
        id: insp.inspector_id,
        name: insp.inspector_name,
        availability: insp.availability_status,
        vehicleAccess: insp.vehicle_access,
        capacity: insp.capacity,
        currentWorkload: insp.current_workload,
        totalProjects: rows.length,
        projects: rows.map((r) => ({
          assignmentId: r.assignment_id,
          projectId: r.project_id,
          name: r.project_id,
          location: r.location,
          type: r.project_type,
          riskScore: r.risk_score,
          riskTier: r.risk_tier,
          priority: r.priority,
          urgency: r.urgency,
          status: r.status ?? "pending",
          hasReport: reportedAssignmentIds.has(r.assignment_id),
        })),
      };
    });

    const assignedProjects = assignments.length;
    const criticalAssignments = assignments.filter((a) => a.risk_tier === "Critical").length;
    const totalScope = profile.role === "inspector" ? assignedProjects : (totalProjects ?? 0);

    return NextResponse.json({
      generatedAt: new Date().toISOString(),
      solver: "PuLP CBC (Integer LP)",
      totalProjects: totalScope,
      assignedProjects,
      unassignedProjects: Math.max(0, totalScope - assignedProjects),
      criticalAssignments,
      inspectors: inspectorPayload,
    });
  } catch (error) {
    console.error("Failed to load assignments from Supabase:", error);
    return NextResponse.json(
      { error: "Assignment data not available. Run the backend pipeline (python main.py) with Supabase configured." },
      { status: 404 },
    );
  }
}
