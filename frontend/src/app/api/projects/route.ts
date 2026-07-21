import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";
import { fetchAllRowsIn } from "@/lib/supabasePaging";
import { getProjectCoordinates } from "@/lib/iloiloMunicipalityCentroids";

interface PredictionRow {
  project_id: string;
  risk_tier: string;
  delay_probability: number;
  cost_overrun_probability: number;
}

interface ProjectRow {
  project_id: string;
  location: string;
  budget_allocated: number;
}

interface InspectionLogRow {
  project_id: string;
  quarter: number;
  actual_physical_pct: number;
}

interface AssignmentRow {
  project_id: string;
  inspector_id: string;
}

interface InspectorNameRow {
  inspector_id: string;
  inspector_name: string;
}

// Serves the currently-monitored project cohort (the ~450 projects scored
// by the most recent pipeline run) from Supabase, in the shape the
// Dashboard/Projects/Forecast Engine pages expect -- replacing the old
// static demo_projects.csv. Inspectors only see their own assigned projects.
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
      .select("project_id, risk_tier, delay_probability, cost_overrun_probability");
    if (predErr) throw predErr;

    let predictions = (predData ?? []) as unknown as PredictionRow[];
    if (ownProjectIds) predictions = predictions.filter((p) => ownProjectIds!.has(p.project_id));
    if (predictions.length === 0) return NextResponse.json([]);

    const projectIds = predictions.map((p) => p.project_id);
    const predByProject = new Map(predictions.map((p) => [p.project_id, p]));

    const [{ data: projData, error: projErr }, logs, { data: asgData, error: asgErr2 }, { data: insData, error: insErr }] =
      await Promise.all([
        supabase.from("projects").select("project_id, location, budget_allocated").in("project_id", projectIds),
        fetchAllRowsIn<InspectionLogRow>(supabase, "inspection_logs", "project_id, quarter, actual_physical_pct", "project_id", projectIds),
        supabase.from("assignments").select("project_id, inspector_id").in("project_id", projectIds),
        supabase.from("inspectors").select("inspector_id, inspector_name"),
      ]);
    if (projErr) throw projErr;
    if (asgErr2) throw asgErr2;
    if (insErr) throw insErr;

    const projects = (projData ?? []) as unknown as ProjectRow[];
    const assignments = (asgData ?? []) as unknown as AssignmentRow[];
    const inspectors = (insData ?? []) as unknown as InspectorNameRow[];

    const inspectorByProject = new Map(assignments.map((a) => [a.project_id, a.inspector_id]));
    const nameByInspector = new Map(inspectors.map((i) => [i.inspector_id, i.inspector_name]));

    // Latest quarter's actual physical accomplishment = current progress.
    const latestProgressByProject = new Map<string, number>();
    const latestQuarterByProject = new Map<string, number>();
    for (const log of logs) {
      const seen = latestQuarterByProject.get(log.project_id) ?? -1;
      if (log.quarter > seen) {
        latestQuarterByProject.set(log.project_id, log.quarter);
        latestProgressByProject.set(log.project_id, log.actual_physical_pct);
      }
    }

    const result = projects.map((p) => {
      const pred = predByProject.get(p.project_id);
      const progress = Math.round(latestProgressByProject.get(p.project_id) ?? 0);
      const delayProb = pred?.delay_probability ?? 0;
      const inspectorId = inspectorByProject.get(p.project_id);
      const [lat, lng] = getProjectCoordinates(p.location, p.project_id);

      let status = "In Progress";
      if (progress >= 100) status = "Completed";
      else if (delayProb > 0.6) status = "Delayed";
      else if (delayProb < 0.4) status = "On Schedule";

      return {
        id: p.project_id,
        name: p.project_id,
        municipality: p.location,
        progress,
        budget: Number(p.budget_allocated || 0).toLocaleString(),
        risk: pred?.risk_tier ?? "Low",
        status,
        inspector: inspectorId ? (nameByInspector.get(inspectorId) ?? inspectorId) : "N/A",
        lat,
        lng,
        delayProb,
        costRisk: pred?.cost_overrun_probability ?? 0,
      };
    });

    return NextResponse.json(result);
  } catch (error) {
    console.error("Failed to load projects from Supabase:", error);
    return NextResponse.json(
      { error: "Project data not available. Run the backend pipeline (python main.py) with Supabase configured." },
      { status: 404 },
    );
  }
}
