import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";
import { fetchAllRowsIn } from "@/lib/supabasePaging";
import { getProjectCoordinates } from "@/lib/iloiloMunicipalityCentroids";
import { PROJECT_TYPES } from "@/lib/ppaOptions";
import { validatePpaRow } from "@/lib/ppaValidation";
import { createProjectIdAllocator } from "@/lib/projectId";

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
  project_name: string | null;
  description: string | null;
}

interface ManualProjectRow {
  project_id: string;
  location: string;
  budget_allocated: number;
  status: string;
  project_name: string | null;
  description: string | null;
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

interface ProjectListItem {
  id: string;
  name: string;
  description: string | null;
  municipality: string;
  progress: number;
  budget: string;
  risk: string;
  status: string;
  inspector: string;
  lat: number;
  lng: number;
  delayProb: number | null;
  costRisk: number | null;
}

// Serves the currently-monitored project cohort (the ~450 projects scored
// by the most recent pipeline run) from Supabase, in the shape the
// Dashboard/Projects/Forecast Engine pages expect -- replacing the old
// static demo_projects.csv. Inspectors only see their own assigned projects.
// Also appends manually-added PPAs ("Add new PPA") that haven't been scored
// by a pipeline run yet, shown with risk "Pending" -- see schema_manual_entry.sql.
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

    const predByProject = new Map(predictions.map((p) => [p.project_id, p]));
    let result: ProjectListItem[] = [];

    if (predictions.length > 0) {
      const projectIds = predictions.map((p) => p.project_id);

      const [{ data: projData, error: projErr }, logs, { data: asgData, error: asgErr2 }, { data: insData, error: insErr }] =
        await Promise.all([
          supabase.from("projects").select("project_id, location, budget_allocated, project_name, description").in("project_id", projectIds),
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

      result = projects.map((p) => {
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
          name: p.project_name || p.project_id,
          description: p.description ?? null,
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
    }

    // Manually-added PPAs not yet scored by a pipeline run -- "Pending
    // Assessment". Not shown to Inspectors: a pending entry has no
    // assignment yet, so it isn't theirs to see.
    // Isolated from the main try/catch: if schema_manual_entry.sql hasn't
    // been applied yet (is_manual_entry column doesn't exist), this should
    // degrade to "no pending entries" rather than breaking the scored
    // ~450-project data above, which doesn't depend on this column at all.
    if (profile.role !== "inspector") {
      try {
        const { data: manualData, error: manualErr } = await supabase
          .from("projects")
          .select("project_id, location, budget_allocated, status, project_name, description")
          .eq("is_manual_entry", true);
        if (manualErr) throw manualErr;

        const manualRows = (manualData ?? []) as unknown as ManualProjectRow[];
        for (const p of manualRows) {
          if (predByProject.has(p.project_id)) continue; // scored by a later pipeline run
          const [lat, lng] = getProjectCoordinates(p.location, p.project_id);
          result.push({
            id: p.project_id,
            name: p.project_name || p.project_id,
            description: p.description ?? null,
            municipality: p.location,
            progress: 0,
            budget: Number(p.budget_allocated || 0).toLocaleString(),
            risk: "Pending",
            status: p.status ?? "Not Started",
            inspector: "N/A",
            lat,
            lng,
            delayProb: null,
            costRisk: null,
          });
        }
      } catch (manualEntryError) {
        console.error("Failed to load pending manual PPAs (schema_manual_entry.sql applied?):", manualEntryError);
      }
    }

    return NextResponse.json(result);
  } catch (error) {
    console.error("Failed to load projects from Supabase:", error);
    return NextResponse.json(
      { error: "Project data not available. Run the backend pipeline (python main.py) with Supabase configured." },
      { status: 404 },
    );
  }
}

// POST: Manager/Admin manually add a new PPA. Cannot get an immediate risk
// score -- the ML pipeline is a batch process, not on-demand -- so it's
// created with is_manual_entry=true and shows as "Pending Assessment"
// (via GET above) until the next full pipeline run scores it.
export async function POST(request: NextRequest) {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }
  if (profile.role === "inspector") {
    return NextResponse.json({ error: "Only a Manager or Admin can add a new PPA" }, { status: 403 });
  }

  const body = await request.json();
  const result = validatePpaRow(body);
  if (!result.valid || !result.normalized) {
    return NextResponse.json({ error: result.error }, { status: 400 });
  }
  const { projectName, description, projectType, category, location, budgetAllocated, startDate, fundingSource } = result.normalized;

  const supabase = getSupabaseServerClient();
  const typeConfig = PROJECT_TYPES.find((t) => t.value === projectType)!;

  const start = new Date(startDate);
  const plannedDurationMonths = typeConfig.durationMonths;
  const plannedEnd = new Date(start);
  plannedEnd.setMonth(plannedEnd.getMonth() + plannedDurationMonths);

  const nextProjectId = await createProjectIdAllocator(supabase);
  const projectId = nextProjectId(start.getFullYear());

  const { error: insertErr } = await supabase
    .from("projects")
    .insert({
      project_id: projectId,
      project_name: projectName,
      description: description || null,
      project_type: projectType,
      category,
      location,
      budget_allocated: budgetAllocated,
      planned_duration_months: plannedDurationMonths,
      start_date: start.toISOString().slice(0, 10),
      planned_end_date: plannedEnd.toISOString().slice(0, 10),
      funding_source: fundingSource,
      status: "Not Started",
      is_manual_entry: true,
      created_by: profile.userId,
    } as never);
  if (insertErr) {
    return NextResponse.json({ error: insertErr.message }, { status: 500 });
  }

  const [lat, lng] = getProjectCoordinates(location, projectId);
  const created: ProjectListItem = {
    id: projectId,
    name: projectName,
    description: description || null,
    municipality: location,
    progress: 0,
    budget: budgetAllocated.toLocaleString(),
    risk: "Pending",
    status: "Not Started",
    inspector: "N/A",
    lat,
    lng,
    delayProb: null,
    costRisk: null,
  };

  return NextResponse.json(created, { status: 201 });
}
