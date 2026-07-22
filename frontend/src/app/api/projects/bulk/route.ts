import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";
import { PROJECT_TYPES } from "@/lib/ppaOptions";
import { validatePpaRow } from "@/lib/ppaValidation";
import { createProjectIdAllocator } from "@/lib/projectId";

// POST: Manager/Admin bulk-import PPAs from a CSV (already parsed and
// previewed client-side -- see ImportPpaModal). Same "Pending Assessment"
// framing as the single-add route: these rows cannot get a risk score until
// the next full pipeline run. Re-validates every row server-side -- the
// client-side preview is a UX aid, never trusted as the sole gate.
export async function POST(request: NextRequest) {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }
  if (profile.role === "inspector") {
    return NextResponse.json({ error: "Only a Manager or Admin can import PPAs" }, { status: 403 });
  }

  const body = await request.json();
  const rows = (body as { rows?: unknown[] }).rows;
  if (!Array.isArray(rows) || rows.length === 0) {
    return NextResponse.json({ error: "No rows to import" }, { status: 400 });
  }

  const supabase = getSupabaseServerClient();
  const nextProjectId = await createProjectIdAllocator(supabase);

  const errors: { row: number; error: string }[] = [];
  const inserts: Record<string, unknown>[] = [];

  rows.forEach((row, index) => {
    const result = validatePpaRow(row as Record<string, string | number | undefined>);
    if (!result.valid || !result.normalized) {
      errors.push({ row: index + 1, error: result.error ?? "Invalid row" });
      return;
    }
    const { projectName, description, projectType, category, location, budgetAllocated, startDate, fundingSource } = result.normalized;
    const typeConfig = PROJECT_TYPES.find((t) => t.value === projectType)!;

    const start = new Date(startDate);
    const plannedEnd = new Date(start);
    plannedEnd.setMonth(plannedEnd.getMonth() + typeConfig.durationMonths);
    const projectId = nextProjectId(start.getFullYear());

    inserts.push({
      project_id: projectId,
      project_name: projectName,
      description: description || null,
      project_type: projectType,
      category,
      location,
      budget_allocated: budgetAllocated,
      planned_duration_months: typeConfig.durationMonths,
      start_date: start.toISOString().slice(0, 10),
      planned_end_date: plannedEnd.toISOString().slice(0, 10),
      funding_source: fundingSource,
      status: "Not Started",
      is_manual_entry: true,
      created_by: profile.userId,
    });
  });

  if (inserts.length > 0) {
    const { error: insertErr } = await supabase.from("projects").insert(inserts as never);
    if (insertErr) {
      return NextResponse.json({ error: insertErr.message }, { status: 500 });
    }
  }

  return NextResponse.json({ created: inserts.length, errors });
}
