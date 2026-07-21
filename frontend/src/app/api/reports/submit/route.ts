import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";

interface AssignmentRow {
  assignment_id: string;
  project_id: string;
  inspector_id: string;
  status: string;
}

// POST: an Inspector submits a real inspection report for one of their own
// *accepted* assignments (photos are uploaded directly to Supabase Storage
// by the client beforehand; this route only stores the resulting URLs).
export async function POST(request: NextRequest) {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }
  if (profile.role !== "inspector") {
    return NextResponse.json({ error: "Only an assigned inspector can submit a report" }, { status: 403 });
  }

  const body = await request.json();
  const {
    assignmentId,
    physicalAccomplishmentPct,
    financialAccomplishmentPct,
    issuesNoted,
    notes,
    photoUrls,
  } = body as {
    assignmentId?: string;
    physicalAccomplishmentPct?: number;
    financialAccomplishmentPct?: number;
    issuesNoted?: string;
    notes?: string;
    photoUrls?: string[];
  };

  if (!assignmentId) {
    return NextResponse.json({ error: "assignmentId is required" }, { status: 400 });
  }

  const supabase = getSupabaseServerClient();

  const { data, error } = await supabase
    .from("assignments")
    .select("assignment_id, project_id, inspector_id, status")
    .eq("assignment_id", assignmentId)
    .single();
  if (error || !data) {
    return NextResponse.json({ error: "Assignment not found" }, { status: 404 });
  }

  const assignment = data as unknown as AssignmentRow;
  if (assignment.inspector_id !== profile.inspectorId) {
    return NextResponse.json({ error: "This assignment is not yours" }, { status: 403 });
  }
  if (assignment.status !== "accepted") {
    return NextResponse.json({ error: "Accept this assignment before submitting a report" }, { status: 409 });
  }

  const { data: inserted, error: insertError } = await supabase
    .from("inspection_reports")
    .insert({
      assignment_id: assignment.assignment_id,
      project_id: assignment.project_id,
      inspector_id: assignment.inspector_id,
      submitted_by: profile.userId,
      physical_accomplishment_pct: physicalAccomplishmentPct ?? null,
      financial_accomplishment_pct: financialAccomplishmentPct ?? null,
      issues_noted: issuesNoted ?? null,
      notes: notes ?? null,
      photo_urls: photoUrls ?? [],
    } as never)
    .select()
    .single();
  if (insertError) {
    return NextResponse.json({ error: insertError.message }, { status: 500 });
  }

  return NextResponse.json(inserted, { status: 201 });
}
