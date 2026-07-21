import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";

interface AssignmentRow {
  assignment_id: string;
  inspector_id: string;
  status: string;
}

// PATCH: an Inspector accepts one of their own assigned visits. Only the
// assigned inspector may accept -- Manager/Admin can view status but not
// act on someone else's assignment.
export async function PATCH(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }
  if (profile.role !== "inspector") {
    return NextResponse.json({ error: "Only the assigned inspector can accept a visit" }, { status: 403 });
  }

  const { id } = await params;
  const supabase = getSupabaseServerClient();

  const { data, error } = await supabase
    .from("assignments")
    .select("assignment_id, inspector_id, status")
    .eq("assignment_id", id)
    .single();
  if (error || !data) {
    return NextResponse.json({ error: "Assignment not found" }, { status: 404 });
  }

  const assignment = data as unknown as AssignmentRow;
  if (assignment.inspector_id !== profile.inspectorId) {
    return NextResponse.json({ error: "This assignment is not yours to accept" }, { status: 403 });
  }
  if (assignment.status === "accepted") {
    return NextResponse.json({ ok: true, status: "accepted" });
  }

  const { error: updateError } = await supabase
    .from("assignments")
    .update({ status: "accepted", accepted_at: new Date().toISOString() } as never)
    .eq("assignment_id", id);
  if (updateError) {
    return NextResponse.json({ error: updateError.message }, { status: 500 });
  }

  return NextResponse.json({ ok: true, status: "accepted" });
}
