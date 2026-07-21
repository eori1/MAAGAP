import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";

// PATCH: a Manager/Admin approves or requests revision on a real,
// inspector-submitted report. Inspectors cannot review their own (or
// anyone's) reports -- read-only for that role.
export async function PATCH(request: NextRequest, { params }: { params: Promise<{ reportId: string }> }) {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }
  if (profile.role !== "manager" && profile.role !== "admin") {
    return NextResponse.json({ error: "Only a Manager or Admin can review reports" }, { status: 403 });
  }

  const { reportId } = await params;
  const body = await request.json();
  const { action, comment } = body as { action?: string; comment?: string };

  if (action !== "approve" && action !== "request_revision") {
    return NextResponse.json({ error: "action must be 'approve' or 'request_revision'" }, { status: 400 });
  }
  if (action === "request_revision" && !comment?.trim()) {
    return NextResponse.json({ error: "A comment is required when requesting revision" }, { status: 400 });
  }

  const supabase = getSupabaseServerClient();

  const { data, error } = await supabase
    .from("inspection_reports")
    .update({
      review_status: action === "approve" ? "approved" : "needs_revision",
      review_comment: action === "request_revision" ? comment!.trim() : null,
      reviewed_by: profile.userId,
      reviewed_at: new Date().toISOString(),
    } as never)
    .eq("report_id", reportId)
    .select()
    .single();

  if (error || !data) {
    return NextResponse.json({ error: error?.message ?? "Report not found" }, { status: 404 });
  }

  return NextResponse.json(data);
}
