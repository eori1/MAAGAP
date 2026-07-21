import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";

// PATCH: update a user's role / inspector link (Admin only).
export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const profile = await getSessionProfile();
  if (!profile) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  if (profile.role !== "admin") return NextResponse.json({ error: "Admin access required" }, { status: 403 });

  const { id } = await params;
  const body = await request.json();
  const { role, inspectorId } = body as { role?: string; inspectorId?: string | null };

  if (role && !["manager", "inspector", "admin"].includes(role)) {
    return NextResponse.json({ error: "Invalid role" }, { status: 400 });
  }

  const supabase = getSupabaseServerClient();
  const update: Record<string, unknown> = {};
  if (role) update.role = role;
  if (inspectorId !== undefined) update.inspector_id = inspectorId;

  const { error } = await supabase.from("profiles").update(update as never).eq("id", id);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ ok: true });
}
