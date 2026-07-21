import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";

interface ProfileRow {
  id: string;
  email: string;
  full_name: string | null;
  role: "manager" | "inspector" | "admin";
  inspector_id: string | null;
  created_at: string;
}

// GET: list accounts (Manager + Admin can view, per the Use Case Diagram's
// "PPDO Manager... full oversight" and "System Administrator manages user
// credentials"). POST: create a new account (Admin only -- AuthService).
export async function GET() {
  const profile = await getSessionProfile();
  if (!profile) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  if (profile.role === "inspector") return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase
    .from("profiles")
    .select("id, email, full_name, role, inspector_id, created_at")
    .order("created_at", { ascending: true });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json((data ?? []) as unknown as ProfileRow[]);
}

export async function POST(request: NextRequest) {
  const profile = await getSessionProfile();
  if (!profile) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  if (profile.role !== "admin") return NextResponse.json({ error: "Admin access required" }, { status: 403 });

  const body = await request.json();
  const { email, password, fullName, role, inspectorId } = body as {
    email?: string;
    password?: string;
    fullName?: string;
    role?: string;
    inspectorId?: string | null;
  };

  if (!email || !password) {
    return NextResponse.json({ error: "Email and password are required" }, { status: 400 });
  }
  if (role && !["manager", "inspector", "admin"].includes(role)) {
    return NextResponse.json({ error: "Invalid role" }, { status: 400 });
  }

  const supabase = getSupabaseServerClient();

  const { data: created, error: createErr } = await supabase.auth.admin.createUser({
    email,
    password,
    email_confirm: true,
    user_metadata: { full_name: fullName ?? null },
  });
  if (createErr || !created.user) {
    return NextResponse.json({ error: createErr?.message ?? "Failed to create user" }, { status: 500 });
  }

  // The on_auth_user_created trigger already inserted a default profile row;
  // update it with the requested role/inspector link.
  if (role || inspectorId !== undefined) {
    const { error: updateErr } = await supabase
      .from("profiles")
      .update({ role: role ?? "inspector", inspector_id: inspectorId ?? null } as never)
      .eq("id", created.user.id);
    if (updateErr) return NextResponse.json({ error: updateErr.message }, { status: 500 });
  }

  return NextResponse.json({ id: created.user.id }, { status: 201 });
}
