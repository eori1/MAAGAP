import { NextResponse } from "next/server";
import { getSessionProfile } from "@/lib/supabaseSessionServer";

// Returns the current session's identity + role for client components
// (Sidebar, page-level role gates) that can't call server-only helpers directly.
export async function GET() {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }
  return NextResponse.json(profile);
}
