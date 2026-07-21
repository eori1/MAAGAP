import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";

interface RiskAlertRow {
  id: string;
  type: string;
  project_id: string;
  from_tier: string | null;
  to_tier: string;
  risk_score: number;
  message: string;
  alert_date: string;
}

// Serves risk-tier escalation and critical-risk alerts persisted in
// Supabase (computed by the backend pipeline diffing successive runs).
// Inspectors only see alerts for projects assigned to them.
export async function GET() {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json([]);
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

    const { data, error } = await supabase
      .from("risk_alerts")
      .select("*")
      .order("risk_score", { ascending: false });
    if (error) throw error;

    let rows = (data ?? []) as unknown as RiskAlertRow[];
    if (ownProjectIds) rows = rows.filter((a) => ownProjectIds!.has(a.project_id));

    const alerts = rows.map((a) => ({
      id: a.id,
      type: a.type,
      projectId: a.project_id,
      fromTier: a.from_tier,
      toTier: a.to_tier,
      riskScore: a.risk_score,
      message: a.message,
      date: a.alert_date,
    }));

    return NextResponse.json(alerts);
  } catch (error) {
    // No alerts table/data yet is not an error condition for the bell icon — just empty.
    console.error("Failed to load alerts from Supabase:", error);
    return NextResponse.json([]);
  }
}
