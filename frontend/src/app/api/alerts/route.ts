import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";

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
export async function GET() {
  try {
    const supabase = getSupabaseServerClient();
    const { data, error } = await supabase
      .from("risk_alerts")
      .select("*")
      .order("risk_score", { ascending: false });
    if (error) throw error;

    const rows = (data ?? []) as unknown as RiskAlertRow[];
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
