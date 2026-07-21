import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";

// Serves per-project actual-vs-scheduled deviation data from Supabase,
// joining predictions (risk tier, forecast) with the project registry.
export async function GET() {
  try {
    const supabase = getSupabaseServerClient();

    const { data, error } = await supabase
      .from("predictions")
      .select(
        "project_id, predicted_delay_days, risk_tier, projects(project_id, project_type, location, project_year, start_date, planned_end_date, planned_duration_months, is_delayed, actual_delay_days)",
      );
    if (error) throw error;

    type Row = {
      project_id: string;
      predicted_delay_days: number;
      risk_tier: string;
      projects: {
        project_type: string;
        location: string;
        project_year: number;
        start_date: string;
        planned_end_date: string;
        planned_duration_months: number;
        is_delayed: boolean;
        actual_delay_days: number;
      } | null;
    };

    const rows = (data ?? []) as unknown as Row[];
    const maxYear = Math.max(...rows.map((r) => r.projects?.project_year ?? 0));

    const timeline = rows
      .filter((r) => r.projects)
      .map((r) => {
        const p = r.projects!;
        const status = p.is_delayed ? "Delayed" : p.project_year === maxYear ? "Ongoing" : "Completed";
        return {
          id: r.project_id,
          name: r.project_id,
          location: p.location,
          type: p.project_type,
          year: p.project_year,
          startDate: p.start_date,
          plannedEndDate: p.planned_end_date,
          plannedMonths: p.planned_duration_months,
          actualDelayDays: p.actual_delay_days,
          predictedDelayDays: r.predicted_delay_days,
          riskTier: r.risk_tier,
          status,
        };
      });

    return NextResponse.json(timeline);
  } catch (error) {
    console.error("Failed to load timeline from Supabase:", error);
    return NextResponse.json(
      { error: "Timeline data not available. Run the backend pipeline (python main.py) with Supabase configured." },
      { status: 404 },
    );
  }
}
