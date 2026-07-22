import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";
import { fetchAllRowsIn } from "@/lib/supabasePaging";
import { pickLatestByKey } from "@/lib/inspectionReports";
import { expectedProgressPct } from "@/lib/projectProgress";

interface PredictionRow {
  project_id: string;
  risk_tier: string;
  delay_probability: number;
  predicted_delay_days: number;
}

interface InspectionReportRow {
  project_id: string;
  inspector_id: string;
  physical_accomplishment_pct: number | null;
  financial_accomplishment_pct: number | null;
  submitted_at: string;
}

interface ProjectRow {
  project_id: string;
  start_date: string;
  planned_duration_months: number;
}

interface InspectorNameRow {
  inspector_id: string;
  inspector_name: string;
}

type Agreement = "confirmed" | "contradicted" | "inconclusive";

// Compares what the model predicted (risk_tier) against what an inspector
// actually reported (physical_accomplishment_pct vs. the expected-by-now
// progress) -- a real-world validation view, not a retraining loop. Real
// inspection_reports never carry a final "was this project delayed"/"final
// cost overrun" outcome (only knowable once a project closes out), so this
// deliberately compares delay/progress risk only, and only for projects that
// have at least one real report -- nothing to validate against otherwise.
export async function GET() {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
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

    const { data: predData, error: predErr } = await supabase
      .from("predictions")
      .select("project_id, risk_tier, delay_probability, predicted_delay_days");
    if (predErr) throw predErr;

    let predictions = (predData ?? []) as unknown as PredictionRow[];
    if (ownProjectIds) predictions = predictions.filter((p) => ownProjectIds!.has(p.project_id));

    const projectIds = predictions.map((p) => p.project_id);
    const predictionByProject = new Map(predictions.map((p) => [p.project_id, p]));
    if (projectIds.length === 0) return NextResponse.json([]);

    const [realReports, projData, { data: insData, error: insErr }] = await Promise.all([
      fetchAllRowsIn<InspectionReportRow>(supabase, "inspection_reports", "project_id, inspector_id, physical_accomplishment_pct, financial_accomplishment_pct, submitted_at", "project_id", projectIds),
      fetchAllRowsIn<ProjectRow>(supabase, "projects", "project_id, start_date, planned_duration_months", "project_id", projectIds),
      supabase.from("inspectors").select("inspector_id, inspector_name"),
    ]);
    if (insErr) throw insErr;

    const inspectors = (insData ?? []) as unknown as InspectorNameRow[];
    const nameByInspector = new Map(inspectors.map((i) => [i.inspector_id, i.inspector_name]));
    const projectById = new Map((projData as unknown as ProjectRow[]).map((p) => [p.project_id, p]));

    // Only projects with at least one real report are relevant here.
    const latestRealByProject = pickLatestByKey(
      realReports as unknown as InspectionReportRow[],
      (r) => r.project_id,
      (r) => r.submitted_at,
    );

    const rows = Array.from(latestRealByProject.entries()).map(([projectId, real]) => {
      const prediction = predictionByProject.get(projectId);
      const submittedAt = new Date(real.submitted_at);
      const expectedProgress = expectedProgressPct(projectById.get(projectId), submittedAt);
      const actualProgress = real.physical_accomplishment_pct;
      const slippage = actualProgress !== null && expectedProgress !== null ? expectedProgress - actualProgress : null;

      const predictedBucket: "risk" | "safe" = prediction && (prediction.risk_tier === "High" || prediction.risk_tier === "Critical") ? "risk" : "safe";
      const actualBucket: "risk" | "on_track" | "inconclusive" = slippage === null ? "inconclusive" : slippage > 5 ? "risk" : "on_track";

      let agreement: Agreement;
      if (actualBucket === "inconclusive") agreement = "inconclusive";
      else if ((predictedBucket === "risk" && actualBucket === "risk") || (predictedBucket === "safe" && actualBucket === "on_track")) agreement = "confirmed";
      else agreement = "contradicted";

      return {
        projectId,
        riskTier: prediction?.risk_tier ?? "Low",
        delayProbability: prediction?.delay_probability ?? null,
        predictedDelayDays: prediction?.predicted_delay_days ?? null,
        expectedProgress,
        actualProgress,
        financialAccomplishmentPct: real.financial_accomplishment_pct,
        slippage,
        agreement,
        reportDate: real.submitted_at.slice(0, 10),
        inspectorName: nameByInspector.get(real.inspector_id) ?? real.inspector_id,
      };
    });

    rows.sort((a, b) => (a.reportDate < b.reportDate ? 1 : -1));
    return NextResponse.json(rows);
  } catch (error) {
    console.error("Failed to load model validation data from Supabase:", error);
    return NextResponse.json(
      { error: "Validation data not available. Run the backend pipeline (python main.py) with Supabase configured." },
      { status: 404 },
    );
  }
}
