import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";
import { getSessionProfile } from "@/lib/supabaseSessionServer";
import { friendlyFeatureLabel } from "@/lib/shapLabels";

interface PredictionRow {
  project_id: string;
  risk_tier: string;
  delay_probability: number;
  cost_overrun_probability: number;
  predicted_delay_days: number;
  shap_explanation: string | null;
}

interface ShapAttribution {
  feature: string;
  shap_value: number;
}

interface ShapJson {
  base_value: number;
  predicted_contribution: number;
  top_contributing_factors: ShapAttribution[];
}

// Serves one project's real prediction detail + SHAP feature attributions
// on demand -- fetched only when a project is selected (e.g. on the
// Forecast Engine page), rather than bloating the /api/projects list
// payload every consumer pays for. The underlying data (predictions.shap_explanation)
// is already computed by the backend pipeline for every monitored project;
// this route just exposes it.
export async function GET(request: NextRequest, { params }: { params: Promise<{ projectId: string }> }) {
  const profile = await getSessionProfile();
  if (!profile) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  const { projectId } = await params;
  const supabase = getSupabaseServerClient();

  if (profile.role === "inspector") {
    const { data: asg } = await supabase
      .from("assignments")
      .select("project_id")
      .eq("inspector_id", profile.inspectorId ?? "__none__")
      .eq("project_id", projectId)
      .maybeSingle();
    if (!asg) {
      return NextResponse.json({ error: "This project is not assigned to you" }, { status: 403 });
    }
  }

  const { data, error } = await supabase
    .from("predictions")
    .select("project_id, risk_tier, delay_probability, cost_overrun_probability, predicted_delay_days, shap_explanation")
    .eq("project_id", projectId)
    .single();

  if (error || !data) {
    return NextResponse.json({ error: "Prediction not found for this project" }, { status: 404 });
  }

  const prediction = data as unknown as PredictionRow;

  let shap: { baseValue: number; predictedContribution: number; factors: { feature: string; friendlyLabel: string; shapValue: number }[] } | null = null;
  if (prediction.shap_explanation) {
    try {
      const parsed = JSON.parse(prediction.shap_explanation) as ShapJson;
      const sorted = [...parsed.top_contributing_factors].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
      shap = {
        baseValue: parsed.base_value,
        predictedContribution: parsed.predicted_contribution,
        factors: sorted.slice(0, 6).map((f) => ({
          feature: f.feature,
          friendlyLabel: friendlyFeatureLabel(f.feature),
          shapValue: f.shap_value,
        })),
      };
    } catch {
      shap = null;
    }
  }

  return NextResponse.json({
    projectId: prediction.project_id,
    riskTier: prediction.risk_tier,
    delayProbability: prediction.delay_probability,
    costOverrunProbability: prediction.cost_overrun_probability,
    predictedDelayDays: prediction.predicted_delay_days,
    shap,
  });
}
