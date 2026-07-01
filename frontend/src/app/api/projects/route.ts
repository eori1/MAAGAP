import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import Papa from "papaparse";

interface CsvRow {
  'no.'?: string | number;
  project_name?: string;
  approved_budget?: string | number;
  physical_accomplishment?: string | number;
  status?: string;
  delay_probability?: number;
  contractor?: string;
  location?: string;
  inferred_muni?: string;
  risk_category?: string;
  lat?: number;
  lng?: number;
  overrun_probability?: number;
  [key: string]: unknown;
}

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), "public", "data", "demo_projects.csv");
    const fileContents = fs.readFileSync(filePath, "utf8");

    const parsed = Papa.parse(fileContents, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
    });

    const projects = parsed.data
      .filter((row: unknown) => {
        const r = row as CsvRow;
        return r.project_name && r['no.'];
      })
      .map((row: unknown) => {
      const r = row as CsvRow;
      const budgetStr = Number(r.approved_budget || 0).toLocaleString();
      
      let progress = 50;
      if (r.physical_accomplishment) {
        const parsedProg = parseInt(strVal(r.physical_accomplishment).replace('%',''));
        if (!isNaN(parsedProg)) progress = parsedProg;
      } else {
        if (r.status?.toLowerCase().includes("completed")) progress = 100;
        else if (r.delay_probability && r.delay_probability > 0.6) progress = Math.max(10, 50 - Math.floor(r.delay_probability * 40));
        else progress = 75 + Math.floor(Math.random() * 20);
      }

      const inspector = r.contractor && strVal(r.contractor).length > 2 ? r.contractor : "N/A";
      const locStr = strVal(r.inferred_muni || r.location).split(',')[0];

      let status = "In Progress";
      if (progress >= 100) status = "Completed";
      else if (r.delay_probability && r.delay_probability > 0.6) status = "Delayed";
      else if (r.delay_probability && r.delay_probability < 0.4) status = "On Schedule";

      return {
        id: `PROJ-${r['no.']}`,
        name: strVal(r.project_name),
        municipality: locStr || "Unknown",
        progress: progress,
        budget: budgetStr,
        risk: r.risk_category || "Low",
        status: status,
        inspector: inspector,
        lat: r.lat,
        lng: r.lng,
        delayProb: r.delay_probability,
        costRisk: r.overrun_probability
      };
    });

    return NextResponse.json(projects);
  } catch (error) {
    console.error("Failed to parse projects CSV:", error);
    return NextResponse.json({ error: "Failed to load data" }, { status: 500 });
  }
}

function strVal(v: unknown) {
  if (v === null || v === undefined) return "";
  return String(v);
}
