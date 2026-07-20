import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

// Serves risk-tier escalation and critical-risk alerts computed by the
// backend pipeline by diffing successive runs
// (backend/main.py -> frontend/public/data/alerts.json).
export async function GET() {
  try {
    const filePath = path.join(process.cwd(), "public", "data", "alerts.json");
    const raw = fs.readFileSync(filePath, "utf8");
    return NextResponse.json(JSON.parse(raw));
  } catch {
    // No alerts file yet is not an error condition for the bell icon — just empty.
    return NextResponse.json([]);
  }
}
