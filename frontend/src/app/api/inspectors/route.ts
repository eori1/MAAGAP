import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabaseServer";

interface InspectorRow {
  inspector_id: string;
  inspector_name: string;
  availability_status: string;
  vehicle_access: boolean;
  capacity: number;
}

interface AssignmentInspectorIdRow {
  inspector_id: string;
}

// Serves the PPDO inspector roster with LP-computed capacity/workload from
// Supabase, joined with a live count of current assignments.
export async function GET() {
  try {
    const supabase = getSupabaseServerClient();

    const [{ data: insData, error: insErr }, { data: asgData, error: asgErr }] = await Promise.all([
      supabase.from("inspectors").select("*"),
      supabase.from("assignments").select("inspector_id"),
    ]);
    if (insErr) throw insErr;
    if (asgErr) throw asgErr;

    const inspectors = (insData ?? []) as unknown as InspectorRow[];
    const assignments = (asgData ?? []) as unknown as AssignmentInspectorIdRow[];

    const assignedCount = new Map<string, number>();
    for (const a of assignments) {
      assignedCount.set(a.inspector_id, (assignedCount.get(a.inspector_id) ?? 0) + 1);
    }

    const roster = inspectors.map((r) => {
      const handle = r.inspector_name.split(".").pop()!.trim().toLowerCase().replace(/\s+/g, "");
      return {
        id: r.inspector_id,
        name: r.inspector_name,
        email: `${handle}@iloilo.gov.ph`,
        position: "Project Inspector",
        role: "Inspector",
        status: String(r.availability_status).toLowerCase() === "available" ? "Active" : "On Duty",
        vehicleAccess: r.vehicle_access,
        capacity: r.capacity,
        assigned: assignedCount.get(r.inspector_id) ?? 0,
      };
    });

    return NextResponse.json(roster);
  } catch (error) {
    console.error("Failed to load inspectors from Supabase:", error);
    return NextResponse.json(
      { error: "Inspector roster not available. Run the backend pipeline (python main.py) with Supabase configured." },
      { status: 404 },
    );
  }
}
