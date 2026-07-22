import type { SupabaseClient } from "@supabase/supabase-js";

// Fetches all existing PROJ-{year}-#### ids once and returns an allocator
// that hands out the next sequential id for a given year, incrementing
// in-memory -- so a single bulk-import batch spanning multiple years never
// collides with itself, and a single-add call is just one call to it.
export async function createProjectIdAllocator(supabase: SupabaseClient): Promise<(year: number) => string> {
  const { data, error } = await supabase.from("projects").select("project_id").like("project_id", "PROJ-%");
  if (error) throw error;

  const maxSeqByYear = new Map<number, number>();
  for (const row of (data ?? []) as { project_id: string }[]) {
    const match = /PROJ-(\d{4})-(\d{4})/.exec(row.project_id);
    if (!match) continue;
    const year = parseInt(match[1], 10);
    const seq = parseInt(match[2], 10);
    maxSeqByYear.set(year, Math.max(maxSeqByYear.get(year) ?? 0, seq));
  }

  return function nextProjectId(year: number): string {
    const next = (maxSeqByYear.get(year) ?? 0) + 1;
    maxSeqByYear.set(year, next);
    return `PROJ-${year}-${String(next).padStart(4, "0")}`;
  };
}
