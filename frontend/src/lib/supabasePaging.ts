import type { SupabaseClient } from "@supabase/supabase-js";

const PAGE_SIZE = 1000;

// Supabase/PostgREST enforces a server-side max-rows cap (commonly 1000)
// that a single request's .range() cannot exceed regardless of the span
// requested, so result sets larger than that must be paged client-side.
export async function fetchAllRowsIn<T>(
  supabase: SupabaseClient,
  table: string,
  columns: string,
  inColumn: string,
  inValues: string[],
): Promise<T[]> {
  const rows: T[] = [];
  for (let start = 0; ; start += PAGE_SIZE) {
    const { data, error } = await supabase
      .from(table)
      .select(columns)
      .in(inColumn, inValues)
      .range(start, start + PAGE_SIZE - 1);
    if (error) throw error;
    if (!data || data.length === 0) break;
    rows.push(...(data as unknown as T[]));
    if (data.length < PAGE_SIZE) break;
  }
  return rows;
}
