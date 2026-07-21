// Shared "latest submission wins" dedup, used by every route that reads
// inspection_reports (a project/assignment can have more than one row over
// time -- resubmissions after a "needs revision" review, for example).
export function pickLatestByKey<T>(
  rows: T[],
  keyOf: (row: T) => string,
  dateOf: (row: T) => string,
): Map<string, T> {
  const latest = new Map<string, T>();
  for (const row of rows) {
    const key = keyOf(row);
    const existing = latest.get(key);
    if (!existing || new Date(dateOf(row)) > new Date(dateOf(existing))) {
      latest.set(key, row);
    }
  }
  return latest;
}
