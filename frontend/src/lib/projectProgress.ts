// Shared "how far along should this project be by now" estimate, derived
// from planned duration rather than a real progress curve (the synthetic
// cohort has no milestone schedule, just a start date + planned duration).
export function expectedProgressPct(
  project: { start_date: string | null; planned_duration_months: number | null } | undefined,
  asOf: Date,
): number | null {
  if (!project || !project.start_date || !project.planned_duration_months) return null;
  const start = new Date(project.start_date);
  const elapsedMonths = (asOf.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 30);
  return Math.max(0, Math.min(100, (elapsedMonths / project.planned_duration_months) * 100));
}
