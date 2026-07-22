import { PROJECT_TYPES, IMPLEMENTING_AGENCIES, FUNDING_SOURCES, MUNICIPALITIES } from "@/lib/ppaOptions";

export interface PpaInput {
  projectName: string;
  description: string;
  projectType: string;
  category: string;
  location: string;
  budgetAllocated: number;
  startDate: string;
  fundingSource: string;
}

export interface PpaValidationResult {
  valid: boolean;
  error?: string;
  normalized?: PpaInput;
}

function findCaseInsensitive(list: readonly string[], value: string | undefined): string | null {
  if (!value) return null;
  const match = list.find((v) => v.toLowerCase() === value.trim().toLowerCase());
  return match ?? null;
}

// Shared by POST /api/projects (single add) and POST /api/projects/bulk
// (CSV import) -- one set of rules, not duplicated in two places. Pure
// (no server-only imports), so it's also safe to run client-side for the
// bulk-import preview table before anything is submitted.
export function validatePpaRow(input: Partial<Record<keyof PpaInput, string | number | undefined>>): PpaValidationResult {
  const projectName = String(input.projectName ?? "").trim();
  if (!projectName) return { valid: false, error: "Project name is required" };

  const typeConfig = PROJECT_TYPES.find((t) => t.value.toLowerCase() === String(input.projectType ?? "").trim().toLowerCase());
  if (!typeConfig) return { valid: false, error: `Project type must be one of: ${PROJECT_TYPES.map((t) => t.value).join(", ")}` };

  const category = findCaseInsensitive(IMPLEMENTING_AGENCIES, input.category as string);
  if (!category) return { valid: false, error: "Implementing agency not recognized" };

  const location = findCaseInsensitive(MUNICIPALITIES, input.location as string);
  if (!location) return { valid: false, error: "Municipality not recognized" };

  const fundingSource = findCaseInsensitive(FUNDING_SOURCES, input.fundingSource as string);
  if (!fundingSource) return { valid: false, error: "Funding source not recognized" };

  const budgetAllocated = Number(input.budgetAllocated);
  if (!budgetAllocated || Number.isNaN(budgetAllocated) || budgetAllocated <= 0) {
    return { valid: false, error: "Budget must be a number greater than 0" };
  }

  const startDate = String(input.startDate ?? "").trim();
  const parsedDate = new Date(startDate);
  if (!startDate || Number.isNaN(parsedDate.getTime())) {
    return { valid: false, error: "Start date is invalid (expected YYYY-MM-DD)" };
  }

  return {
    valid: true,
    normalized: {
      projectName,
      description: String(input.description ?? "").trim(),
      projectType: typeConfig.value,
      category,
      location,
      budgetAllocated,
      startDate,
      fundingSource,
    },
  };
}
