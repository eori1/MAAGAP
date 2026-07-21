# Decisions Log

Related: [[00-Overview]] · [[01-Architecture]] · [[03-Progress-Log]]

Chronological record of non-obvious choices and why they were made. Each entry: what was decided, why, what alternatives were considered.

## Inspector-assignment LP over budget-knapsack (kept both)

**Decision**: Replace the original `ResourceOptimizer` (budget-knapsack: "which projects get funded under a budget cap") with a new `InspectorAssignmentOptimizer` (assignment problem: "which inspector visits which project") as the one `main.py` actually uses.

**Why**: The manuscript and the PPDO stakeholders' actual problem is inspector routing/scheduling under a 5-6-person roster, not project funding selection. The original code solved a different, easier problem that didn't match the thesis's stated Objective 4 or the DFD Level 2 (Optimization Engine) diagram.

**Alternative considered**: Delete `ResourceOptimizer` entirely. **Rejected** — kept it in `optimization.py` for comparison/historical reference since it doesn't hurt to keep, and removing working code the user didn't ask to remove is unnecessary churn.

## Manual round-robin + Monte Carlo random, not hardcoded stats, as baselines

**Decision**: Benchmark the LP against (a) a manual round-robin baseline (mirrors current PPDO practice — visit projects in list order) and (b) a 100-iteration Monte Carlo random-assignment baseline, both computed for real every run.

**Why**: The original code had `mc_results = {"n_successful": 100, "std_improvement_pct": 3.2, ...}` hardcoded as constants, misrepresenting a specific run's numbers as if they were general properties of the algorithm. This would not survive a thesis defense.

## Supabase over SQLite/better-sqlite3

**Decision**: User explicitly chose Supabase (hosted Postgres) when asked how to implement the "replace CSV/JSON with a database" objective, overriding the initially-proposed local SQLite + `better-sqlite3` approach.

**Why** (inferred from the ask): matches the manuscript's own "PostgreSQL available for production deployment" line, gives free hosted Postgres + Auth + RLS in one product, avoids native-module install friction (`better-sqlite3` requires prebuilt binaries).

**Consequence**: I cannot run schema DDL directly (no direct Postgres connection, only the service-role REST key) — every schema change requires the user to paste SQL into the Supabase dashboard manually. See [[04-Workflows-and-Gotchas#Schema changes require a manual step]].

## supabase-py for the backend write path (not raw psycopg2)

**Decision**: User chose the `supabase-py` client library over a direct `psycopg2`/SQLAlchemy Postgres connection for the pipeline's write path.

**Why**: Matches Supabase's officially recommended pattern; the backend already writes via a service-role REST client rather than needing raw SQL/connection-string management.

## Login page: reuse the existing mock UI, delete the newly-built duplicate

**Decision**: Discovered mid-implementation that a polished login/landing page already existed at the site root (`src/app/page.tsx`) — hardcoded fake credentials (`admin@iloilo.gov.ph`), no real auth, just `router.push("/dashboard")` on submit. I had already built a separate, plainer `/login` page wired to real Supabase Auth. Once the mismatch surfaced (user tried logging in via the root page, got confused by a proxy redirect to the different-looking `/login`), the decision was: wire real Supabase Auth into the **existing, nicer** root page, and delete the newly-built `/login` page rather than keeping two login screens.

**Why**: One login screen, using the design that was already built and presumably intentional (branding, copy, footer disclaimer already written). `src/proxy.ts`'s public path became `/` instead of `/login`.

## Role visibility: Manager sees User Management too, not Admin-only

**Decision**: Both Manager and Admin can view the Users page / Account Access panel; only Inspector is excluded from nav. Only Admin can create accounts or change roles (Manager can view the account list but not mutate it).

**Why**: User's own phrasing when specifying role scope was "Manager sees everything (dashboard, allocation, reports, user management)" — explicitly listing user management as part of Manager's full access, with Admin's distinguishing responsibility being *managing credentials* specifically (create/role-change), not exclusive page access. Confirmed via an explicit follow-up question when the ambiguity was noticed.

## Dashboard/Projects/Forecast Engine scoped to the same 450-project cohort as everything else

**Decision**: When migrating `/api/projects` off the static `demo_projects.csv` (150 hand-picked rows), scope it to the same ~450-project "currently monitored" cohort that `predictions`/`assignments` already cover — not all 3000 rows in the `projects` table.

**Why**: Consistency. Timeline/Reports/Allocation already only show this cohort (the projects that have been scored by the most recent pipeline run). Showing a different, larger, unscored set on Dashboard/Projects would mean some pages have risk tiers for a project and others don't — confusing and inconsistent. It also means Inspector-role scoping (via the `assignments` table) works identically across every page with no special-casing.

## Map pins via deterministic centroid jitter, not real per-project coordinates

**Decision**: Since the synthetic `projects` table only stores a municipality name (not coordinates), and the original `demo_projects.csv` turned out to already be using municipality-centroid + jitter (verified by averaging its per-project lat/lng per municipality), replicate that same approach: a static centroid lookup (`iloiloMunicipalityCentroids.ts`) + a deterministic hash-based jitter keyed on `project_id`, so each project always renders at the same pin across requests.

**Why**: Cheapest way to keep the map feature working without adding real geocoding, backend schema changes, or a new data source, and it matches the fidelity level the original demo data already had (it was never real per-project GPS data either).

## Obsidian knowledge-base vault, git-tracked, auto-updated

**Decision**: Build `/knowledge-base` in the repo root as a git-tracked Obsidian vault, and update it automatically after significant work rather than waiting to be asked.

**Why**: User's stated goal is surviving `/compact` and fresh sessions without re-deriving context or hallucinating prior decisions. Git-tracked means it travels with clones/branches and stays versioned alongside the code it describes.
