# Architecture

Related: [[00-Overview]] · [[04-Workflows-and-Gotchas]] · [[05-Manuscript-Alignment]]

## Backend pipeline

Entry point: `backend/main.py` → `MainPipeline.run()`. Run with `python main.py` from `backend/`, using the venv at `backend/.venv` (Python 3.12 — created because no system Python had the ML deps; see [[04-Workflows-and-Gotchas]]).

Sequence (all in one `run()` call):

1. **Data pipeline** — `DataPreprocessor` cleans the real PPDO Excel workbook (`data/raw/`), extracts statistical distributions, then `SyntheticDataGenerator.generate_synthetic_dataset()` generates 3000 synthetic projects + quarterly monitoring records seeded from those distributions (deterministic, `SEED=42`). Writes `synthetic_projects.csv`, `synthetic_quarterly.csv`, and the ERD tables (`tbl_contractor.csv`, `tbl_inspector.csv`, `tbl_project.csv`, `tbl_inspection_log.csv`, `tbl_external_context.csv`) to `data/processed/`.
2. **Objective 1 — predictive framework** — `TreeModelTrainer` (RF, XGBoost), `LSTMTrainer`, `MetaEnsembleTrainer` (stacking meta-ensemble) in `maagap/models.py`. Trained on a 70/15/15 split via `feature_engineering.split_data`.
3. **Cost overrun + MAE regression** (added in the alignment fixes, not original scaffolding) — dedicated XGBoost classifier for `is_cost_overrun`, plus `RegressionModelTrainer` (XGBoost regressors) for `delay_days` and `cost_overrun_pct`, satisfying the manuscript's MAE requirement.
4. **Objective 3 — risk scoring** — `maagap/risk_scoring.py`. Weighted composite of RF/XGB/LSTM delay probabilities → 4-tier score (Low/Medium/High/Critical, thresholds 0.3/0.7/0.9). `check_logic_consistency()` verifies every tier assignment matches its threshold.
5. **Objective 4 — inspector deployment optimization** — `maagap/optimization.py`. Two classes:
   - `InspectorAssignmentOptimizer` — **the real one**, used in production. Integer LP (PuLP/CBC): assigns inspectors to projects to maximize captured risk utility, subject to per-inspector visit capacity (derived from vehicle access + availability + current workload) and guaranteed Critical-tier coverage. Benchmarked against a manual round-robin baseline and a Monte Carlo random-assignment baseline (both computed for real, not hardcoded).
   - `ResourceOptimizer` — the original budget-knapsack formulation (picks *which projects* to fund under a budget cap). Kept for comparison/history; not used by `main.py` anymore.
6. **Explainability** — `maagap/explainability.py`, SHAP attributions per prediction.
7. **Exports** — CSVs to `data/processed/` (`tbl_predictions.csv`, `tbl_assignments.csv`), JSON files to `frontend/public/data/` (legacy path, superseded by Supabase — see below), and finally **`_sync_to_supabase()`** which pushes everything to Postgres.

## Supabase schema

Two SQL files in `backend/supabase/`, both must be run once via the Supabase SQL Editor (paste-and-run — no DB password needed, only the dashboard):

- **`schema.sql`** — core ERD tables: `contractors`, `inspectors`, `projects`, `inspection_logs`, `external_context`, `predictions`, `assignments`, plus `risk_alerts` (persisted so tier-escalations can be diffed across pipeline runs). RLS enabled, no public policies — only the service-role key can read/write until `schema_auth.sql` adds real users.
- **`schema_auth.sql`** — `profiles` table (`id` → `auth.users.id`, `role` ∈ {manager, inspector, admin}, `inspector_id` → `inspectors.inspector_id`). A Postgres trigger (`handle_new_user`) auto-creates a profile (default role `inspector`) whenever a new Supabase Auth user signs up.

`backend/maagap/database.py` wraps `supabase-py`: `get_client()` (reads `backend/.env`, normalizes the URL), `sync_all()` (FK-safe delete-then-insert across all 7 core tables in one call), `sync_table()` (single table, used for `risk_alerts` since its diff logic needs the pre-sync snapshot fetched first).

**Scope note**: `predictions`/`assignments`/`risk_alerts` only ever cover the ~450-project *test-set cohort* from the most recent pipeline run (the "currently monitored" projects), not all 3000 rows in `projects`. Every frontend route that shows project-level data (Timeline, Reports, Allocation, Dashboard, Projects, Forecast Engine) is scoped to this same 450-project cohort for consistency — see [[03-Progress-Log]] entry on the `/api/projects` migration.

## Frontend

Next.js 16 App Router (`frontend/src/app/`). **Next.js 16 renamed Middleware to Proxy** — see [[04-Workflows-and-Gotchas]], this matters if you're used to older Next.js conventions.

### Pages
Dashboard, Projects (table + Leaflet map view), Forecast Engine, Allocation, Reports, Project Timeline, Users. All are client components (`"use client"`) that fetch from same-origin `/api/*` routes.

### Auth
- Login screen lives at the site **root** (`/`, `src/app/page.tsx`) — NOT a separate `/login` route (one was built then deleted; see [[02-Decisions-Log]]).
- `src/proxy.ts` — refreshes the Supabase session cookie every request, redirects unauthenticated requests to `/`, redirects authenticated requests away from `/` to `/dashboard`. Excludes `/api/*` (those do their own 401 JSON, not HTML redirects).
- `src/lib/supabaseBrowserClient.ts` — anon-key client for client components (login form, logout button).
- `src/lib/supabaseSessionServer.ts` — cookie-aware anon-key client + `getSessionProfile()` (resolves `{userId, email, role, inspectorId, fullName}` from the session), used in every API route for auth + role checks.
- `src/lib/supabaseServer.ts` — service-role client (bypasses RLS), used for all actual data queries once a route has verified the caller via `getSessionProfile()`.
- `src/lib/supabaseUrl.ts` — `normalizeSupabaseUrl()`, strips an accidentally-included `/rest/v1` (or `/auth/v1`) suffix. Every Supabase client construction site uses this. See [[04-Workflows-and-Gotchas]].

### API routes (`src/app/api/*/route.ts`)
All query Supabase directly (no more file-based JSON reads — that was the pre-Supabase design, since removed as the source of truth though `_export_frontend_*` in `main.py` still writes the JSON files as a secondary artifact).

| Route | Scoping |
|---|---|
| `/api/me` | returns the caller's session profile |
| `/api/projects` | Dashboard/Projects/Forecast Engine data. Inspector → own assigned projects only |
| `/api/assignments` | Allocation page. Inspector → own record only |
| `/api/timeline` | Timeline page. Inspector → own assigned projects |
| `/api/reports` | Reports page. Inspector → own assigned projects. **Inspector name comes from `assignments.inspector_id`, not `inspection_logs.inspector_id`** — see [[04-Workflows-and-Gotchas]] |
| `/api/inspectors` | Users page roster. Inspector → own record only |
| `/api/alerts` | Notification bell. Inspector → alerts for own projects only |
| `/api/admin/users` (GET/POST) | Account list (Manager+Admin can view) / create account (Admin only) |
| `/api/admin/users/[id]` (PATCH) | Change a user's role (Admin only) |

Shared helpers: `src/lib/supabasePaging.ts` (`fetchAllRowsIn` — see [[04-Workflows-and-Gotchas]] on the PostgREST row cap), `src/lib/iloiloMunicipalityCentroids.ts` (map pin coordinates — synthetic data has no lat/lng, only municipality names, so pins are a deterministic jitter around each municipality's centroid).

### Roles
- **admin** — everything, plus create accounts / change roles
- **manager** — everything except account creation/role changes (can view the account list)
- **inspector** — scoped to their own `inspector_id`'s assigned projects everywhere; "User Management" hidden from nav (`src/components/Sidebar.tsx` filters `excludeInspector` items)

Test accounts exist for all three roles — see [[04-Workflows-and-Gotchas#Test accounts]].
