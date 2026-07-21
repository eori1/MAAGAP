# Progress Log

Related: [[02-Decisions-Log]] · [[06-Current-State-and-Next-Steps]]

Chronological, commit-by-commit record of what happened on `fix/inspector-assignment-alignment`. Newest at bottom. This is the "what" — see [[02-Decisions-Log]] for the "why" behind the non-obvious calls.

## Session 1 — Initial gap assessment

Read the manuscript (`UG-CICT-THESIS-MANUSCRIPT.md`) end to end and audited the backend/frontend against it. Findings (full detail was reported to the user, condensed here):

- Objective 4's optimizer solved a budget-knapsack ("which projects get funded"), not the manuscript's inspector-assignment problem.
- Monte Carlo baseline stats were hardcoded constants, not computed.
- Cost-overrun probability was the delay model's output relabeled — no real cost-overrun model existed. No MAE regression existed at all (Objective 2 requirement).
- Frontend: Allocation, Timeline, Reports, Users pages were 100% hardcoded mock data; only Dashboard/Projects/Forecast Engine read anything real (a static demo CSV).
- Manuscript internally inconsistent: 3-tier vs 4-tier risk scoring, duplicate Objective 4/5, Streamlit/Flask described vs Next.js actually built.

## Commit `5ddfdf4` — fix: align Objective 4 with manuscript

- `maagap/optimization.py`: added `InspectorAssignmentOptimizer` (real LP assignment problem, capacity derived from vehicle access/availability/workload, guaranteed Critical coverage). Kept old `ResourceOptimizer` for reference.
- `maagap/models.py`: added `RegressionModelTrainer` (XGBoost regressors for delay days / cost overrun pct) and wired a dedicated cost-overrun classifier into `main.py`.
- `main.py`: computes manual round-robin baseline + real Monte Carlo random baseline (no hardcoded stats); exports `tbl_assignments.csv` and a frontend `assignments.json`.
- Frontend: `allocation/page.tsx` rewritten to consume the real assignment schedule instead of 5 fictional hardcoded inspectors.
- Manuscript: reconciled to 4-tier risk scoring throughout, deduped Objective 4/5, Next.js stack references replacing Streamlit/Flask mentions.
- Added `tests/test_inspector_assignment.py` (5 new tests). Total 25/25 passing.

## Commit `1c796c6` — feat: wire Timeline, Reports, Users to real data; add alerts

- `main.py`: new `_export_frontend_extras()` producing `timeline.json`, `reports.json`, `inspectors.json`, `alerts.json` (tier-escalation diff against the previous run's `tbl_predictions.csv` snapshot, plus standing Critical-tier alerts).
- Frontend: Timeline (Gantt-style, elapsed-months-since-start axis rather than calendar, since projects span 2016-2025 non-uniformly), Reports (real quarterly inspection logs + a print-based PDF export), Users (real inspector roster) all rewritten off real data.
- `TopRight.tsx` notification bell wired to real alerts with a live badge + dropdown.

## Commit `ecf666a` — feat: Supabase as source of truth (replacing CSV/JSON)

User chose Supabase over SQLite (see [[02-Decisions-Log]]).

- `backend/supabase/schema.sql`: 8-table ERD-aligned schema, RLS on, no public policies.
- `backend/maagap/database.py`: `supabase-py` wrapper, `sync_all`/`sync_table`, FK-safe delete-then-insert, URL normalization.
- `main.py`: `_sync_to_supabase()` — snapshots previous risk tiers before overwrite (for alert diffing), pushes all 8 tables.
- Frontend: `/api/assignments`, `/api/timeline`, `/api/reports`, `/api/inspectors`, `/api/alerts` rewritten to query Supabase directly instead of reading the JSON files (those are still written as a secondary artifact, not read anymore).
- Bugs found/fixed during this work: URL normalization (`/rest/v1` suffix), a schema/DataFrame column mismatch, the pytest/main.py concurrent-write race condition, PostgREST's silent row cap on `.range()`. All documented in [[04-Workflows-and-Gotchas]].

## Commit `637bc00` — feat: Supabase Auth with RBAC

- `backend/supabase/schema_auth.sql`: `profiles` table, roles, auto-provisioning trigger.
- `@supabase/ssr` wired into Next.js: browser client, session-server client (`getSessionProfile()`), `src/proxy.ts` (discovered Next.js 16 renamed Middleware → Proxy mid-implementation by checking `node_modules/next/dist/docs`).
- Discovered a pre-existing, unwired mock login page at the site root; wired real auth into it instead of keeping a separately-built `/login` page (deleted the duplicate). See [[02-Decisions-Log]].
- Role-based data scoping added to 5 API routes (Inspector → own `assignments.inspector_id` only).
- Admin user-management panel (Account Access) + `/api/admin/users` routes (list/create/PATCH role).
- Bootstrapped first admin account (`kirkgamo@gmail.com`), created test manager + test inspector accounts, all three roles manually verified end-to-end in-browser.
- Bug found via user's own testing: `/api/reports` displayed the wrong inspector name (`inspection_logs.inspector_id`, a synthetic placeholder, instead of `assignments.inspector_id`, the real assignment) — fixed. See [[04-Workflows-and-Gotchas]].
- Second instance of the test-pollution bug, this time in model checkpoints (`test_models.py` default filenames matched real ones) — fixed with a `models_dir` override, mirroring the earlier CSV fix.

## Commit `100c360` — feat: consolidate /api/projects onto Supabase

- `/api/projects` (Dashboard, Projects, Forecast Engine) migrated off the static `demo_projects.csv` (150 rows) onto the same Supabase 450-project cohort the rest of the app uses, with the same Inspector-role scoping.
- `iloiloMunicipalityCentroids.ts`: map pins via deterministic centroid+jitter (synthetic data has no lat/lng). Centroids extracted by averaging the old demo CSV's own coordinates.
- Extracted the PostgREST-pagination workaround into a shared `fetchAllRowsIn()` helper (`src/lib/supabasePaging.ts`), used by both `/api/reports` and the new `/api/projects`.
- Removed the now-unused `papaparse`/`@types/papaparse` dependencies.
- Verified end-to-end in-browser across all three roles.

## Knowledge-base vault created

This vault (`/knowledge-base`) was set up at this point, git-tracked, to be updated automatically after future significant work. See [[06-Current-State-and-Next-Steps]] for what's next.
