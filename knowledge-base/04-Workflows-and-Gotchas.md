# Workflows and Gotchas

Related: [[01-Architecture]] · [[06-Current-State-and-Next-Steps]]

Read this before touching the backend or Supabase. Everything here was learned the hard way once already — don't relearn it.

## Never run pytest and `main.py` concurrently

`SyntheticDataGenerator.generate_synthetic_dataset()` writes to fixed filenames in `backend/data/processed/` (`synthetic_projects.csv`, `tbl_project.csv`, etc.) by default. Test fixtures used to call this with small `n_projects` (50-80) **without overriding the output directory**, so running `pytest` while `python main.py` was also running clobbered the production 3000-row dataset mid-run with a tiny test dataset. This actually happened once and produced a `KeyError` deep in `main.py` from a project ID that existed in memory but not in the just-overwritten CSV.

**Fixed at the root**: `generate_synthetic_dataset()` now takes an `output_dir` param; `tests/test_optimization.py`, `tests/test_models.py`, `tests/test_preprocessing.py` all pass a `tmp_path_factory` directory. Same issue existed for **model checkpoints**: `TreeModelTrainer`/`LSTMTrainer`/`MetaEnsembleTrainer` now take a `models_dir` param for the same reason (test defaults `task="binary"` / `artifact_name="meta_ensemble.pkl"` happened to match real, if superseded, production model filenames).

**Still true going forward**: don't run `pytest` and `python main.py` as truly concurrent background tasks. Run one, wait for it to finish, then the other. The fixture fix prevents *this specific* pollution, but treat shared `data/processed/` and `models/` as a resource one process should own at a time.

## Supabase URL must not include a path suffix

If you (or the user) copy the Supabase project URL from somewhere that includes `/rest/v1/` or `/auth/v1/` appended, every client silently fails with `PGRST125: Invalid path specified in request URL` — because the client library appends its own path on top. This bit us **four separate times** (backend `database.py`, then three different frontend clients added later for auth).

**Fixed**: `backend/maagap/database.py::get_client()` strips it; frontend has a shared `src/lib/supabaseUrl.ts::normalizeSupabaseUrl()` used by every client construction site (`supabaseServer.ts`, `supabaseBrowserClient.ts`, `supabaseSessionServer.ts`, `proxy.ts`). **If you add a new Supabase client anywhere, route it through `normalizeSupabaseUrl()`.**

## PostgREST caps rows per request (~1000), silently

`.range(start, end)` does NOT let you fetch more than ~1000 rows in one request even if you ask for more — the server enforces its own max-rows cap regardless of the requested range. A single `.select().in('project_id', bigArray)` call over, say, 1200 `inspection_logs` rows will silently return only ~1000 and drop the rest — no error, just fewer rows than expected. This caused `/api/reports` to return 369 reports instead of 450 the first time.

**Fixed**: `frontend/src/lib/supabasePaging.ts::fetchAllRowsIn()` — loops `.range()` in pages of 1000 until an empty page comes back. **Use this helper for any query that could plausibly exceed 1000 rows** (anything joined against `inspection_logs` or `external_context`, which have ~7944 rows total).

## Next.js 16 renamed Middleware to Proxy

This repo's `frontend/AGENTS.md` warns "this is NOT the Next.js you know" for a reason. `middleware.ts` doesn't exist in this version — it's `src/proxy.ts`, exporting a function named `proxy` (or default export), same `config.matcher` convention. If you're about to write `middleware.ts`, stop and check `node_modules/next/dist/docs/01-app/01-getting-started/16-proxy.md` first.

## Password-recovery links land on a page with no session cookie yet

Clicking a Supabase "reset password" email link takes the browser to `/reset-password?code=...`. The one-time code is exchanged for a session **client-side**, by the Supabase browser JS running after the page loads — not by the server. That means:

1. `proxy.ts` must treat `/reset-password` as a public path, or it'll redirect the very first request away before the code can ever be exchanged (losing the code param).
2. The page component can't assume a session exists on mount — it has to listen for the `PASSWORD_RECOVERY` auth event (`supabase.auth.onAuthStateChange`), and *also* call `getSession()` directly as a fallback, because the event can fire before your listener finishes attaching. `src/app/reset-password/page.tsx` does both, plus a timeout that shows an "invalid/expired link" state if neither resolves within 5s.

## `/api/reports` inspector-name bug (fixed, but the underlying data ambiguity remains)

Two different columns look like "who's the inspector for this project" and mean different things:
- `inspection_logs.inspector_id` — a synthetic round-robin placeholder from data generation, meaning "who logged this quarterly report" in the fake dataset. **Not meaningful for anything real.**
- `assignments.inspector_id` — the actual LP-optimizer output, "who is assigned to inspect this project right now." **This is the authoritative one.**

`/api/reports` originally displayed the former; a Test Inspector account correctly had its *project list* filtered to its own 10 projects, but the *displayed inspector name* on each row showed other inspectors, because it was reading the wrong column. Fixed by joining `assignments` instead. **If you add any new inspector-display logic, use `assignments.inspector_id`, never `inspection_logs.inspector_id`.**

## Test accounts

Bootstrapped for manual role verification (all in the real Supabase Auth, not mocked):

| Role | Email | Password | Notes |
|---|---|---|---|
| admin | `kirkgamo@gmail.com` | (set by user, not recorded here) | first/bootstrap admin |
| manager | `test.manager@maagap.local` | `TestManager123!` | no `inspector_id` link |
| inspector | `test.inspector@maagap.local` | `TestInspector123!` | linked to `INSP-001` (Engr. Juan Dela Cruz) |

All three roles were manually verified end-to-end in-browser (login → correct nav → correct data scoping) as of the Supabase Auth + `/api/projects` migration commits.

## Environment files (never commit, always check `.gitignore` covers them)

- `backend/.env` — `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
- `frontend/.env.local` — `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `backend/.venv/` — Python 3.12 venv, created because no system Python had the ML deps (`xgboost`, `tensorflow`, `pulp`, `supabase`, etc.) installed. Use `.venv/Scripts/python.exe` on Windows.

Confirmed gitignored via `git check-ignore -v` — re-check this if `.gitignore` ever changes.

## Schema changes require a manual step

I (Claude) cannot run `CREATE TABLE`/`ALTER TABLE` against Supabase directly — the service-role key only grants REST/data access (PostgREST), not DDL. Any schema change means: write/update the `.sql` file in `backend/supabase/`, tell the user to paste-and-run it in the Supabase SQL Editor, then verify with a `SELECT` before proceeding. Don't assume a schema file matches the live database until confirmed.

## General process notes

- Full pipeline run (`python main.py`) takes ~5–8 minutes (mostly LSTM/XGBoost training). Run it in the background and wait for the actual completion notification — don't poll the output file repeatedly, and don't fabricate results while waiting.
- After any pipeline run that should reach Supabase, verify row counts per table rather than trusting the log alone (`SELECT count(*)` per table matches expectations: `projects`=3000, `predictions`=`assignments`≤450, etc.)
- IDE diagnostics shown in tool results after an `Edit` are sometimes **stale by one edit** (reflecting the state before the just-applied change) — if a diagnostic looks wrong given what you just wrote, re-run `npx tsc --noEmit` fresh rather than trusting it blindly.
