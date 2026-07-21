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

## Commit `6b422a9` — docs: knowledge-base vault created

This vault (`/knowledge-base`) was set up at this point, git-tracked, to be updated automatically after future significant work. Root `CLAUDE.md` added pointing Claude Code at it.

## Commit `ea3ff1a` — feat: password self-service

- Login page's pre-existing "Forgot Password?" button (previously a no-op placeholder) wired to `supabase.auth.resetPasswordForEmail()`.
- New `/reset-password` page: listens for the `PASSWORD_RECOVERY` auth event (also checks `getSession()` directly, in case the event fired before the listener attached — a real timing race, not a hypothetical one) and lets the user set a new password. Added to `proxy.ts`'s public paths, since the first request after clicking the emailed link has no session cookie yet — the client-side code-exchange happens after the page loads.
- New `/account` page: shows the logged-in user's email/name/role and a self-service change-password form (`supabase.auth.updateUser({password})`, no re-authentication required — Supabase's default). Reachable from all three roles via the profile icon in `TopRight.tsx`, which was previously an inert placeholder button.
- Verified end-to-end by the user: both change-password (logged in) and forgot-password (real email received, link clicked, password reset, redirected to dashboard) confirmed working.

## Full codebase audit: PPDO workflow alignment + frontend UI/UX

Requested by the user: audit the whole codebase against the described PPDO operational workflow (Admin/Manager tracks backlogs, assigns projects, Inspector accepts and submits reports) and general frontend UI/UX. Verified every finding against actual code (API routes, schema, page source), not memory. Key findings:
- The entire frontend was read-only against Supabase except `/api/admin/users` — no accept/decline persistence, no report-submission path, decorative buttons ("Manual Edit", "Add new PPA", "Import Data") with no handlers.
- Dashboard still had leftover mock data (hardcoded greeting/date/trend badges/alerts) missed during the earlier Supabase migration.
- No dedicated "backlog" view/metric.
- No responsive/mobile support anywhere in the app.
- No loading states (pages render empty until fetch resolves).

Full findings reported to the user; priority order agreed: (1) Dashboard mock-data cleanup, (2) scope and build the accept/submit-report workflow, (3) mobile responsiveness (not started as of this entry) — see [[06-Current-State-and-Next-Steps]].

## Commit `06ef5cd` — fix: Dashboard mock-data cleanup

Real name/time-of-day greeting (via `/api/me`), real date, fake "+5.3%" badges removed (no historical data existed to compute a real trend from), "AI Forecast Alerts" wired to real `/api/alerts`.

## Commit `00836f6` — feat: operational assign→accept→submit-report workflow

The largest finding from the audit. See [[02-Decisions-Log]] for the three scoping decisions (photo upload, accept-only, report gated on acceptance).

- `backend/supabase/schema_workflow.sql` (additive): `assignments.status`/`accepted_at`; new `inspection_reports` table; public `inspection-photos` Storage bucket + policies.
- `PATCH /api/assignments/[id]/accept`, `POST /api/reports/submit` — both enforce the caller is the assigned inspector; submit also enforces `status === 'accepted'`.
- `GET /api/assignments` extended with `status`/`hasReport` per project.
- Allocation page redesigned: per-project-row Accept/Submit Report/Reported controls (Inspector, actionable) vs. read-only Pending/Accepted/Reported badges (Manager/Admin) — replacing the old non-functional per-card button.
- New `SubmitReportModal` component: accomplishment %, issues, notes, multi-photo upload directly to Storage from the browser client.
- Verified end-to-end by the user as both Inspector (accept → submit with photo → "Reported" badge) and Manager (same state, read-only). Report row confirmed correctly persisted in Supabase.

## Commit `89c5de6` — feat: merge real submitted reports into the Reports page

Next audit item tackled after the workflow feature. `/api/reports` now joins `inspection_reports` alongside `inspection_logs`/`projects`; a project's row prefers its latest real submission when one exists. New "Source" column/filter ("Field Report" vs "Pipeline Estimate"); issues/notes/photos surface for real reports.

Bug caught by the user testing the very first real submission: `physical_accomplishment_pct` is an optional field on the submit form, and treating a missing value as `0` (rather than "not reported") produced a nonsensical "-100.0 pts / Flagged" result. Fixed by keeping `null` as `null` throughout the response and rendering: progress shows "—", slippage shows "Not reported", and a new neutral "Submitted" status badge replaces the false "Flagged" when nothing was actually measured.

User then asked (exploratory, not yet actioned) whether reports should be clickable for Manager approve/comment. Agreed to finish already-scoped items (mobile responsiveness, loading states) first — see [[06-Current-State-and-Next-Steps]].

## Commit `e11ee9b` — feat: mobile responsiveness baseline

Sidebar off-canvas drawer + hamburger, `.main` margin collapse on all 8 authenticated pages, stat-row/two-column-layout stacking, Gantt horizontal scroll, table `overflow-x: auto`.

First pass didn't actually fix table scrolling: `.table { width: 100% }` forced tables to compress to fit rather than overflow, so `overflow-x: auto` on the parent had nothing to scroll — columns were squishing/truncating instead. Fixed with a mobile-only `min-width` on `.table`. Also fixed the Projects page's action-button row overflowing off-screen (`.pageHeader`/`.actionBtns` needed `flex-wrap`).

**User tested again and found table/chart rendering still not fully satisfactory on mobile.** Decision: stop iterating on the current design's mobile behavior and defer to a planned full UI revamp later — see [[02-Decisions-Log]] and [[06-Current-State-and-Next-Steps]]. User then redirected focus to backend/core-workflow items instead of further frontend polish.

## Commit `86910b9` — docs: add PRD/system specification document

Created `07-PRD.md`: 15 functional requirements (FR-1 to FR-15), each tagged Built/Partial/Planned, plus stakeholders/roles, non-functional requirements, out-of-scope items, data model reference, and open questions. Requested by the user before scoping further backend work, "so that we will not be confused or get lost on the development." Cross-linked from `00-Overview.md`, `06-Current-State-and-Next-Steps.md`, and root `CLAUDE.md`.

## PR #1 — merged into `main` (2026-07-21)

All 16 commits from `5ddfdf4` through `86910b9` (the entire Supabase migration, Auth/RBAC, password self-service, operational workflow, Reports-page merge, mobile-responsiveness baseline, and the knowledge-base vault + PRD) were opened as a single PR and merged into `main` via **squash-and-merge** as one commit, `7932d3e`. See [[02-Decisions-Log]] for why the branch was renamed before opening the PR and why squash (not a real merge commit) was chosen.

`main` is now up to date with all of this work; the feature branch `feat/supabase-migration-and-operational-workflow` still exists on `origin` post-merge (not deleted).

## FR-13 — report review/approval workflow

Built directly after PR #1, per the PRD's flagged "Planned, not scoped" items. See [[02-Decisions-Log]] for the scoping decisions and [[04-Workflows-and-Gotchas]] for the `risk_alerts`-overwrite gotcha this surfaced.

- `backend/supabase/schema_review.sql` (additive): `inspection_reports.review_status` (`pending`/`approved`/`needs_revision`, default `pending`), `review_comment`, `reviewed_by`, `reviewed_at`.
- New `frontend/src/lib/inspectionReports.ts::pickLatestByKey()` — shared "latest report wins" dedup, now used by `/api/reports`, `/api/assignments`, and `/api/alerts` (previously each route had its own copy of this logic, or in `/api/assignments`' case, didn't need it until now).
- New `PATCH /api/reports/[reportId]/review` — Manager/Admin only; `approve` (no comment) or `request_revision` (comment required, 400 otherwise).
- `/api/reports` GET: exposes `reportId`/`reviewStatus`/`reviewComment` per row (`null` for pipeline-estimate rows — nothing to review).
- `/api/assignments` GET: exposes `reviewStatus` per project, keyed off the latest report per assignment.
- `/api/alerts` GET: derives a `REPORT_NEEDS_REVISION` alert on read for an Inspector's own `needs_revision` reports — not stored in `risk_alerts` (see gotcha).
- Allocation page: a `needs_revision` report now shows **Resubmit Report** (Inspector) or a **Needs Revision** badge (Manager/Admin) instead of the old terminal "✓ Reported" — resubmission reuses the existing `SubmitReportModal` unchanged.
- Reports page: new **Review Status** column (Awaiting Review/Approved/Needs Revision, kept visually distinct from the existing progress-based Status column).
- `TopRight.tsx` notification bell: new alert type with a distinct dot color, alongside the existing tier-escalation/critical-risk alerts.
- Committed to `feat/report-review-workflow` (`313976d`), pushed, PR #2 opened against `main`. Verified: `npx tsc --noEmit` clean, `pytest` still 25/25 (no Python touched), ESLint clean (two pre-existing `<img>` warnings unrelated to this change).

### Follow-up (same PR, commit `47b3c44`) — full report detail modal

User caught a real gap after the first pass: the Reports table row only has room for 32×32 photo thumbnails and truncated notes — not enough for a Manager/Admin to actually judge a submission before approving or requesting revision. Fixed:

- Added `financialAccomplishmentPct` to the `/api/reports` response (the field was already being fetched from `inspection_reports` but had never been surfaced to the frontend at all — a pre-existing gap, not something FR-13 introduced).
- New `ReportDetailModal` (+ `.module.css`) replaces the removed `ReportReviewModal`: full-size photos, complete issues/notes text, physical **and** financial accomplishment % side-by-side, and Approve/Request Revision live inline right below the evidence — folding the revision-comment flow into the same read-first surface instead of stacking two separate modals.
- Reports page: the Review Status column now shows one **Review Report** (Manager/Admin, pending) / **View Report** (everyone else, or already decided) button that opens the modal, instead of separate inline Approve/Request Revision buttons disconnected from the report content.
- `tsc`/lint re-verified clean after this change.
