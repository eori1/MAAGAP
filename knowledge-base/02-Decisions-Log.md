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

## Operational workflow: photo upload, accept-only (no decline), report gated on acceptance

**Decision**: When scoping the assign→accept→submit-report feature (surfaced by a full codebase audit against the PPDO's stated workflow), three explicit choices were made via direct questions rather than assumed:
1. Reports include photo upload (not just data fields) — requires a Supabase Storage bucket, an extra manual setup step for the user.
2. Inspectors can only **accept** an assignment, not decline it. If an inspector genuinely can't do a visit, that's handled outside the app (contact the Manager) rather than a formal decline/reassignment flow. Simpler, and avoids needing to model "assignment goes back to the pool" without re-running the offline LP.
3. Submitting a report **requires** the assignment to already be `accepted` (enforced server-side, 409 if not) — matches the real-world sequence (accept the visit → go inspect → submit) rather than allowing a report for any assigned project regardless of acceptance state.

**Why**: These aren't obvious defaults — a simpler v1 could've skipped photos, allowed decline, or left reports ungated — so the choices are recorded here to avoid re-litigating or "simplifying" them away in a future session.

**Consequence**: `inspection_reports.photo_urls` is a `text[]` column, uploaded to a public `inspection-photos` Storage bucket directly from the browser client (using the user's own session, not the service-role key — Storage RLS policies, not table RLS, gate this). `assignments.status` has only two values (`pending`, `accepted`), no `declined`. `POST /api/reports/submit` returns 409 if the assignment isn't accepted yet.

## Old per-inspector-card "Accept AI Allocation" button replaced with per-project-row controls

**Decision**: The original Allocation page (inherited from the pre-Supabase mockup) had one "Accept AI Allocation" button per inspector card — accepting an entire day's schedule as one blob, and not wired to anything (local component state only, reset on refresh). Redesigned to per-project-row Accept/Submit Report/Reported controls instead.

**Why**: A real PPDO inspector doesn't accept or decline their *entire schedule* as a unit — acceptance is naturally per-assignment (per-project). The audit that surfaced this also found the button did nothing persistent, so it was actively misleading about state, not just wrong-grained.

## Defer further mobile/table UI polish to a future full UI revamp

**Decision**: After a baseline mobile-responsiveness pass (sidebar drawer, table horizontal scroll, layout stacking — commit `e11ee9b`), the user tested again and found table/chart rendering on mobile still not fully satisfactory. Rather than keep iterating incrementally on the *current* design's responsive behavior, the user decided to defer further polish to a planned full UI revamp, and redirected focus to backend/core-workflow items.

**Why**: The current page designs (wide multi-column tables, dense stat rows, Gantt charts) were built mockup-first without mobile in mind; incremental CSS patches (horizontal scroll, stacking, wrapping) improve things but don't fundamentally solve the mismatch between "desktop dashboard density" and "phone screen." A future revamp can redesign these views mobile-first (e.g. card-based lists instead of tables) rather than continuing to retrofit.

**Consequence**: Don't re-attempt incremental mobile CSS fixes on the current page designs unless explicitly asked — the baseline (`e11ee9b`) is considered "good enough for now," and the user wants a redesign, not further patching, when this comes up again. Loading states (a related, smaller polish item) were also left unstarted for the same reason — likely folds into the same future revamp.

## Branch renamed before opening the PR

**Decision**: Renamed `fix/inspector-assignment-alignment` → `feat/supabase-migration-and-operational-workflow` before pushing/opening the PR.

**Why**: The branch name was accurate for its first commit only. By the time it was ready to merge it carried 16 commits spanning the entire Supabase migration, Auth/RBAC, password self-service, the operational workflow, Reports-page merge, mobile responsiveness, and the PRD — "inspector assignment" would have badly undersold the PR's actual scope to a reviewer. Safe to rename locally since the branch hadn't been pushed to `origin` yet at the time.

## PR #1 merged via squash-and-merge

**Decision**: Merged PR #1 (the entire branch above, 16 commits) into `main` using GitHub's "Squash and merge," collapsing everything into a single commit (`7932d3e`) on `main`.

**Why**: The branch's 16 commits mixed real feature commits with small `docs: update knowledge-base vault` housekeeping commits after nearly every feature — squashing avoids `main`'s log being cluttered with those interstitial doc-sync commits, at the cost of losing per-feature `git bisect`/`git blame` granularity for this PR.

**Consequence**: The squash commit's message follows the template (Description + a trimmed "Changes made" bullet list) — not a copy-paste of the full PR body (no how-to-test steps, no screenshot links, no files-changed list; those either don't render in a terminal or are already available via `git log --stat`).

## FR-13 (report review/approval): scope and design

**Decision**: Chosen as the next priority after PR #1 merged, over FR-14 (ML feedback loop) and FR-15 (ISO/IEC 25010) — smallest, best-bounded scope, builds directly on the existing accept/submit workflow rather than needing more real data volume or being a non-coding deliverable. Scoped via explicit questions:
1. **Who can act**: Manager and Admin both (matches their existing shared full-read-access pattern; only Inspector is excluded).
2. **Notification**: reuse the existing notification-bell/`risk_alerts` system rather than building a new channel — but see [[04-Workflows-and-Gotchas]] for why this had to be *derived on read*, not stored in `risk_alerts` directly.
3. **Resubmission**: allowed. An Inspector can resubmit after "needs revision," which resets review to pending.
4. **Naming**: a new, separate "Review Status" badge (Awaiting Review/Approved/Needs Revision) rather than overloading the existing progress-based `status` field (Validated/Pending Review/Flagged/Submitted) — both use "Pending Review"-shaped language but mean different things, so keeping them visually and structurally distinct avoids confusing a reviewer who sees both on the same row.

**Consequence**: Resubmission needed no explicit "reset review status" logic — a resubmission is just a new `inspection_reports` row, which defaults to `review_status = 'pending'` like any other insert, and the existing "latest report wins" merge (now `pickLatestByKey()`, see [[04-Workflows-and-Gotchas]]) naturally surfaces it as the current, actionable one.

## FR-13 review actions moved from inline table buttons to a full detail modal

**Decision**: The first pass put Approve/Request Revision buttons directly in the Reports table row (`ReportReviewModal` handled only the revision comment). After shipping it, the user asked how a Manager/Admin would actually see the notes and photos before deciding — and the honest answer was "barely": the row only had space for 32×32 thumbnails and truncated/italic text, disconnected from the review action next to it. Replaced with `ReportDetailModal`: a single modal showing full-size photos, complete text, and both accomplishment percentages, with Approve/Request Revision inline below the evidence. `ReportReviewModal` was deleted rather than kept alongside it, since its one job (the revision comment) is now a mode within the bigger modal — no reason to keep two overlapping components.

**Why**: A review action needs to be next to what's being reviewed, not just next to a badge. Approve requires no comment (single click); Request Revision requires a non-empty comment, enforced both client-side (`ReportDetailModal`) and server-side (`PATCH /api/reports/[reportId]/review`).

**Consequence**: Surfaced and fixed an unrelated pre-existing gap while at it — `financial_accomplishment_pct` was already being fetched by `/api/reports` from `inspection_reports` but never included in the API response, so it was invisible to every frontend page, not just the review flow. Now exposed alongside `actualProgress` (physical %). No `Co-Authored-By` trailer on this or future commits — the user asked for it to be dropped.
