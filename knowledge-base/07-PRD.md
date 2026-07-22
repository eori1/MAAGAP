# MAAGAP — Product Requirements / System Specification Document

**Status:** Living document. Update when scope changes, a feature moves between Planned/Partial/Built, or a new requirement is agreed with the user. This is the reference to consult *before* scoping new work — if a feature discussion isn't covered here, add it here first, then build.

Related: [[00-Overview]] · [[01-Architecture]] · [[05-Manuscript-Alignment]] · [[06-Current-State-and-Next-Steps]]

---

## 1. Purpose

This document is the single authoritative reference for what MAAGAP is supposed to do — for both the thesis manuscript's stated objectives and the operational system built around them. It exists so that feature discussions ("should reports support X", "who can do Y") start from an agreed spec instead of being re-litigated ad hoc each session. See [[04-Workflows-and-Gotchas]] for implementation-level pitfalls; this document stays at the requirements level.

## 2. Product summary

MAAGAP (Machine Analytics for Allocation, Governance and Assessment of Projects) is a decision-support system for the Iloilo Provincial Planning and Development Office (PPDO). It has two purposes:

1. **Predict** which government infrastructure/non-infrastructure projects (PPAs) are at risk of delay or cost overrun, before it happens.
2. **Optimize and operationalize** the deployment of PPDO's limited field inspectors (5-6 people) to the highest-risk projects, and give those inspectors a real digital path to accept assignments and log field results — replacing what would otherwise be manual scheduling and paper/spreadsheet reporting.

Full background: [[00-Overview]].

## 3. Stakeholders and roles

Three roles, enforced via Supabase Auth + RLS-adjacent application-layer checks (see [[01-Architecture#Roles]]):

| Role | Who | Core responsibilities |
|---|---|---|
| **Admin** (System Administrator) | PPDO IT/admin staff | Manage user accounts (create, assign roles). Full data visibility. |
| **Manager** (PPDO Manager) | PPDO planning officers | Full data visibility (dashboard, allocation, reports, timeline, user list). Cannot create accounts or change roles. |
| **Inspector** (Field Inspector) | PPDO field personnel | Scoped to their own assigned projects only, everywhere. Accepts assignments, submits inspection reports with photos. Cannot see User Management. |

## 4. Functional requirements

Each requirement is tagged **Built**, **Partial**, or **Planned**. "Built" means implemented and manually verified working; "Partial" means implemented but with known gaps; "Planned" means agreed as in-scope but not yet started.

### FR-1: Predictive risk assessment — **Built**
- Ensemble ML pipeline (Random Forest, XGBoost, LSTM, stacking meta-ensemble) forecasts project delay probability from static + temporal (quarterly) features.
- Dedicated XGBoost classifier for cost-overrun probability; XGBoost regressors for delay-days and cost-overrun-% (MAE-evaluated).
- SHAP explainability: per-prediction feature attributions.
- Manuscript Objectives 1 & 2. See [[01-Architecture#Backend pipeline]], [[05-Manuscript-Alignment]].

### FR-2: Dynamic risk scoring — **Built**
- Weighted composite score → 4 tiers (Low/Medium/High/Critical), thresholds 0.3/0.7/0.9.
- Logic-consistency check confirms every tier assignment matches its threshold (0 violations required).
- Manuscript Objective 3.

### FR-3: Inspector deployment optimization — **Built**
- Integer LP (PuLP/CBC): assigns inspectors to projects to maximize captured risk utility, under per-inspector visit-capacity constraints (derived from vehicle access, availability, current workload) and guaranteed Critical-tier coverage.
- Benchmarked against a manual round-robin baseline and a Monte Carlo random-assignment baseline (both computed for real).
- Manuscript Objective 4. See [[02-Decisions-Log]] for why this replaced the original budget-knapsack formulation.

### FR-4: Data pipeline & database — **Built**
- `python main.py` orchestrates: real PPDO data ingestion → synthetic data generation (seeded, deterministic) → model training → risk scoring → LP optimization → Supabase sync.
- Supabase Postgres is the single source of truth for the frontend (`backend/supabase/schema.sql`, `schema_auth.sql`, `schema_workflow.sql`).
- **Scope delimitation**: predictions/assignments/reports only ever cover the ~450-project cohort scored by the most recent pipeline run (the "currently monitored" set), not the full 3000-row historical `projects` table. This is intentional — see [[02-Decisions-Log]].

### FR-5: Authentication & role-based access — **Built**
- Supabase Auth (email/password), 3 roles, auto-provisioning `profiles` trigger on signup.
- Every API route scopes Inspector-role requests to their own `inspector_id`'s data.
- Manuscript's AuthService / Use Case Diagram (Manager, Inspector, System Administrator actors).

### FR-6: Password self-service — **Built**
- Logged-in "change password" (`/account`, any role).
- "Forgot password" email flow (`/reset-password`), verified with real email delivery.

### FR-7: Operational workflow — assign → accept → submit report — **Built**
This is the core PPDO-facing loop and the most important functional area. Requirements, each individually confirmed working:

- **7a. View assignment.** An Inspector sees their AI-optimized visit schedule (Allocation page), scoped to their own `inspector_id`. Manager/Admin see everyone's.
- **7b. Accept assignment.** An Inspector can accept one of their own pending assignments (`assignments.status`: `pending` → `accepted`). Only the assigned Inspector may accept theirs — enforced server-side (403 otherwise). **No decline path** — an inspector who can't do a visit handles it outside the app (contact the Manager). This was an explicit scoping choice, not an oversight — see [[02-Decisions-Log]].
- **7c. Submit report.** Once accepted, an Inspector submits a report: physical accomplishment %, financial accomplishment %, issues noted (free text), notes (free text), and photo uploads (to a public Supabase Storage bucket). **Gated on acceptance** — submitting for a not-yet-accepted assignment returns 409. Both fields are optional; a missing accomplishment % must be treated as "not reported," never coerced to 0 (a real bug caught during testing — see [[04-Workflows-and-Gotchas]]).
- **7d. Manager/Admin visibility.** Manager/Admin see the same pending/accepted/reported states as read-only badges on the Allocation page — they cannot act on someone else's assignment.
- **7e. Reports page integration.** The Reports page shows one row per monitored project, preferring the latest real `inspection_reports` submission over the synthetic pipeline baseline (`inspection_logs`) when one exists. A "Source" badge distinguishes "Field Report" from "Pipeline Estimate".

### FR-8: Dashboards & analytics — **Built**
- **Dashboard**: portfolio-level stats (total/completed/ongoing/critical projects), delay trend chart, AI forecast alerts (from `risk_alerts`), recent projects table. All real data, no mock content remaining (see [[03-Progress-Log]] for the cleanup that removed the last hardcoded leftovers).
- **Projects**: table + Leaflet map view of the monitored cohort. Map pins are deterministic centroid+jitter (no real per-project GPS data exists — see [[02-Decisions-Log]]).
- **Forecast Engine**: per-project delay/cost-risk detail view with a forecast chart.
- **Project Timeline**: Gantt-style view, elapsed-months-since-start axis (not calendar, since the cohort spans non-uniform years).
- **Reports**: see FR-7e.
- All of the above are scoped to Inspector's own projects when viewed by an Inspector.

### FR-9: User/account management — **Built**
- Users page: "Account Access" panel (Manager+Admin can view; only Admin can create accounts / change roles) + "Inspector Roster" panel (LP-computed visit capacity per inspector).

### FR-10: Alerts & notifications — **Built**
- `risk_alerts` table: tier-escalation alerts (computed by diffing successive pipeline runs) + standing Critical-tier alerts.
- Notification bell (all pages) + Dashboard's "AI Forecast Alerts" card, both scoped to Inspector's own projects.

### FR-11: PDF report export — **Built (lightweight)**
- Browser `window.print()`-based export on the Reports page, styled via a `data-print-hide`/`data-print-area` CSS convention. Not a server-generated PDF library — a deliberate lightweight choice for a thesis prototype, not a placeholder.

### FR-12: Mobile responsiveness / UI revamp — **Partial, in progress**
- Baseline mobile support done: off-canvas sidebar drawer, tables scroll horizontally, stat rows/layouts stack on narrow viewports.
- User-tested and found still unsatisfactory for dense tables/charts. **Decision: defer further polish to a full UI revamp** rather than keep patching the old design — see [[02-Decisions-Log]].
- **UI revamp Phase 1 now built and approved**: a formal/enterprise design-token system, Framer Motion animation, and a fully rebuilt, triage-first Dashboard (real loading/error/empty states, a real Risk Tier Distribution chart replacing a fake hardcoded one, reordered "attention → context → browse" layout instead of the old stats-then-charts-then-table template). Reusable primitives (`Skeleton`/`Badge`/`EmptyState`/`StatCard`) and tokens exist for the remaining 8 pages + login, but their rollout (and whether each page's *layout*, not just its skin, gets rethought the same way) is not yet scoped. See [[03-Progress-Log]] and [[06-Current-State-and-Next-Steps]].

### FR-13: Report review / approval workflow — **Built**
- Manager and Admin can approve or request revision on a submitted report (`inspection_reports.review_status`: `pending`/`approved`/`needs_revision`, plus `review_comment`, `reviewed_by`, `reviewed_at`).
- "Needs revision" surfaces to the affected Inspector via the existing notification bell (derived on read from `inspection_reports`, not persisted into `risk_alerts` — see [[04-Workflows-and-Gotchas]]) and a "Needs Revision" badge/Resubmit button on the Allocation page.
- An Inspector can resubmit after "needs revision," which resets review to pending (a fresh row, no explicit reset logic needed).
- Review happens via a dedicated `ReportDetailModal` on the Reports page (full-size photos, complete notes/issues text, physical **and** financial accomplishment % side-by-side) rather than inline table buttons — the table row alone doesn't have room to actually judge a submission.
- See [[02-Decisions-Log]] for the scoping decisions and [[03-Progress-Log]] for what was built.

### FR-14: Model validation against real report data — **Built (reframed)**
- Originally scoped as an "ML feedback loop" (real report data informing retraining). Reframed after finding two blockers: real `inspection_reports` only ever capture an interim signal (physical/financial accomplishment % at one point in time), never the final delay-days/cost-overrun-% outcome the models actually train on; and the manuscript's own delimitation (§6) excludes "real-time/continuous model retraining" outright.
- Built instead as a **prediction-vs-reality tracking view**: a new "Model Validation" page comparing each project's predicted risk tier against its latest real report's progress slippage, with a Confirmed/Contradicted/Inconclusive read per project. No retraining, no model changes — a read-only evaluation angle distinct from Objective 2's existing synthetic-test-set metrics.
- Deliberately scoped to delay/progress only (not cost-overrun) and to projects with at least one real report, with no filters yet — see [[02-Decisions-Log]].

### FR-15: ISO/IEC 25010 software quality evaluation — **Planned, not scoped**
- Manuscript Objective 5. Likely a UAT/survey-design task (stakeholder or faculty evaluation instrument covering Functional Suitability, Usability, Reliability), not a coding task. Scope not yet clarified with the user.

## 5. Non-functional requirements

- **Security**: RLS enabled on every Supabase table with no public policies — all data access goes through server-side Next.js API routes using the service-role key, gated by `getSessionProfile()` role checks. The anon key (browser-exposed) is only used for Auth operations (login, password reset, session refresh) and Storage uploads (photo bucket, policy-gated to authenticated users).
- **Data scope**: system operates on the ~450-project "currently monitored" cohort per pipeline run, not the full historical registry. See FR-4.
- **Determinism**: synthetic data generation is seeded (`SEED=42`) and fully reproducible.
- **Pipeline performance**: a full `python main.py` run takes ~5-8 minutes (dominated by LSTM/XGBoost training). Not designed for real-time/on-demand re-scoring.
- **Schema changes**: require a manual step (paste-and-run SQL in the Supabase dashboard) — no programmatic DDL access. See [[04-Workflows-and-Gotchas]].
- **Browser/device support**: desktop-first; mobile is partially supported (FR-12) with further work deferred by design.

## 6. Out of scope (per manuscript delimitation)

- Modifying the existing PPDO Management Information System (MIS) — MAAGAP is a complementary decision-support tool, not a replacement.
- Highly specialized/classified infrastructure (defense, aerospace).
- Real-time/continuous model retraining within the research timeline.
- Post-occupancy facility quality or environmental sustainability scoring.
- Non-linear/heuristic optimization methods (LP via PuLP only, per delimitation).

## 7. Data model reference

See [[01-Architecture#Supabase schema]] for the full table list. Core entities: `projects`, `contractors`, `inspectors`, `inspection_logs` (synthetic), `predictions`, `assignments`, `risk_alerts`, `profiles` (auth/roles), `inspection_reports` (real, FR-7).

## 8. Open questions (resolve before building the corresponding feature)

- FR-15: is this a coding deliverable at all, or purely a research-methodology task for the manuscript's defense?
- UI revamp rollout scope (FR-12): Phase 1 (design system + Dashboard) is done; whether/how to extend it to the other 8 pages + login, and whether each page's layout gets rethought (not just re-skinned), is not yet decided — a future conversation.
