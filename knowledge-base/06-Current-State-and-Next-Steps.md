# Current State and Next Steps

**Last updated:** 2026-07-21, after PR #2 (FR-13) merged into `main`.

Related: [[03-Progress-Log]] · [[05-Manuscript-Alignment]] · [[04-Workflows-and-Gotchas]] · [[07-PRD]]

> This is the one file in the vault meant to change often. Update the "Current state" and "Next steps" sections after each significant task; leave the rest of the vault's structure alone unless something structural actually changed.

## Current state

**Merged.** All work described below landed on `main` via PR #1 (`https://github.com/eori1/MAAGAP/pull/1`), merged via **squash-and-merge** as a single commit, `7932d3e` — see [[02-Decisions-Log]]. The feature branch was renamed from `fix/inspector-assignment-alignment` to `feat/supabase-migration-and-operational-workflow` before opening the PR, since its scope had grown far beyond the original name. That branch still exists on `origin` post-merge (not deleted), but since the merge was a squash, `main`'s history is now one clean commit, not 16 — don't expect `git log` on `main` to show the individual commits below; they only exist on the feature branch. Local and remote `main` are in sync as of this update — start a new session by checking out `main` fresh, not the old feature branch, unless there's a specific reason to keep working on it.

Commits that were on the feature branch (now squashed into `7932d3e` on `main`), in order: `5ddfdf4`, `1c796c6`, `ecf666a`, `637bc00`, `100c360`, `6b422a9` (knowledge-base vault), `ea3ff1a` (password self-service), `06ef5cd` (Dashboard mock-data cleanup), `00836f6` (operational workflow), `1308faf` (vault update), `89c5de6` (Reports-page real-data merge), `f623eae` (vault update), `e11ee9b` (mobile responsiveness baseline), `c3332b0` (UI-revamp deferral decision), `86910b9` (PRD).

The system is fully functional end-to-end and manually verified:
- Full ML pipeline runs (`python main.py`, ~5-8 min) and syncs to Supabase.
- All 7 frontend pages query Supabase live, no more static/mock data anywhere (Dashboard's remaining mock leftovers — fake greeting/date/trend badges/alerts — were the last of these, now fixed).
- Supabase Auth with 3 roles (manager/inspector/admin), verified in-browser for all three.
- Password self-service: logged-in "change password" (`/account`) and "forgot password" email flow (`/reset-password`), both verified end-to-end including real email delivery.
- **Operational workflow (new)**: Inspectors can accept an assigned visit and submit a real inspection report (physical/financial accomplishment %, issues, notes, photos) directly in the app — the assign→accept→inspect→report loop from the PPDO's actual workflow now has a real implementation, not just AI-generated read-only schedules. See [[01-Architecture]] and [[02-Decisions-Log]].
- Reports page now shows real inspector-submitted reports (with a "Field Report" badge, photos, notes) in preference to the synthetic pipeline baseline, per project.
- 25/25 backend tests passing, frontend builds clean, no known type errors.

**FR-13 (report review/approval) is merged.** PR #2 (`https://github.com/eori1/MAAGAP/pull/2`, branch `feat/report-review-workflow`) squash-merged into `main` as `a20ddfa`. User verified the full walkthrough in-browser (Manager approve/request-revision → Inspector sees alert + resubmit → Manager sees it pending again) before merging. See [[03-Progress-Log]] for what was built and [[02-Decisions-Log]] for the scoping decisions, including the follow-up that replaced inline table review actions with a full `ReportDetailModal` (full-size photos, complete notes, both accomplishment percentages) and the "Pending Review"→"At Risk" rename that resolved a naming collision the user caught via screenshot. `npx tsc --noEmit` clean, `pytest` 25/25, ESLint clean throughout. The Supabase schema migration (`schema_review.sql`) has been applied. The `feat/report-review-workflow` branch still exists on `origin` post-merge (not deleted), same pattern as PR #1.

See [[05-Manuscript-Alignment]] for the objective-by-objective status.

## Immediate next steps

1. **Mobile responsiveness** — ✅ baseline done (`e11ee9b`): sidebar off-canvas drawer, tables scroll horizontally, stat rows/two-column layouts stack, Projects header wraps. **User tested and confirmed it's a real improvement but tables/charts still aren't fully resolved on mobile.** **Decision: defer further polish to a planned full UI revamp later.** Don't re-attempt incremental mobile CSS fixes on the current design unless explicitly asked.
2. **Loading states** — no page shows a spinner/skeleton while its initial fetch is in flight. Not started; likely folds into the same future UI revamp.
3. **FR-14 (ML feedback loop)** and **FR-15 (ISO/IEC 25010 evaluation)** — still "Planned, not scoped" per [[07-PRD]]; ask the user before starting either.

## Things that are known-fine, don't re-litigate

- The `ResourceOptimizer` (budget-knapsack) class still exists in `optimization.py` alongside `InspectorAssignmentOptimizer` — this is intentional, not dead code to clean up. See [[02-Decisions-Log]].
- Dashboard/Projects/Forecast Engine showing ~450 projects (not 3000) is intentional scope, not a bug.
- PDF export via `window.print()` (not a server PDF library) was a deliberate lightweight choice for a thesis prototype, not a placeholder that needs upgrading unless asked.
- The map's project pins are jittered centroids, not real GPS coordinates — this is a known, accepted approximation (the original demo data had the same limitation).
- Password self-service exists and is verified working — don't rebuild it if asked about "forgot password" or "change password" again; check [[01-Architecture]] and [[04-Workflows-and-Gotchas]] first.
- The accept/submit-report workflow exists and is verified working (both Inspector and Manager/Admin views tested in-browser) — don't rebuild it if asked about "assignment status" or "inspection reports" again; check [[01-Architecture]] and [[02-Decisions-Log]] first. The old ResourceOptimizer-era `ASSIGN-INSP-00X` fake refs are long gone; `assignments.status` and `inspection_reports` are the real thing now.
- Reports support photo upload (chosen over data-fields-only), there's no decline path (accept-only was chosen over accept/decline), and report submission is gated on acceptance (chosen over ungated) — all explicit scoping decisions made with the user, not oversights. See [[02-Decisions-Log]] before "fixing" any of them.
- The Reports page now merges real `inspection_reports` with synthetic `inspection_logs` (real takes priority per project) — don't rebuild this if asked about "why does Reports still show fake data," check first whether the project actually has a real submission yet (most of the 450-project cohort still won't, since only test accounts have submitted so far).
- FR-13 (report review/approval) exists and is verified working end-to-end (Manager/Admin approve/request-revision, Inspector notification + resubmit) — don't rebuild it if asked about "report approval" again; check [[02-Decisions-Log]] and [[03-Progress-Log]] first. Review happens via `ReportDetailModal` (full photos/notes/both accomplishment %s), not inline table buttons — that was a deliberate redesign after the first pass, not the current state to "simplify." The Reports page's older slippage-based status is "At Risk" (not "Pending Review") specifically to avoid colliding with the newer Review Status column.

## If you're picking this up cold (after `/compact` or a new session)

1. `git log --oneline -10` and `git status` — confirm you're on the right branch and nothing is uncommitted that this vault doesn't know about.
2. Read [[04-Workflows-and-Gotchas]] fully before running anything — several of those gotchas will silently corrupt data or waste a long pipeline run if ignored.
3. Check whether `backend/.env` and `frontend/.env.local` exist and have real values (they're gitignored, so a fresh clone won't have them — the user has these values, don't try to reconstruct or guess them).
4. Don't assume the Supabase schema matches `backend/supabase/*.sql` — those are the intended schema, but applying them is a manual step the user does in the Supabase dashboard. Verify with a live query before relying on a table/column existing.
