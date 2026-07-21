# Current State and Next Steps

**Last updated:** 2026-07-21, end of the Reports-page real-data merge.

Related: [[03-Progress-Log]] · [[05-Manuscript-Alignment]] · [[04-Workflows-and-Gotchas]]

> This is the one file in the vault meant to change often. Update the "Current state" and "Next steps" sections after each significant task; leave the rest of the vault's structure alone unless something structural actually changed.

## Current state

Branch `fix/inspector-assignment-alignment` (off `main`, not yet merged, not confirmed pushed to remote — check `git status`/`git log` at session start). Commits in order: `5ddfdf4`, `1c796c6`, `ecf666a`, `637bc00`, `100c360`, `6b422a9` (knowledge-base vault), `ea3ff1a` (password self-service), `06ef5cd` (Dashboard mock-data cleanup), `00836f6` (operational workflow), `1308faf` (vault update), `89c5de6` (Reports-page real-data merge).

The system is fully functional end-to-end and manually verified:
- Full ML pipeline runs (`python main.py`, ~5-8 min) and syncs to Supabase.
- All 7 frontend pages query Supabase live, no more static/mock data anywhere (Dashboard's remaining mock leftovers — fake greeting/date/trend badges/alerts — were the last of these, now fixed).
- Supabase Auth with 3 roles (manager/inspector/admin), verified in-browser for all three.
- Password self-service: logged-in "change password" (`/account`) and "forgot password" email flow (`/reset-password`), both verified end-to-end including real email delivery.
- **Operational workflow (new)**: Inspectors can accept an assigned visit and submit a real inspection report (physical/financial accomplishment %, issues, notes, photos) directly in the app — the assign→accept→inspect→report loop from the PPDO's actual workflow now has a real implementation, not just AI-generated read-only schedules. See [[01-Architecture]] and [[02-Decisions-Log]].
- Reports page now shows real inspector-submitted reports (with a "Field Report" badge, photos, notes) in preference to the synthetic pipeline baseline, per project.
- 25/25 backend tests passing, frontend builds clean, no known type errors.

See [[05-Manuscript-Alignment]] for the objective-by-objective status.

## Immediate next steps

A full codebase audit against the PPDO's described operational workflow and general frontend UI/UX turned up several items, tackled in priority order. Remaining:

1. **Mobile responsiveness** — only 2 of 12+ page CSS modules have any `@media` query, and neither is an app page. Fixed 210px sidebar, wide multi-column layouts. Matters because Inspectors are field workers who'd realistically check schedules/submit reports from a phone. Not yet started.
2. **Loading states** — no page shows a spinner/skeleton while its initial fetch is in flight; tables/lists just render empty until data arrives. Not yet started.
3. **ISO/IEC 25010 evaluation (manuscript Objective 5)** — entirely unaddressed, see [[05-Manuscript-Alignment]]. Likely a UAT/survey-design task, not a coding task — clarify scope with the user before assuming it's a dev task.

**Explicitly discussed and deferred, not forgotten**: a Manager-facing report approve/comment workflow (click a report row → approve or request revision) was proposed by the user as a possible next feature after the Reports-page merge. Agreed to finish the already-scoped items above first, then revisit — this needs its own scoping pass (a `review_status` on `inspection_reports`, who can act on it, whether "needs revision" surfaces back to the inspector) once there's more real report data to design around.

No other next steps identified as of this update — ask the user what they want to tackle next rather than assuming.

## Things that are known-fine, don't re-litigate

- The `ResourceOptimizer` (budget-knapsack) class still exists in `optimization.py` alongside `InspectorAssignmentOptimizer` — this is intentional, not dead code to clean up. See [[02-Decisions-Log]].
- Dashboard/Projects/Forecast Engine showing ~450 projects (not 3000) is intentional scope, not a bug.
- PDF export via `window.print()` (not a server PDF library) was a deliberate lightweight choice for a thesis prototype, not a placeholder that needs upgrading unless asked.
- The map's project pins are jittered centroids, not real GPS coordinates — this is a known, accepted approximation (the original demo data had the same limitation).
- Password self-service exists and is verified working — don't rebuild it if asked about "forgot password" or "change password" again; check [[01-Architecture]] and [[04-Workflows-and-Gotchas]] first.
- The accept/submit-report workflow exists and is verified working (both Inspector and Manager/Admin views tested in-browser) — don't rebuild it if asked about "assignment status" or "inspection reports" again; check [[01-Architecture]] and [[02-Decisions-Log]] first. The old ResourceOptimizer-era `ASSIGN-INSP-00X` fake refs are long gone; `assignments.status` and `inspection_reports` are the real thing now.
- Reports support photo upload (chosen over data-fields-only), there's no decline path (accept-only was chosen over accept/decline), and report submission is gated on acceptance (chosen over ungated) — all explicit scoping decisions made with the user, not oversights. See [[02-Decisions-Log]] before "fixing" any of them.
- The Reports page now merges real `inspection_reports` with synthetic `inspection_logs` (real takes priority per project) — don't rebuild this if asked about "why does Reports still show fake data," check first whether the project actually has a real submission yet (most of the 450-project cohort still won't, since only test accounts have submitted so far).

## If you're picking this up cold (after `/compact` or a new session)

1. `git log --oneline -10` and `git status` — confirm you're on the right branch and nothing is uncommitted that this vault doesn't know about.
2. Read [[04-Workflows-and-Gotchas]] fully before running anything — several of those gotchas will silently corrupt data or waste a long pipeline run if ignored.
3. Check whether `backend/.env` and `frontend/.env.local` exist and have real values (they're gitignored, so a fresh clone won't have them — the user has these values, don't try to reconstruct or guess them).
4. Don't assume the Supabase schema matches `backend/supabase/*.sql` — those are the intended schema, but applying them is a manual step the user does in the Supabase dashboard. Verify with a live query before relying on a table/column existing.
