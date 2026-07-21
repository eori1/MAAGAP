# Current State and Next Steps

**Last updated:** 2026-07-21, end of the password self-service work.

Related: [[03-Progress-Log]] · [[05-Manuscript-Alignment]] · [[04-Workflows-and-Gotchas]]

> This is the one file in the vault meant to change often. Update the "Current state" and "Next steps" sections after each significant task; leave the rest of the vault's structure alone unless something structural actually changed.

## Current state

Branch `fix/inspector-assignment-alignment` (off `main`, not yet merged, not confirmed pushed to remote — check `git status`/`git log` at session start). Six feature commits so far, in order: `5ddfdf4`, `1c796c6`, `ecf666a`, `637bc00`, `100c360`, `6b422a9` (knowledge-base vault), `ea3ff1a` (password self-service).

The system is fully functional end-to-end and manually verified:
- Full ML pipeline runs (`python main.py`, ~5-8 min) and syncs to Supabase.
- All 7 frontend pages query Supabase live, no more static/mock data anywhere.
- Supabase Auth with 3 roles (manager/inspector/admin), verified in-browser for all three.
- Password self-service: logged-in "change password" (`/account`) and "forgot password" email flow (`/reset-password`), both verified end-to-end by the user including real email delivery.
- 25/25 backend tests passing, frontend builds clean, no known type errors.

See [[05-Manuscript-Alignment]] for the objective-by-objective status.

## Immediate next steps

Only one open item remains from the original gap list:

1. **ISO/IEC 25010 evaluation (manuscript Objective 5)** — entirely unaddressed, see [[05-Manuscript-Alignment]]. This is likely a UAT/survey-design task, not a coding task — clarify scope with the user before assuming it's a dev task.

No other next steps have been identified as of this update — ask the user what they want to tackle next rather than assuming.

## Things that are known-fine, don't re-litigate

- The `ResourceOptimizer` (budget-knapsack) class still exists in `optimization.py` alongside `InspectorAssignmentOptimizer` — this is intentional, not dead code to clean up. See [[02-Decisions-Log]].
- Dashboard/Projects/Forecast Engine showing ~450 projects (not 3000) is intentional scope, not a bug.
- PDF export via `window.print()` (not a server PDF library) was a deliberate lightweight choice for a thesis prototype, not a placeholder that needs upgrading unless asked.
- The map's project pins are jittered centroids, not real GPS coordinates — this is a known, accepted approximation (the original demo data had the same limitation).
- Password self-service exists and is verified working — don't rebuild it if asked about "forgot password" or "change password" again; check [[01-Architecture]] and [[04-Workflows-and-Gotchas]] first.

## If you're picking this up cold (after `/compact` or a new session)

1. `git log --oneline -10` and `git status` — confirm you're on the right branch and nothing is uncommitted that this vault doesn't know about.
2. Read [[04-Workflows-and-Gotchas]] fully before running anything — several of those gotchas will silently corrupt data or waste a long pipeline run if ignored.
3. Check whether `backend/.env` and `frontend/.env.local` exist and have real values (they're gitignored, so a fresh clone won't have them — the user has these values, don't try to reconstruct or guess them).
4. Don't assume the Supabase schema matches `backend/supabase/*.sql` — those are the intended schema, but applying them is a manual step the user does in the Supabase dashboard. Verify with a live query before relying on a table/column existing.
