# MAAGAP — Instructions for Claude Code

## Read this first, every session

Before doing any work in this repo, read `knowledge-base/00-Overview.md` and `knowledge-base/06-Current-State-and-Next-Steps.md`. They contain the project context, architecture, decision history, and known gotchas accumulated across prior sessions — don't re-derive this from scratch or guess at it.

The knowledge base is an Obsidian vault at `/knowledge-base` (plain markdown, no special tooling needed to read it):

- `00-Overview.md` — what MAAGAP is, who it's for, read-order for the rest of the vault
- `01-Architecture.md` — backend pipeline, Supabase schema, frontend structure, API routes
- `02-Decisions-Log.md` — why things are the way they are, alternatives considered
- `03-Progress-Log.md` — commit-by-commit history of what's been done
- `04-Workflows-and-Gotchas.md` — **read this before running the pipeline or touching Supabase** — several documented failure modes will silently corrupt data or waste a multi-minute pipeline run if you skip this
- `05-Manuscript-Alignment.md` — how the implementation maps to the undergraduate thesis manuscript's objectives, including known gaps
- `06-Current-State-and-Next-Steps.md` — the living document; check this for what's actually pending right now
- `07-PRD.md` — **the requirements reference; check this before scoping any new feature** — every feature area tagged Built/Partial/Planned, so scope discussions start from an agreed spec instead of being re-litigated ad hoc

## Keep the knowledge base current

After completing significant work (a feature, a meaningful bug fix, a non-obvious decision), update the relevant vault file(s) — don't wait to be asked. In particular:

- Append to `03-Progress-Log.md` for anything commit-worthy
- Add an entry to `02-Decisions-Log.md` if you made a non-obvious choice or the user overrode a suggestion
- Add an entry to `04-Workflows-and-Gotchas.md` if you hit and fixed a bug that could recur (race conditions, silent API limits, config gotchas — the kind of thing that wastes time if rediscovered)
- Update `06-Current-State-and-Next-Steps.md`'s "Current state" and "Next steps" sections — this file should always reflect where things actually stand
- Update `07-PRD.md` whenever a feature's status changes (Planned → Built, etc.) or a new requirement is agreed — this is the spec, so it must stay in sync with reality

Keep entries concise and link between notes with `[[wikilink]]` syntax (matches Obsidian's linking convention) rather than duplicating content across files.

## Project-specific reminders

- Full pipeline run (`python main.py` from `backend/`) takes ~5-8 minutes. Run it in background, wait for the real completion notification, don't poll.
- Never run `pytest` concurrently with `python main.py` — see `04-Workflows-and-Gotchas.md`.
- Schema changes to Supabase require the user to manually run SQL in their dashboard — there's no way for Claude to run DDL directly. See the same file.
