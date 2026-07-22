-- MAAGAP manual PPA entry ("Add new PPA").
-- Run once in the Supabase SQL Editor, after the earlier schema files.
--
-- Distinguishes a manually-added, not-yet-scored PPA from the ~2550
-- historical (non-monitored) rows already in `projects` -- without this
-- flag, a naive "project has no prediction yet" query would incorrectly
-- surface that entire historical backlog too, not just genuinely new entries.

alter table projects add column if not exists is_manual_entry boolean not null default false;
alter table projects add column if not exists created_by uuid references profiles(id);
alter table projects add column if not exists created_at timestamptz default now();

-- The synthetic/historical cohort never had a real name or description --
-- every page just displays project_id as the "name" (see /api/projects).
-- A real, manually-added PPA needs an actual identifying title, so these
-- are nullable additions: null for existing rows (frontend falls back to
-- project_id, preserving today's display for the synthetic cohort).
alter table projects add column if not exists project_name text;
alter table projects add column if not exists description text;
