-- MAAGAP report review/approval workflow (FR-13).
-- Run once in the Supabase SQL Editor, after schema.sql, schema_auth.sql,
-- and schema_workflow.sql.
--
-- Lets a Manager/Admin mark a submitted inspection_reports row approved or
-- needing revision. Deliberately does NOT touch risk_alerts -- that table is
-- fully deleted-and-reinserted by every `python main.py` pipeline run, so a
-- revision-needed notification is derived on read (see /api/alerts) rather
-- than persisted there.

alter table inspection_reports add column if not exists review_status text not null default 'pending';
alter table inspection_reports add column if not exists review_comment text;
alter table inspection_reports add column if not exists reviewed_by uuid references profiles(id);
alter table inspection_reports add column if not exists reviewed_at timestamptz;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'inspection_reports_review_status_check'
  ) then
    alter table inspection_reports add constraint inspection_reports_review_status_check
      check (review_status in ('pending', 'approved', 'needs_revision'));
  end if;
end $$;
