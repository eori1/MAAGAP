-- MAAGAP operational workflow schema (Accept Assignment -> Submit Report).
-- Run once in the Supabase SQL Editor, after schema.sql and schema_auth.sql.
--
-- Closes the loop the manuscript's Use Case Diagram describes for the PPDO
-- Inspector actor ("accesses AI-optimized deployment schedules and logs the
-- actual physical and financial accomplishments gathered during site
-- visits") which, until now, only existed as synthetic pipeline data with
-- no real operational path in the app.

-- ---------------------------------------------------------------------
-- 1. Assignment acceptance status
-- ---------------------------------------------------------------------
alter table assignments add column if not exists status text not null default 'pending';
alter table assignments add column if not exists accepted_at timestamptz;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'assignments_status_check'
  ) then
    alter table assignments add constraint assignments_status_check
      check (status in ('pending', 'accepted'));
  end if;
end $$;

-- ---------------------------------------------------------------------
-- 2. Real inspection reports (submitted by inspectors, not the ML pipeline)
-- ---------------------------------------------------------------------
create table if not exists inspection_reports (
  report_id uuid primary key default gen_random_uuid(),
  assignment_id text references assignments(assignment_id) on delete cascade,
  project_id text references projects(project_id) on delete cascade,
  inspector_id text references inspectors(inspector_id),
  submitted_by uuid references profiles(id),
  physical_accomplishment_pct numeric,
  financial_accomplishment_pct numeric,
  issues_noted text,
  notes text,
  photo_urls text[],
  submitted_at timestamptz default now()
);
create index if not exists idx_inspection_reports_project on inspection_reports(project_id);
create index if not exists idx_inspection_reports_assignment on inspection_reports(assignment_id);

alter table inspection_reports enable row level security;

-- ---------------------------------------------------------------------
-- 3. Storage bucket for site photos attached to a report
-- ---------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('inspection-photos', 'inspection-photos', true)
on conflict (id) do nothing;

drop policy if exists "Authenticated users can upload inspection photos" on storage.objects;
create policy "Authenticated users can upload inspection photos"
  on storage.objects for insert
  to authenticated
  with check (bucket_id = 'inspection-photos');

drop policy if exists "Anyone can view inspection photos" on storage.objects;
create policy "Anyone can view inspection photos"
  on storage.objects for select
  to public
  using (bucket_id = 'inspection-photos');
