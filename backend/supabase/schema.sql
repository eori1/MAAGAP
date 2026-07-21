-- MAAGAP Supabase (Postgres) schema.
-- Run once in the Supabase SQL Editor (Dashboard -> SQL Editor -> New query -> Run).
-- Mirrors the ERD in the manuscript (Figure 4): PROJECT, CONTRACTOR,
-- INSPECTION_LOG, PPDO_INSPECTOR, EXTERNAL_CONTEXT, MAAGAP_PREDICTIONS,
-- plus the LP deployment schedule (assignments) and a persisted risk_alerts
-- table (tier-escalation alerts must survive across pipeline runs to be
-- diffable, so they cannot be a recomputed view).
--
-- Safe to re-run: uses IF NOT EXISTS / DROP ... CASCADE guards.

drop table if exists risk_alerts cascade;
drop table if exists assignments cascade;
drop table if exists predictions cascade;
drop table if exists inspection_logs cascade;
drop table if exists external_context cascade;
drop table if exists projects cascade;
drop table if exists inspectors cascade;
drop table if exists contractors cascade;

create table contractors (
  contractor_id text primary key,
  contractor_name text not null,
  past_project_count int,
  past_delay_count int,
  past_average_slippage numeric,
  reliability_score numeric
);

create table inspectors (
  inspector_id text primary key,
  inspector_name text not null,
  availability_status text,
  current_workload int,
  vehicle_access boolean,
  capacity int  -- derived by the LP optimizer from availability/workload/vehicle access
);

create table projects (
  project_id text primary key,
  project_type text,
  category text,  -- implementing agency (renamed to match tbl_project.csv)
  location text,
  budget_allocated numeric,
  planned_duration_months int,
  start_date date,
  planned_end_date date,
  funding_source text,
  status text,
  -- Simulation ground-truth fields (outside the strict ERD, needed by the
  -- Timeline view to compare predicted vs. actual outcomes):
  project_year int,
  is_delayed boolean,
  actual_delay_days int
);

create table inspection_logs (
  log_id text primary key,
  project_id text references projects(project_id) on delete cascade,
  inspector_id text references inspectors(inspector_id),
  quarter int,
  total_quarters int,
  target_physical_pct numeric,
  actual_physical_pct numeric,
  slippage_pct numeric,
  target_financial_accomplishment numeric,
  actual_financial_accomplishment numeric,
  expenditure_ratio numeric,
  issues_noted int,
  rainfall_mm numeric,
  typhoon_days numeric,
  cpi_quarterly numeric,
  cmrpi_quarterly numeric,
  report_date date
);
create index idx_inspection_logs_project on inspection_logs(project_id);
create index idx_inspection_logs_inspector on inspection_logs(inspector_id);

create table external_context (
  context_id text primary key,
  quarter int,
  rainfall_mm numeric,
  typhoon_days numeric,
  cpi_quarterly numeric,
  cmrpi_quarterly numeric
);

create table predictions (
  prediction_id text primary key,
  project_id text references projects(project_id) on delete cascade,
  prediction_date date,
  delay_probability numeric,
  cost_overrun_probability numeric,
  predicted_delay_days numeric,
  risk_score numeric,
  risk_tier text,
  shap_explanation text,  -- pre-serialized JSON string (parsed client-side)
  optimized_assignment_ref text
);
create index idx_predictions_project on predictions(project_id);

create table assignments (
  assignment_id text primary key,
  project_id text references projects(project_id) on delete cascade,
  inspector_id text references inspectors(inspector_id),
  project_type text,
  location text,
  risk_score numeric,
  risk_tier text,
  priority text,
  urgency text
);
create index idx_assignments_project on assignments(project_id);
create index idx_assignments_inspector on assignments(inspector_id);

create table risk_alerts (
  id text primary key,
  type text,  -- 'TIER_ESCALATION' | 'CRITICAL_RISK'
  project_id text references projects(project_id) on delete cascade,
  from_tier text,
  to_tier text,
  risk_score numeric,
  message text,
  alert_date date,
  created_at timestamptz default now()
);
create index idx_risk_alerts_project on risk_alerts(project_id);

-- Row-Level Security: enabled with no policies, so only the service_role key
-- (used server-side by the backend pipeline and Next.js API routes) can
-- read/write. No anon/public access until an auth layer defines policies.
alter table contractors enable row level security;
alter table inspectors enable row level security;
alter table projects enable row level security;
alter table inspection_logs enable row level security;
alter table external_context enable row level security;
alter table predictions enable row level security;
alter table assignments enable row level security;
alter table risk_alerts enable row level security;
