-- MAAGAP authentication schema (AuthService, Use Case Diagram Fig. 6).
-- Run once in the Supabase SQL Editor, after schema.sql.
--
-- Defines the three actors from the manuscript's Use Case Diagram:
--   - manager   (PPDO Manager: full read access, approves allocations)
--   - inspector (PPDO Inspector: sees only their own assigned projects)
--   - admin     (System Administrator: manages accounts, batch uploads)
--
-- Every Supabase Auth user automatically gets a `profiles` row (role
-- defaults to 'inspector'; promote via the app's admin panel or manually
-- here). `inspector_id` links a login to a row in the `inspectors` table
-- so an Inspector's queries can be scoped to their own workload.

drop table if exists profiles cascade;

create table profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  full_name text,
  role text not null default 'inspector' check (role in ('manager', 'inspector', 'admin')),
  inspector_id text references inspectors(inspector_id),
  created_at timestamptz default now()
);

alter table profiles enable row level security;

-- Users can always read their own profile (needed to resolve their role).
create policy "Users read own profile"
  on profiles for select
  using (auth.uid() = id);

-- Auto-create a profile row whenever a new Supabase Auth user is created.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, email, full_name)
  values (new.id, new.email, new.raw_user_meta_data->>'full_name');
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- Bootstrap: promote the first registered user to admin so there is always
-- at least one account that can manage the rest. Run this manually after
-- your first signup:
--   update profiles set role = 'admin' where email = 'you@example.com';
