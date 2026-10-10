-- BioDog: percorsi di 7 giorni. Da eseguire una volta nel SQL Editor di Supabase (è idempotente).
-- Il backend crea i percorsi (service role, per applicare i limiti del piano gratuito);
-- il browser legge i propri percorsi e segna i giorni completati.

create table if not exists public.care_paths (
    id          bigint generated always as identity primary key,
    user_id     uuid        not null references auth.users(id) on delete cascade,
    situation   text        not null check (char_length(situation) <= 600),
    plan        jsonb       not null,
    completed   integer[]   not null default '{}',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index if not exists care_paths_user_created_idx on public.care_paths (user_id, created_at desc);

alter table public.care_paths enable row level security;

do $$
declare pol record;
begin
    for pol in
        select policyname, tablename from pg_policies
        where schemaname = 'public' and tablename = 'care_paths'
    loop
        execute format('drop policy %I on public.%I', pol.policyname, pol.tablename);
    end loop;
end $$;

create policy "read own paths" on public.care_paths
    for select to authenticated using (auth.uid() = user_id);
create policy "update own paths" on public.care_paths
    for update to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "delete own paths" on public.care_paths
    for delete to authenticated using (auth.uid() = user_id);

-- Il browser può aggiornare solo i giorni completati, non il contenuto del piano.
revoke update on public.care_paths from authenticated;
grant select, delete on public.care_paths to authenticated;
grant update (completed, updated_at) on public.care_paths to authenticated;
