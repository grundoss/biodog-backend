-- BioDog: contatori di utilizzo lato server + RLS restrittive su user_subscriptions.
-- Da eseguire una volta nel SQL Editor di Supabase (è idempotente).

-- 1) Contatori di utilizzo (prove gratuite, quota mensile Premium, gettone della domenica)
create table if not exists public.usage_counters (
    user_id    uuid        not null references auth.users(id) on delete cascade,
    bucket     text        not null,
    count      integer     not null default 0,
    updated_at timestamptz not null default now(),
    primary key (user_id, bucket)
);

alter table public.usage_counters enable row level security;

-- Incremento atomico usato dal backend (service role).
create or replace function public.biodog_increment_usage(p_user_id uuid, p_bucket text)
returns integer
language sql
security definer
set search_path = public
as $$
    insert into public.usage_counters (user_id, bucket, count, updated_at)
    values (p_user_id, p_bucket, 1, now())
    on conflict (user_id, bucket)
    do update set count = public.usage_counters.count + 1, updated_at = now()
    returning count;
$$;

revoke all on function public.biodog_increment_usage(uuid, text) from public, anon, authenticated;
grant execute on function public.biodog_increment_usage(uuid, text) to service_role;

-- 2) user_subscriptions: un abbonamento per utente (serve all'upsert on_conflict=user_id)
create unique index if not exists user_subscriptions_user_id_key
    on public.user_subscriptions (user_id);
create index if not exists user_subscriptions_stripe_subscription_id_idx
    on public.user_subscriptions (stripe_subscription_id);

-- 3) RLS: il browser (chiave pubblica) può solo LEGGERE la propria riga.
--    Tutte le scritture passano dal backend con la service role key, che bypassa RLS.
alter table public.user_subscriptions enable row level security;

do $$
declare pol record;
begin
    for pol in
        select policyname, tablename from pg_policies
        where schemaname = 'public' and tablename in ('user_subscriptions', 'usage_counters')
    loop
        execute format('drop policy %I on public.%I', pol.policyname, pol.tablename);
    end loop;
end $$;

create policy "read own subscription" on public.user_subscriptions
    for select to authenticated using (auth.uid()::text = user_id::text);

create policy "read own usage" on public.usage_counters
    for select to authenticated using (auth.uid() = user_id);

-- 4) Diario delle analisi e check-in dell'umore: sincronizzati tra dispositivi.
--    Il browser legge e scrive solo le proprie righe.
create table if not exists public.analyses (
    id         bigint generated always as identity primary key,
    user_id    uuid        not null default auth.uid() references auth.users(id) on delete cascade,
    query      text        not null check (char_length(query) <= 600),
    synth      jsonb       not null,
    created_at timestamptz not null default now()
);
create index if not exists analyses_user_created_idx on public.analyses (user_id, created_at desc);

create table if not exists public.mood_checkins (
    user_id    uuid        not null default auth.uid() references auth.users(id) on delete cascade,
    day        date        not null,
    mood       text        not null check (mood in ('calm', 'reactive', 'anxious', 'hyper')),
    updated_at timestamptz not null default now(),
    primary key (user_id, day)
);

alter table public.analyses enable row level security;
alter table public.mood_checkins enable row level security;

do $$
declare pol record;
begin
    for pol in
        select policyname, tablename from pg_policies
        where schemaname = 'public' and tablename in ('analyses', 'mood_checkins')
    loop
        execute format('drop policy %I on public.%I', pol.policyname, pol.tablename);
    end loop;
end $$;

create policy "own analyses" on public.analyses
    for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "own moods" on public.mood_checkins
    for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);

grant select, insert, delete on public.analyses to authenticated;
grant select, insert, update, delete on public.mood_checkins to authenticated;
