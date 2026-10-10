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
