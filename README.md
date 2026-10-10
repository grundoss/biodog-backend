# biodog-backend

Backend FastAPI di BioDog.io (analisi Gemini, abbonamenti Stripe, quote d'uso) e,
in `frontend/index.html`, la webapp che lo usa.

## Piani e limiti (applicati dal server)

| Piano | Prezzo | Analisi testuali | Video | Configuratore razza |
|---|---|---|---|---|
| Ospite (senza account) | – | 1 anteprima ogni 24h per IP, parte pratica nascosta | – | – |
| Free (registrato) | 0 € | 2 in totale + 1 omaggio ogni domenica (Europe/Rome) | – | – |
| Premium | 1,99 € | 100 / mese | – | – |
| Standard | 3,49 € | illimitate | – | – |
| Vision PRO | 3,99 € | illimitate | ✓ | ✓ |

Se il motore Gemini non risponde, l'API restituisce `502 engine_unavailable` e non consuma crediti.

## Deploy

Fai i passi **in quest'ordine**: il nuovo frontend richiede il nuovo backend e viceversa.

1. **Supabase → SQL Editor**: esegui `supabase/migrations/20261010_usage_counters_and_rls.sql`.
   Crea `usage_counters`, `analyses` (diario), `mood_checkins` (check-in) e imposta le RLS:
   il browser può solo leggere la propria riga di `user_subscriptions`.
   Se l'indice unico su `user_subscriptions.user_id` fallisce, ci sono righe duplicate da unire prima.
2. **Render → Environment** del backend:
   - `SUPABASE_SERVICE_KEY` (**obbligatoria**: senza, login e quote rispondono 503)
   - `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`
   - `STRIPE_PRICE_ID_PREMIUM`, `STRIPE_PRICE_ID_STANDARD`, `STRIPE_PRICE_ID_PRO` (tutti e tre: un piano senza prezzo ora dà errore invece di addebitare il Premium)
   - `GEMINI_API_KEY`, opzionale `GEMINI_MODEL`
   - opzionali `ALLOWED_ORIGINS` (default `https://biodog.io,https://www.biodog.io`) e `FRONTEND_URL` (default `https://biodog.io`)
3. **Stripe → Developers → Webhooks**: sull'endpoint `/api/v1/stripe/webhook` abilita, oltre a
   `checkout.session.completed`, anche `customer.subscription.created`,
   `customer.subscription.updated` e `customer.subscription.deleted`.
4. Pubblica `frontend/index.html` su biodog.io.

## Test

```bash
pip install -r requirements.txt pytest "httpx<0.28"
SUPABASE_SERVICE_KEY=x pytest tests
```
