import asyncio
import base64
import json
import os
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Literal, Optional, Tuple

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from zoneinfo import ZoneInfo
    APP_TZ = ZoneInfo("Europe/Rome")
except Exception:
    APP_TZ = None

# Libreria Stripe ufficiale
try:
    import stripe
except ImportError:
    stripe = None

app = FastAPI(title="BioDog.io Neural Engine", version="5.0")

# Origini autorizzate a chiamare l'API dal browser (separate da virgola).
ALLOWED_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.getenv("ALLOWED_ORIGINS", "https://biodog.io,https://www.biodog.io").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ==================== VARIABILI D'AMBIENTE ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'").replace("[", "").replace("]", "")
ACTIVE_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_PRICE_ID_PREMIUM = os.getenv("STRIPE_PRICE_ID_PREMIUM", os.getenv("STRIPE_PRICE_ID", "")).strip()
STRIPE_PRICE_ID_STANDARD = os.getenv("STRIPE_PRICE_ID_STANDARD", "").strip()
STRIPE_PRICE_ID_PRO = os.getenv("STRIPE_PRICE_ID_PRO", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://itjfyjyzaornkintpefd.supabase.co").strip().rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://biodog.io").strip().rstrip("/")

if stripe and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# ==================== PIANI E LIMITI ====================
FREE_TRIAL_LIMIT = 2          # traduzioni gratuite a vita per utente registrato
PREMIUM_MONTHLY_LIMIT = 100   # piano Premium (1,99 €)
SUNDAY_TOKEN_LIMIT = 1        # consulto omaggio della domenica per utenti free
GUEST_LIMIT = 1               # anteprima senza account, per IP
GUEST_WINDOW_SECONDS = 24 * 3600
MAX_VIDEO_BYTES = 25 * 1024 * 1024

PAID_TIERS = ("premium", "standard", "pro")
ACTIVE_SUB_STATUSES = {"active", "trialing"}
ALLOWED_VIDEO_MIME = {"video/mp4", "video/quicktime", "video/webm", "video/x-m4v"}

DEFAULT_MORPHOLOGY = {"snout": "normal", "ears": "prick", "size": "medium", "tail": "long"}

# ==================== SCHEMI DATI PYDANTIC ====================
class TransductionRequest(BaseModel):
    user_text: str = Field(..., min_length=2, max_length=500)
    snout: Literal["flat", "normal", "long"] = "normal"
    ears: Literal["prick", "drop"] = "prick"
    size: Literal["small", "medium", "large"] = "medium"
    tail: Literal["long", "curled", "short"] = "long"
    lang: Optional[str] = Field(default="it", description="Lingua di output richiesta ('it' oppure 'en')")
    observed_time_hours: Optional[float] = 0.0

class CreateCheckoutRequest(BaseModel):
    tier: Literal["premium", "standard", "pro"] = "premium"

# ==================== ERRORI API ====================
def api_error(status: int, code: str, message: str) -> HTTPException:
    """Errore con un codice stabile che il frontend usa per decidere cosa mostrare."""
    return HTTPException(status_code=status, detail={"code": code, "message": message})

# ==================== CALCOLO SENSORIALE FISICO ====================
class SensoryEngine:
    @staticmethod
    def compute(req: TransductionRequest):
        turbinates = 170.0 if req.snout == "long" else (45.0 if req.snout == "flat" else 100.0)
        fov = 270 if req.snout == "long" else (220 if req.snout == "flat" else 250)
        cpd = 12.5 if req.snout == "long" else (9.5 if req.snout == "flat" else 11.5)
        eye_height = 25 if req.size == "small" else (75 if req.size == "large" else 45)
        mobility = "Flessibilità 180° indipendente" if req.ears == "prick" else "Assorbimento passivo frontale"

        return {
            "turbinates_cm2": turbinates,
            "fov_degrees": fov,
            "acuity_cpd": cpd,
            "eye_height_cm": eye_height,
            "ear_mobility": mobility,
            "tail_bias": req.tail
        }

SYSTEM_PROMPT = """Sei il motore di intelligenza artificiale biologica ed evolutiva BioDog.io.
Trasduci il comportamento del cane descritto (o mostrato nel video) dall'umano nella prospettiva etologica, neurobiologica, prossemica ed evolutiva del cane.

REGOLE CRITICHE (ANTI-ANTROPOMORFISMO, EVOLUZIONE E PROSSEMICA):
1. DIVIETO ASSOLUTO di attribuire concetti morali umani: dispetto, vendetta, senso di colpa, prevaricazione etica o dominio gerarchico alfa.
2. Radica sempre il comportamento nei 7 circuiti emotivi primari di Jaak Panksepp: SEEKING, RAGE, FEAR, PANIC/GRIEF, PLAY, CARE, LUST.
3. Decodifica l'esperienza in 4 canali sensoriali principali: 
   - Olfatto (molecole, decadimento VOC, feromoni)
   - Vista (movimento, deuteranopia, altezza da terra, campo visivo)
   - Udito (frequenze, prosodia)
   - Tatto & Prossemica (fibre C-tattili, vibrisse, tolleranza manipolativa).
4. LETTURA DELLO SPAZIO E DISTANZE (Hediger & Prossemica):
   - Distanze di Hediger: Identifica se l'umano si trova a Distanza Sociale (sicurezza), Distanza di Fuga (innesco evitamento/stress) o Distanza Critica (messa all'angolo, innesco fear-biting).
   - Effetto Barriera (Frustrazione Territoriale): Se il cane ringhia/abbaia a una recinzione, cancello o finestra (che si trova TRA LUI E L'ESTERNO), decodificalo come "Frustrazione da Barriera Frontale" (circuito RAGE/Difesa). La barriera fisica lo costringe a una reazione stanziale sul confine contro lo stimolo esterno. NON dire che il cane "è messo all'angolo" o "non ha via di fuga" (poiché la barriera è davanti a lui, non dietro di lui).
   - Geometria dell'Avvicinamento: L'approccio frontale e lo sguardo fisso (Staring) sono minacce predatorie spaziali (Emisfero Destro). Suggerisci sempre l'approccio indiretto a curva ("Curving") e il posizionamento di fianco.
   - Referenza Sociale (Base Sicura): Se il cane osserva l'umano o si nasconde dietro di lui in presenza di estranei, non definirlo solo "pauroso", ma spiega che sta mappando la reazione della sua "Base Sicura" (Attaccamento).
   - Ossitocina vs Fissazione: Spiega che lo sguardo morbido (Mutual Gaze) rilascia ossitocina, mentre lo sguardo fisso imposto è pressione spaziale.
   - Asimmetria Caudale: Coda a destra = emisfero sinistro (approccio/positivo). Coda a sinistra = emisfero destro (allarme/evitamento).
   - Regola dell'Abbraccio: Costringere collo/spalle blocca l'istinto cursore di fuga e alza il cortisolo (Stress Simpatico).
   - Tocco Medico/Dolore (OA): Se c'è rifiuto al tatto o freezing, suggerisci iperalgesia o allodinia da possibile osteoartrite.
5. IL MOTORE EVOLUTIVO (Neotenia e Sindrome da Domesticazione):
   - Faccia da Colpevole / Puppy Dog Eyes: Se l'umano descrive il cane come "colpevole", spiega l'azione del muscolo facciale LAOM (AU101). Spiega che non è morale umana, ma un micro-movimento neotenico evolutosi per attivare il loop materno dell'ossitocina umana.
   - PMP (Pattern Motorio Predatorio): Se il cane rincorre, morde caviglie o punta, decodifica il blocco eterocronico (Orient, Stalk, Chase, Grab). I cani da pastore o ferma sono "congelati" in queste fasi della caccia.
   - Deficit Brachicefali: Se il cane ha il "Muso Schiacciato", spiega il Collasso dei Segnali Visivi Agonistici. La sua anatomia (no coda, no muso lungo) impedisce di mostrare i 15 segnali di de-escalation del lupo.
   - Ipertrofia dell'Abbaio: Spiega che l'abbaio incessante non è "dominanza", ma un tratto paedomorfico/infantile (il lupo adulto abbaia raramente).
6. Fornisci indicazioni precise sulla mimica corporea e sul tono vocale che l'umano deve assumere per la de-escalation spaziale ed emotiva.
7. DIRETTIVA SULLA LINGUA DI OUTPUT (BILINGUAL DIRECTIVE):
   - Se lang == 'en' (o se il testo/input dell'utente è redatto in inglese): genera TUTTI i campi del JSON (situation_title, panksepp_label, thought, sensory, human_body_language, explanation, steps, forbidden) RIGOROSAMENTE in INGLESE fluente, naturale ed etologicamente accurato (adottando la corretta terminologia scientifica: Frontal Barrier Frustration, Hediger Distances, Curving Approach, LAOM Neoteny, ecc.).
   - Altrimenti (se lang == 'it'): genera TUTTI i campi del JSON in ITALIANO.
8. Genera ESCLUSIVAMENTE un JSON valido (senza testo introduttivo o markdown) con questa struttura esatta:
{
  "situation_title": "Titolo etologico breve",
  "panksepp": "CARE | RAGE | FEAR | PANIC/GRIEF | PLAY | SEEKING | LUST",
  "panksepp_label": "Nome del circuito ed etichetta emotiva (es. CARE / Ricongiungimento Affiliativo)",
  "arousal": numero intero da 0 a 100,
  "valence": numero intero da -50 a +50,
  "thought": "pensiero del cane in prima persona: rapido, sensoriale, spaziale, privo di morale umana",
  "sensory": {
    "smell": "Cosa percepisce il tartufo in questo momento",
    "sight": "Cosa vede dagli occhi (prospettiva, geometria dell'avvicinamento, sagoma incombente)",
    "hearing": "Cosa sente con le orecchie",
    "touch": "Cosa percepiscono i recettori tattili e calcolo prossemico dello spazio vitale (Hediger/Barriera)"
  },
  "human_body_language": {
    "voice": "Tono di voce raccomandato",
    "posture": "Postura corporea (es. Fianco a 45°, Curving, evitare sguardi fissi frontali)"
  },
  "explanation": "Spiegazione etologica profonda (Usa concetti: Neotenia, LAOM, PMP, Distanze Hediger, Effetto Barriera, Allodinia)",
  "steps": ["passo pratico 1 da fare subito per modulare lo spazio", "passo pratico 2", "passo pratico 3"],
  "forbidden": ["errore tipico 1 da evitare (es. avanzare frontalmente)", "errore tipico 2"]
}"""

def _extract_clean_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    marker = chr(96) * 3
    if marker in cleaned:
        cleaned = cleaned.replace(marker + "json", "").replace(marker, "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]
    return json.loads(cleaned)

def _clamp_int(value, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(round(float(value)))))
    except (TypeError, ValueError):
        return default

def _normalize_synthesis(raw: dict) -> dict:
    """Garantisce che la risposta del modello abbia sempre la forma attesa dal frontend."""
    if not isinstance(raw, dict):
        raise ValueError("Risposta del modello non è un oggetto JSON")

    def text(v) -> str:
        return v.strip() if isinstance(v, str) else ""

    def text_list(v) -> list:
        return [s.strip() for s in v if isinstance(s, str) and s.strip()] if isinstance(v, list) else []

    sensory = raw.get("sensory") if isinstance(raw.get("sensory"), dict) else {}
    body = raw.get("human_body_language") if isinstance(raw.get("human_body_language"), dict) else {}

    synth = {
        "situation_title": text(raw.get("situation_title")),
        "panksepp": text(raw.get("panksepp")),
        "panksepp_label": text(raw.get("panksepp_label")),
        "arousal": _clamp_int(raw.get("arousal"), 0, 100, 50),
        "valence": _clamp_int(raw.get("valence"), -50, 50, 0),
        "thought": text(raw.get("thought")),
        "sensory": {k: text(sensory.get(k)) for k in ("smell", "sight", "hearing", "touch")},
        "human_body_language": {k: text(body.get(k)) for k in ("voice", "posture")},
        "explanation": text(raw.get("explanation")),
        "steps": text_list(raw.get("steps")),
        "forbidden": text_list(raw.get("forbidden")),
    }
    if not synth["thought"] and not synth["explanation"]:
        raise ValueError("Risposta del modello vuota")
    return synth

# Campi pratici riservati agli utenti registrati: agli ospiti non vengono inviati.
GUEST_LOCKED_FIELDS = ("human_body_language", "explanation", "steps", "forbidden")

def _lock_for_guest(synth: dict) -> dict:
    locked = dict(synth)
    locked["human_body_language"] = {"voice": "", "posture": ""}
    locked["explanation"] = ""
    locked["steps"] = []
    locked["forbidden"] = []
    return locked

# ==================== CHIAMATE GEMINI API ====================

def _gemini_generate(api_key: str, parts: list) -> Tuple[Optional[dict], Optional[str]]:
    if not api_key:
        return None, "Chiave API mancante"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "BioDog/5.0", "x-goog-api-key": api_key},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body["candidates"][0]["content"]["parts"][0].get("text", "")
        return _normalize_synthesis(_extract_clean_json(text)), None
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        return None, f"HTTP {e.code} {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def _call_gemini_api(api_key: str, full_prompt: str) -> Tuple[Optional[dict], Optional[str]]:
    return _gemini_generate(api_key, [{"text": full_prompt}])

def _call_gemini_api_video(api_key: str, full_prompt: str, video_bytes: bytes, mime_type: str) -> Tuple[Optional[dict], Optional[str]]:
    b64_data = base64.b64encode(video_bytes).decode("utf-8")
    return _gemini_generate(api_key, [
        {"text": full_prompt},
        {"inline_data": {"mimeType": mime_type, "data": b64_data}}
    ])

# ==================== SUPABASE (SERVICE ROLE) ====================

class SupabaseError(Exception):
    pass

def _supabase_request(method: str, path: str, payload=None, extra_headers: Optional[dict] = None):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise SupabaseError("SUPABASE_SERVICE_KEY non configurata")

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"{SUPABASE_URL}{path}", data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        raise SupabaseError(f"HTTP {e.code} su {path}: {body}") from e
    except Exception as e:
        raise SupabaseError(f"{type(e).__name__} su {path}: {e}") from e

def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def _upsert_subscription(user_id: str, fields: dict) -> None:
    payload = {"user_id": user_id, **fields, "updated_at": _now_iso()}
    _supabase_request(
        "POST", "/rest/v1/user_subscriptions?on_conflict=user_id", payload,
        {"Prefer": "resolution=merge-duplicates,return=minimal"},
    )

def _update_subscription_by_stripe_id(sub_id: str, fields: dict) -> list:
    rows = _supabase_request(
        "PATCH", f"/rest/v1/user_subscriptions?stripe_subscription_id=eq.{urllib.parse.quote(sub_id)}",
        {**fields, "updated_at": _now_iso()},
        {"Prefer": "return=representation"},
    )
    return rows or []

def _get_subscription(user_id: str) -> Optional[dict]:
    rows = _supabase_request(
        "GET",
        f"/rest/v1/user_subscriptions?user_id=eq.{urllib.parse.quote(user_id)}"
        "&select=is_active,plan_tier,stripe_customer_id,stripe_subscription_id&limit=1",
    )
    return rows[0] if rows else None

def _tier_from_subscription(sub: Optional[dict]) -> str:
    if sub and sub.get("is_active") is True:
        tier = sub.get("plan_tier") or "premium"
        return tier if tier in PAID_TIERS else "premium"
    return "free"

def _get_usage(user_id: str, bucket: str) -> int:
    rows = _supabase_request(
        "GET",
        f"/rest/v1/usage_counters?user_id=eq.{urllib.parse.quote(user_id)}"
        f"&bucket=eq.{urllib.parse.quote(bucket)}&select=count&limit=1",
    )
    return int(rows[0]["count"]) if rows else 0

def _increment_usage(user_id: str, bucket: str) -> int:
    result = _supabase_request("POST", "/rest/v1/rpc/biodog_increment_usage", {"p_user_id": user_id, "p_bucket": bucket})
    return int(result) if result is not None else 0

# ==================== AUTENTICAZIONE ====================
_TOKEN_CACHE_TTL = 60
_token_cache: dict = {}
_token_cache_lock = threading.Lock()

def _fetch_supabase_user(token: str) -> Optional[dict]:
    """Verifica il JWT di Supabase chiedendo a Supabase Auth chi è l'utente."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise SupabaseError("SUPABASE_SERVICE_KEY non configurata")
    req = urllib.request.Request(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return None
        raise SupabaseError(f"Auth HTTP {e.code}") from e
    except Exception as e:
        raise SupabaseError(f"Auth {type(e).__name__}: {e}") from e
    if not data or not data.get("id"):
        return None
    return {"id": data["id"], "email": data.get("email") or ""}

def _resolve_user(token: str) -> Optional[dict]:
    now = time.time()
    with _token_cache_lock:
        cached = _token_cache.get(token)
        if cached and cached[0] > now:
            return cached[1]
    user = _fetch_supabase_user(token)
    if user:
        with _token_cache_lock:
            if len(_token_cache) > 5000:
                _token_cache.clear()
            _token_cache[token] = (now + _TOKEN_CACHE_TTL, user)
    return user

async def optional_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    if not token:
        return None
    try:
        user = await asyncio.to_thread(_resolve_user, token)
    except SupabaseError as e:
        print(f"[BioDog Auth] {e}")
        raise api_error(503, "service_unavailable", "Servizio di autenticazione non disponibile.")
    if not user:
        raise api_error(401, "invalid_session", "Sessione scaduta: accedi di nuovo.")
    return user

async def required_user(user: Optional[dict] = Depends(optional_user)) -> dict:
    if not user:
        raise api_error(401, "auth_required", "Accedi per usare questa funzione.")
    return user

# ==================== QUOTE DI UTILIZZO ====================

def _local_now() -> datetime:
    return datetime.now(APP_TZ) if APP_TZ else datetime.utcnow()

def _month_bucket(now: datetime) -> str:
    return f"month-{now:%Y-%m}"

def _sunday_bucket(now: datetime) -> str:
    return f"sunday-{now:%Y-%m-%d}"

def decide_quota(tier: str, trial_used: int, month_used: int, sunday_used: int, is_sunday: bool) -> Tuple[bool, Optional[str], Optional[str]]:
    """Ritorna (consentito, bucket da incrementare, codice di blocco)."""
    if tier in ("standard", "pro"):
        return True, None, None
    if tier == "premium":
        if month_used < PREMIUM_MONTHLY_LIMIT:
            return True, "month", None
        return False, None, "monthly_limit"
    if trial_used < FREE_TRIAL_LIMIT:
        return True, "trial", None
    if is_sunday and sunday_used < SUNDAY_TOKEN_LIMIT:
        return True, "sunday", None
    return False, None, "trial_exhausted"

def _usage_snapshot(user_id: str) -> dict:
    now = _local_now()
    sub = _get_subscription(user_id)
    tier = _tier_from_subscription(sub)
    is_sunday = now.weekday() == 6
    snap = {
        "tier": tier,
        "has_customer": bool(sub and sub.get("stripe_customer_id")),
        "is_sunday": is_sunday,
        "trial_used": 0, "trial_limit": FREE_TRIAL_LIMIT,
        "month_used": 0, "month_limit": PREMIUM_MONTHLY_LIMIT if tier == "premium" else None,
        "sunday_used": 0,
        "buckets": {"trial": "trial", "month": _month_bucket(now), "sunday": _sunday_bucket(now)},
    }
    if tier == "free":
        snap["trial_used"] = _get_usage(user_id, "trial")
        if is_sunday:
            snap["sunday_used"] = _get_usage(user_id, snap["buckets"]["sunday"])
    elif tier == "premium":
        snap["month_used"] = _get_usage(user_id, snap["buckets"]["month"])
    return snap

def _public_usage(snap: dict) -> dict:
    tier = snap["tier"]
    if tier in ("standard", "pro"):
        remaining = None
    elif tier == "premium":
        remaining = max(0, PREMIUM_MONTHLY_LIMIT - snap["month_used"])
    else:
        remaining = max(0, FREE_TRIAL_LIMIT - snap["trial_used"])
    return {
        "tier": tier,
        "remaining": remaining,
        "trial_used": snap["trial_used"],
        "trial_limit": FREE_TRIAL_LIMIT,
        "month_used": snap["month_used"],
        "month_limit": snap["month_limit"],
        "sunday_token_available": tier == "free" and snap["is_sunday"] and snap["sunday_used"] < SUNDAY_TOKEN_LIMIT,
        "has_customer": snap["has_customer"],
    }

# Anteprima ospiti: contatore in memoria per IP (si azzera al riavvio del server).
_guest_hits: dict = {}
_guest_lock = threading.Lock()

def _client_ip(request: Request) -> str:
    # Render aggiunge l'IP reale in coda a X-Forwarded-For: l'ultimo valore è quello affidabile.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"

def _guest_allowed(ip: str) -> bool:
    cutoff = time.time() - GUEST_WINDOW_SECONDS
    with _guest_lock:
        hits = [t for t in _guest_hits.get(ip, []) if t > cutoff]
        _guest_hits[ip] = hits
        return len(hits) < GUEST_LIMIT

def _record_guest(ip: str) -> None:
    with _guest_lock:
        _guest_hits.setdefault(ip, []).append(time.time())
        if len(_guest_hits) > 20000:
            cutoff = time.time() - GUEST_WINDOW_SECONDS
            for key in [k for k, v in _guest_hits.items() if not v or v[-1] <= cutoff]:
                del _guest_hits[key]

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {"status": "BioDog Neural Engine Online", "model": ACTIVE_MODEL}

@app.get("/api/v1/me")
async def me(user: dict = Depends(required_user)):
    try:
        snap = await asyncio.to_thread(_usage_snapshot, user["id"])
    except SupabaseError as e:
        print(f"[BioDog] /me: {e}")
        raise api_error(503, "service_unavailable", "Impossibile leggere l'abbonamento. Riprova tra poco.")
    return {"user_id": user["id"], "email": user["email"], **_public_usage(snap)}

def _lang_directive(lang: Optional[str]) -> Tuple[str, str]:
    target_lang = "en" if (lang and lang.lower().strip() == "en") else "it"
    directive = "OUTPUT IN NATURAL ENGLISH (lang=en)" if target_lang == "en" else "OUTPUT IN ITALIAN (lang=it)"
    return target_lang, directive

@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest, request: Request, user: Optional[dict] = Depends(optional_user)):
    snap = None
    bucket_key = None
    guest_ip = None

    if user:
        try:
            snap = await asyncio.to_thread(_usage_snapshot, user["id"])
        except SupabaseError as e:
            print(f"[BioDog] quota: {e}")
            raise api_error(503, "service_unavailable", "Servizio temporaneamente non disponibile. Riprova tra poco.")
        allowed, bucket_key, block_code = decide_quota(
            snap["tier"], snap["trial_used"], snap["month_used"], snap["sunday_used"], snap["is_sunday"]
        )
        if not allowed:
            msg = ("Hai esaurito le analisi del mese." if block_code == "monthly_limit"
                   else "Hai completato le traduzioni gratuite.")
            raise api_error(402, block_code, msg)
        tier = snap["tier"]
    else:
        guest_ip = _client_ip(request)
        if not _guest_allowed(guest_ip):
            raise api_error(401, "auth_required", "Registrati gratis per continuare.")
        tier = "guest"

    # La calibrazione morfologica è una funzione del piano PRO.
    if tier != "pro":
        req = req.model_copy(update=DEFAULT_MORPHOLOGY)

    bio = SensoryEngine.compute(req)
    target_lang, lang_directive = _lang_directive(req.lang)

    user_prompt = f"""{SYSTEM_PROMPT}\n\nTARGET LANGUAGE: {lang_directive}\n\nComportamento osservato / Observed behavior: "{req.user_text}"\nProfilo biologico / Biological profile:\n- Cranio/Skull: {req.snout} (Turbinati olfattivi / Olfactory turbinates: {bio['turbinates_cm2']} cm²)\n- Campo Visivo / FOV: {bio['fov_degrees']}° (Acuità / Acuity: {bio['acuity_cpd']} cpd)\n- Occhi da terra / Eye height: {bio['eye_height_cm']} cm\n- Coda / Tail: {bio['tail_bias']}\n- Orecchie / Ears: {bio['ear_mobility']}"""
    synth, api_err = await asyncio.to_thread(_call_gemini_api, GEMINI_API_KEY, user_prompt)
    if not synth:
        # Nessun credito consumato se il motore non risponde.
        print(f"[BioDog] Gemini testo fallito: {api_err}")
        raise api_error(502, "engine_unavailable", "Il motore di analisi non risponde. Riprova tra poco: nessun credito è stato usato.")

    sunday_token_used = False
    if user:
        if bucket_key:
            try:
                await asyncio.to_thread(_increment_usage, user["id"], snap["buckets"][bucket_key])
                if bucket_key == "trial":
                    snap["trial_used"] += 1
                elif bucket_key == "month":
                    snap["month_used"] += 1
                elif bucket_key == "sunday":
                    snap["sunday_used"] += 1
                    sunday_token_used = True
            except SupabaseError as e:
                print(f"[BioDog] Incremento quota fallito per {user['id']}: {e}")
        usage = _public_usage(snap)
    else:
        _record_guest(guest_ip)
        synth = _lock_for_guest(synth)
        usage = None

    return {
        "status": "success",
        "engine": ACTIVE_MODEL,
        "lang": target_lang,
        "locked": user is None,
        "sunday_token_used": sunday_token_used,
        "usage": usage,
        "neural_synthesis": synth,
    }

@app.post("/api/v1/umwelt/transduce-video")
async def transduce_video(
    video: UploadFile = File(...),
    user_text: str = Form("", max_length=500),
    lang: str = Form("it"),
    user: dict = Depends(required_user),
):
    try:
        sub = await asyncio.to_thread(_get_subscription, user["id"])
    except SupabaseError as e:
        print(f"[BioDog] video sub: {e}")
        raise api_error(503, "service_unavailable", "Servizio temporaneamente non disponibile. Riprova tra poco.")
    if _tier_from_subscription(sub) != "pro":
        raise api_error(402, "pro_required", "La video-analisi è inclusa in Vision PRO.")

    mime_type = (video.content_type or "").lower()
    if mime_type not in ALLOWED_VIDEO_MIME:
        raise api_error(415, "unsupported_video", "Formato video non supportato (usa MP4, MOV o WebM).")

    video_bytes = await video.read(MAX_VIDEO_BYTES + 1)
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise api_error(413, "video_too_large", "Video troppo pesante (max 25MB).")
    if not video_bytes:
        raise api_error(400, "empty_video", "Il video è vuoto.")

    target_lang, lang_directive = _lang_directive(lang)
    user_prompt = f"{SYSTEM_PROMPT}\n\nTARGET LANGUAGE: {lang_directive}\n\nAnalizza i fotogrammi di questo video per decodificare il comportamento del cane."
    if user_text:
        user_prompt += f"\nContesto / Context: '{user_text}'"
    synth, api_err = await asyncio.to_thread(_call_gemini_api_video, GEMINI_API_KEY, user_prompt, video_bytes, mime_type)
    if not synth:
        print(f"[BioDog] Gemini video fallito: {api_err}")
        raise api_error(502, "engine_unavailable", "Impossibile analizzare il video in questo momento. Riprova tra poco.")
    return {"status": "success", "engine": f"{ACTIVE_MODEL}-vision", "lang": target_lang, "locked": False, "neural_synthesis": synth}

# ==================== STRIPE ====================

def _price_for_tier(tier: str) -> str:
    return {"premium": STRIPE_PRICE_ID_PREMIUM, "standard": STRIPE_PRICE_ID_STANDARD, "pro": STRIPE_PRICE_ID_PRO}.get(tier, "")

def _tier_for_price(price_id: str) -> Optional[str]:
    for tier in PAID_TIERS:
        if price_id and price_id == _price_for_tier(tier):
            return tier
    return None

@app.post("/api/v1/stripe/create-checkout-session")
async def create_checkout_session(req: CreateCheckoutRequest, user: dict = Depends(required_user)):
    if not stripe or not STRIPE_SECRET_KEY:
        raise api_error(500, "stripe_not_configured", "Stripe non configurato.")

    price_to_use = _price_for_tier(req.tier)
    if not price_to_use:
        # Mai addebitare il prezzo di un altro piano: meglio un errore esplicito.
        raise api_error(500, "price_not_configured", f"Prezzo Stripe per il piano '{req.tier}' non configurato.")

    try:
        sub = await asyncio.to_thread(_get_subscription, user["id"])
    except SupabaseError as e:
        print(f"[BioDog] checkout sub: {e}")
        raise api_error(503, "service_unavailable", "Servizio temporaneamente non disponibile. Riprova tra poco.")
    if _tier_from_subscription(sub) != "free":
        raise api_error(409, "already_subscribed", "Hai già un abbonamento attivo: puoi cambiarlo da 'Abbonamento'.")

    metadata = {"user_id": user["id"], "plan_tier": req.tier}
    params = dict(
        client_reference_id=user["id"],
        metadata=metadata,
        subscription_data={"metadata": metadata},
        line_items=[{"price": price_to_use, "quantity": 1}],
        mode="subscription",
        success_url=f"{FRONTEND_URL}/?payment=success",
        cancel_url=f"{FRONTEND_URL}/",
    )
    if sub and sub.get("stripe_customer_id"):
        params["customer"] = sub["stripe_customer_id"]
    elif user["email"]:
        params["customer_email"] = user["email"]

    try:
        session = await asyncio.to_thread(lambda: stripe.checkout.Session.create(**params))
        return {"checkout_url": session.url}
    except Exception as e:
        print(f"[BioDog] Checkout Stripe fallito: {e}")
        raise api_error(400, "checkout_failed", "Impossibile avviare il pagamento. Riprova tra poco.")

@app.post("/api/v1/stripe/create-portal-session")
async def create_portal_session(user: dict = Depends(required_user)):
    if not stripe or not STRIPE_SECRET_KEY:
        raise api_error(500, "stripe_not_configured", "Stripe non configurato.")
    try:
        sub = await asyncio.to_thread(_get_subscription, user["id"])
    except SupabaseError as e:
        print(f"[BioDog] portal sub: {e}")
        raise api_error(503, "service_unavailable", "Servizio temporaneamente non disponibile. Riprova tra poco.")
    customer_id = (sub or {}).get("stripe_customer_id")
    if not customer_id:
        raise api_error(404, "no_customer", "Nessun abbonamento Stripe collegato a questo account.")
    try:
        session = await asyncio.to_thread(
            lambda: stripe.billing_portal.Session.create(customer=customer_id, return_url=f"{FRONTEND_URL}/")
        )
        return {"portal_url": session.url}
    except Exception as e:
        print(f"[BioDog] Portale Stripe fallito: {e}")
        raise api_error(400, "portal_failed", "Impossibile aprire la gestione abbonamento. Riprova tra poco.")

def _handle_checkout_completed(data_obj: dict) -> None:
    metadata = data_obj.get("metadata") or {}
    customer_details = data_obj.get("customer_details") or {}

    user_id = data_obj.get("client_reference_id") or metadata.get("user_id")
    plan_tier = metadata.get("plan_tier")
    if plan_tier not in PAID_TIERS:
        plan_tier = "premium"
    if not user_id:
        print("[BioDog Webhook] checkout.session.completed senza user_id: ignorato.")
        return

    _upsert_subscription(user_id, {
        "email": customer_details.get("email") or data_obj.get("customer_email") or "",
        "stripe_customer_id": data_obj.get("customer") or "",
        "stripe_subscription_id": data_obj.get("subscription") or "",
        "is_active": True,
        "plan_tier": plan_tier,
    })
    print(f"[BioDog Webhook] Abbonamento attivato: user_id={user_id}, tier={plan_tier}")

def _handle_subscription_change(data_obj: dict, deleted: bool) -> None:
    sub_id = data_obj.get("id") or ""
    status = data_obj.get("status") or ""
    is_active = (not deleted) and status in ACTIVE_SUB_STATUSES

    fields = {"is_active": is_active}
    items = ((data_obj.get("items") or {}).get("data") or [])
    price_id = ((items[0].get("price") or {}).get("id") if items else "") or ""
    tier = _tier_for_price(price_id)
    if tier:
        fields["plan_tier"] = tier

    updated = _update_subscription_by_stripe_id(sub_id, fields) if sub_id else []
    if not updated:
        user_id = (data_obj.get("metadata") or {}).get("user_id")
        if user_id:
            _upsert_subscription(user_id, {
                **fields,
                "stripe_customer_id": data_obj.get("customer") or "",
                "stripe_subscription_id": sub_id,
            })
        else:
            print(f"[BioDog Webhook] Nessuna riga per subscription {sub_id}: ignorato.")
            return
    print(f"[BioDog Webhook] Subscription {sub_id}: status={status}, deleted={deleted}, attivo={is_active}")

@app.post("/api/v1/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    if not stripe or not STRIPE_WEBHOOK_SECRET:
        print("[BioDog Webhook] Errore: libreria Stripe o STRIPE_WEBHOOK_SECRET non presenti.")
        raise HTTPException(status_code=500, detail="Webhook non configurato")

    payload = await request.body()

    try:
        # Verifica della firma crittografica con la chiave segreta Stripe
        stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        print(f"[BioDog Webhook] Errore verifica firma HMAC: {e}")
        raise HTTPException(status_code=400, detail="Webhook signature verification failed")

    try:
        # Decodifica JSON standard del payload verificato per evitare incompatibilità di versione dell'SDK
        event_dict = json.loads(payload.decode("utf-8"))
        event_type = event_dict.get("type", "")
        data_obj = (event_dict.get("data") or {}).get("object") or {}

        print(f"[BioDog Webhook] Evento verificato ricevuto: {event_type}")

        if event_type == "checkout.session.completed":
            await asyncio.to_thread(_handle_checkout_completed, data_obj)
        elif event_type in ("customer.subscription.created", "customer.subscription.updated"):
            await asyncio.to_thread(_handle_subscription_change, data_obj, False)
        elif event_type == "customer.subscription.deleted":
            await asyncio.to_thread(_handle_subscription_change, data_obj, True)

        return {"status": "success"}
    except Exception as e:
        # Un 500 fa ritentare l'evento a Stripe: meglio di un abbonamento non registrato.
        traceback.print_exc()
        print(f"[BioDog Webhook] Errore interno durante l'elaborazione dell'evento: {e}")
        raise HTTPException(status_code=500, detail="Errore interno webhook")
