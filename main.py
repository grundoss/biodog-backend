import asyncio
import json
import os
import time
import traceback
from typing import Optional, Tuple
import urllib.error
import urllib.request
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Libreria Stripe ufficiale
try:
    import stripe
except ImportError:
    stripe = None

app = FastAPI(title="BioDog.io Neural Engine", version="3.8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== VARIABILI D'AMBIENTE ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'").replace("[", "").replace("]", "")
ACTIVE_MODEL = None

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://itjfyjyzaornkintpefd.supabase.co").strip().rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

if stripe and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# ==================== SCHEMI DATI ====================
class TransductionRequest(BaseModel):
    user_text: str = Field(..., min_length=2, max_length=500)
    snout: str = Field(default="normal")
    ears: str = Field(default="prick")
    size: str = Field(default="medium")
    tail: str = Field(default="long")
    observed_time_hours: Optional[float] = 0.0

class CreateCheckoutRequest(BaseModel):
    user_id: str
    user_email: str

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

SYSTEM_PROMPT = """Sei il motore di intelligenza artificiale biologica BioDog.io.
Trasduci il comportamento del cane descritto dall'umano nella prospettiva etologica e percettiva del cane (Umwelt di Jakob von Uexküll).

REGOLE CRITICHE (ANTI-ANTROPOMORFISMO DPO):
1. DIVIETO ASSOLUTO di attribuire concetti morali umani: dispetto, vendetta, senso di colpa, prevaricazione etica o dominio gerarchico alfa.
2. Radica sempre il comportamento nei 7 circuiti emotivi primari di Jaak Panksepp: SEEKING, RAGE, FEAR, PANIC/GRIEF, PLAY, CARE, LUST.
3. Decodifica l'esperienza nei tre canali sensoriali principali del cane: Olfatto (molecole, decadimento VOC), Vista (movimento, deuteranopia, altezza da terra), Udito (frequenze, prosodia).
4. Fornisci indicazioni precise sulla mimica corporea e sul tono vocale che l'umano deve assumere.
5. Genera ESCLUSIVAMENTE un JSON valido (senza testo introduttivo o markdown) con questa struttura esatta:
{
  "situation_title": "Titolo etologico breve",
  "panksepp": "CARE | RAGE | FEAR | PANIC/GRIEF | PLAY | SEEKING | LUST",
  "panksepp_label": "Nome del circuito ed etichetta emotiva (es. CARE / Ricongiungimento Affiliativo)",
  "arousal": numero intero da 0 a 100,
  "valence": numero intero da -50 a +50,
  "thought": "pensiero del cane in prima persona: rapido, sensoriale, privo di morale umana",
  "sensory": {
    "smell": "Cosa percepisce il tartufo in questo momento (es. molecole odorose, assenza di feromoni)",
    "sight": "Cosa vede dagli occhi (prospettiva bassa, movimento rapido, sagoma incombente)",
    "hearing": "Cosa sente con le orecchie (toni acuti, frequenze gravi, passi, rumori improvvisi)"
  },
  "human_body_language": {
    "voice": "Tono di voce raccomandato per l'umano (es. Silenzio assoluto, tono basso e distensivo)",
    "posture": "Postura corporea raccomandata (es. Fianco a 45°, spalle rilassate, evitare contatto visivo diretto)"
  },
  "explanation": "spiegazione etologica chiara e accessibile per il proprietario",
  "steps": ["passo pratico 1 da fare subito", "passo pratico 2", "passo pratico 3"],
  "forbidden": ["errore tipico 1 da evitare assolutamente", "errore tipico 2"]
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

def _find_live_model(api_key: str) -> str:
    global ACTIVE_MODEL
    if ACTIVE_MODEL:
        return ACTIVE_MODEL

    pt = "https"
    dm = "generativelanguage.googleapis.com"
    list_url = f"{pt}://{dm}/v1beta/models?key={api_key}"

    try:
        req = urllib.request.Request(list_url, headers={"User-Agent": "BioDog/3.8"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            available = [
                m["name"].replace("models/", "")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            
            # Priorità esclusiva alle nuove versioni 3.8 indicate dall'errore
            for priority in ["3.8-flash", "3.5-flash", "1.5-flash"]:
                for m in available:
                    if priority in m.lower():
                        ACTIVE_MODEL = m
                        return ACTIVE_MODEL
    except Exception as e:
        print(f"ListModels error: {e}")

    ACTIVE_MODEL = "gemini-3.8-flash"
    return ACTIVE_MODEL

def _call_gemini_api(api_key: str, full_prompt: str) -> Tuple[Optional[dict], Optional[str], str]:
    if not api_key:
        return None, "Chiave API mancante", "none"

    discovered = _find_live_model(api_key)
    
    # Pool purificato centrato sulla versione 3.8
    models_pool = [
        "gemini-3.8-flash",
        discovered
    ]
    candidate_models = list(dict.fromkeys([m for m in models_pool if m]))

    last_err = "Nessun modello valido ha risposto"
    pt = "https"
    dm = "generativelanguage.googleapis.com"

    for m in candidate_models:
        url = f"{pt}://{dm}/v1beta/models/{m}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "BioDog/3.8"},
            method="POST"
        )
        
        for attempt in range(2):
            try:
                # TIMEOUT ESTESO A 60 SECONDI PER EVITARE TIMEOUT DI RETE
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status == 200:
                        body = json.loads(response.read().decode("utf-8"))
                        parts = body["candidates"][0]["content"]["parts"]
                        text = ""
                        for p in reversed(parts):
                            if "text" in p and "{" in p["text"]:
                                text = p["text"]
                                break
                        if not text and parts:
                            text = parts[-1].get("text", "")
                        
                        global ACTIVE_MODEL
                        ACTIVE_MODEL = m
                        return _extract_clean_json(text), None, m
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8")
                last_err = f"{m} HTTP {he.code}: {err_body}"
                
                if he.code in [400, 403, 404]:
                    break 
                
                if he.code in [503, 429]:
                    time.sleep(1.0)
                    continue
                break
            except TimeoutError:
                last_err = f"{m} err: Timeout superato (60s)"
                break
            except Exception as e:
                last_err = f"{m} err: {str(e)}"
                break

    return None, last_err, "failed"

# ==================== SUPABASE HELPER PER STRIPE ====================
def _update_supabase_subscription(user_id: str, email: str, customer_id: str, sub_id: str, is_active: bool) -> bool:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[Supabase Warning] Chiavi non configurate!")
        return False

    url = f"{SUPABASE_URL}/rest/v1/user_subscriptions?on_conflict=user_id"
    payload = [{
        "user_id": user_id,
        "email": email or "",
        "stripe_customer_id": customer_id or "",
        "stripe_subscription_id": sub_id or "",
        "is_active": is_active
    }]

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[Supabase] Abbonamento OK per {email}: active={is_active}")
            return resp.status in [200, 201]
    except Exception as e:
        print(f"[Supabase Error]: {e}")
        return False

def _deactivate_subscription_by_stripe_id(sub_id: str, customer_id: str = "") -> bool:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False

    url = f"{SUPABASE_URL}/rest/v1/user_subscriptions?stripe_subscription_id=eq.{sub_id}"
    payload = {"is_active": False}
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json"
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="PATCH")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True
    except Exception:
        return False

# ==================== ENDPOINT STRIPE ====================
@app.post("/api/v1/stripe/create-checkout-session")
async def create_checkout_session(req: CreateCheckoutRequest):
    if not stripe or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe non configurato.")
    
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        session = stripe.checkout.Session.create(
            customer_email=req.user_email,
            client_reference_id=req.user_id,
            metadata={"user_id": req.user_id, "user_email": req.user_email},
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            mode="subscription",
            success_url="https://biodog.io/?payment=success&session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://biodog.io/?payment=cancelled",
            allow_promotion_codes=True
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    try:
        payload = await request.body()
        if not stripe:
            raise HTTPException(status_code=500, detail="Stripe non disponibile.")

        event = None
        if STRIPE_WEBHOOK_SECRET and stripe_signature:
            try:
                event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Firma invalida: {str(e)}")
        else:
            event = json.loads(payload.decode("utf-8"))

        event_dict = event.to_dict() if hasattr(event, "to_dict") else (event if isinstance(event, dict) else json.loads(payload.decode("utf-8")))
        event_type = event_dict.get("type", "")
        data_obj = event_dict.get("data", {}).get("object", {})

        if event_type == "checkout.session.completed":
            metadata = data_obj.get("metadata", {})
            user_id = data_obj.get("client_reference_id") or metadata.get("user_id")
            email = data_obj.get("customer_details", {}).get("email") or metadata.get("user_email") or ""
            customer_id = data_obj.get("customer", "")
            sub_id = data_obj.get("subscription", "")

            if user_id:
                _update_supabase_subscription(str(user_id), email, str(customer_id), str(sub_id), True)

        elif event_type in ["customer.subscription.deleted", "customer.subscription.paused"]:
            sub_id = data_obj.get("id", "")
            customer_id = data_obj.get("customer", "")
            if sub_id:
                _deactivate_subscription_by_stripe_id(str(sub_id), str(customer_id))

        return {"status": "success", "processed_event": event_type}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENDPOINT PRINCIPALE TRADUZIONE ====================
@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    bio = SensoryEngine.compute(req)

    user_prompt = f"""{SYSTEM_PROMPT}\n\nComportamento osservato: "{req.user_text}"\nProfilo biologico:\n- Cranio: {req.snout} (Turbinati olfattivi: {bio['turbinates_cm2']} cm²)\n- Campo Visivo: {bio['fov_degrees']}° (Acuità: {bio['acuity_cpd']} cpd)\n- Occhi da terra: {bio['eye_height_cm']} cm\n- Coda: {bio['tail_bias']}\n- Orecchie: {bio['ear_mobility']}"""

    synth, api_err, used_model = await asyncio.to_thread(_call_gemini_api, GEMINI_API_KEY, user_prompt)

    if synth:
        engine_used = f"neural_{used_model}"
    else:
        engine_used = "fallback_local"
        synth = {
            "situation_title": "Valutazione Etologica",
            "panksepp": "SEEKING",
            "panksepp_label": "SEEKING / Analisi Ambientale",
            "arousal": 50,
            "valence": 10,
            "thought": f"Analizzo la situazione '{req.user_text}' con i miei recettori sensoriali.",
            "sensory": {
                "smell": "Scansione olfattiva di routine dell'ambiente circostante.",
                "sight": "Messa a fuoco frontale bilanciata.",
                "hearing": "Percezione dei rumori di fondo ambientali."
            },
            "human_body_language": {
                "voice": "Tono neutro, calmo e rassicurante.",
                "posture": "Postura morbida ed eretta senza incombenza fisica."
            },
            "explanation": f"Elaborazione di sicurezza (Dettaglio API: {api_err}).",
            "steps": ["Osserva la postura generale.", "Mantieni calma e spazio vitale."],
            "forbidden": ["Evita reazioni improvvise o rimproveri."]
        }

    return {
        "status": "success",
        "engine": engine_used,
        "diagnostic_error": api_err,
        "morphology_profile": {
            "snout": req.snout,
            "ears": req.ears,
            "size": req.size,
            "tail": req.tail
        },
        "sensory_telemetry": bio,
        "neural_synthesis": synth
    }

@app.get("/")
async def root():
    return {
        "status": "BioDog Neural Engine Online",
        "modello_rilevato": ACTIVE_MODEL or "Attesa prima chiamata",
        "stripe_abilitato": bool(stripe and STRIPE_SECRET_KEY and STRIPE_PRICE_ID)
    }
