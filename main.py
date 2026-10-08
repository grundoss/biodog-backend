import asyncio
import json
import os
import time
import base64
from typing import Optional, Tuple
import urllib.error
import urllib.request
from fastapi import FastAPI, Request, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Libreria Stripe ufficiale
try:
    import stripe
except ImportError:
    stripe = None

app = FastAPI(title="BioDog.io Neural Engine", version="3.9")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== VARIABILI D'AMBIENTE ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'").replace("[", "").replace("]", "")
ACTIVE_MODEL = "gemini-3.8-flash"

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://itjfyjyzaornkintpefd.supabase.co").strip().rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

if stripe and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# ==================== SCHEMI DATI PYDANTIC ====================
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
Trasduci il comportamento del cane descritto (o mostrato nel video) dall'umano nella prospettiva etologica, percettiva e prossemica del cane (Umwelt di Jakob von Uexküll e distanze di Hediger).

REGOLE CRITICHE (ANTI-ANTROPOMORFISMO E NEUROBIOLOGIA):
1. DIVIETO ASSOLUTO di attribuire concetti morali umani: dispetto, vendetta, senso di colpa, prevaricazione etica o dominio gerarchico alfa.
2. Radica sempre il comportamento nei 7 circuiti emotivi primari di Jaak Panksepp: SEEKING, RAGE, FEAR, PANIC/GRIEF, PLAY, CARE, LUST.
3. Decodifica l'esperienza in 4 canali sensoriali principali: 
   - Olfatto (molecole, decadimento VOC, feromoni)
   - Vista (movimento, deuteranopia, altezza da terra, campo visivo)
   - Udito (frequenze, prosodia)
   - Tatto & Prossemica (fibre C-tattili, vibrisse, tolleranza manipolativa, costrizione fisica).
4. LETTURA DEL CORPO, PROSSEMICA E DISTANZE (Regole Fisse):
   - Distanze di Hediger: Valuta se l'umano è a Distanza Sociale (sicurezza), di Fuga (stress da avvicinamento) o Critica (messa all'angolo, fear-biting).
   - Geometria dell'Avvicinamento: L'approccio frontale e lo sguardo fisso sono minacce (Simpatico/Attacco-Fuga). Suggerisci sempre l'approccio curvo ("Curving") e il fianco.
   - Referenza Sociale: Se il cane osserva l'umano prima di reagire a uno stimolo o si nasconde dietro di lui, decodifica il bisogno di "Base Sicura" (Sistema di Attaccamento).
   - Ossitocina vs Minaccia: Sguardo morbido reciproco (Mutual Gaze) innalza l'ossitocina; fissare negli occhi innesca minaccia predatoria.
   - Segnali Calmanti (Turid Rugaas): identifica tongue flick, whale eye, sbadigli fuori contesto, curving o freezing come tentativi di de-escalation, non testardaggine.
   - Asimmetria Caudale: coda verso destra = emisfero sinistro (approccio/positivo); coda verso sinistra = emisfero destro (evitamento/timore).
   - Regola dell'Abbraccio: se l'umano abbraccia o costringe le spalle/testa, classificalo come innesco di Stress Simpatico (blocco della fuga). Il tocco laterale sul petto invece innesca il Parasimpatico.
   - Variabile Dolore (OA): se c'è un rifiuto improvviso al tocco o aggressività improvvisa, includi l'ipotesi di allodinia/iperalgesia (es. osteoartrite).
5. Fornisci indicazioni precise sulla mimica corporea e sul tono vocale che l'umano deve assumere per la de-escalation spaziale ed emotiva.
6. Genera ESCLUSIVAMENTE un JSON valido (senza testo introduttivo o markdown) con questa struttura esatta:
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
    "touch": "Cosa percepiscono i recettori tattili e percezione dello spazio vitale (Distanze di Hediger)"
  },
  "human_body_language": {
    "voice": "Tono di voce raccomandato",
    "posture": "Postura corporea (es. Fianco a 45°, approccio a curva, evitare sguardi fissi)"
  },
  "explanation": "spiegazione etologica e prossemica (riferimenti a stress, Hediger, referenza sociale o segnali calmanti)",
  "steps": ["passo pratico 1 da fare subito per modulare lo spazio", "passo pratico 2", "passo pratico 3"],
  "forbidden": ["errore tipico 1 da evitare (es. avvicinamento frontale diretto)", "errore tipico 2"]
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
    if ACTIVE_MODEL: return ACTIVE_MODEL
    ACTIVE_MODEL = "gemini-3.8-flash"
    return ACTIVE_MODEL

# ==================== CHIAMATE GEMINI API (TESTO E VIDEO) ====================

def _call_gemini_api(api_key: str, full_prompt: str) -> Tuple[Optional[dict], Optional[str], str]:
    if not api_key: return None, "Chiave API mancante", "none"
    model = _find_live_model(api_key)
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "BioDog/3.9"}, method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status == 200:
                body = json.loads(response.read().decode("utf-8"))
                text = body["candidates"][0]["content"]["parts"][0].get("text", "")
                return _extract_clean_json(text), None, model
    except Exception as e:
        return None, f"Errore API: {str(e)}", "failed"

def _call_gemini_api_video(api_key: str, full_prompt: str, video_bytes: bytes, mime_type: str) -> Tuple[Optional[dict], Optional[str], str]:
    if not api_key: return None, "Chiave API mancante", "none"
    model = _find_live_model(api_key)
    
    b64_data = base64.b64encode(video_bytes).decode("utf-8")
    
    if mime_type not in ["video/mp4", "video/quicktime", "video/webm", "video/x-m4v"]:
        mime_type = "video/mp4"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [
            {"text": full_prompt},
            {"inline_data": {"mimeType": mime_type, "data": b64_data}}
        ]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "BioDog/3.9"}, method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status == 200:
                body = json.loads(response.read().decode("utf-8"))
                text = body["candidates"][0]["content"]["parts"][0].get("text", "")
                return _extract_clean_json(text), None, model
    except Exception as e:
        return None, f"Errore API Video: {str(e)}", "failed"

# ==================== SCRITTURA DIRETTA SU SUPABASE ====================

def _update_supabase_subscription(user_id: str, email: str, customer_id: str, sub_id: str, is_active: bool):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[BioDog] Credenziali Supabase mancanti per aggiornare l'abbonamento.")
        return False

    url = f"{SUPABASE_URL}/rest/v1/user_subscriptions"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    payload = {
        "user_id": user_id,
        "email": email,
        "stripe_customer_id": customer_id,
        "stripe_subscription_id": sub_id,
        "is_active": is_active,
        "updated_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[BioDog] Supabase aggiornato con successo per {email} (is_active={is_active})")
            return resp.status in [200, 201]
    except Exception as e:
        print(f"[BioDog] Errore aggiornamento Supabase: {e}")
        return False

# ==================== ENDPOINT RADICE & STATO ====================

@app.get("/")
async def root():
    return {"status": "BioDog Neural Engine Online", "model": ACTIVE_MODEL}

# ==================== ENDPOINT 1: TRADUZIONE TESTO (USATA DA INDEX E TEST) ====================

@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    bio = SensoryEngine.compute(req)
    user_prompt = f"""{SYSTEM_PROMPT}\n\nComportamento osservato: "{req.user_text}"\nProfilo biologico:\n- Cranio: {req.snout} (Turbinati olfattivi: {bio['turbinates_cm2']} cm²)\n- Campo Visivo: {bio['fov_degrees']}° (Acuità: {bio['acuity_cpd']} cpd)\n- Occhi da terra: {bio['eye_height_cm']} cm\n- Coda: {bio['tail_bias']}\n- Orecchie: {bio['ear_mobility']}"""

    synth, api_err, used_model = await asyncio.to_thread(_call_gemini_api, GEMINI_API_KEY, user_prompt)
    
    if not synth:
        synth = {
            "situation_title": "Elaborazione di sicurezza",
            "panksepp": "CARE",
            "panksepp_label": "Ricongiungimento Affiliativo",
            "arousal": 50,
            "valence": 0,
            "thought": "Sto cercando di elaborare i segnali dell'ambiente...",
            "sensory": {
                "smell": "Percezione ordinaria delle molecole d'aria.",
                "sight": "Messa a fuoco frontale bilanciata.",
                "hearing": "Percezione dei rumori di fondo ambientali.",
                "touch": "Percezione tattile e recettoriale neutra."
            },
            "human_body_language": {
                "voice": "Tono neutro, calmo e rassicurante.",
                "posture": "Postura morbida ed eretta senza incombenza fisica."
            },
            "explanation": f"Elaborazione di sicurezza (Dettaglio API: {api_err})",
            "steps": ["Osserva la postura generale.", "Mantieni calma e spazio vitale."],
            "forbidden": ["Non forzare il contatto.", "Evita movimenti bruschi."]
        }

    return {"status": "success", "engine": used_model, "neural_synthesis": synth}

# ==================== ENDPOINT 2: TRADUZIONE VIDEO ====================

@app.post("/api/v1/umwelt/transduce-video")
async def transduce_video(video: UploadFile = File(...), user_text: str = Form("")):
    video_bytes = await video.read()
    
    # Limite di sicurezza: max 25 MB
    if len(video_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video troppo pesante (max 25MB). Carica una clip di massimo 10 secondi.")

    user_prompt = f"{SYSTEM_PROMPT}\n\nAnalizza minuziosamente i fotogrammi e l'audio di questo video per decodificare il comportamento del cane. Ricerca attivamente la presenza di segnali calmanti (sbadigli, tongue flick, rotazione testa, freezing), tensioni muscolari, e il posizionamento della coda e delle orecchie in relazione allo spazio umano."
    if user_text:
        user_prompt += f"\nContesto aggiunto dall'umano: '{user_text}'"

    synth, api_err, used_model = await asyncio.to_thread(_call_gemini_api_video, GEMINI_API_KEY, user_prompt, video_bytes, video.content_type)
    
    if not synth:
        synth = {
            "situation_title": "Elaborazione Video",
            "panksepp": "CARE",
            "panksepp_label": "Analisi Posturale",
            "arousal": 50,
            "valence": 0,
            "thought": "Sto osservando e analizzando il movimento...",
            "sensory": {
                "smell": "Scambio chimico attivo durante l'azione.",
                "sight": "Assetto visivo orientato allo stimolo ripreso nel video.",
                "hearing": "Frequenze audio registrate nella clip.",
                "touch": "Interazione somatosensoriale o posturale in corso."
            },
            "human_body_language": {
                "voice": "Tono distensivo e pacato.",
                "posture": "Fianco a 45 gradi, non invadere lo spazio vitale del cane."
            },
            "explanation": f"Analisi video completata con modello di fallback (Dettaglio API: {api_err})",
            "steps": ["Valuta la reazione del cane.", "Offri spazio di de-escalation."],
            "forbidden": ["Non bloccare i movimenti del cane.", "Evita urla o rimproveri concitati."]
        }

    return {"status": "success", "engine": f"{used_model}-vision", "neural_synthesis": synth}

# ==================== ENDPOINT 3: STRIPE CREATE CHECKOUT SESSION ====================

@app.post("/api/v1/stripe/create-checkout-session")
async def create_checkout_session(req: CreateCheckoutRequest):
    if not stripe or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe non configurato sul server.")
    if not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="ID Prezzo Stripe mancante.")

    try:
        session = stripe.checkout.Session.create(
            customer_email=req.user_email,
            client_reference_id=req.user_id,
            metadata={"user_id": req.user_id},
            line_items=[{
                'price': STRIPE_PRICE_ID,
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://biodog.io/?payment=success&session_id={CHECKOUT_SESSION_ID}',
            cancel_url='https://biodog.io/',
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== ENDPOINT 4: STRIPE WEBHOOK ====================

@app.post("/api/v1/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    if not stripe or not STRIPE_WEBHOOK_SECRET:
        return {"status": "ignored", "reason": "Webhook secret non configurato"}

    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore firma Webhook: {str(e)}")

    event_type = event.get("type", "")
    data_obj = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        user_id = data_obj.get("client_reference_id") or (data_obj.get("metadata") or {}).get("user_id")
        email = data_obj.get("customer_details", {}).get("email") or data_obj.get("customer_email") or ""
        customer_id = data_obj.get("customer", "")
        sub_id = data_obj.get("subscription", "")

        if user_id:
            _update_supabase_subscription(user_id, email, customer_id, sub_id, True)

    elif event_type == "invoice.payment_succeeded":
        sub_id = data_obj.get("subscription", "")
        customer_id = data_obj.get("customer", "")
        email = data_obj.get("customer_email", "")

    elif event_type == "customer.subscription.deleted":
        sub_id = data_obj.get("id", "")
        customer_id = data_obj.get("customer", "")

    return {"status": "success"}
