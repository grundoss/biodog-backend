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

app = FastAPI(title="BioDog.io Neural Engine", version="4.1")

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
# Nuove variabili per i due piani
STRIPE_PRICE_ID_PREMIUM = os.getenv("STRIPE_PRICE_ID_PREMIUM", os.getenv("STRIPE_PRICE_ID", "")).strip()
STRIPE_PRICE_ID_PRO = os.getenv("STRIPE_PRICE_ID_PRO", "").strip()
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
    tier: str = "premium" # Accetta "premium" o "pro"

class PortalRequest(BaseModel):
    customer_id: str

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
7. Genera ESCLUSIVAMENTE un JSON valido (senza testo introduttivo o markdown) con questa struttura esatta:
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

def _find_live_model(api_key: str) -> str:
    global ACTIVE_MODEL
    if ACTIVE_MODEL: return ACTIVE_MODEL
    ACTIVE_MODEL = "gemini-3.8-flash"
    return ACTIVE_MODEL

# ==================== CHIAMATE GEMINI API ====================

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
        headers={"Content-Type": "application/json", "User-Agent": "BioDog/4.1"}, method="POST"
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
        headers={"Content-Type": "application/json", "User-Agent": "BioDog/4.1"}, method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status == 200:
                body = json.loads(response.read().decode("utf-8"))
                text = body["candidates"][0]["content"]["parts"][0].get("text", "")
                return _extract_clean_json(text), None, model
    except Exception as e:
        return None, f"Errore API Video: {str(e)}", "failed"

# ==================== SCRITTURA SU SUPABASE ====================

def _update_supabase_subscription(user_id: str, email: str, customer_id: str, sub_id: str, is_active: bool, plan_tier: str):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[BioDog] Credenziali Supabase mancanti.")
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
        "plan_tier": plan_tier,
        "updated_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[BioDog] Supabase aggiornato per {email} (tier={plan_tier})")
            return resp.status in [200, 201]
    except Exception as e:
        print(f"[BioDog] Errore aggiornamento Supabase: {e}")
        return False

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {"status": "BioDog Neural Engine Online", "model": ACTIVE_MODEL}

@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    bio = SensoryEngine.compute(req)
    user_prompt = f"""{SYSTEM_PROMPT}\n\nComportamento osservato: "{req.user_text}"\nProfilo biologico:\n- Cranio: {req.snout} (Turbinati olfattivi: {bio['turbinates_cm2']} cm²)\n- Campo Visivo: {bio['fov_degrees']}° (Acuità: {bio['acuity_cpd']} cpd)\n- Occhi da terra: {bio['eye_height_cm']} cm\n- Coda: {bio['tail_bias']}\n- Orecchie: {bio['ear_mobility']}"""
    synth, api_err, used_model = await asyncio.to_thread(_call_gemini_api, GEMINI_API_KEY, user_prompt)
    if not synth:
        synth = {
            "situation_title": "Elaborazione di sicurezza", "panksepp": "CARE", "panksepp_label": "Ricongiungimento",
            "arousal": 50, "valence": 0, "thought": "Sto cercando di elaborare i segnali dell'ambiente...",
            "sensory": {"smell": "-", "sight": "-", "hearing": "-", "touch": "-"},
            "human_body_language": {"voice": "-", "posture": "-"},
            "explanation": f"Errore: {api_err}", "steps": ["-"], "forbidden": ["-"]
        }
    return {"status": "success", "engine": used_model, "neural_synthesis": synth}

@app.post("/api/v1/umwelt/transduce-video")
async def transduce_video(video: UploadFile = File(...), user_text: str = Form("")):
    video_bytes = await video.read()
    if len(video_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video troppo pesante (max 25MB).")
    user_prompt = f"{SYSTEM_PROMPT}\n\nAnalizza i fotogrammi di questo video per decodificare il comportamento del cane."
    if user_text: user_prompt += f"\nContesto: '{user_text}'"
    synth, api_err, used_model = await asyncio.to_thread(_call_gemini_api_video, GEMINI_API_KEY, user_prompt, video_bytes, video.content_type)
    if not synth: raise HTTPException(status_code=500, detail=f"Errore API Video: {api_err}")
    return {"status": "success", "engine": f"{used_model}-vision", "neural_synthesis": synth}

# ==================== STRIPE ====================

@app.post("/api/v1/stripe/create-checkout-session")
async def create_checkout_session(req: CreateCheckoutRequest):
    if not stripe or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe non configurato.")
    
    # Assegna il prezzo in base al piano scelto (premium o pro)
    if req.tier == "pro":
        if not STRIPE_PRICE_ID_PRO:
            raise HTTPException(status_code=500, detail="ID Prezzo PRO mancante su Render.")
        price_to_use = STRIPE_PRICE_ID_PRO
    else:
        if not STRIPE_PRICE_ID_PREMIUM:
            raise HTTPException(status_code=500, detail="ID Prezzo PREMIUM mancante su Render.")
        price_to_use = STRIPE_PRICE_ID_PREMIUM

    try:
        session = stripe.checkout.Session.create(
            customer_email=req.user_email,
            client_reference_id=req.user_id,
            metadata={"user_id": req.user_id, "plan_tier": req.tier},
            line_items=[{'price': price_to_use, 'quantity': 1}],
            mode='subscription',
            success_url='https://biodog.io/?payment=success',
            cancel_url='https://biodog.io/',
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/stripe/create-portal-session")
async def create_portal_session(req: PortalRequest):
    if not stripe or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe non configurato.")
    try:
        session = stripe.billing_portal.Session.create(customer=req.customer_id, return_url='https://biodog.io/')
        return {"portal_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    if not stripe or not STRIPE_WEBHOOK_SECRET: return {"status": "ignored"}
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))

    event_type = event.get("type", "")
    data_obj = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        user_id = data_obj.get("client_reference_id") or (data_obj.get("metadata") or {}).get("user_id")
        plan_tier = (data_obj.get("metadata") or {}).get("plan_tier", "premium")
        email = data_obj.get("customer_details", {}).get("email") or data_obj.get("customer_email") or ""
        customer_id = data_obj.get("customer", "")
        sub_id = data_obj.get("subscription", "")
        if user_id:
            _update_supabase_subscription(user_id, email, customer_id, sub_id, True, plan_tier)

    return {"status": "success"}
```eof

```html:index.html
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BioDog.io | Traduttore Etologico e Psicologia Canina</title>
    
    <!-- ==================== META TAG SEO ESSENZIALI ==================== -->
    <meta name="description" content="Entra nella mente del tuo cane. BioDog.io decodifica comportamenti, emozioni primarie e percezioni sensoriali (olfatto, vista, udito, tatto) su base scientifica.">
    <meta name="keywords" content="comportamento cane, psicologia canina, traduttore cane, etologia canina, capire il cane, bio dog, linguaggio cani">
    <meta name="robots" content="index, follow">
    <link rel="canonical" href="https://biodog.io/">

    <!-- ==================== ANTEPRIME SOCIAL ==================== -->
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://biodog.io/">
    <meta property="og:title" content="BioDog.io | Il Traduttore Etologico del tuo Cane">
    <meta property="og:description" content="Descrivi qualsiasi comportamento: scopri cosa prova, cosa fiuta e come porsi con voce e corpo per capirsi subito.">
    <meta property="og:image" content="https://biodog.io/1239.png">
    <meta property="og:locale" content="it_IT">

    <!-- Supabase JS Client v2 -->
    <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>

    <!-- Tailwind & Font -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        body { 
            background-color: #f8fafc;
            color: #0f172a;
            font-family: 'Inter', sans-serif; 
            background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
            background-size: 20px 20px;
        }
        h1, h2, h3, h4, h5, .font-heading { font-family: 'Nunito', sans-serif; }
        
        .fade-in { animation: fadeIn 0.35s ease-out forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        
        .search-container {
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.02);
        }
        .search-container:focus-within {
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15), 0 10px 25px -5px rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
            border-color: #34d399;
        }

        /* Effetto sfocatura elegante per gli ospiti */
        .blur-content {
            filter: blur(6px);
            user-select: none;
            pointer-events: none;
            opacity: 0.6;
        }

        .custom-scroll::-webkit-scrollbar { width: 5px; }
        .custom-scroll::-webkit-scrollbar-track { background: #f1f5f9; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        .custom-scroll::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    </style>
</head>
<body class="min-h-screen flex flex-col selection:bg-emerald-200">

    <!-- Banner Notifica Pagamento Stripe Riuscito -->
    <div id="stripeSuccessBanner" class="hidden bg-gradient-to-r from-emerald-600 to-teal-600 text-white text-xs sm:text-sm font-bold py-2.5 px-4 text-center shadow-sm flex items-center justify-center gap-2 fade-in z-50 relative">
        <span>🎉</span>
        <span>Complimenti! Il tuo abbonamento a <strong>BioDog Premium</strong> è attivo.</span>
        <button onclick="dismissStripeBanner()" class="ml-2 text-white/80 hover:text-white"><i class="fa-solid fa-xmark"></i></button>
    </div>

    <!-- Header con Autenticazione & Info -->
    <header class="p-3 sm:p-5 flex justify-between items-center max-w-5xl mx-auto w-full gap-2">
        <!-- Area Utente -->
        <div id="authHeaderContainer">
            <!-- Tasto Accedi -->
            <button onclick="openAuthModal()" id="loginBtn" class="text-xs sm:text-sm font-bold text-slate-700 bg-white hover:bg-slate-50 px-3.5 sm:px-4 py-2 rounded-full border border-slate-200 shadow-xs transition flex items-center gap-2">
                <i class="fa-solid fa-user text-emerald-600"></i> Accedi / Registrati
            </button>
            
            <!-- Profilo Connesso -->
            <div id="userProfile" class="hidden flex items-center gap-2 sm:gap-2.5">
                <!-- Tasto Diario -->
                <button onclick="openHistoryDrawer()" class="text-xs font-bold text-emerald-800 bg-emerald-50 hover:bg-emerald-100 border border-emerald-300 px-3 sm:px-3.5 py-1.5 rounded-full flex items-center gap-2 transition shadow-xs">
                    <i class="fa-solid fa-book-bookmark text-emerald-600"></i>
                    <span class="hidden xs:inline sm:inline">Diario</span>
                    <span id="historyCountBadge" class="bg-emerald-600 text-white text-[10px] px-1.5 py-0.5 rounded-full font-mono">0</span>
                </button>

                <!-- Gestisci Abbonamento -->
                <button onclick="openStripePortal()" id="btnManageSub" class="hidden text-[11px] sm:text-xs font-bold text-slate-600 bg-white hover:bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-full items-center gap-1.5 shadow-xs transition">
                    <i class="fa-solid fa-gear"></i> <span class="hidden sm:inline">Abbonamento</span>
                </button>

                <!-- Badge Abbonamento / Tasto Upgrade -->
                <div id="premiumBadgeArea">
                    <button onclick="openPaywallModal()" id="btnUpgradeHeader" class="text-[11px] sm:text-xs font-bold text-amber-800 bg-amber-50 hover:bg-amber-100 border border-amber-300 px-3 py-1.5 rounded-full flex items-center gap-1.5 transition shadow-xs">
                        <i class="fa-solid fa-crown text-amber-500"></i>
                        <span>Piani PRO</span>
                    </button>
                    <span id="badgePremiumActive" class="hidden text-[11px] sm:text-xs font-bold text-emerald-800 bg-emerald-100 border border-emerald-300 px-3 py-1.5 rounded-full flex items-center gap-1.5 shadow-xs">
                        <i class="fa-solid fa-crown text-amber-500"></i>
                        <span>Premium Attivo</span>
                    </span>
                </div>

                <!-- Tasto Esci -->
                <button onclick="handleLogout()" title="Esci dall'account" class="text-xs font-bold text-rose-600 bg-white hover:bg-rose-50 border border-rose-200 px-3 py-1.5 rounded-full transition flex items-center gap-1.5 shadow-xs">
                    <i class="fa-solid fa-arrow-right-from-bracket"></i>
                </button>
            </div>
        </div>

        <!-- Info BioDog -->
        <button onclick="openWelcomeModal()" class="text-slate-500 hover:text-emerald-600 transition flex items-center gap-1.5 sm:gap-2 text-xs sm:text-sm font-semibold bg-white hover:bg-emerald-50 px-3.5 sm:px-4 py-2 rounded-full border border-slate-200 shadow-xs shrink-0">
            <i class="fa-solid fa-circle-info"></i> <span class="hidden sm:inline">Cos'è BioDog?</span><span class="sm:hidden">Info</span>
        </button>
    </header>

    <!-- Main Content -->
    <main class="flex-grow flex flex-col items-center justify-start pt-2 sm:pt-4 px-4 pb-16">
        
        <!-- Logo -->
        <div class="mb-4 w-44 sm:w-52 flex justify-center">
            <img src="1239.png" alt="BioDog.io Logo" class="w-full h-auto drop-shadow-sm">
        </div>

        <!-- ==================== PILLOLA ETOLOGICA DEL GIORNO (HOOK) ==================== -->
        <div id="dailyPill" class="hidden w-full max-w-2xl bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100 rounded-2xl p-3 sm:p-4 mb-6 items-start gap-3 shadow-sm fade-in cursor-pointer" onclick="openWelcomeModal()">
            <span class="text-xl sm:text-2xl mt-0.5">💡</span>
            <div>
                <strong class="text-[10px] sm:text-xs text-indigo-900 font-black uppercase tracking-wider block mb-0.5">La pillola del giorno</strong>
                <p id="dailyPillText" class="text-xs sm:text-sm text-indigo-800 font-medium leading-relaxed"></p>
            </div>
        </div>

        <!-- Titolo -->
        <div class="text-center mb-6 max-w-xl px-2">
            <h1 class="text-2xl sm:text-3xl font-black text-slate-800 font-heading tracking-tight leading-tight">
                Entra nella mente del tuo cane.
            </h1>
        </div>

        <!-- Barra di Ricerca -->
        <div class="w-full max-w-2xl relative search-container bg-white rounded-full border border-slate-200">
            <div class="absolute inset-y-0 left-0 pl-6 flex items-center pointer-events-none">
                <i class="fa-solid fa-magnifying-glass text-slate-400 text-lg"></i>
            </div>
            
            <input type="text" id="queryInput" 
                placeholder="Es: Quando torno a casa mi salta addosso..." 
                class="w-full pl-14 pr-36 py-4 sm:py-5 bg-transparent text-slate-800 placeholder-slate-400 text-base sm:text-lg focus:outline-none rounded-full"
                onkeypress="if(event.key === 'Enter') runTranslation()">
                
            <!-- Tasto Traduci -->
            <button onclick="runTranslation()" id="translateBtn" class="absolute inset-y-1.5 right-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold px-6 rounded-2xl transition-all duration-300 text-base flex items-center gap-2 shadow-sm transform hover:scale-[1.02] active:scale-95">
                <span>Traduci</span>
                <i class="fa-solid fa-microchip text-emerald-100/90 text-lg"></i>
            </button>
        </div>

        <!-- Notifiche Dinamiche -->
        <div id="saveToast" class="hidden mt-4 bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold px-4 py-2 rounded-full flex items-center gap-2 fade-in shadow-xs">
            <i class="fa-solid fa-bookmark text-emerald-600"></i>
            <span>Traduzione salvata nel tuo Diario Etologico!</span>
        </div>
        <div id="sundayToast" class="hidden mt-4 bg-amber-50 border border-amber-200 text-amber-800 text-xs font-semibold px-4 py-2 rounded-full flex items-center gap-2 fade-in shadow-xs">
            <i class="fa-solid fa-gift text-amber-600"></i>
            <span>Hai appena usato il tuo <strong>Gettone OMAGGIO della Domenica!</strong></span>
        </div>

        <!-- Indicatore di caricamento -->
        <div id="loadingIndicator" class="hidden mt-10 flex flex-col items-center gap-3">
            <i class="fa-solid fa-circle-notch fa-spin text-emerald-500 text-4xl"></i>
            <p class="text-emerald-600 text-sm font-semibold animate-pulse font-mono bg-emerald-50 px-4 py-1.5 rounded-full border border-emerald-100">Decodifica etologica in corso...</p>
        </div>

        <!-- ==================== SCHEDA RISULTATO ETOLOGICO ==================== -->
        <div id="resultContainer" class="hidden w-full max-w-3xl mt-10 space-y-5 fade-in">
            <!-- 1. TERMOMETRO EMOTIVO -->
            <div class="bg-white p-5 sm:p-6 rounded-3xl border border-slate-200 shadow-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div class="space-y-1">
                    <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest block font-mono">Stato Neuroaffettivo (Panksepp):</span>
                    <div id="resPankseppBadge" class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs sm:text-sm font-bold">
                        <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                        <span id="resPankseppText">CARE / Ricongiungimento</span>
                    </div>
                </div>
                <div class="w-full sm:w-64 space-y-1.5">
                    <div class="flex justify-between items-center text-xs font-bold text-slate-700">
                        <span class="flex items-center gap-1.5"><i class="fa-solid fa-gauge-high text-emerald-600"></i> Agitazione Motoria (Arousal):</span>
                        <span id="resArousalNumber" class="font-mono text-emerald-700">80%</span>
                    </div>
                    <div class="w-full bg-slate-100 h-3 rounded-full overflow-hidden p-0.5 border border-slate-200">
                        <div id="resArousalBar" class="h-full bg-gradient-to-r from-emerald-500 to-amber-500 rounded-full transition-all duration-700" style="width: 80%"></div>
                    </div>
                </div>
            </div>

            <!-- 2. PENSIERO DEL CANE (Umwelt) -->
            <div class="relative bg-white p-6 sm:p-8 rounded-3xl border border-slate-200 shadow-sm">
                <div class="absolute -top-5 left-6 w-11 h-11 bg-emerald-500 border-4 border-white rounded-full flex items-center justify-center text-white text-lg shadow-xs">
                    <i class="fa-solid fa-paw"></i>
                </div>
                <h3 class="text-xs font-black text-slate-400 uppercase tracking-widest mb-3 mt-1 ml-1 font-heading">Nella mente del cane in questo istante:</h3>
                <p id="resThought" class="text-base sm:text-lg text-slate-800 italic font-heading leading-relaxed ml-1 border-l-4 border-emerald-400 pl-4 sm:pl-5"></p>
            </div>

            <!-- SEZIONE SOGGETTA AL BLOCCO "VEDO-NON-VEDO" (BLUR) -->
            <div id="practicalSectionWrapper" class="relative group mt-2">
                <div id="practicalContent" class="space-y-5 transition-all duration-300">
                    
                    <!-- 3. LO SPACCATO SENSORIALE -->
                    <div class="bg-slate-50 p-5 sm:p-6 rounded-3xl border border-slate-200/80 space-y-3">
                        <h4 class="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                            <i class="fa-solid fa-eye text-emerald-600"></i> Spaccato Sensoriale Biologico
                        </h4>
                        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                            <div class="bg-white p-4 rounded-2xl border border-slate-200 shadow-xs space-y-1.5">
                                <span class="text-xs font-bold text-amber-700 flex items-center gap-1.5"><span class="text-base">👃</span> Olfatto Chimico</span>
                                <p id="resSmell" class="text-xs text-slate-600 leading-relaxed font-medium"></p>
                            </div>
                            <div class="bg-white p-4 rounded-2xl border border-slate-200 shadow-xs space-y-1.5">
                                <span class="text-xs font-bold text-blue-700 flex items-center gap-1.5"><span class="text-base">👀</span> Vista da Terra</span>
                                <p id="resSight" class="text-xs text-slate-600 leading-relaxed font-medium"></p>
                            </div>
                        </div>
                    </div>

                    <!-- 4. SPIEGAZIONE E PROTOCOLLI -->
                    <div class="bg-white p-6 rounded-3xl border border-slate-200 shadow-xs space-y-5">
                        <div>
                            <h4 class="text-sm font-bold text-blue-900 flex items-center gap-2 mb-2 font-heading">
                                <i class="fa-solid fa-lightbulb text-amber-500 text-lg"></i> Perché si comporta così?
                            </h4>
                            <p id="resExplanation" class="text-slate-700 text-sm leading-relaxed font-medium"></p>
                        </div>
                        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4 border-t border-slate-100">
                            <div>
                                <h4 class="text-sm font-bold text-emerald-900 flex items-center gap-2 mb-4 font-heading">
                                    <i class="fa-solid fa-circle-check text-emerald-500 text-lg"></i> Cosa fare adesso:
                                </h4>
                                <ul id="resSteps" class="space-y-3 text-sm text-slate-700 font-medium"></ul>
                            </div>
                            <div>
                                <h4 class="text-sm font-bold text-rose-900 flex items-center gap-2 mb-4 font-heading">
                                    <i class="fa-solid fa-circle-xmark text-rose-500 text-lg"></i> Errori da evitare:
                                </h4>
                                <ul id="resForbidden" class="space-y-3 text-sm text-slate-700 font-medium"></ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- OVERLAY SFOCATURA (BLUR) PER OSPITI -->
                <div id="blurOverlay" class="hidden absolute inset-0 z-10 bg-white/40 backdrop-blur-[6px] flex flex-col items-center justify-center rounded-3xl border border-white/50 cursor-pointer hover:bg-white/30 transition" onclick="openAuthModal(false, '✅ Crea un account gratis per salvare questa traduzione e leggere le istruzioni posturali complete!')">
                    <div class="bg-slate-900/95 text-white px-7 py-6 rounded-2xl shadow-2xl flex flex-col items-center text-center transform group-hover:scale-105 transition-transform border border-slate-700 max-w-sm w-11/12">
                        <i class="fa-solid fa-lock text-3xl mb-3 text-emerald-400 drop-shadow-md"></i>
                        <span class="font-black text-lg mb-1 font-heading tracking-wide">Guida Pratica Bloccata</span>
                        <span class="text-xs text-slate-300 font-medium mb-5">Sblocca il referto completo e salvalo nel tuo Diario Etologico in 5 secondi.</span>
                        <button class="w-full bg-emerald-500 hover:bg-emerald-400 text-white text-sm font-bold py-3 px-6 rounded-xl shadow-md transition flex items-center justify-center gap-2">
                            <i class="fa-brands fa-google"></i> Sblocca Gratis Ora
                        </button>
                    </div>
                </div>

            </div> <!-- Fine Wrapper Pratico -->
        </div>

    </main>

    <!-- ==================== POPUP PAYWALL 2 COLONNE (PREMIUM VS PRO) ==================== -->
    <div id="paywallModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-3 sm:p-4 fade-in overflow-y-auto">
        <div class="bg-white w-full max-w-3xl rounded-3xl border border-slate-200 shadow-2xl relative my-auto">
            <button onclick="closePaywallModal()" class="absolute top-4 right-4 text-slate-400 hover:text-slate-800 w-8 h-8 flex items-center justify-center rounded-full bg-slate-100 transition z-10">
                <i class="fa-solid fa-xmark"></i>
            </button>

            <div class="p-5 sm:p-8">
                <!-- Intestazione -->
                <div class="text-center space-y-2 mb-6 sm:mb-8">
                    <h3 class="text-2xl sm:text-3xl font-black text-slate-800 font-heading" id="paywallTitle">Scegli il tuo piano BioDog</h3>
                    <p class="text-xs sm:text-sm text-slate-500 font-medium" id="paywallSubtitle">
                        Sblocca le funzioni avanzate e smetti di indovinare cosa pensa il tuo cane.
                    </p>
                </div>

                <!-- Griglia Prezzi -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                    
                    <!-- BOX PREMIUM (1,99) -->
                    <div class="bg-slate-50 rounded-2xl border border-slate-200 p-5 sm:p-6 flex flex-col justify-between">
                        <div>
                            <div class="flex items-center gap-2 mb-2">
                                <i class="fa-solid fa-star text-amber-500 text-lg"></i>
                                <h4 class="text-lg font-black text-slate-800 font-heading">Premium</h4>
                            </div>
                            <div class="flex items-baseline gap-1 mb-4">
                                <span class="text-3xl font-black text-slate-800 font-heading">1,99 €</span>
                                <span class="text-xs text-slate-500 font-bold">/ mese</span>
                            </div>
                            
                            <ul class="space-y-3 text-xs text-slate-700 font-medium mb-6">
                                <li class="flex items-start gap-2.5"><i class="fa-solid fa-check text-emerald-500 mt-0.5"></i> <span><strong>100 Analisi Etologiche</strong> Testuali al mese</span></li>
                                <li class="flex items-start gap-2.5"><i class="fa-solid fa-check text-emerald-500 mt-0.5"></i> <span>Accesso completo ai 4 Sensi</span></li>
                                <li class="flex items-start gap-2.5"><i class="fa-solid fa-check text-emerald-500 mt-0.5"></i> <span>Diario Clinico sempre sincronizzato</span></li>
                            </ul>
                        </div>
                        <button onclick="startStripeCheckout('premium')" class="w-full bg-slate-800 hover:bg-slate-700 text-white font-bold py-3 rounded-xl transition text-sm">
                            Scegli Premium
                        </button>
                    </div>

                    <!-- BOX VISION PRO (3,99) -->
                    <div class="bg-emerald-50 rounded-2xl border-2 border-emerald-500 p-5 sm:p-6 flex flex-col justify-between relative shadow-lg">
                        <div class="absolute -top-3 left-1/2 -translate-x-1/2 bg-emerald-500 text-white text-[10px] font-black uppercase px-3 py-1 rounded-full tracking-wider">
                            Consigliato
                        </div>
                        <div>
                            <div class="flex items-center gap-2 mb-2">
                                <i class="fa-solid fa-video text-emerald-600 text-lg"></i>
                                <h4 class="text-lg font-black text-emerald-950 font-heading">Vision PRO</h4>
                            </div>
                            <div class="flex items-baseline gap-1 mb-4">
                                <span class="text-3xl font-black text-emerald-950 font-heading">3,99 €</span>
                                <span class="text-xs text-emerald-700 font-bold">/ mese</span>
                            </div>
                            
                            <ul class="space-y-3 text-xs text-slate-700 font-medium mb-6">
                                <li class="flex items-start gap-2.5"><i class="fa-solid fa-check text-emerald-500 mt-0.5"></i> <span>Analisi Testuali <strong>Illimitate</strong></span></li>
                                <li class="flex items-start gap-2.5 font-bold text-emerald-900"><i class="fa-solid fa-camera text-emerald-600 mt-0.5"></i> <span>Video-Analisi (Registra/Galleria)</span></li>
                                <li class="flex items-start gap-2.5"><i class="fa-solid fa-dna text-emerald-500 mt-0.5"></i> <span>Configuratore Morfologico su misura</span></li>
                            </ul>
                        </div>
                        <button onclick="startStripeCheckout('pro')" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 rounded-xl transition text-sm shadow-md">
                            Scegli Vision PRO
                        </button>
                    </div>

                </div>
                <p class="text-[10px] sm:text-xs text-center text-slate-400 font-medium mt-5">Pagamento sicuro Stripe. Disdici in qualsiasi momento senza vincoli.</p>
            </div>
        </div>
    </div>

    <!-- ==================== POPUP AUTENTICAZIONE (SUPABASE) ==================== -->
    <div id="authModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4 fade-in">
        <div class="bg-white w-full max-w-sm rounded-3xl border border-slate-200 overflow-hidden shadow-2xl relative p-6 sm:p-8 space-y-4">
            <button onclick="closeAuthModal()" class="absolute top-4 right-4 text-slate-400 hover:text-slate-800 w-8 h-8 flex items-center justify-center rounded-full bg-slate-100 transition">
                <i class="fa-solid fa-xmark"></i>
            </button>
            <div class="text-center space-y-2 pt-1">
                <div class="w-24 sm:w-28 h-auto mx-auto mb-2 flex justify-center">
                    <img src="1239.png" alt="BioDog Logo" class="w-full h-auto drop-shadow-xs">
                </div>
                <h3 id="authModalTitle" class="text-xl font-black text-slate-800 font-heading">Accedi a BioDog</h3>
            </div>
            <div id="authPaywallNotice" class="hidden bg-emerald-50 border border-emerald-200 p-3 rounded-xl text-xs text-emerald-900 font-medium text-center shadow-inner"></div>
            <div class="pt-1">
                <button onclick="handleGoogleLogin()" id="btnActionGoogle" class="w-full bg-white hover:bg-slate-50 text-slate-700 font-bold py-2.5 px-4 rounded-xl border border-slate-200 shadow-xs transition text-sm flex items-center justify-center gap-3">
                    <svg class="w-4 h-4 shrink-0" viewBox="0 0 24 24">
                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                    </svg>
                    <span>Continua con Google</span>
                </button>
            </div>
            
            <div class="flex items-center my-2">
                <div class="flex-grow border-t border-slate-200"></div>
                <span class="px-2.5 text-[10px] font-bold text-slate-400 uppercase tracking-widest">oppure con email</span>
                <div class="flex-grow border-t border-slate-200"></div>
            </div>

            <!-- Form Email e Password -->
            <div class="space-y-3">
                <div>
                    <input type="email" id="authEmail" placeholder="Email" class="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-emerald-500 transition">
                </div>
                <div>
                    <input type="password" id="authPassword" placeholder="Password" class="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-emerald-500 transition" onkeypress="if(event.key === 'Enter') handleEmailLogin()">
                </div>
                <p id="authError" class="text-xs text-rose-500 hidden font-semibold bg-rose-50 p-2.5 rounded-lg border border-rose-200"></p>
                <div class="space-y-2 pt-1">
                    <button onclick="handleEmailLogin()" id="btnActionLogin" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2.5 rounded-xl transition text-sm">Accedi con Email</button>
                    <button onclick="handleEmailSignUp()" id="btnActionSignUp" class="w-full bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold py-2.5 rounded-xl transition text-sm">Crea nuovo account</button>
                </div>
            </div>
        </div>
    </div>

    <!-- ==================== AREA RISERVATA: DIARIO TRADUZIONI (DRAWER LATERALE) ==================== -->
    <div id="historyDrawerBackdrop" class="hidden fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs transition-opacity duration-300" onclick="closeHistoryDrawer()"></div>
    
    <div id="historyDrawer" class="fixed inset-y-0 right-0 z-50 w-full max-w-md bg-white shadow-2xl border-l border-slate-200 transform translate-x-full transition-transform duration-300 ease-in-out flex flex-col">
        <!-- Intestazione Drawer -->
        <div class="p-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/70">
            <div class="flex items-center gap-3">
                <div class="w-9 h-9 rounded-2xl bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold text-sm shadow-xs">
                    <i class="fa-solid fa-book-bookmark"></i>
                </div>
                <div>
                    <h3 class="font-heading font-black text-slate-800 text-base">Diario Etologico</h3>
                    <p class="text-[11px] text-slate-400 font-medium">Cronologia delle decodifiche salvate</p>
                </div>
            </div>
            <button onclick="closeHistoryDrawer()" class="w-8 h-8 rounded-full bg-white border border-slate-200 hover:bg-slate-100 text-slate-400 hover:text-slate-700 flex items-center justify-center transition">
                <i class="fa-solid fa-xmark"></i>
            </button>
        </div>

        <!-- Contenuto Lista Traduzioni -->
        <div id="historyListContainer" class="flex-grow overflow-y-auto p-4 space-y-3 custom-scroll">
            <!-- Popolato dinamicamente da JavaScript -->
        </div>

        <!-- Barra Inferiore con Conteggio e Azioni -->
        <div class="p-4 border-t border-slate-100 bg-slate-50 flex items-center justify-between text-xs text-slate-500">
            <span id="historySummaryText" class="font-semibold text-slate-600">0 traduzioni</span>
            <button onclick="clearAllHistory()" class="text-rose-500 hover:text-rose-700 font-bold text-xs transition flex items-center gap-1.5 hover:underline">
                <i class="fa-solid fa-trash-can"></i> Svuota diario
            </button>
        </div>
    </div>

    <!-- Popup di Benvenuto -->
    <div id="welcomeModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4 fade-in">
        <div class="bg-white w-full max-w-md rounded-3xl border border-slate-200 overflow-hidden shadow-2xl relative">
            <button onclick="closeWelcomeModal()" class="absolute top-4 right-4 text-slate-400 hover:text-slate-800 text-xl w-8 h-8 flex items-center justify-center rounded-full bg-slate-100 hover:bg-slate-200 transition">
                <i class="fa-solid fa-xmark"></i>
            </button>
            
            <div class="p-8 space-y-6">
                <div class="flex justify-center mb-1">
                    <div class="w-32 h-auto flex justify-center items-center">
                        <img src="1239.png" alt="BioDog" class="w-full h-auto">
                    </div>
                </div>
                
                <h2 class="text-2xl font-black text-center text-slate-800 font-heading leading-tight">
                    Scopri il mondo attraverso i suoi occhi.
                </h2>
                
                <p class="text-slate-600 text-sm leading-relaxed text-center">
                    Hai mai guardato il tuo cane chiedendoti: <br><i class="text-slate-800 font-medium">"Cosa ti passa per la testa?"</i>
                </p>

                <div class="bg-slate-50 p-5 rounded-2xl border border-slate-100 space-y-3 text-sm text-slate-700">
                    <p>
                        <b class="text-emerald-600">BioDog.io</b> non è il solito addestratore virtuale. È il primo traduttore basato sulle vere neuroscienze canine.
                    </p>
                    <ul class="space-y-2 mt-2">
                        <li class="flex items-start gap-2"><i class="fa-solid fa-microchip text-emerald-500 mt-1"></i> <span>Niente giudizi o dispetti umani. Solo biologia ed etologia.</span></li>
                        <li class="flex items-start gap-2"><i class="fa-solid fa-brain text-emerald-500 mt-1"></i> <span>Scopri cosa fiuta, cosa vede e cosa prova davvero.</span></li>
                        <li class="flex items-start gap-2"><i class="fa-solid fa-shield-dog text-emerald-500 mt-1"></i> <span>Ricevi consigli sulla tua postura per risolvere ogni momento.</span></li>
                    </ul>
                </div>

                <button onclick="closeWelcomeModal()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3.5 rounded-2xl transition shadow-md text-sm">
                    Inizia a tradurre
                </button>
            </div>
        </div>
    </div>

    <!-- Javascript Logic -->
    <script>
        // ==================== CONFIGURAZIONE ====================
        const SUPABASE_URL = "https://itjfyjyzaornkintpefd.supabase.co";
        const SUPABASE_ANON_KEY = "sb_publishable_zLclG7vV8MNdtYFDghH6cg_a7IBTe8E";
        const BACKEND_API_BASE = "https://biodog-api.onrender.com";
        
        let supabaseClient = null;
        if (window.supabase) {
            supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
        }

        let currentUser = null;
        let isUserPremium = false;
        let stripeCustomerId = null; 

        // ==================== DATI PILLOLA DEL GIORNO ====================
        const ethoPills = [
            "I cani hanno la deuteranopia: faticano a distinguere il rosso dal verde. Quel gioco rosso sul prato per loro è quasi invisibile!",
            "Quando un cane starnutisce durante il gioco non ha il raffreddore: è un segnale per dirti che sta solo fingendo di lottare.",
            "Il movimento della coda non significa sempre felicità: una coda rigida che sbatte verso sinistra indica spesso allerta o evitamento.",
            "Da 30 centimetri da terra, una persona che si china frontalmente appare come un grattacielo incombente, scatenando reazioni di difesa.",
            "I cani non provano senso di colpa. La tipica 'faccia colpevole' è solo un gesto di pacificazione per calmare la tensione del momento.",
            "L'abbaio continuo è un tratto 'neotenico' (infantile). I lupi adulti abbaiano raramente: il cane lo usa per chiedere attenzione sociale."
        ];

        window.onload = () => {
            initAuth();
            setupDailyPill();
        };

        function setupDailyPill() {
            const dayOfYear = Math.floor((new Date() - new Date(new Date().getFullYear(), 0, 0)) / 1000 / 60 / 60 / 24);
            document.getElementById('dailyPillText').innerText = ethoPills[dayOfYear % ethoPills.length];
            document.getElementById('dailyPill').classList.remove('hidden');
            document.getElementById('dailyPill').classList.add('flex');
        }

        async function initAuth() {
            if (!supabaseClient) return;
            const { data: { session } } = await supabaseClient.auth.getSession();
            await updateAuthUI(session);
            supabaseClient.auth.onAuthStateChange(async (_e, session) => await updateAuthUI(session));
            checkStripeRedirect();
        }

        async function updateAuthUI(session) {
            currentUser = session?.user || null;
            const loginBtn = document.getElementById('loginBtn');
            const profile = document.getElementById('userProfile');

            if (currentUser) {
                loginBtn.classList.add('hidden');
                profile.classList.remove('hidden');
                await checkSubscriptionStatus();
            } else {
                loginBtn.classList.remove('hidden');
                profile.classList.add('hidden');
                isUserPremium = false;
                stripeCustomerId = null;
                updatePremiumBadgeUI();
                applyBlurToResult();
            }
            updateHistoryBadge();
        }

        async function checkSubscriptionStatus() {
            if (!supabaseClient || !currentUser) {
                isUserPremium = false;
                stripeCustomerId = null;
                updatePremiumBadgeUI(); return;
            }
            try {
                const { data } = await supabaseClient.from('user_subscriptions').select('is_active, stripe_customer_id').eq('user_id', currentUser.id).maybeSingle();
                if (data && data.is_active) {
                    isUserPremium = true;
                    stripeCustomerId = data.stripe_customer_id;
                } else {
                    isUserPremium = false;
                    stripeCustomerId = null;
                }
            } catch (e) { isUserPremium = false; }
            
            updatePremiumBadgeUI();
            if (isUserPremium || currentUser) removeBlurFromResult();
        }

        function updatePremiumBadgeUI() {
            const btnUpgrade = document.getElementById('btnUpgradeHeader');
            const badgePremium = document.getElementById('badgePremiumActive');
            const btnManageSub = document.getElementById('btnManageSub');
            if (isUserPremium) {
                btnUpgrade?.classList.add('hidden');
                badgePremium?.classList.remove('hidden');
                if (stripeCustomerId) btnManageSub?.classList.remove('hidden');
            } else {
                btnUpgrade?.classList.remove('hidden');
                badgePremium?.classList.add('hidden');
                btnManageSub?.classList.add('hidden');
            }
        }

        function checkStripeRedirect() {
            const params = new URLSearchParams(window.location.search);
            if (params.get('payment') === 'success') {
                document.getElementById('stripeSuccessBanner').classList.remove('hidden');
                window.history.replaceState({}, document.title, window.location.pathname);
                setTimeout(checkSubscriptionStatus, 1500);
            }
        }
        function dismissStripeBanner() { document.getElementById('stripeSuccessBanner').classList.add('hidden'); }

        // ==================== GESTIONE PAYWALL A DUE COLONNE ====================
        function openPaywallModal(customMessage = null) {
            if (!currentUser) { openAuthModal(true, customMessage); return; }
            if (customMessage) {
                document.getElementById('paywallTitle').innerText = "Hai raggiunto il limite";
                document.getElementById('paywallSubtitle').innerText = customMessage;
            } else {
                document.getElementById('paywallTitle').innerText = "Scegli il tuo piano BioDog";
                document.getElementById('paywallSubtitle').innerText = "Sblocca le funzioni avanzate e smetti di indovinare cosa pensa il tuo cane.";
            }
            document.getElementById('paywallModal').classList.remove('hidden');
        }

        function closePaywallModal() { document.getElementById('paywallModal').classList.add('hidden'); }

        async function startStripeCheckout(tier) {
            const btn = document.querySelector(`button[onclick="startStripeCheckout('${tier}')"]`);
            const origText = btn.innerHTML;
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Attendere...';
            btn.disabled = true;

            try {
                const response = await fetch(`${BACKEND_API_BASE}/api/v1/stripe/create-checkout-session`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        user_id: currentUser.id, 
                        user_email: currentUser.email || "",
                        tier: tier // Ora il frontend invia la scelta ("premium" o "pro")
                    })
                });
                const data = await response.json();
                if (data.checkout_url) window.location.href = data.checkout_url;
            } catch (error) {
                alert("Impossibile avviare il pagamento. " + error.message);
                btn.innerHTML = origText;
                btn.disabled = false;
            }
        }

        async function openStripePortal() {
            if (!stripeCustomerId) return;
            const btn = document.getElementById('btnManageSub');
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';
            try {
                const response = await fetch(`${BACKEND_API_BASE}/api/v1/stripe/create-portal-session`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ customer_id: stripeCustomerId })
                });
                const data = await response.json();
                if (data.portal_url) window.location.href = data.portal_url;
            } catch (error) { alert("Errore portale."); }
        }

        // ==================== AUTH E SUPABASE ====================
        function openAuthModal(isPaywall = false, msg = null) {
            if(msg) document.getElementById('authPaywallNotice').innerHTML = msg;
            document.getElementById('authPaywallNotice').className = msg ? "bg-emerald-50 border border-emerald-200 p-3 rounded-xl text-xs text-emerald-900 font-medium text-center shadow-inner mb-3" : "hidden";
            document.getElementById('authModal').classList.remove('hidden');
        }
        function closeAuthModal() { document.getElementById('authModal').classList.add('hidden'); }
        async function handleGoogleLogin() {
            await supabaseClient.auth.signInWithOAuth({ provider: 'google', options: { redirectTo: window.location.origin } });
        }
        async function handleEmailLogin() {
            const email = document.getElementById('authEmail').value.trim();
            const password = document.getElementById('authPassword').value;
            const errorBox = document.getElementById('authError');
            errorBox.classList.add('hidden');
            if (!email || !password) return;
            const { data, error } = await supabaseClient.auth.signInWithPassword({ email, password });
            if (error) { errorBox.innerText = "Credenziali non valide."; errorBox.classList.remove('hidden'); }
            else { closeAuthModal(); updateAuthUI(data.session); }
        }
        async function handleEmailSignUp() {
            const email = document.getElementById('authEmail').value.trim();
            const password = document.getElementById('authPassword').value;
            const errorBox = document.getElementById('authError');
            errorBox.classList.add('hidden');
            if (!email || !password) return;
            const { data, error } = await supabaseClient.auth.signUp({ email, password });
            if (error) { errorBox.innerText = error.message; errorBox.classList.remove('hidden'); }
            else { closeAuthModal(); updateAuthUI(data.session); }
        }
        async function handleLogout() {
            await supabaseClient.auth.signOut();
            isUserPremium = false; stripeCustomerId = null; updateAuthUI(null); closeHistoryDrawer();
        }

        // ==================== DIARIO ====================
        function getHistoryKey() { return `biodog_history_${currentUser?.id || 'guest'}`; }
        function getUserHistory() { try { return JSON.parse(localStorage.getItem(getHistoryKey()) || "[]"); } catch (e) { return []; } }
        function saveToHistory(query, synth) {
            const history = getUserHistory();
            const newItem = { id: Date.now(), timestamp: new Date().toISOString(), query: query, synth: synth };
            if (history.length > 0 && history[0].query.toLowerCase() === query.toLowerCase()) history[0] = newItem;
            else history.unshift(newItem);
            if (history.length > 50) history.pop();
            localStorage.setItem(getHistoryKey(), JSON.stringify(history));
            updateHistoryBadge();
            if (currentUser) {
                document.getElementById('saveToast').classList.remove('hidden');
                setTimeout(() => document.getElementById('saveToast').classList.add('hidden'), 4000);
            }
        }
        function updateHistoryBadge() {
            const count = getUserHistory().length;
            const badge = document.getElementById('historyCountBadge');
            const summary = document.getElementById('historySummaryText');
            if (badge) badge.innerText = count.toString();
            if (summary) summary.innerText = `${count} ${count === 1 ? 'traduzione salvata' : 'traduzioni salvate'}`;
        }
        function openHistoryDrawer() { renderHistoryList(); document.getElementById('historyDrawerBackdrop').classList.remove('hidden'); document.getElementById('historyDrawer').classList.remove('translate-x-full'); }
        function closeHistoryDrawer() { document.getElementById('historyDrawerBackdrop').classList.add('hidden'); document.getElementById('historyDrawer').classList.add('translate-x-full'); }
        function renderHistoryList() {
            const container = document.getElementById('historyListContainer');
            const history = getUserHistory();
            if (!history || history.length === 0) { container.innerHTML = `<div class="py-16 px-4 text-center text-slate-400">Diario vuoto.</div>`; return; }
            container.innerHTML = history.map(item => `
                <div class="bg-slate-50 border border-slate-200 rounded-2xl p-4 space-y-2">
                    <h5 class="text-xs font-black text-slate-800 line-clamp-2">"${item.query}"</h5>
                    <button onclick="restoreHistoryItem(${item.id})" class="text-xs font-bold text-emerald-700">Riapri analisi</button>
                </div>
            `).join('');
        }
        function restoreHistoryItem(id) {
            const history = getUserHistory();
            const item = history.find(x => x.id === id);
            if (!item) return;
            document.getElementById('queryInput').value = item.query;
            displayTranslationResult(item.synth);
            closeHistoryDrawer();
            setTimeout(() => document.getElementById('resultContainer').scrollIntoView({ behavior: 'smooth' }), 150);
        }
        function clearAllHistory() { if (confirm("Cancellare diario?")) { localStorage.removeItem(getHistoryKey()); renderHistoryList(); } }

        // ==================== LOGICA TRADUZIONE ====================
        function showFakeLoadingAndTrigger(callback) {
            const btn = document.getElementById('translateBtn');
            const resultBox = document.getElementById('resultContainer');
            const loading = document.getElementById('loadingIndicator');
            resultBox.classList.add('hidden');
            loading.classList.remove('hidden');
            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin text-lg"></i> <span class="ml-1">Calcolo...</span>';
            setTimeout(() => {
                loading.classList.add('hidden');
                btn.disabled = false;
                btn.innerHTML = '<span>Traduci</span><i class="fa-solid fa-microchip text-emerald-100/90 text-lg"></i>';
                callback();
            }, 2500);
        }

        async function runTranslation() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            let guestUsage = parseInt(localStorage.getItem('biodog_free_translations') || "0");
            let userSearchesDone = parseInt(localStorage.getItem(`biodog_user_searches_${currentUser?.id}`) || "0");
            
            const today = new Date();
            const isSunday = today.getDay() === 0;
            const todayStr = today.toISOString().split('T')[0];
            const lastSundayToken = localStorage.getItem(`biodog_sunday_token_${currentUser?.id}`);
            let usingSundayToken = false;

            if (!currentUser && guestUsage >= 1) {
                showFakeLoadingAndTrigger(() => openAuthModal(true, "✅ <strong>L'IA ha decodificato il comportamento!</strong><br>Accedi o registrati gratis per leggere la spiegazione etologica."));
                return;
            }

            if (currentUser && !isUserPremium && userSearchesDone >= 2) {
                if (isSunday && lastSundayToken !== todayStr) {
                    usingSundayToken = true;
                } else {
                    showFakeLoadingAndTrigger(() => openPaywallModal("✅ <strong>Hai esaurito le tue analisi mensili.</strong><br>Scegli un piano per continuare o attendi la prossima Domenica per un'analisi omaggio!"));
                    return;
                }
            }

            const btn = document.getElementById('translateBtn');
            const resultBox = document.getElementById('resultContainer');
            const loading = document.getElementById('loadingIndicator');
            resultBox.classList.add('hidden'); loading.classList.remove('hidden'); btn.disabled = true;

            try {
                const response = await fetch(`${BACKEND_API_BASE}/api/v1/umwelt/transduce`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_text: query, snout: "normal", ears: "prick", size: "medium", tail: "long" })
                });
                
                if (!response.ok) throw new Error("Errore API");
                const synth = (await response.json()).neural_synthesis;

                if (!currentUser) {
                    localStorage.setItem('biodog_free_translations', (guestUsage + 1).toString());
                } else if (!isUserPremium) {
                    if (usingSundayToken) {
                        localStorage.setItem(`biodog_sunday_token_${currentUser.id}`, todayStr);
                        document.getElementById('sundayToast').classList.remove('hidden');
                        setTimeout(() => document.getElementById('sundayToast').classList.add('hidden'), 5000);
                    } else {
                        localStorage.setItem(`biodog_user_searches_${currentUser.id}`, (userSearchesDone + 1).toString());
                    }
                }

                displayTranslationResult(synth);
                saveToHistory(query, synth);
                loading.classList.add('hidden');
                setTimeout(() => resultBox.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
            } catch (error) { alert("Errore di rete. Riprova."); loading.classList.add('hidden'); }
            finally { btn.disabled = false; }
        }

        function applyBlurToResult() {
            document.getElementById('practicalContent')?.classList.add('blur-content');
            document.getElementById('blurOverlay')?.classList.remove('hidden');
        }
        function removeBlurFromResult() {
            document.getElementById('practicalContent')?.classList.remove('blur-content');
            document.getElementById('blurOverlay')?.classList.add('hidden');
        }

        function displayTranslationResult(synth) {
            document.getElementById('resPankseppText').innerText = synth.panksepp_label || synth.panksepp || "SEEKING";
            document.getElementById('resThought').innerText = `"${synth.thought}"`;
            if (synth.sensory) {
                document.getElementById('resSmell').innerText = synth.sensory.smell;
                document.getElementById('resSight').innerText = synth.sensory.sight;
                document.getElementById('resHearing').innerText = synth.sensory.hearing || "-";
                document.getElementById('resTouch').innerText = synth.sensory.touch || "-";
            }
            if (synth.human_body_language) {
                document.getElementById('resHumanVoice').innerText = synth.human_body_language.voice;
                document.getElementById('resHumanPosture').innerText = synth.human_body_language.posture;
            }
            document.getElementById('resExplanation').innerText = synth.explanation;
            document.getElementById('resSteps').innerHTML = (synth.steps || []).map((step, i) => `<li class="flex items-start gap-2"><span class="bg-emerald-200 text-emerald-800 text-xs font-black w-4 h-4 flex items-center justify-center rounded-full shrink-0">${i+1}</span><span>${step}</span></li>`).join('');
            document.getElementById('resForbidden').innerHTML = (synth.forbidden || []).map(f => `<li class="flex items-start gap-2"><i class="fa-solid fa-ban text-rose-500 mt-1"></i><span>${f}</span></li>`).join('');
            
            if (!currentUser) applyBlurToResult(); else removeBlurFromResult();
            document.getElementById('resultContainer').classList.remove('hidden');
        }

        function openWelcomeModal() { document.getElementById('welcomeModal').classList.remove('hidden'); }
        function closeWelcomeModal() { document.getElementById('welcomeModal').classList.add('hidden'); }
        function setQuery(text) { document.getElementById('queryInput').value = text; runTranslation(); }
    </script>
</body>
</html>

