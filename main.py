import asyncio
import json
import math
import os
import re
from typing import Optional, Tuple
import urllib.error
import urllib.request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="BioDog.io Neural Sensory Engine",
    version="2.3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")

# ==================== SCHEMI DATI ====================
class TransductionRequest(BaseModel):
    user_text: str = Field(..., min_length=2, max_length=500)
    snout: str = Field(default="normal")
    ears: str = Field(default="prick")
    size: str = Field(default="medium")
    tail: str = Field(default="long")
    observed_time_hours: Optional[float] = 0.0

# ==================== CALCOLO SENSORIALE FISICO ====================
class SensoryEngine:
    @staticmethod
    def compute(req: TransductionRequest):
        turbinates = 170.0 if req.snout == "long" else (45.0 if req.snout == "flat" else 100.0)
        decay = 0.35
        detected_hours = req.observed_time_hours or 0.0
        match = re.search(r"(\d+)\s*(?:ore|ora|h)", req.user_text, re.IGNORECASE)
        if match:
            detected_hours = float(match.group(1))
        elif re.search(r"tutto il giorno|sempre", req.user_text, re.IGNORECASE):
            detected_hours = 8.0

        residual = max(5.0, 100.0 * math.exp(-decay * detected_hours)) if detected_hours > 0 else 95.0
        fov = 270 if req.snout == "long" else (220 if req.snout == "flat" else 250)
        cpd = 12.5 if req.snout == "long" else (9.5 if req.snout == "flat" else 11.5)
        eye_height = 25 if req.size == "small" else (75 if req.size == "large" else 45)
        mobility = "Flessibilità 180° e orientamento indipendente" if req.ears == "prick" else "Assorbimento passivo frontale"

        return {
            "turbinates_cm2": turbinates,
            "voc_residual_percent": round(residual, 1),
            "fov_degrees": fov,
            "acuity_cpd": cpd,
            "eye_height_cm": eye_height,
            "ear_mobility": mobility,
            "tail_bias": "Coda arricciata" if req.tail == "curly" else ("Coda corta" if req.tail == "short" else "Standard")
        }

# ==================== SYSTEM PROMPT ANTI-ANTROPOMORFISMO ====================
SYSTEM_PROMPT = """Sei il motore di intelligenza artificiale biologica BioDog.io.
Trasduci il comportamento del cane descritto dall'umano nella prospettiva etologica e percettiva del cane (Umwelt di Jakob von Uexküll).

REGOLE CRITICHE (ANTI-ANTROPOMORFISMO DPO):
1. DIVIETO ASSOLUTO di attribuire concetti morali umani: dispetto, vendetta, senso di colpa, prevaricazione etica o dominio gerarchico alfa.
2. Radica sempre il comportamento nei 7 circuiti emotivi primari di Jaak Panksepp: SEEKING, RAGE, FEAR, PANIC/GRIEF, PLAY, CARE, LUST.
3. Considera le costanti morfologiche (olfatto, campo visivo, altezza occhi) per determinare la reattività.
4. Genera ESCLUSIVAMENTE un JSON valido (senza testo introduttivo o markdown) con questa struttura esatta:
{
  "panksepp": "CARE | RAGE | FEAR | PANIC/GRIEF | PLAY | SEEKING | LUST",
  "arousal": numero intero da 0 a 100,
  "valence": numero intero da -50 a +50,
  "thought": "pensiero del cane in prima persona: rapido, sensoriale (odori, suoni, distanze, postura), privo di morale umana",
  "explanation": "spiegazione etologica chiara per il proprietario",
  "steps": ["passo 1 concreto da fare subito", "passo 2", "passo 3"],
  "forbidden": ["errore 1 da non commettere", "errore 2"]
}"""

def _extract_clean_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
        else:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]

    return json.loads(cleaned)

def _call_gemini_api(api_key: str, full_prompt: str) -> Tuple[Optional[dict], Optional[str]]:
    if not api_key:
        return None, "Chiave GEMINI_API_KEY mancante nelle Environment Variables di Render"

    models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    last_error = None

    for model_name in models:
        # Costruzione URL con pulizia automatica di eventuali parentesi quadre
        raw_url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){model_name}:generateContent?key={api_key}"
        url = raw_url.replace("[", "").replace("]", "").strip()

        payload = {
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=14) as response:
                if response.status == 200:
                    body = json.loads(response.read().decode("utf-8"))
                    text = body["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = _extract_clean_json(text)
                    return parsed, None
        except urllib.error.HTTPError as he:
            err_msg = he.read().decode("utf-8")
            last_error = f"{model_name} HTTP {he.code}: {err_msg}"
        except Exception as e:
            last_error = f"{model_name} Errore: {str(e)}"

    return None, last_error

# ==================== ENDPOINT PRINCIPALE ====================
@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    bio = SensoryEngine.compute(req)

    user_prompt = f"""{SYSTEM_PROMPT}

Descrizione della situazione osservata: "{req.user_text}"
Profilo biologico del soggetto:
- Cranio: {req.snout} (Superficie turbinati olfattivi: {bio['turbinates_cm2']} cm²)
- Campo Visivo: {bio['fov_degrees']}° (Acuità: {bio['acuity_cpd']} cpd)
- Altezza occhi da terra: {bio['eye_height_cm']} cm
- Morfologia Coda: {bio['tail_bias']}
- Orientamento Padiglioni Auricolari: {bio['ear_mobility']}"""

    synth, api_err = await asyncio.to_thread(_call_gemini_api, GEMINI_API_KEY, user_prompt)

    if synth:
        engine_used = "neural_gemini_flash"
    else:
        engine_used = "fallback_local"
        lower = req.user_text.lower()
        if any(w in lower for w in ["ulula", "solitudine", "solo"]):
            synth = {
                "panksepp": "PANIC/GRIEF",
                "arousal": 85,
                "valence": -35,
                "thought": "Essere rimasto da solo mi disorienta. L'isolamento dal branco fa crollare la mia sicurezza: ululo per emettere un segnale acustico a lungo raggio e farmi ritrovare!",
                "explanation": "L'ululato in solitudine è l'espressione classica del circuito PANIC/GRIEF: un richiamo di localizzazione per ricongiungersi con la figura di attaccamento.",
                "steps": [
                    "Abitua il cane a micropause di assenza graduali rientrando prima che parta l'ansia.",
                    "Lasciagli un masticativo naturale appetibile prima di uscire per impegnare la bocca.",
                    "Lascia a disposizione un tuo indumento indossato nella sua cuccia preferita."
                ],
                "forbidden": [
                    "Non sgridarlo mai al rientro se ha ululato: assocerà il ritorno alla paura.",
                    "Non fare saluti enfatici o prolungati prima di varcare la porta."
                ]
            }
        else:
            synth = {
                "panksepp": "SEEKING",
                "arousal": 50,
                "valence": 10,
                "thought": f"Analizzo la situazione '{req.user_text}'. Scansione sensoriale attiva.",
                "explanation": f"Elaborazione locale temporanea. (Dettaglio: {api_err})",
                "steps": ["Mantieni la calma e osserva la postura del cane."],
                "forbidden": ["Evita movimenti bruschi."]
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
    clean_key = GEMINI_API_KEY.replace("[", "").replace("]", "").strip()
    return {
        "status": "BioDog Neural Engine Online",
        "has_key": bool(clean_key),
        "key_prefix": clean_key[:6] + "..." if clean_key else "MANCANTE"
    }
