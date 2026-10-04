import asyncio
import json
import math
import os
import re
from typing import List, Optional
import urllib.error
import urllib.request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="BioDog.io Neural Sensory Engine",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

class TransductionRequest(BaseModel):
    user_text: str = Field(..., min_length=2, max_length=500)
    snout: str = Field(default="normal")
    ears: str = Field(default="prick")
    size: str = Field(default="medium")
    tail: str = Field(default="long")
    observed_time_hours: Optional[float] = 0.0

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

SYSTEM_PROMPT = """Sei il motore di intelligenza artificiale biologica BioDog.io.
Il tuo compito è trasdurre la situazione vissuta dal cane e descritta dal proprietario nella prospettiva etologica e percettiva canina (Umwelt di Jakob von Uexküll).

REGOLE CRITICHE (ANTI-ANTROPOMORFISMO DPO):
1. DIVIETO ASSOLUTO di attribuire concetti morali umani: dispetto, vendetta, senso di colpa, prevaricazione etica o gerarchia alfa.
2. Radica sempre il comportamento nei 7 circuiti emotivi primari di Jaak Panksepp: SEEKING, RAGE, FEAR, PANIC/GRIEF, PLAY, CARE, LUST.
3. Considera le costanti morfologiche (olfatto, campo visivo, altezza occhi) per determinare la reattività.
4. Genera ESCLUSIVAMENTE un JSON valido (senza blocchi di codice markdown) con questa struttura esatta:
{
  "panksepp": "FEAR",
  "arousal": 85,
  "valence": -40,
  "thought": "pensiero in prima persona del cane, focalizzato su minaccia visiva/acustica e istinto di sicurezza",
  "explanation": "spiegazione etologica chiara per il proprietario",
  "steps": ["passo 1 concreto da fare subito", "passo 2", "passo 3"],
  "forbidden": ["errore grave 1 da evitare", "errore grave 2"]
}"""

def _call_gemini_raw(api_key: str, prompt_text: str):
    # Proviamo gemini-2.5-flash e fallback su gemini-1.5-flash
    models = ["gemini-2.5-flash", "gemini-1.5-flash"]
    last_err = None

    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                if response.status == 200:
                    raw = json.loads(response.read().decode("utf-8"))
                    text = raw["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text), None
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8")
            last_err = f"HTTP {he.code}: {err_body}"
        except Exception as e:
            last_err = str(e)

    return None, last_err

@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    bio = SensoryEngine.compute(req)

    user_prompt = f"""{SYSTEM_PROMPT}

Situazione descritta: "{req.user_text}"
Profilo cane:
- Muso: {req.snout} (Turbinati: {bio['turbinates_cm2']} cm²)
- Campo Visivo: {bio['fov_degrees']}° (Acuità: {bio['acuity_cpd']} cpd)
- Occhi da terra: {bio['eye_height_cm']} cm
- Coda: {bio['tail_bias']}
- Orecchie: {bio['ear_mobility']}"""

    synth = None
    diag_error = None

    if GEMINI_API_KEY:
        synth, diag_error = await asyncio.to_thread(_call_gemini_raw, GEMINI_API_KEY, user_prompt)

    if not synth:
        # Fallback contestuale dinamico
        lower = req.user_text.lower()
        if any(w in lower for w in ["aspirapolvere", "botti", "tuoni", "paura", "nasconde", "trema"]):
            synth = {
                "panksepp": "FEAR", "arousal": 85, "valence": -40,
                "thought": "Un oggetto mobile estraneo emette ultrasuoni minacciosi sul mio stesso livello visivo. Non posso controllarlo: salto in alto per trovare un rifugio sicuro ed emetto abbai difensivi per fermarlo!",
                "explanation": "Il robot aspirapolvere si muove a terra (nell'altezza del campo visivo del cane) ed emette frequenze acustiche ad alto volume. Salire sul divano è una strategia di fuga verso l'alto (safe-place) tipica del circuito FEAR.",
                "steps": [
                    "Spegni subito l'elettrodomestico ed evita di farlo avvicinare al divano.",
                    "Non toccare il cane mentre è in allerta alta: dagli tempo di scendere spontaneamente quando si calma.",
                    "Abitualo a motore spento spargendo bocconi intorno all'aspirapolvere per de-sensibilizzarlo."
                ],
                "forbidden": [
                    "Non sgridarlo mentre abbaia: la punizione conferma la minaccia del robot.",
                    "Non forzarlo a scendere dal divano o ad avvicinarsi all'oggetto acceso."
                ]
            }
        else:
            synth = {
                "panksepp": "SEEKING", "arousal": 50, "valence": 10,
                "thought": f"Analizzo la situazione '{req.user_text}'. Scansione sensoriale attiva.",
                "explanation": "Reazione a stimolo ambientale non standard.",
                "steps": ["Mantieni la calma e osserva la postura del cane."],
                "forbidden": ["Evita movimenti bruschi o grida."]
            }

    return {
        "status": "success",
        "engine": "neural_gemini" if (synth and not diag_error and GEMINI_API_KEY) else "rule_fallback",
        "diagnostic_error": diag_error,
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
        "api_key_configured": bool(GEMINI_API_KEY),
        "key_prefix": GEMINI_API_KEY[:6] + "..." if GEMINI_API_KEY else "MISSING"
    }
