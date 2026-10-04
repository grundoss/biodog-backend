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

app = FastAPI(title="BioDog.io Neural Engine", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
CACHED_MODEL_NAME = None

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
3. Considera le costanti morfologiche (olfatto, campo visivo, altezza occhi) per determinare la reattività.
4. Genera ESCLUSIVAMENTE un JSON valido (senza testo introduttivo o markdown) con questa struttura esatta:
{
  "situation_title": "Breve titolo etologico della situazione",
  "panksepp": "CARE | RAGE | FEAR | PANIC/GRIEF | PLAY | SEEKING | LUST",
  "arousal": numero intero da 0 a 100,
  "valence": numero intero da -50 a +50,
  "thought": "pensiero del cane in prima persona: rapido, sensoriale (odori, suoni, distanze, posture), privo di morale umana",
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

def _discover_active_model(api_key: str) -> str:
    global CACHED_MODEL_NAME
    if CACHED_MODEL_NAME:
        return CACHED_MODEL_NAME

    list_url = f"[https://generativelanguage.googleapis.com/v1beta/models?key=](https://generativelanguage.googleapis.com/v1beta/models?key=){api_key}"
    try:
        req = urllib.request.Request(list_url, headers={"User-Agent": "BioDog/2.4"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            available = [
                m["name"].replace("models/", "")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            for target in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash"]:
                for m in available:
                    if target in m:
                        CACHED_MODEL_NAME = m
                        return CACHED_MODEL_NAME
            if available:
                CACHED_MODEL_NAME = available[0]
                return CACHED_MODEL_NAME
    except Exception as e:
        print(f"Errore autodiscovery modelli: {e}")

    # Fallback predefinito se la lista non risponde
    CACHED_MODEL_NAME = "gemini-2.0-flash"
    return CACHED_MODEL_NAME

def _call_gemini_api(api_key: str, full_prompt: str) -> Tuple[Optional[dict], Optional[str], str]:
    global CACHED_MODEL_NAME
    if not api_key:
        return None, "Chiave GEMINI_API_KEY non presente", "none"

    model_name = _discover_active_model(api_key)
    candidate_models = [model_name, "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    last_err = None

    for m in candidate_models:
        url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){m}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "BioDog/2.4"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                if response.status == 200:
                    body = json.loads(response.read().decode("utf-8"))
                    text = body["candidates"][0]["content"]["parts"][0]["text"]
                    CACHED_MODEL_NAME = m
                    return _extract_clean_json(text), None, m
        except urllib.error.HTTPError as he:
            last_err = f"{m} HTTP {he.code}"
            CACHED_MODEL_NAME = None
        except Exception as e:
            last_err = f"{m} err: {str(e)}"

    return None, last_err, "failed"

# ==================== ENDPOINT PRINCIPALE ====================
@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    bio = SensoryEngine.compute(req)

    user_prompt = f"""{SYSTEM_PROMPT}

Comportamento osservato: "{req.user_text}"
Profilo biologico:
- Cranio: {req.snout} (Turbinati olfattivi: {bio['turbinates_cm2']} cm²)
- Campo Visivo: {bio['fov_degrees']}° (Acuità: {bio['acuity_cpd']} cpd)
- Occhi da terra: {bio['eye_height_cm']} cm
- Coda: {bio['tail_bias']}
- Orecchie: {bio['ear_mobility']}"""

    synth, api_err, used_model = await asyncio.to_thread(_call_gemini_api, GEMINI_API_KEY, user_prompt)

    if synth:
        engine_used = f"neural_{used_model}"
    else:
        engine_used = "fallback_local"
        lower = req.user_text.lower()
        if "recinzione" in lower or "avanti e indietro" in lower or "rete" in lower:
            synth = {
                "situation_title": "Pacing da Barriera / Frustrazione",
                "panksepp": "RAGE", "arousal": 85, "valence": -35,
                "thought": "Vedo e sento stimoli al di là della rete ma non posso raggiungerli. La barriera mi blocca: cammino avanti e indietro senza sosta per sfogare la tensione muscolare!",
                "explanation": "La recinzione crea una barriera visivo-motoria. Il movimento continuo (pacing) è un comportamento compulsivo stereotipato causato dall'impossibilità di esplorare o controllare l'ambiente.",
                "steps": [
                    "Scherma la rete con teli oscuranti per interrompere la stimolazione visiva continua.",
                    "Porta il cane fuori dal giardino in passeggiata per soddisfare il suo bisogno di esplorare.",
                    "Interrompi il loop spargendo cibo nell'erba per stimolare il fiuto e abbassare l'agitazione."
                ],
                "forbidden": [
                    "Non sgridarlo mentre fa avanti e indietro: lo stress aumenterà.",
                    "Non lasciarlo ore da solo in giardino credendo che 'faccia esercizio'."
                ]
            }
        else:
            synth = {
                "situation_title": "Valutazione Etologica",
                "panksepp": "SEEKING", "arousal": 50, "valence": 10,
                "thought": f"Analizzo la situazione '{req.user_text}' attraverso i miei sensi.",
                "explanation": f"Elaborazione di sicurezza (Dettaglio API: {api_err}).",
                "steps": ["Osserva la postura complessiva del cane.", "Mantieni calma e spazio vitale."],
                "forbidden": ["Evita movimenti improvvisi o punizioni verbali."]
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
        "active_model": CACHED_MODEL_NAME or _discover_active_model(GEMINI_API_KEY) if GEMINI_API_KEY else "No key",
        "key_ready": bool(GEMINI_API_KEY)
    }
