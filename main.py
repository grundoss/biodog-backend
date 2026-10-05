import asyncio
import json
import math
import os
import re
import time
from typing import Optional, Tuple
import urllib.error
import urllib.request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="BioDog.io Neural Engine", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'").replace("[", "").replace("]", "")
ACTIVE_MODEL = None

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
  "situation_title": "Titolo etologico breve",
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
        req = urllib.request.Request(list_url, headers={"User-Agent": "BioDog/3.2"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            available = [
                m["name"].replace("models/", "")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            # Cerca SOLO i modelli moderni in ordine di preferenza
            for priority in ["3.8-flash", "3.6-flash", "flash-latest", "2.5-flash", "2.0-flash"]:
                for m in available:
                    if priority in m.lower():
                        ACTIVE_MODEL = m
                        return ACTIVE_MODEL
            if available:
                ACTIVE_MODEL = available[0]
                return ACTIVE_MODEL
    except Exception as e:
        print(f"ListModels error: {e}")

    ACTIVE_MODEL = "gemini-flash-latest"
    return ACTIVE_MODEL

def _call_gemini_api(api_key: str, full_prompt: str) -> Tuple[Optional[dict], Optional[str], str]:
    if not api_key:
        return None, "Chiave API mancante", "none"

    discovered = _find_live_model(api_key)
    
    # Pool di modelli RIGOROSAMENTE aggiornati, senza alcun 1.5 o 1.0
    models_pool = [
        discovered,
        "gemini-flash-latest",
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]
    # Rimuove eventuali duplicati preservando l'ordine
    candidate_models = list(dict.fromkeys([m for m in models_pool if m]))

    last_err = "Nessun modello ha risposto"
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
            headers={"Content-Type": "application/json", "User-Agent": "BioDog/3.2"},
            method="POST"
        )
        
        # Fino a 2 tentativi per modello per bypassare i picchi 503 momentanei
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=12) as response:
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
                
                # Chiave invalida: interrompiamo subito per tutti i modelli
                if he.code in [400, 403]:
                    return None, f"Chiave non autorizzata (HTTP {he.code})", m
                
                # Sovraccarico Google (503) o rate limit (429): pausa di 1 secondo e riprova
                if he.code in [503, 429]:
                    time.sleep(1.0)
                    continue
                
                # Se è 404 (modello non esiste) esce dal tentativo e passa al modello successivo
                break
            except Exception as e:
                last_err = f"{m} err: {str(e)}"
                break

    return None, last_err, "failed"

# ==================== ENDPOINT PRINCIPALE ====================
@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    bio = SensoryEngine.compute(req)

    user_prompt = f"""{SYSTEM_PROMPT}\n\nComportamento osservato: "{req.user_text}"\nProfilo biologico:\n- Cranio: {req.snout} (Turbinati olfattivi: {bio['turbinates_cm2']} cm²)\n- Campo Visivo: {bio['fov_degrees']}° (Acuità: {bio['acuity_cpd']} cpd)\n- Occhi da terra: {bio['eye_height_cm']} cm\n- Coda: {bio['tail_bias']}\n- Orecchie: {bio['ear_mobility']}"""

    synth, api_err, used_model = await asyncio.to_thread(_call_gemini_api, GEMINI_API_KEY, user_prompt)

    if synth:
        engine_used = f"neural_{used_model}"
    else:
        engine_used = "fallback_local"
        lower = req.user_text.lower()
        if any(w in lower for w in ["slego", "torna", "scappa", "fugge", "libero"]):
            synth = {
                "situation_title": "Scarica Cinetica & Circuito SEEKING",
                "panksepp": "SEEKING",
                "arousal": 85,
                "valence": 15,
                "thought": "La rimozione improvvisa del vincolo attiva una scarica dopaminergica ed esplorativa massiccia! L'odore dell'aria aperta crea tunnel attentivo e sordità selettiva al richiamo vocale umano.",
                "explanation": "La fuga momentanea post-sgancio non è disubbidienza né dispetto: è l'effetto fionda da confinamento prossemico (SEEKING cinestesico ad altissimo arousal).",
                "steps": [
                    "Non inseguire il cane correndogli dietro: aumenteresti la fuga predatoria fittizia.",
                    "Fermati, accovacciati di lato a 45 gradi e richiama muovendoti all'indietro.",
                    "Premia sempre il ritorno spontaneo senza rimproveri a posteriori."
                ],
                "forbidden": [
                    "Non punire mai il cane quando finalmente torna da te.",
                    "Non urlare con tono rabbioso: inibirebbe il rientro."
                ]
            }
        else:
            synth = {
                "situation_title": "Valutazione Etologica",
                "panksepp": "SEEKING",
                "arousal": 50,
                "valence": 10,
                "thought": f"Analizzo la situazione '{req.user_text}' con i miei sensi.",
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

# ==================== ENDPOINT DIAGNOSTICO ====================
@app.get("/test-gemini")
async def test_gemini():
    synth, err, model = await asyncio.to_thread(
        _call_gemini_api,
        GEMINI_API_KEY,
        'Rispondi SOLO con questo JSON: {"status": "ok", "test": "success"}'
    )
    return {
        "ok": bool(synth),
        "modello_agganciato": model,
        "risposta_gemini": synth,
        "eventuale_errore": err
    }

@app.get("/")
async def root():
    return {
        "status": "BioDog Neural Engine Online",
        "modello_rilevato": ACTIVE_MODEL or _find_live_model(GEMINI_API_KEY) if GEMINI_API_KEY else "Nessuna chiave"
    }

