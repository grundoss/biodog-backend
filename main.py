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

app = FastAPI(title="BioDog.io Neural Engine", version="3.3.0")

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
        req = urllib.request.Request(list_url, headers={"User-Agent": "BioDog/3.3"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            available = [
                m["name"].replace("models/", "")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            for priority in ["3.8-flash", "3.6-flash", "flash-latest", "2.5-flash"]:
                for m in available:
                    if priority in m.lower():
                        ACTIVE_MODEL = m
                        return ACTIVE_MODEL
            if available:
                ACTIVE_MODEL = available[0]
                return ACTIVE_MODEL
    except Exception as e:
        print(f"ListModels error: {e}")

    ACTIVE_MODEL = "gemini-3.8-flash"
    return ACTIVE_MODEL

def _call_gemini_api(api_key: str, full_prompt: str) -> Tuple[Optional[dict], Optional[str], str]:
    if not api_key:
        return None, "Chiave API mancante", "none"

    discovered = _find_live_model(api_key)
    
    models_pool = [
        discovered,
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash"
    ]
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
            headers={"Content-Type": "application/json", "User-Agent": "BioDog/3.3"},
            method="POST"
        )
        
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
                
                if he.code in [400, 403]:
                    return None, f"Chiave non autorizzata (HTTP {he.code})", m
                
                if he.code in [503, 429]:
                    time.sleep(1.0)
                    continue
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
                "panksepp_label": "SEEKING / Esplorazione Cinetica",
                "arousal": 85,
                "valence": 15,
                "thought": "Lo sgancio improvviso dal vincolo attiva una scarica dopaminergica massiccia! L'odore dell'aria aperta crea tunnel attentivo e sordità selettiva al richiamo vocale.",
                "sensory": {
                    "smell": "Flusso massiccio di molecole ambientali nell'aria fresca che saturano i turbinati.",
                    "sight": "Orizzonte aperto ad ampio raggio; movimento rapido di stimoli periferici.",
                    "hearing": "La voce umana distante è filtrata dal rumore del vento e dal battito cardiaco elevato."
                },
                "human_body_language": {
                    "voice": "Tono grave, calmo e cadenzato; non gridare mai per non stimolare la fuga predatoria fittizia.",
                    "posture": "Accovacciati di lato a 45 gradi, muoviti all'indietro per invitarlo al rientro spontaneo."
                },
                "explanation": "La fuga post-sgancio non è disubbidienza né dispetto: è l'effetto fionda da confinamento prossemico (SEEKING cinestesico ad alto arousal).",
                "steps": [
                    "Non inseguire il cane correndogli dietro: aumenteresti la fuga predatoria fittizia.",
                    "Fermati, accovacciati di lato a 45 gradi e richiama muovendoti all'indietro.",
                    "Premia sempre il ritorno spontaneo senza rimproveri a posteriori."
                ],
                "forbidden": [
                    "Non punire mai il cane quando finalmente torna da te.",
                    "Non urlare con tono rabbioso o concitato: inibirebbe il rientro."
                ]
            }
        else:
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
