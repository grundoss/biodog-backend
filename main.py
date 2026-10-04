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
    title="BioDog.io Neural Sensory Engine (Gemini)",
    description="Backend neurale multi-agente per la simulazione dell'Umwelt canino",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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
            "tail_bias": "Coda arricciata di natura" if req.tail == "curly" else ("Coda corta" if req.tail == "short" else "Standard")
        }

# ==================== SYSTEM PROMPT ANTI-ANTROPOMORFISMO ====================
SYSTEM_PROMPT = """Sei il motore di intelligenza artificiale biologica BioDog.io.
Trasduci il comportamento canino descritto dall'umano nella prospettiva etologica e percettiva del cane (Umwelt di Jakob von Uexküll).

REGOLE CRITICHE (ANTI-ANTROPOMORFISMO DPO):
1. DIVIETO ASSOLUTO di attribuire concetti morali umani: dispetto, vendetta, senso di colpa, prevaricazione etica o dominio gerarchico alfa.
2. Radica sempre il comportamento nei 7 circuiti emotivi primari di Jaak Panksepp: SEEKING, RAGE, FEAR, PANIC/GRIEF, PLAY, CARE, LUST.
3. Considera le costanti morfologiche (olfatto, campo visivo, altezza occhi) per determinare la reattività.
4. Genera una risposta valida conforme a questo schema JSON:
{
  "panksepp": "CARE | RAGE | FEAR | PANIC/GRIEF | PLAY | SEEKING | LUST",
  "arousal": numero intero da 0 a 100,
  "valence": numero intero da -50 a +50,
  "thought": "pensiero del cane in prima persona: rapido, sensoriale (odori, suoni, distanze, postura), privo di morale umana",
  "explanation": "spiegazione etologica chiara per il proprietario",
  "steps": ["passo 1 concreto da fare subito", "passo 2", "passo 3"],
  "forbidden": ["errore 1 da non commettere", "errore 2"]
}"""

def _call_gemini_native(url: str, payload_bytes: bytes) -> Optional[dict]:
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=12) as res:
        if res.status == 200:
            raw_body = res.read().decode("utf-8")
            data = json.loads(raw_body)
            text_content = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text_content)
    return None

async def query_gemini(req: TransductionRequest, bio: dict) -> Optional[dict]:
    if not GEMINI_API_KEY:
        return None

    user_prompt = f"""Descrizione del comportamento: "{req.user_text}"
Profilo biologico:
- Forma cranio: {req.snout} (Turbinati olfattivi: {bio['turbinates_cm2']} cm²)
- Vista: {bio['fov_degrees']}° (Acuità: {bio['acuity_cpd']} cpd)
- Altezza da terra: {bio['eye_height_cm']} cm
- Coda: {bio['tail_bias']}
- Orecchie: {bio['ear_mobility']}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json"
        }
    }
    payload_bytes = json.dumps(payload).encode("utf-8")

    try:
        return await asyncio.to_thread(_call_gemini_native, url, payload_bytes)
    except Exception as e:
        print(f"Errore chiamata Gemini: {e}")
        return None

# ==================== FALLBACK LOCALE ====================
def fallback_synthesis(text: str):
    text_lower = text.lower()
    if any(w in text_lower for w in ["salta", "festa", "torno", "rientro"]):
        return {
            "panksepp": "CARE", "arousal": 80, "valence": 35,
            "thought": "Sei tornato! Il rientro della mia persona preferita mi fa esplodere il cuore di gioia. La mia eccitazione è altissima: salto per starti vicino!",
            "explanation": "Il cane manifesta sollievo ed eccitazione affiliativa per la riunione sociale del branco.",
            "steps": ["Ruota il corpo a 45 gradi con le braccia conserte.", "Attendi in silenzio quattro zampe a terra.", "Accarezzalo sul petto quando calmo."],
            "forbidden": ["Non spingerlo con le mani (invita al gioco).", "Non urlare 'NO!' o 'GIÙ!'."]
        }
    elif any(w in text_lower for w in ["ringhia", "litiga", "aggressivo", "cane"]):
        return {
            "panksepp": "RAGE", "arousal": 88, "valence": -35,
            "thought": "Sagoma estranea frontale troppo vicina. Ringhio per fermare l'avanzata e difendere il mio spazio vitale!",
            "explanation": "L'avvicinamento frontale inibisce le traiettorie ad arco e innesca la difesa ritualizzata dello spazio.",
            "steps": ["Allarga la traiettoria compiendo un arco ampio.", "Fai da scudo visivo con il tuo corpo.", "Mantieni il guinzaglio morbido a 'U'."],
            "forbidden": ["Non punire il ringhio: togliere il segnale porta al morso improvviso.", "Non strattonare il guinzaglio."]
        }
    return {
        "panksepp": "SEEKING", "arousal": 45, "valence": 15,
        "thought": "Scansione dell'ambiente attiva. Analizzo gradienti olfattivi a terra e mantengo l'equilibrio.",
        "explanation": "Il cane si trova in uno stato di perlustrazione serena e monitoraggio degli stimoli.",
        "steps": ["Concedigli il tempo di analizzare gli odori a terra.", "Mantieni un passo calmo e fluido."],
        "forbidden": ["Non strapparlo via bruscamente mentre annusa."]
    }

# ==================== ENDPOINT API ====================
@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    bio = SensoryEngine.compute(req)
    synth = await query_gemini(req, bio)
    engine_used = "neural_gemini_flash"
    
    if not synth:
        synth = fallback_synthesis(req.user_text)
        engine_used = "rule_fallback"

    return {
        "status": "success",
        "engine": engine_used,
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
        "gemini_active": bool(GEMINI_API_KEY)
    }
