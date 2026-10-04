from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import math
import re

app = FastAPI(
    title="BioDog.io Neural Sensory Engine",
    description="Backend multi-agente per la trasduzione dell'Umwelt canino",
    version="1.0.0"
)

# Abilitazione CORS per consentire le chiamate da www.biodog.io
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== SCHEMI DATI ====================
class TransductionRequest(BaseModel):
    user_text: str = Field(..., min_length=2, max_length=500)
    snout: str = Field(default="normal")    # flat, normal, long
    ears: str = Field(default="prick")      # prick, drop, long
    size: str = Field(default="medium")     # small, medium, large
    tail: str = Field(default="long")       # long, curly, short
    observed_time_hours: Optional[float] = 0.0

# ==================== AGENTI SENSORIALI COCHAIN ====================
class OlfactoryAgent:
    @staticmethod
    def process(text: str, snout: str, hours: float):
        turbinates = 170.0 if snout == "long" else (45.0 if snout == "flat" else 100.0)
        decay = 0.35
        detected_hours = hours
        
        match = re.search(r"(\d+)\s*(?:ore|ora|h)", text, re.IGNORECASE)
        if match:
            detected_hours = float(match.group(1))
        elif re.search(r"tutto il giorno|sempre", text, re.IGNORECASE):
            detected_hours = 8.0

        if detected_hours > 0:
            residual = max(5.0, 100.0 * math.exp(-decay * detected_hours))
            delta = -(100.0 - residual)
            desc = f"Decadimento esponenziale VOC umano su {detected_hours}h. Concentrazione residua al {residual:.1f}%."
        else:
            residual = 95.0
            delta = 95.0
            desc = "Saturazione molecolare al picco: transito chimico recente del referente umano."

        return {
            "turbinates_cm2": turbinates,
            "residual_density": round(residual, 1),
            "delta_voc": round(delta, 1),
            "description": desc
        }

class VisualAgent:
    @staticmethod
    def process(text: str, snout: str, size: str):
        fov = 270 if snout == "long" else (220 if snout == "flat" else 250)
        cpd = 12.5 if snout == "long" else (9.5 if snout == "flat" else 11.5)
        eye_height = 25 if size == "small" else (75 if size == "large" else 45)
        
        motion_focus = "Scansione basale orizzontale."
        if any(w in text.lower() for w in ["corre", "salta", "scatta", "bici", "gatto", "monopattino", "mosche"]):
            motion_focus = "Priorità magnocellulare: tracking rapido del movimento periferico."

        return {
            "fov_degrees": fov,
            "acuity_cpd": cpd,
            "eye_height_cm": eye_height,
            "motion_salience": motion_focus,
            "spectrum": "Dicromatico attivo (429-555 nm: contrasto blu-giallo e assenza canale rosso)."
        }

class AcousticAgent:
    @staticmethod
    def process(text: str, ears: str):
        text_lower = text.lower()
        pitch_hz = 250.0
        urgency = "Normale / Relazionale"

        if any(w in text_lower for w in ["festa", "bravo", "bello", "amore", "gioia"]) or "!" in text:
            pitch_hz = 420.0
            urgency = "Bassa / Affiliativa (Dog-Directed Speech accogliente)"
        elif any(w in text_lower for w in ["no", "smettila", "basta", "fermo", "ringhia"]):
            pitch_hz = 110.0
            urgency = "Alta / Allarme Minaccia (Frequenza aspramente grave < 150 Hz)"
        elif any(w in text_lower for w in ["ulula", "piange", "solo"]):
            pitch_hz = 550.0
            urgency = "Distress Acuto / Richiamo a Lungo Raggio"

        mobility = "Flessibilità 180° e scansione stereofonica indipendente" if ears == "prick" else "Assorbimento passivo frontale"
        return {
            "pitch_hz": pitch_hz,
            "urgency": urgency,
            "ear_mobility": mobility
        }

class SynthesisAgent:
    @staticmethod
    def synthesize(text: str, snout: str, tail: str):
        text_lower = text.lower()

        if any(w in text_lower for w in ["salta", "festa", "torno", "rientro"]):
            return {
                "panksepp": "CARE",
                "arousal": 80,
                "valence": 35,
                "thought": "Porta si apre. Profilo chimico tuo al massimo. Voce acuta e familiare. Non riesco a contenere il corpo: salto in alto per intercettare il tuo viso e salutarti!",
                "buttons": ["PORTA APRE", "ODORE TUO", "SALTO SU", "CUORE VELOCE"],
                "explanation": "Il cane prova un sollievo travolgente e una forte eccitazione affettiva per il rientro della figura di riferimento.",
                "steps": [
                  "Ruota il corpo di lato a 45 gradi e incrocia le braccia. Ignora i salti con calma e senza parlare.",
                  "Attendi in silenzio finché non poggia tutte e quattro le zampe sul pavimento.",
                  "Appena si è calmato, abbassati tu al suo livello e accarezzagli delicatamente il petto."
                ],
                "forbidden": [
                  "Non spingerlo con le mani: per lui toccarlo è un invito a giocare alla lotta e salterà ancora di più.",
                  "Non urlare 'NO!' o 'GIÙ!': la voce concitata alza ancora di più la sua agitazione."
                ]
            }
        elif any(w in text_lower for w in ["ringhia", "litiga", "aggressivo", "attacca", "altro cane"]):
            return {
                "panksepp": "RAGE",
                "arousal": 88,
                "valence": -35,
                "thought": "Sagoma estranea frontale troppo vicina. Il guinzaglio mi blocca la fuga. Mostro i denti anteriori e ringhio per fermare la sua avanzata prima che sia troppo tardi.",
                "buttons": ["ALTRO CANE", "SPAZIO CHIUSO", "RINGHIO STOP", "CREA DISTANZA"],
                "explanation": "L'avvicinamento frontale diretto inibisce le traiettorie naturali ad arco, attivando una risposta difensiva ritualizzata per proteggere lo spazio vitale.",
                "steps": [
                  "Allarga subito la traiettoria compiendo un arco ampio di almeno 5-6 metri.",
                  "Mettiti fisicamente tra il tuo cane e l'estraneo facendo da scudo visivo.",
                  "Mantieni il guinzaglio morbido a 'U' per non attivare il riflesso di opposizione."
                ],
                "forbidden": [
                  "Non punire il ringhio: se gli togli l'avvertimento, in futuro morderà direttamente!",
                  "Non strattonare il guinzaglio con violenza stringendogli la gola."
                ]
            }
        elif any(w in text_lower for w in ["lecca", "zampe", "sangue"]):
            return {
                "panksepp": "PANIC/GRIEF",
                "arousal": 75,
                "valence": -30,
                "thought": "Ansia interna costante. Leccarmi le zampe rilascia piccole dosi di endorfine che mi calmano. Continuo a farlo per sopportare il disagio.",
                "buttons": ["ANSIA INTERNA", "LECCA ZAMPA", "CERCO CALMA", "STRESS"],
                "explanation": "Il leccamento compulsivo focale funge da autolenimento (displacement) per contrastare stati di stress cronico o noia prolungata.",
                "steps": [
                  "Fai visitare le zampe dal veterinario per escludere dermatiti, corpi estranei o allergie.",
                  "Offrigli masticativi naturali duraturi (es. corno di cervo o legno d'ulivo) per scaricare lo stress.",
                  "Aumenta le uscite dedicate al fiuto libero nella natura per liberare la mente."
                ],
                "forbidden": [
                  "Non sgridarlo mentre si lecca: la punizione alimenta l'ansia e lo farà di nascosto.",
                  "Non limitarti a fasciargli la zampa senza curare la causa psicologica."
                ]
            }
        else:
            return {
                "panksepp": "SEEKING",
                "arousal": 45,
                "valence": 15,
                "thought": "Scansione dell'ambiente attiva. Analizzo tracce olfattive a terra e monitoro i suoni intorno. Mi sento curioso e vigile.",
                "buttons": ["AMBIENTE", "ODORI NUOVI", "ESPLORO", "CALMO"],
                "explanation": "Il cane si trova in uno stato di perlustrazione serena, guidato dalla naturale curiosità e dal fiuto.",
                "steps": [
                  "Lascialo annusare i punti che attirano la sua attenzione durante la passeggiata.",
                  "Cammina a passo calmo e asseconda le sue esplorazioni."
                ],
                "forbidden": [
                  "Non strattonarlo via mentre è assorto nell'analisi di un marcatore olfattivo."
                ]
            }

# ==================== ENDPOINT PRINCIPALE ====================
@app.post("/api/v1/umwelt/transduce")
async def transduce(req: TransductionRequest):
    olf = OlfactoryAgent.process(req.user_text, req.snout, req.observed_time_hours)
    vis = VisualAgent.process(req.user_text, req.snout, req.size)
    acu = AcousticAgent.process(req.user_text, req.ears)
    synth = SynthesisAgent.synthesize(req.user_text, req.snout, req.tail)

    return {
        "status": "success",
        "morphology_profile": {
            "snout": req.snout,
            "ears": req.ears,
            "size": req.size,
            "tail": req.tail
        },
        "sensory_telemetry": {
            "olfactory": olf,
            "visual": vis,
            "acoustic": acu
        },
        "neural_synthesis": synth
    }

@app.get("/")
async def root():
    return {"status": "BioDog Neural Engine Online"}
