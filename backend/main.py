from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from backend.model import predict_heavy_user, predict_heavy_user_proba
from backend.analytics import usage_summary
from backend.storage import save_prediction, load_history
from backend.nlp import analyze_sentiment, detect_mood, detect_toxicity
from backend.nlp_clustering import cluster_texts
app = FastAPI(
    title="Privacy-Preserving Social Media Analytics",
    version="4.0"
)

# =========================
# ===== SCHEMAS ===========
# =========================

class UserInput(BaseModel):
    Posts_Per_Day: int
    Likes_Per_Day: int
    Follows_Per_Day: int
    App: str


class TextInput(BaseModel):
    text: str


class TextBatch(BaseModel):
    texts: List[str]
    n_clusters: int = 3


# =========================
# ===== HEALTH CHECK ======
# =========================

@app.get("/health")
def health():
    return {"status": "Backend running successfully"}


# =========================
# ===== DATA SUMMARY ======
# =========================

@app.get("/summary")
def summary():
    return usage_summary()


# =========================
# ===== HEAVY USER PREDICT
# =========================

@app.post("/predict")
def predict(data: UserInput):
    payload = data.dict()

    prediction = predict_heavy_user(payload)
    confidence = predict_heavy_user_proba(payload)

    # Phase 2 storage
    save_prediction(payload, prediction, confidence)

    return {
        "prediction": "Heavy User" if prediction == 1 else "Normal User",
        "confidence_score": round(confidence, 2)
    }


# =========================
# ===== SENTIMENT =========
# =========================

@app.post("/sentiment")
def sentiment(data: TextInput):
    return analyze_sentiment(data.text)


# =========================
# ===== MOOD DETECTION ====
# =========================

@app.post("/mood")
def mood(data: TextInput):
    return detect_mood(data.text)


# =========================
# ===== TOXICITY ==========
# =========================

@app.post("/toxicity")
def toxicity(data: TextInput):
    return detect_toxicity(data.text)


# =========================
# ===== CLUSTERING ========
# =========================

@app.post("/cluster")
def cluster(data: TextBatch):
    return cluster_texts(data.texts, data.n_clusters)


# =========================
# ===== HISTORY ===========
# =========================

@app.get("/history")
def history():
    return load_history()
