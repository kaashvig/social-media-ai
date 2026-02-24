from transformers import pipeline
sentiment_model = pipeline("sentiment-analysis")

def analyze_sentiment(text: str):
    result = sentiment_model(text)[0]
    return {
        "label": result["label"],
        "score": round(float(result["score"]), 2)
    }

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)


def detect_mood(text: str):
    result = emotion_model(text)[0]
    return {
        "mood": result["label"],
        "confidence": round(float(result["score"]), 2)
    }

# Load once globally
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None   
)

def analyze_toxicity(text: str):
    try:
        results = toxicity_model(text)[0]

        toxic_score = 0.0

        for r in results:
            if "toxic" in r["label"].lower():
                toxic_score = float(r["score"])

        prediction = "Toxic" if toxic_score >= 0.5 else "Non-Toxic"

        return {
            "prediction": prediction,
            "confidence": round(toxic_score, 4)
        }

    except Exception as e:
        return {
            "error": str(e)
        }

def detect_toxicity(text: str):
    result = toxicity_model(text)[0]
    return {
        "toxicity": result["label"],
        "confidence": round(float(result["score"]), 2)
    }
