import json
import os
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
STORAGE_PATH = os.path.join(PROJECT_ROOT, "data", "predictions.json")


def save_prediction(input_data: dict, prediction: int, confidence: float):
    record = {
        "timestamp": datetime.now().isoformat(),
        "Posts_Per_Day": input_data["Posts_Per_Day"],
        "Likes_Per_Day": input_data["Likes_Per_Day"],
        "Follows_Per_Day": input_data["Follows_Per_Day"],
        "App": input_data["App"],
        "prediction": "Heavy User" if prediction == 1 else "Normal User",
        "confidence": round(float(confidence), 2)
    }

    if os.path.exists(STORAGE_PATH):
        with open(STORAGE_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(STORAGE_PATH, "w") as f:
        json.dump(data, f, indent=4)


def load_history():
    if not os.path.exists(STORAGE_PATH):
        return []

    with open(STORAGE_PATH, "r") as f:
        return json.load(f)
