import joblib
import pandas as pd
import os

# Resolve project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "final_heavy_user_model1.pkl")
model = joblib.load(MODEL_PATH)

# Feature names used during training
FEATURE_NAMES = list(model.feature_names_in_)
def _prepare_input(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=["App"], drop_first=False)

    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_NAMES]


def predict_heavy_user(data: dict):
    df = _prepare_input(data)
    return int(model.predict(df)[0])


def predict_heavy_user_proba(data: dict):
    df = _prepare_input(data)
    return float(model.predict_proba(df)[0][1])