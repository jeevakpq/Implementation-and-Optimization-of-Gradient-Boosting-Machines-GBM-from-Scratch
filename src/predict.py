import os
import pickle
import numpy as np
from src.model import GradientBoostingClassifierScratch

# ✅ Robust path (production-safe)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gbm_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict(data):
    data = np.array(data)
    prob = model.predict_proba(data)
    pred = model.predict(data)

    return prob.tolist(), pred.tolist()