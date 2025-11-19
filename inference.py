# inference.py — FINAL PRODUCTION VERSION (No CLI testing)
"""
Fast single-text inference for PolyDetect.
- CPU DistilGPT-2 for perplexity
- GPU for embeddings (if available)
- Uses metadata + correct scalers
"""

import os, json, joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

MODEL_DIR = "models"

# -----------------------------------------------------
# DEVICE SETUP
# -----------------------------------------------------
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

# -----------------------------------------------------
# LOAD MODELS (cached)
# -----------------------------------------------------
# GPT-2 for perplexity (CPU always)
gen_pipe = pipeline(
    "text-generation",
    model=os.path.join(MODEL_DIR, "distilgpt2"),
    device=-1
)
gpt_tok = gen_pipe.tokenizer
gpt_mdl = gen_pipe.model.cpu().eval()

if gpt_tok.pad_token is None:
    gpt_tok.pad_token = gpt_tok.eos_token
gpt_mdl.config.pad_token_id = gpt_tok.pad_token_id

# XLM-R encoder (GPU)
xlmr_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "xlm-roberta-base"))
xlmr_mdl = AutoModel.from_pretrained(os.path.join(MODEL_DIR, "xlm-roberta-base")).to(device).eval()

# MiniLM encoder (GPU)
minilm_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384"))
minilm_mdl = AutoModel.from_pretrained(os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384")).to(device).eval()


# -----------------------------------------------------
# FEATURE FUNCTIONS
# -----------------------------------------------------
def compute_perplexity(text):
    enc = gpt_tok([text], return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
    with torch.no_grad():
        out = gpt_mdl(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).cpu().item())


def compute_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def embed_xlmr(text):
    enc = xlmr_tok(text, return_tensors="pt", padding=True,
                   truncation=True, max_length=256).to(device)
    with torch.no_grad():
        rep = xlmr_mdl(**enc)
    return rep.last_hidden_state[:, 0, :].cpu().numpy().flatten()


def embed_minilm(text):
    enc = minilm_tok(text, return_tensors="pt", padding=True,
                     truncation=True, max_length=256).to(device)
    with torch.no_grad():
        rep = minilm_mdl(**enc)
    return rep.last_hidden_state[:, 0, :].cpu().numpy().flatten()


def build_features(text, encoder):
    perp = compute_perplexity(text)
    div  = compute_diversity(text)

    if encoder == "minilm":
        emb = embed_minilm(text)
    else:
        emb = embed_xlmr(text)

    return np.hstack([perp, div, emb]).reshape(1, -1)


# -----------------------------------------------------
# MAIN INFERENCE FUNCTION
# -----------------------------------------------------
def predict_text(text, model_name):
    """
    Predict human/AI probability using a trained PolyDetect model.

    Parameters:
        text (str): Input text to classify
        model_name (str): Model filename WITHOUT .joblib extension
                          e.g. "polydetect_minilm_xgboost"

    Returns:
        dict = {
            "ai_probability": float,
            "human_probability": float,
            "predicted_label": int
        }
    """

    clf_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    meta_path = os.path.join(MODEL_DIR, f"{model_name}.meta.json")

    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Model not found: {clf_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    clf  = joblib.load(clf_path)
    meta = json.load(open(meta_path))

    encoder = meta["encoder"]

    # Load correct scaler
    scaler_path = os.path.join(MODEL_DIR, f"{encoder}_global_scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Build features
    X = build_features(text, encoder)

    # Scale perplexity + diversity
    X[:, :2] = scaler.transform(X[:, :2])

    # Predict
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        human_p, ai_p = proba[0], proba[1]
    else:
        pred = clf.predict(X)[0]
        human_p, ai_p = (1.0, 0.0) if pred == 0 else (0.0, 1.0)

    return {
        "ai_probability": float(ai_p),
        "human_probability": float(human_p),
        "predicted_label": int(ai_p >= 0.5)
    }
