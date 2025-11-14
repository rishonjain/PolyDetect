# inference.py
"""
Unified inference script for PolyDetect.
Supports:
- XLM-R embeddings (768-d)
- MiniLM embeddings (384-d)
- Perplexity + Lexical Diversity
- All classifiers polydetect_*.joblib
"""

import os
import sys
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel, pipeline

# ---------------- GPU Check ----------------
if not torch.cuda.is_available():
    print("⚠️ GPU not found — embeddings computed on CPU (slower).")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

# ---------------- Paths ----------------
MODEL_DIR = "models"
XLMR_PATH = os.path.join(MODEL_DIR, "xlm-roberta-base")
MINILM_PATH = os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384")
GPT2_PATH = os.path.join(MODEL_DIR, "distilgpt2")

# ---------------- Load Embedding Models ----------------
print("\n📥 Loading XLM-R...")
xlmr_tok = AutoTokenizer.from_pretrained(XLMR_PATH)
xlmr_model = AutoModel.from_pretrained(XLMR_PATH).to(DEVICE).eval()

print("📥 Loading MiniLM...")
minilm_tok = AutoTokenizer.from_pretrained(MINILM_PATH)
minilm_model = AutoModel.from_pretrained(MINILM_PATH).to(DEVICE).eval()
MINILM_DIM = minilm_model.config.hidden_size  # 384

# ---------------- Load GPT-2 Perplexity Model (CPU) ----------------
print("\n📥 Loading distilgpt2 (CPU) for perplexity...")
gen_pipe = pipeline("text-generation", model=GPT2_PATH, device=-1)
gen_tok = gen_pipe.tokenizer
gen_mdl = gen_pipe.model.cpu().eval()

if gen_tok.pad_token is None:
    gen_tok.pad_token = gen_tok.eos_token
gen_mdl.config.pad_token_id = gen_tok.pad_token_id

# ---------------- Feature Functions ----------------
def compute_perplexity(text):
    enc = gen_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
    for k in enc: enc[k] = enc[k].cpu()  # ensure CPU
    with torch.no_grad():
        out = gen_mdl(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).cpu().item())

def compute_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0

def embed_xlmr(text):
    enc = xlmr_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = xlmr_model(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy().flatten()

def embed_minilm(text):
    enc = minilm_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = minilm_model(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy().flatten()

# ---------------- Build Feature Vectors ----------------
def build_features(text, encoder_type):
    """Return proper feature vector for each classifier type."""
    perp = compute_perplexity(text)
    div = compute_diversity(text)

    if encoder_type == "minilm":
        emb = embed_minilm(text)
    else:
        emb = embed_xlmr(text)

    return np.hstack([[perp], [div], emb]).reshape(1, -1)

# ---------------- Load Classifiers ----------------
print("\n🔎 Loading classifiers...")
classifiers = {}
for f in os.listdir(MODEL_DIR):
    if f.startswith("polydetect_") and f.endswith(".joblib"):
        name = f.replace("polydetect_", "").replace(".joblib", "")
        classifiers[name] = joblib.load(os.path.join(MODEL_DIR, f))
        print(f"  ✔ Loaded {name}")

# ---------------- Inference Function ----------------
def predict(text):
    results = {}

    for name, clf in classifiers.items():

        # detect encoder type
        if "minilm" in name:
            X = build_features(text, encoder_type="minilm")
            encoder = "MiniLM (384d)"
        else:
            X = build_features(text, encoder_type="xlmr")
            encoder = "XLM-R (768d)"

        # predict label
        try:
            y_pred = clf.predict(X)[0]
        except:
            y_pred = clf.predict(X.astype(np.float32))[0]

        # predict confidence if available
        if hasattr(clf, "predict_proba"):
            conf = float(np.max(clf.predict_proba(X)))
        elif hasattr(clf, "decision_function"):
            df = clf.decision_function(X)
            conf = float(df if np.isscalar(df) else np.max(df))
        else:
            conf = None

        results[name] = {
            "prediction": int(y_pred),
            "confidence": conf,
            "encoder": encoder,
            "features_dim": X.shape[1]
        }

    return results

# ---------------- CLI Interface ----------------
if __name__ == "__main__":
    print("\n📝 PolyDetect Inference")
    while True:
        text = input("\nEnter text (or 'quit'): ")
        if text.lower() == "quit":
            break

        res = predict(text)

        print("\n=== RESULTS ===")
        for model_name, info in res.items():
            print(f"\n🔹 Model: {model_name}")
            print(f"   Encoder: {info['encoder']}")
            print(f"   Prediction: {info['prediction']}")
            print(f"   Confidence: {info['confidence']}")
            print(f"   Feature dim: {info['features_dim']}")