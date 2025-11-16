# check_models.py — FINAL VERSION (Metadata-aware)
import os, json, joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

MODEL_DIR = "models"

# ---------------- DEVICE ----------------
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")
print("Device set to use", device)

# ---------------- LOAD ENCODERS ----------------
xlmr_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "xlm-roberta-base"))
xlmr_model = AutoModel.from_pretrained(os.path.join(MODEL_DIR, "xlm-roberta-base")).to(device).eval()

minilm_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384"))
minilm_model = AutoModel.from_pretrained(os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384")).to(device).eval()

# ---------------- DISTILGPT2 (CPU PERPLEXITY) ----------------
gen_pipe = pipeline("text-generation",
                    model=os.path.join(MODEL_DIR, "distilgpt2"),
                    device=-1)
gen_tok = gen_pipe.tokenizer
gen_mdl = gen_pipe.model.cpu().eval()
if gen_tok.pad_token is None:
    gen_tok.pad_token = gen_tok.eos_token
gen_mdl.config.pad_token_id = gen_tok.pad_token_id

def compute_perp(t):
    enc = gen_tok([t], return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
    with torch.no_grad():
        out = gen_mdl(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).cpu().item())

def div(t):
    w = t.split()
    return len(set(w))/len(w) if w else 0

def embed_xlmr(t):
    enc = xlmr_tok(t, return_tensors="pt", padding=True,
                   truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = xlmr_model(**enc)
    return out.last_hidden_state[:,0,:].cpu().numpy().flatten()

def embed_minilm(t):
    enc = minilm_tok(t, return_tensors="pt", padding=True,
                     truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = minilm_model(**enc)
    return out.last_hidden_state[:,0,:].cpu().numpy().flatten()

# ---------------- CHECK MODELS ----------------
sample = "This is a simple test sentence to verify model feature dimensions."

clf_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")])

for file in clf_files:
    name = file.replace(".joblib","")
    print("\n==============================")
    print("Checking", name)
    print("==============================")

    model_path = os.path.join(MODEL_DIR, file)
    meta_path = os.path.join(MODEL_DIR, f"{name}.meta.json")

    clf = joblib.load(model_path)

    if not os.path.exists(meta_path):
        print("❌ Missing metadata! Skipping.")
        continue

    meta = json.load(open(meta_path))

    encoder = meta["encoder"]
    expected_dim = meta["feature_dim"]

    print("Encoder:", encoder)
    print("Stored feature_dim:", expected_dim)
    print("Model n_features_in_:", clf.n_features_in_)

    # Build test vectors
    perp = compute_perp(sample)
    dv = div(sample)

    feat_minilm = np.hstack([[perp, dv], embed_minilm(sample)])
    feat_xlmr   = np.hstack([[perp, dv], embed_xlmr(sample)])

    # Check correct encoder
    if encoder == "minilm":
        correct_vec = feat_minilm
    else:
        correct_vec = feat_xlmr

    print("Correct vector dim:", correct_vec.shape[0])

    # Validate
    if correct_vec.shape[0] != expected_dim:
        print("❌ DIMENSION MISMATCH!")
    else:
        print("✅ Dimension OK.")

    # Test prediction
    try:
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(correct_vec.reshape(1,-1))[0]
        else:
            pred = clf.predict(correct_vec.reshape(1,-1))[0]
            proba = [1-pred, pred]

        print("Predict-Proba:", proba)
        print("✅ Model functional.")
    except Exception as e:
        print("❌ Prediction FAILED:", e)