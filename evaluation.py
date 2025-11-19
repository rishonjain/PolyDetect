"""
FINAL Evaluation Script for PolyDetect (GPU + Metadata + Metrics Folder)

Generates:
    - metrics/metrics_language.csv
    - metrics/metrics_combined.csv
    - confusion_matrices/{model}_{lang}.png
    - roc_curves/{model}_{lang}.png

Notes:
    - Perplexity computed on CPU (safe on Windows)
    - Embeddings computed on GPU (fast)
    - Reads model metadata + correct scalers
"""

import os
import glob
import json
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve
)

import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# -------------------------------------------------
# PATHS & CONFIG
# -------------------------------------------------
DATA_PATH = "data/multitude_v3_clean.csv"
MODEL_DIR = "models"
CLASSIFIER_PATTERN = os.path.join(MODEL_DIR, "polydetect_*.joblib")

METRICS_DIR = "metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

OUT_LANG = os.path.join(METRICS_DIR, "metrics_language.csv")
OUT_COMB = os.path.join(METRICS_DIR, "metrics_combined.csv")

CM_DIR = "confusion_matrices"
ROC_DIR = "roc_curves"
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)

LANGS = ["en", "es", "de", "ru", "zh"]
N_PER_LANG = 200

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")
print("Using device:", device)
if USE_GPU:
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset missing:", DATA_PATH)

df = pd.read_csv(DATA_PATH).dropna(subset=["text", "language", "label"])
df = df[df.language.isin(LANGS)]
print("Dataset size:", len(df))

# Sample balanced subset
lang_samples = {}
for lang in LANGS:
    subset = df[df.language == lang]
    sample = subset.sample(n=min(N_PER_LANG, len(subset)), random_state=42)
    lang_samples[lang] = (sample.text.tolist(), sample.label.astype(int).tolist())
    print(f"{lang}: {len(sample)} samples")

# -------------------------------------------------
# LOAD GPT-2 ON CPU (SAFE)
# -------------------------------------------------
print("\nLoading DistilGPT-2 (CPU only) for perplexity...")
gen_pipe = pipeline("text-generation",
                    model=os.path.join(MODEL_DIR, "distilgpt2"),
                    device=-1)

gen_tok = gen_pipe.tokenizer
gen_mdl = gen_pipe.model.cpu().eval()
if gen_tok.pad_token is None:
    gen_tok.pad_token = gen_tok.eos_token
gen_mdl.config.pad_token_id = gen_tok.pad_token_id

def compute_perplexity(texts):
    vals = []
    for t in tqdm(texts, desc="Perplexity", unit="it"):
        enc = gen_tok([t], return_tensors="pt", padding=True,
                       truncation=True, max_length=256)
        with torch.no_grad():
            out = gen_mdl(**enc, labels=enc["input_ids"])
        vals.append(float(torch.exp(out.loss).cpu().item()))
    return np.array(vals)

def compute_diversity(texts):
    return np.array([
        (len(set(t.split())) / len(t.split())) if t.split() else 0
        for t in texts
    ])

# -------------------------------------------------
# LOAD EMBEDDING MODELS (GPU)
# -------------------------------------------------
print("\nLoading embedding models...")

# XLM-R
xlmr_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "xlm-roberta-base"))
xlmr_model = AutoModel.from_pretrained(os.path.join(MODEL_DIR, "xlm-roberta-base")).to(device).eval()

# MiniLM
minilm_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384"))
minilm_model = AutoModel.from_pretrained(os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384")).to(device).eval()

def get_emb_xlmr(texts):
    out = []
    for t in tqdm(texts, desc="XLM-R embeddings"):
        enc = xlmr_tok(t, return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(device)
        with torch.no_grad():
            rep = xlmr_model(**enc)
        out.append(rep.last_hidden_state[:,0,:].cpu().numpy().flatten())
        if USE_GPU: torch.cuda.empty_cache()
    return np.vstack(out)

def get_emb_minilm(texts):
    out = []
    for t in tqdm(texts, desc="MiniLM embeddings"):
        enc = minilm_tok(t, return_tensors="pt", padding=True,
                         truncation=True, max_length=256).to(device)
        with torch.no_grad():
            rep = minilm_model(**enc)
        out.append(rep.last_hidden_state[:,0,:].cpu().numpy().flatten())
        if USE_GPU: torch.cuda.empty_cache()
    return np.vstack(out)

# -------------------------------------------------
# LOAD CLASSIFIERS
# -------------------------------------------------
clf_files = sorted(glob.glob(CLASSIFIER_PATTERN))
if not clf_files:
    raise RuntimeError("No classifiers found!")

print("\nClassifiers detected:")
for f in clf_files:
    print(" -", os.path.basename(f))

# -------------------------------------------------
# MAIN EVALUATION LOOP
# -------------------------------------------------
records = []

for lang in LANGS:
    texts, labels = lang_samples[lang]
    print(f"\n=== Evaluating {lang} ({len(texts)} samples) ===")

    perps = compute_perplexity(texts)
    divs  = compute_diversity(texts)
    emb_xlm = get_emb_xlmr(texts)
    emb_min = get_emb_minilm(texts)

    for clf_path in tqdm(clf_files, desc=f"Models [{lang}]", unit="model"):
        base = os.path.basename(clf_path).replace(".joblib","")
        meta_path = os.path.join(MODEL_DIR, f"{base}.meta.json")

        if not os.path.exists(meta_path):
            print("⚠ Missing metadata:", meta_path)
            continue

        clf = joblib.load(clf_path)
        meta = json.load(open(meta_path))

        encoder = meta["encoder"]
        scaler_path = os.path.join(MODEL_DIR, f"{encoder}_global_scaler.pkl")
        scaler = joblib.load(scaler_path)

        # Pick correct embedding
        if encoder == "minilm":
            X = np.hstack([perps[:,None], divs[:,None], emb_min])
        else:
            X = np.hstack([perps[:,None], divs[:,None], emb_xlm])

        # Scale only first two features
        X[:, :2] = scaler.transform(X[:, :2])

        # Predict
        try:
            if hasattr(clf, "predict_proba"):
                score = clf.predict_proba(X)[:,1]
                preds = (score >= 0.5).astype(int)
            else:
                preds = clf.predict(X)
                score = preds.astype(float)
        except Exception as e:
            print("Prediction failed:", e)
            continue

        # Metrics
        acc = accuracy_score(labels, preds)
        f1_m = f1_score(labels, preds, average="macro")
        f1_w = f1_score(labels, preds, average="weighted")
        prec = precision_score(labels, preds, average="weighted", zero_division=0)
        rec  = recall_score(labels, preds, average="weighted")

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)

        # AUC
        try:
            auc = roc_auc_score(labels, score)
        except:
            auc = None

        # Save Confusion Matrix
        cm_path = os.path.join(CM_DIR, f"{base}_{lang}.png")
        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap='Greys')
        plt.title(f"{base} [{lang}] — CM")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(cm_path, dpi=130)
        plt.close()

        # Save ROC curve
        roc_path = None
        if auc is not None:
            fpr_vals, tpr_vals, _ = roc_curve(labels, score)
            roc_path = os.path.join(ROC_DIR, f"{base}_{lang}.png")
            plt.figure(figsize=(5,4))
            plt.plot(fpr_vals, tpr_vals)
            plt.plot([0,1],[0,1],'--')
            plt.title(f"{base} [{lang}] — ROC (AUC={auc:.3f})")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=130)
            plt.close()

        # Add to records
        records.append({
            "Language": lang,
            "Model": base,
            "Accuracy": acc,
            "F1_macro": f1_m,
            "F1_weighted": f1_w,
            "Precision_weighted": prec,
            "Recall_weighted": rec,
            "FPR": fpr,
            "FNR": fnr,
            "AUC_ROC": auc,
            "ConfusionMatrix": cm_path,
            "ROC": roc_path
        })

        gc.collect()
        if USE_GPU: torch.cuda.empty_cache()

# -------------------------------------------------
# SAVE METRICS
# -------------------------------------------------
if len(records) == 0:
    print("\n❌ No records generated!")
else:
    df_lang = pd.DataFrame(records)
    df_lang.to_csv(OUT_LANG, index=False)
    print("\nSaved:", OUT_LANG)

    df_comb = df_lang.groupby("Model").agg({
        "Accuracy":"mean",
        "F1_macro":"mean",
        "F1_weighted":"mean",
        "Precision_weighted":"mean",
        "Recall_weighted":"mean",
        "FPR":"mean",
        "FNR":"mean",
        "AUC_ROC":"mean"
    }).reset_index()

    df_comb.to_csv(OUT_COMB, index=False)
    print("Saved:", OUT_COMB)

print("\n✔ Evaluation completed.")