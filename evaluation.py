# evaluation.py
"""
Evaluation (L2) for PolyDetect
- Option A: clean white-style plots for confusion matrices and ROC curves
- Produces:
    - metrics_language.csv (per-model per-language metrics)
    - metrics_combined.csv (averaged across languages per model)
    - confusion_matrices/{model}_{lang}.png
    - roc_curves/{model}_{lang}.png
Notes:
- Uses MiniLM (384-d) embeddings for minilm classifiers, XLM-R (768-d) for others
- Uses distilgpt2 on CPU for perplexity to avoid GPU OOM
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve
)

import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# ----------------- Config -----------------
DATA_PATH = "data/multitude_v3_clean.csv"
MODEL_DIR = "models"
CLASSIFIER_GLOB = os.path.join(MODEL_DIR, "polydetect_*.joblib")

# embedding model folders (local)
XLMR_PATH = os.path.join(MODEL_DIR, "xlm-roberta-base")
MINILM_PATH = os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384")

GEN_MODEL_NAME = os.path.join(MODEL_DIR, "distilgpt2")  # local
OUT_LANG_METRICS = "metrics_language.csv"
OUT_COMBINED = "metrics_combined.csv"

# output dirs for images
CM_DIR = "confusion_matrices"
ROC_DIR = "roc_curves"
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)

LANGS = ["en", "es", "de", "ru", "zh"]
N_PER_LANG = 200  # L2 fast mode

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Load dataset -----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Put multitude_v3_clean.csv in data/")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "language", "label"])
df = df[df["language"].isin(LANGS)]
print(f"Total filtered samples: {len(df)}")

# ----------------- Sample by language -----------------
lang_samples = {}
for lang in LANGS:
    df_lang = df[df["language"] == lang]
    sample = df_lang.sample(n=min(N_PER_LANG, len(df_lang)), random_state=42)
    lang_samples[lang] = (sample["text"].tolist(), sample["label"].astype(int).tolist())
    print(f"{lang}: using {len(sample)} samples")

# ----------------- Load distilgpt2 on CPU for perplexity -----------------
print("Loading distilgpt2 on CPU to avoid GPU OOM (perplexity on CPU)...")
gen_pipe = pipeline("text-generation", model=GEN_MODEL_NAME, device=-1)
gen_tok = gen_pipe.tokenizer
gen_mdl = gen_pipe.model.cpu().eval()

if gen_tok.pad_token is None:
    gen_tok.pad_token = gen_tok.eos_token
gen_mdl.config.pad_token_id = gen_tok.pad_token_id

# ----------------- Feature function settings -----------------
EMB_BATCH = 8      # embedding batch to fit into 4GB GPU
PPL_MAX_LEN = 256  # max length for perplexity tokens

def compute_perplexity(texts):
    """Compute perplexity on CPU per text (safe for low-VRAM GPUs)."""
    vals = []
    # tqdm for live ETA
    for t in tqdm(texts, desc="Perplexity", unit="it"):
        # tokenize -> BatchEncoding; move tensors individually to CPU
        enc = gen_tok([t], return_tensors="pt", padding=True, truncation=True, max_length=PPL_MAX_LEN)
        for k in enc:
            enc[k] = enc[k].cpu()
        with torch.no_grad():
            out = gen_mdl(**enc, labels=enc["input_ids"])
        perp = float(torch.exp(out.loss).cpu().item())
        vals.append(perp)
        gc.collect()
        torch.cuda.empty_cache()
    return np.array(vals)

def compute_diversity(texts):
    arr = []
    for t in tqdm(texts, desc="Diversity", unit="it"):
        words = str(t).split()
        arr.append(len(set(words))/len(words) if words else 0.0)
    return np.array(arr)

def compute_embeddings(texts, tokenizer, model):
    """Compute CLS embeddings with a tqdm progress bar (EMB_BATCH)."""
    outs = []
    it = range(0, len(texts), EMB_BATCH)
    for i in tqdm(it, desc=f"Embeddings ({tokenizer.name_or_path})", unit="batch"):
        batch = texts[i:i+EMB_BATCH]
        inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model(**inp)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        outs.append(cls)
        torch.cuda.empty_cache()
    if len(outs) == 0:
        return np.zeros((0, model.config.hidden_size))
    return np.vstack(outs)

# ----------------- Load classifier files -----------------
clf_files = sorted(glob.glob(CLASSIFIER_GLOB))
if not clf_files:
    raise FileNotFoundError(f"No classifiers found matching {CLASSIFIER_GLOB}")
print(f"Found {len(clf_files)} classifier(s):")
for f in clf_files:
    print(" -", os.path.basename(f))

# ----------------- Load embedding models -----------------
print("Loading XLM-R tokenizer + model ...")
xlmr_tok = AutoTokenizer.from_pretrained(XLMR_PATH)
xlmr_model = AutoModel.from_pretrained(XLMR_PATH).to(device).eval()

print("Loading MiniLM tokenizer + model ...")
minilm_tok = AutoTokenizer.from_pretrained(MINILM_PATH)
minilm_model = AutoModel.from_pretrained(MINILM_PATH).to(device).eval()
MINILM_DIM = minilm_model.config.hidden_size  # should be 384

# ----------------- Evaluation loop -----------------
records = []

for lang in LANGS:
    texts, labels = lang_samples[lang]
    n = len(texts)
    if n == 0:
        continue

    print(f"\n=== Evaluating language: {lang} (n={n}) ===")

    # Shared features (with tqdm)
    perps = compute_perplexity(texts)         # shape (n,)
    divs = compute_diversity(texts)          # shape (n,)

    # Embeddings (precompute both so each classifier can use correct encoder)
    embs_xlmr = compute_embeddings(texts, xlmr_tok, xlmr_model)   # (n, 768)
    embs_minilm = compute_embeddings(texts, minilm_tok, minilm_model)  # (n, MINILM_DIM)

    # Loop classifiers with progress bar (per-language)
    for clf_path in tqdm(clf_files, desc=f"Models ({lang})", unit="model"):
        model_base = os.path.basename(clf_path)
        model_name = model_base.replace("polydetect_", "").replace(".joblib", "")
        try:
            clf = joblib.load(clf_path)
        except Exception as e:
            print(f"Failed to load {clf_path}: {e}")
            continue

        # Select proper embedding based on filename
        if "minilm" in model_name:
            X = np.hstack([perps.reshape(-1,1), divs.reshape(-1,1), embs_minilm])
        else:
            X = np.hstack([perps.reshape(-1,1), divs.reshape(-1,1), embs_xlmr])

        # Prediction (with safe fallback)
        try:
            y_pred = clf.predict(X)
        except Exception as e:
            # try casting to float32
            try:
                y_pred = clf.predict(X.astype(np.float32))
            except Exception as e2:
                print(f"Prediction failed for {model_name} on {lang}: {e2}")
                continue

        # Compute metrics
        acc = accuracy_score(labels, y_pred)
        f1_macro = f1_score(labels, y_pred, average="macro")
        f1_weighted = f1_score(labels, y_pred, average="weighted")
        prec_weighted = precision_score(labels, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(labels, y_pred, average="weighted")

        # Confusion matrix (handle case where not binary)
        cm = confusion_matrix(labels, y_pred)
        # If binary shape (2,2), unpack; else compute FPR/FNR using first positive class if possible
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # fallback: sum off-diagonals
            tp = np.trace(cm)
            fp = cm.sum(axis=0).sum() - np.trace(cm)
            fn = cm.sum(axis=1).sum() - np.trace(cm)
            tn = 0

        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)

        # AUC-ROC and ROC curve plotting (only when binary or classifier provides scores)
        auc_roc = None
        roc_fig_path = None
        try:
            # prefer predict_proba
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X)
                # choose positive class as column 1 if binary; else try first non-zero column
                if prob.shape[1] == 2:
                    y_score = prob[:,1]
                    auc_roc = roc_auc_score(labels, y_score)
                else:
                    # multiclass: compute macro average using one-vs-rest
                    try:
                        auc_roc = roc_auc_score(labels, prob, multi_class="ovr", average="macro")
                        # create a simple macro ROC (not plotted per-class)
                        y_score = None
                    except:
                        y_score = None
            elif hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X)
                auc_roc = roc_auc_score(labels, y_score)
            else:
                y_score = None
        except Exception:
            y_score = None
            auc_roc = None

        # Save confusion matrix plot (Option A: clean white)
        try:
            cm_fname = f"{model_name}_{lang}.png".replace(" ", "_")
            cm_path = os.path.join(CM_DIR, cm_fname)
            plt.figure(figsize=(4,4))
            plt.imshow(cm, interpolation='nearest', cmap='Greys')
            plt.title(f"Confusion Matrix: {model_name} [{lang}]")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar(shrink=0.6)
            # annotate cells
            thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Failed to save confusion matrix for {model_name} {lang}: {e}")
            cm_path = None

        # Save ROC curve if we have scores and it's binary (y_score or prob with two columns)
        if (('y_score' in locals() and y_score is not None) or
            (hasattr(clf, "predict_proba") and hasattr(prob, 'shape') and prob.shape[1] == 2)):
            try:
                # determine score array
                if 'y_score' in locals() and y_score is not None:
                    score_arr = y_score
                else:
                    score_arr = prob[:,1]

                fpr_vals, tpr_vals, _ = roc_curve(labels, score_arr)
                auc_val = auc_roc if auc_roc is not None else roc_auc_score(labels, score_arr)

                roc_fname = f"{model_name}_{lang}.png".replace(" ", "_")
                roc_path = os.path.join(ROC_DIR, roc_fname)
                plt.figure(figsize=(5,4))
                plt.plot(fpr_vals, tpr_vals, linewidth=2)
                plt.plot([0,1], [0,1], linestyle='--', linewidth=1, alpha=0.6)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC: {model_name} [{lang}] (AUC={auc_val:.3f})")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(roc_path, dpi=150, bbox_inches='tight')
                plt.close()
                roc_fig_path = roc_path
            except Exception as e:
                roc_fig_path = None
                # don't fail the run for plotting errors
                print(f"Could not plot/save ROC for {model_name} {lang}: {e}")

        # Print summary line
        print(f" {model_name} -> acc:{acc:.4f} f1_w:{f1_weighted:.4f} AUC:{(auc_roc if auc_roc is not None else 'N/A')}")

        # Add record
        records.append({
            "Language": lang,
            "Model": model_name,
            "Accuracy": acc,
            "F1_macro": f1_macro,
            "F1_weighted": f1_weighted,
            "Precision_weighted": prec_weighted,
            "Recall_weighted": rec_weighted,
            "FPR": fpr,
            "FNR": fnr,
            "AUC_ROC": auc_roc,
            "ConfusionMatrixImage": cm_path,
            "ROCImage": roc_fig_path
        })

        # cleanup per-model
        gc.collect()
        torch.cuda.empty_cache()

# ----------------- Save metrics -----------------
if records:
    df_lang = pd.DataFrame(records)
    df_lang.to_csv(OUT_LANG_METRICS, index=False)
    print(f"\nSaved per-language metrics to {OUT_LANG_METRICS}")

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
    df_comb.to_csv(OUT_COMBINED, index=False)
    print(f"Saved combined metrics to {OUT_COMBINED}")
else:
    print("No records to save.")

print("\nEvaluation (L2) complete.")