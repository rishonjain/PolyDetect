# app.py
import os, glob, json, math
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
MODEL_DIR = "models"
XLMR_PATH = os.path.join(MODEL_DIR, "xlm-roberta-base")
MINILM_PATH = os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384")
BERT_PATH = os.path.join(MODEL_DIR, "bert-base-multilingual-cased")
DISTILBERT_PATH = os.path.join(MODEL_DIR, "distilbert-base-multilingual-cased")
GPT2_PATH = os.path.join(MODEL_DIR, "distilgpt2")

CLASSIFIER_GLOB = os.path.join(MODEL_DIR, "polydetect_*.joblib")
METRICS_COMBINED = "metrics_combined.csv"
METRICS_LANG = "metrics_language.csv"
CM_DIR = "confusion_matrices"
ROC_DIR = "roc_curves"
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.markdown(f"**Device:** {device}")
st.title("🧠 PolyDetect — Multilingual AI Text Detector")

# ---------------- LOAD ENCODERS (CACHED) ----------------
@st.cache_resource
def load_embedding_models():
    xlmr_tok = AutoTokenizer.from_pretrained(XLMR_PATH)
    xlmr_model = AutoModel.from_pretrained(XLMR_PATH).to(device).eval()

    minilm_tok = AutoTokenizer.from_pretrained(MINILM_PATH)
    minilm_model = AutoModel.from_pretrained(MINILM_PATH).to(device).eval()

    bert_tok = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModel.from_pretrained(BERT_PATH).to(device).eval()

    distil_tok = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
    distil_model = AutoModel.from_pretrained(DISTILBERT_PATH).to(device).eval()

    return {
        "xlmr": (xlmr_tok, xlmr_model),
        "minilm": (minilm_tok, minilm_model),
        "bert": (bert_tok, bert_model),
        "distilbert": (distil_tok, distil_model)
    }

@st.cache_resource
def load_gpt2():
    pipe = pipeline("text-generation", model=GPT2_PATH, device=-1)  # ensure CPU pipeline
    tok = pipe.tokenizer
    mdl = pipe.model.cpu().eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.config.pad_token_id = tok.pad_token_id
    return tok, mdl

encoders = load_embedding_models()
ppl_tok, ppl_mdl = load_gpt2()

# ---------------- METRICS UTIL ----------------
def load_metrics(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Model" in df.columns and "Encoder" not in df.columns:
        df["Encoder"] = df["Model"].apply(lambda x: x.split("_")[0] if "_" in x else "unknown")
    return df

metrics_combined = load_metrics(METRICS_COMBINED)
metrics_lang = load_metrics(METRICS_LANG)

# ---------------- FEATURE HELPERS ----------------
def stable_perplexity(p):
    # log scale & clip to reduce domination
    if p <= 0:
        return 0.0
    return float(max(0.0, min(math.log(p + 1.0), 10.0)))

def compute_perplexity(text):
    enc = ppl_tok([text], return_tensors="pt", padding=True,
                   truncation=True, max_length=256)
    for k in enc:
        enc[k] = enc[k].cpu()
    with torch.no_grad():
        out = ppl_mdl(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).cpu().item())

def compute_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0

def embed_text(text, encoder_key):
    tok, mdl = encoders[encoder_key]
    enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = mdl(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy().flatten()

def build_features_for_encoder(text, encoder):
    perp = compute_perplexity(text)
    perp = stable_perplexity(perp)
    div = compute_diversity(text)
    emb = embed_text(text, encoder)
    return np.hstack([[perp, div], emb]).reshape(1, -1)

# ---------------- LOAD CLASSIFIERS + META ----------------
classifiers = {}
meta_index = {}  # name -> meta dict

for f in sorted(glob.glob(CLASSIFIER_GLOB)):
    name = os.path.basename(f).replace(".joblib", "")
    clf = joblib.load(f)
    classifiers[name] = clf
    meta_path = os.path.join(MODEL_DIR, f"{name}.meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
    else:
        # fallback: try parse name: polydetect_{encoder}_{model}
        parts = name.split("_")
        meta = {
            "encoder": parts[1] if len(parts) > 2 else ("minilm" if "minilm" in name else "xlmr"),
            "embedding_dim": None,
            "feature_order": ["perplexity", "diversity", "embedding"],
            "feature_dim": getattr(clf, "n_features_in_", None)
        }
    meta_index[name] = meta

# ---------------- UI: Metrics ----------------
st.header("📊 Combined Model Performance")
if metrics_combined is None:
    st.warning("Run evaluation.py to generate metrics_combined.csv")
else:
    st.dataframe(metrics_combined)
    melt_vars = [c for c in ["Accuracy", "F1_macro", "F1_weighted", "Precision_weighted", "Recall_weighted", "AUC_ROC"] if c in metrics_combined.columns]
    melt = metrics_combined.melt(id_vars=["Model"], value_vars=melt_vars)
    fig = px.bar(melt, x="Model", y="value", color="variable", barmode="group", title="Model Performance")
    fig.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

st.header("🌍 Per-Language Evaluation")
if metrics_lang is None:
    st.warning("Run evaluation.py to generate metrics_language.csv")
else:
    lang = st.selectbox("Language", ["All"] + sorted(metrics_lang["Language"].unique()))
    df = metrics_lang if lang == "All" else metrics_lang[metrics_lang["Language"] == lang]
    st.dataframe(df)

# ---------------- UI: Confusion/ROC preview ----------------
st.header("🖼 Confusion Matrices & ROC Curves")
model_names = sorted(list(classifiers.keys()))
sel_model_img = st.selectbox("Choose model image", model_names)
cm_path = os.path.join(CM_DIR, f"{sel_model_img}_en.png")
roc_path = os.path.join(ROC_DIR, f"{sel_model_img}_en.png")
if os.path.exists(cm_path):
    st.subheader("Confusion Matrix (EN)")
    st.image(cm_path)
else:
    st.info("No confusion matrix found for this model.")
if os.path.exists(roc_path):
    st.subheader("ROC Curve (EN)")
    st.image(roc_path)
else:
    st.info("No ROC curve found for this model.")

# ---------------- UI: Live Inference ----------------
st.header("🧪 Live Inference (robust)")
sel_clf = st.selectbox("Choose classifier", model_names)
text_input = st.text_area("Enter text", height=160)

if st.button("Analyze Text"):
    if len(text_input.strip().split()) < 5:
        st.warning("Please enter at least 5 words.")
    else:
        model_key = sel_clf
        clf = classifiers[model_key]
        meta = meta_index[model_key]
        expected_dim = meta.get("feature_dim") or getattr(clf, "n_features_in_", None)
        preferred = meta.get("encoder", None)

        # build using preferred encoder
        if preferred is None:
            preferred = "minilm" if "minilm" in model_key else "xlmr"
        feats = build_features_for_encoder(text_input, encoder=preferred)

        # attempt alt encoder if dimension mismatch
        if expected_dim is not None and feats.shape[1] != expected_dim:
            alt = "xlmr" if preferred == "minilm" else "minilm"
            st.warning(f"Feature dim mismatch: built {feats.shape[1]} but model expects {expected_dim}. Trying {alt} encoder.")
            feats_alt = build_features_for_encoder(text_input, encoder=alt)
            if feats_alt.shape[1] == expected_dim:
                feats = feats_alt
                st.success(f"Switched encoder to {alt}.")
            else:
                st.error("Neither encoder produced expected feature dimension. Aborting.")
                st.stop()

        X = feats.astype(np.float32)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            human_p, ai_p = float(proba[0]), float(proba[1])
        else:
            pred = clf.predict(X)[0]
            human_p = 1.0 if pred == 0 else 0.0
            ai_p = 1.0 - human_p

        st.metric("🤖 AI Probability", f"{ai_p*100:.2f}%")
        st.metric("👤 Human Probability", f"{human_p*100:.2f}%")

        df_plot = pd.DataFrame({"Class":["Human","AI"], "Score":[human_p*100, ai_p*100]})
        fig = px.bar(df_plot, x="Class", y="Score", color="Class", text="Score", range_y=[0,100])
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

st.caption("PolyDetect ©2025")