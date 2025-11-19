# app.py — FINAL PRODUCTION VERSION
import os
import glob
import json
import streamlit as st
import pandas as pd
import plotly.express as px

from inference import predict_text  # <-- use central inference pipeline

# -----------------------------------------
# PATHS
# -----------------------------------------
MODEL_DIR = "models"
METRICS_DIR = "metrics"
CM_DIR = "confusion_matrices"
ROC_DIR = "roc_curves"

os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)

METRICS_COMBINED = os.path.join(METRICS_DIR, "metrics_combined.csv")
METRICS_LANG     = os.path.join(METRICS_DIR, "metrics_language.csv")

# -----------------------------------------
# UI HEADER
# -----------------------------------------
st.set_page_config(layout="wide")
st.title("🧠 PolyDetect — Multilingual AI Text Detector")

# -----------------------------------------
# LOAD CLASSIFIERS + METADATA
# -----------------------------------------
models = []
for f in sorted(glob.glob(os.path.join(MODEL_DIR, "polydetect_*.joblib"))):
    name = os.path.basename(f).replace(".joblib","")
    meta_path = os.path.join(MODEL_DIR, f"{name}.meta.json")
    if os.path.exists(meta_path):
        models.append(name)

if not models:
    st.error("No trained PolyDetect models found in /models/. Run training first.")
    st.stop()

# -----------------------------------------
# LOAD METRICS CSVs
# -----------------------------------------
def load_metrics(path):
    return pd.read_csv(path) if os.path.exists(path) else None

metrics_combined = load_metrics(METRICS_COMBINED)
metrics_lang = load_metrics(METRICS_LANG)

# -----------------------------------------
# METRICS (COMBINED)
# -----------------------------------------
st.header("📊 Combined Model Performance")
if metrics_combined is None:
    st.warning("metrics/metrics_combined.csv not found. Run evaluation.py.")
else:
    st.dataframe(metrics_combined)

    melt = metrics_combined.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    fig = px.bar(melt, x="Model", y="Score", color="Metric", title="Model Performance", barmode="group")
    fig.update_layout(yaxis_range=[0,1.05])
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# METRICS (PER-LANGUAGE)
# -----------------------------------------
st.header("🌍 Per-Language Evaluation")
if metrics_lang is None:
    st.warning("metrics/metrics_language.csv not found. Run evaluation.py.")
else:
    lang = st.selectbox("Select language", ["All"] + sorted(metrics_lang["Language"].unique()))
    df_lang = metrics_lang if lang == "All" else metrics_lang[metrics_lang["Language"] == lang]
    st.dataframe(df_lang)

# -----------------------------------------
# CONFUSION MATRIX / ROC PREVIEW
# -----------------------------------------
st.header("🖼 Confusion Matrices & ROC Curves")

sel_model_img = st.selectbox("Choose model image", models)

cm_path  = os.path.join(CM_DIR,  f"{sel_model_img}_en.png")
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

# -----------------------------------------
# LIVE INFERENCE
# -----------------------------------------
st.header("🧪 Live Inference")
sel_model = st.selectbox("Choose classifier", models)
text_input = st.text_area("Enter text", height=180)

if st.button("Analyze Text"):
    if len(text_input.strip().split()) < 5:
        st.warning("Please enter at least 5 words.")
    else:
        result = predict_text(text_input, sel_model)

        human_p = result["human_probability"] * 100
        ai_p = result["ai_probability"] * 100

        st.metric("👤 Human Probability", f"{human_p:.2f}%")
        st.metric("🤖 AI Probability", f"{ai_p:.2f}%")

        df_plot = pd.DataFrame({
            "Class": ["Human", "AI"],
            "Score": [human_p, ai_p]
        })

        fig = px.bar(df_plot, x="Class", y="Score", color="Class", text="Score", range_y=[0,100])
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

st.caption("PolyDetect © 2025 — Accurate Multilingual AI Text Detection")