# 🧠 PolyDetect — Multilingual AI-Generated Text Detector  
**A GPU-accelerated hybrid detector for AI-generated text across 5 languages (EN/ES/DE/RU/ZH).**  
Combines transformer embeddings (XLM-R, MiniLM, mBERT, DistilBERT) with statistical-lexical cues and classical ML models.

---

## 🚀 Features

### **Multilingual Support**
- English  
- Spanish  
- German  
- Russian  
- Chinese  

### **Hybrid Detection Architecture**
PolyDetect uses a combination of:
- **Transformer embeddings** (XLM-R / MiniLM / BERT / DistilBERT)
- **Perplexity** (DistilGPT-2)
- **Lexical diversity**
- **Classical ML models**
  - XGBoost (GPU-accelerated)
  - Logistic Regression  
  - Random Forest  
  - SVM  
  - FFNN (MLPClassifier)

### **Model Formats**
Each trained model saves:
- `polydetect_{encoder}_{clf}.joblib`
- `polydetect_{encoder}_{clf}.meta.json`
- `{encoder}_global_scaler.pkl`

This guarantees **100% reproducible** inference.

### **GPU Optimization**
- Embeddings → **GPU**
- XGBoost → **GPU** (if available)
- Perplexity → **CPU** (avoids Windows CUDA deadlocks)
- Automatic fallback to CPU when GPU not available.

---

## 🏗 Project Structure

```
PolyDetect/
│
├── app.py                      # Streamlit frontend (GPU-enabled)
├── inference.py                # Generates CM & ROC (GPU)
├── evaluation.py               # Language-wise evaluation → metrics/
│
├── train_xlmr.py               # Train XLM-R models
├── train_minilm_multilingual.py
├── train_bert_multilingual.py
├── train_distilbert_multilingual.py
│
├── models/                     # Encoder folders + generated classifiers
│   ├── xlm-roberta-base/
│   ├── microsoft-MiniLM-L12-H384/
│   ├── distilgpt2/
│   ├── bert-base-multilingual-cased/
│   ├── distilbert-base-multilingual-cased/
│   ├── polydetect_xlmr_xgboost.joblib
│   ├── ...
│
├── metrics/                    # NEW (metrics CSVs stored here)
│   ├── metrics_combined.csv
│   ├── metrics_language.csv
│
├── confusion_matrices/         # Generated CM pngs
├── roc_curves/                 # Generated ROC pngs
│
├── data/
│   ├── multitude_v3_clean.csv
│
├── requirements.txt
└── README.md
```

---

## ⚡ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/PolyDetect
cd PolyDetect
```

### 2. Install dependencies

GPU-enabled PyTorch (CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install everything else:

```bash
pip install -r requirements.txt
```

---

## 🧪 Training All Models

Each training script automatically:
- Computes features  
- Runs embeddings on GPU  
- Runs perplexity on CPU  
- Saves scaler + metadata  
- Trains 5 different classifiers  

Example:

```bash
python train_xlmr.py
python train_minilm_multilingual.py
python train_bert_multilingual.py
python train_distilbert_multilingual.py
```

---

## 📊 Evaluation

To create:
- confusion_matrices/
- roc_curves/
- metrics/metrics_combined.csv
- metrics/metrics_language.csv

Run:

```bash
python evaluation.py
```

---

## 🎛 Running Streamlit App

```bash
streamlit run app.py
```

---

## 🧩 Model Metadata Structure

Every model has a JSON metadata file:

```json
{
  "encoder": "minilm",
  "embedding_dim": 384,
  "feature_order": ["perplexity", "diversity", "embedding"],
  "feature_dim": 386
}
```

This ensures the correct encoder, scaler, and feature pipeline are used during inference.

---

## 🧠 Live Inference (API-Ready)

```python
from inference import predict_text

prob = predict_text("your text here", model="polydetect_minilm_xgboost")
print(prob)
```

---

## 🛡 Safety Notes

Perplexity is always computed on **CPU** due to instability of GPT-2 pipeline on CUDA/Windows.  
Embeddings always run on **GPU** for speed.

---

## 📄 License
MIT License.

---

## 👨‍💻 Authors
- Rishon Jain (Lead Engineer, Researcher)
- PolyDetect Research Team (Bennett University)
