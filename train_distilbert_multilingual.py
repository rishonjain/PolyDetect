# train_distilbert_multilingual.py
import os, sys, joblib, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel, pipeline
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# ---------------- GPU Enforcement ----------------
if not torch.cuda.is_available():
    sys.exit("❌ GPU with CUDA required.")
DEVICE = torch.device("cuda")
print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

# ---------------- Config ----------------
DATASET = "data/multitude_v3_clean.csv"
MODEL_DIR = "models/distilbert-base-multilingual-cased"   # ensure exists
BATCH_SIZE = 16
os.makedirs("precomputed_features/distilbert_multilingual", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------- Load Models ----------------
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE).eval()

gen_pipe = pipeline("text-generation", model=os.path.join("models", "distilgpt2"), device=0)
gen_tok = gen_pipe.tokenizer
gen_mdl = gen_pipe.model.to(DEVICE).eval()
if gen_tok.pad_token is None:
    gen_tok.pad_token = gen_tok.eos_token
gen_mdl.config.pad_token_id = gen_tok.pad_token_id

# ---------------- Dataset ----------------
df = pd.read_csv(DATASET)
df = df[df["language"].isin(["en","es","de","zh","ru"])].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], stratify=df["label"], test_size=0.2, random_state=42
)

# ---------------- Feature Functions ----------------
def get_perplexity(texts):
    vals = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Perplexity"):
        batch = texts[i:i+BATCH_SIZE]
        inp = gen_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = gen_mdl(**inp, labels=inp["input_ids"])
        batch_perp = float(torch.exp(out.loss).cpu().item())
        vals.extend([batch_perp] * len(batch))
    return np.array(vals)

def get_div(texts):
    res = []
    for t in texts:
        words = str(t).split()
        res.append(len(set(words)) / len(words) if words else 0.0)
    return np.array(res)

def get_embeds(texts):
    outs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embeddings"):
        batch = texts[i:i+BATCH_SIZE]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = mdl(**inp)
        outs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        torch.cuda.empty_cache()
    return np.vstack(outs)

def cached(path, func, *a):
    if os.path.exists(path):
        print(f"Loading cached {path}")
        return np.load(path, allow_pickle=False)
    arr = func(*a)
    np.save(path, arr)
    return arr

# ---------------- Compute / Load ----------------
train_p = cached("precomputed_features/distilbert_multilingual/train_perp.npy", get_perplexity, X_train.tolist())
test_p  = cached("precomputed_features/distilbert_multilingual/test_perp.npy", get_perplexity, X_test.tolist())
train_d = cached("precomputed_features/distilbert_multilingual/train_div.npy", get_div, X_train.tolist())
test_d  = cached("precomputed_features/distilbert_multilingual/test_div.npy", get_div, X_test.tolist())
train_e = cached("precomputed_features/distilbert_multilingual/train_emb.npy", get_embeds, X_train.tolist())
test_e  = cached("precomputed_features/distilbert_multilingual/test_emb.npy", get_embeds, X_test.tolist())

Xtr = np.hstack([train_p.reshape(-1, 1), train_d.reshape(-1, 1), train_e])
Xte = np.hstack([test_p.reshape(-1, 1), test_d.reshape(-1, 1), test_e])

# ---------------- Models ----------------
models = {
    "XGBoost": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", device="cuda"),
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10),
    "SVM": SVC(probability=True),
    "FFNN": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=50)
}

results = []
for n, m in models.items():
    print(f"\nTraining {n} (DistilBERT encoder)...")
    m.fit(Xtr, y_train)
    preds = m.predict(Xte)
    rep = classification_report(y_test, preds, output_dict=True)
    results.append({
        "Model": n,
        "Encoder": "distilbert-base-multilingual-cased",
        "Accuracy": rep["accuracy"],
        "F1": rep["weighted avg"]["f1-score"],
        "Precision": rep["weighted avg"]["precision"],
        "Recall": rep["weighted avg"]["recall"]
    })
    joblib.dump(m, f"models/polydetect_distilbert_{n.lower()}.joblib")

df_res = pd.DataFrame(results)
df_res.to_csv("metrics_distilbert_multilingual.csv", index=False)

combined_path = "metrics_combined.csv"
if os.path.exists(combined_path):
    df_all = pd.read_csv(combined_path)
    df_all = pd.concat([df_all, df_res], ignore_index=True)
    df_all.to_csv(combined_path, index=False)
else:
    df_res.to_csv(combined_path, index=False)

print("\n--- Results ---")
print(df_res)
print("✅ Training complete for distilbert-base-multilingual-cased.")