# inference.py
import os, glob, json, joblib, math
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

MODEL_DIR = "models"
CM_DIR = "confusion_matrices"; ROC_DIR = "roc_curves"
os.makedirs(CM_DIR, exist_ok=True); os.makedirs(ROC_DIR, exist_ok=True)

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")
print("Using GPU:", USE_GPU)
if USE_GPU: print("GPU:", torch.cuda.get_device_name(0))

# load encoders
xlmr_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "xlm-roberta-base"))
xlmr_model = AutoModel.from_pretrained(os.path.join(MODEL_DIR, "xlm-roberta-base")).to(device).eval()
minilm_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384"))
minilm_model = AutoModel.from_pretrained(os.path.join(MODEL_DIR, "microsoft-MiniLM-L12-H384")).to(device).eval()

# GPT2 pipeline
gpt_device = 0 if USE_GPU else -1
gen_pipe = pipeline("text-generation", model=os.path.join(MODEL_DIR, "distilgpt2"), device=gpt_device,
                    torch_dtype=torch.float16 if USE_GPU else torch.float32)
gen_tok = gen_pipe.tokenizer
gen_mdl = gen_pipe.model
if gen_tok.pad_token is None: gen_tok.pad_token = gen_tok.eos_token
gen_mdl.config.pad_token_id = gen_tok.pad_token_id
if USE_GPU:
    try: gen_mdl = gen_mdl.to(device).half()
    except: gen_mdl = gen_mdl.to(device)

def stable_perp(p): return float(max(0.0, min(math.log(p+1.0), 10.0)))

def compute_perplexity(texts):
    vals=[]
    for t in texts:
        enc = gen_tok([t], return_tensors="pt", padding=True, truncation=True, max_length=256)
        if USE_GPU:
            for k in enc: enc[k] = enc[k].to(device)
        with torch.no_grad():
            out = gen_mdl(**{k:v for k,v in enc.items()}, labels=enc["input_ids"])
        vals.append(float(torch.exp(out.loss).cpu().item()))
        if USE_GPU: torch.cuda.empty_cache()
    return np.array(vals)

def compute_div(texts):
    return np.array([len(set(t.split()))/len(t.split()) if len(t.split()) else 0 for t in texts])

def embed_xlmr(texts):
    outs=[]
    for t in texts:
        enc = xlmr_tok(t, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = xlmr_model(**enc)
        outs.append(out.last_hidden_state[:,0,:].cpu().numpy().flatten())
        if USE_GPU: torch.cuda.empty_cache()
    return np.vstack(outs)

def embed_minilm(texts):
    outs=[]
    for t in texts:
        enc = minilm_tok(t, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = minilm_model(**enc)
        outs.append(out.last_hidden_state[:,0,:].cpu().numpy().flatten())
        if USE_GPU: torch.cuda.empty_cache()
    return np.vstack(outs)

# load data for evaluation (adjust as needed)
DATA_CSV = "data/multitude_v3_clean.csv"
df = pd.read_csv(DATA_CSV)
df = df[df["language"].isin(["en"])].dropna()
texts = df["text"].tolist()
labels = df["label"].tolist()

print("Computing perplexity (this may take time)...")
perps = compute_perplexity(texts)
perps = np.array([stable_perp(p) for p in perps])
divs = compute_div(texts)

for f in sorted(glob.glob(os.path.join(MODEL_DIR, "polydetect_*.joblib"))):
    name = os.path.basename(f).replace(".joblib","")
    print("Evaluating", name)
    clf = joblib.load(f)
    meta_path = os.path.join(MODEL_DIR, f"{name}.meta.json")
    if not os.path.exists(meta_path):
        print("Missing meta for", name, "— skipping")
        continue
    meta = json.load(open(meta_path,"r"))
    encoder = meta.get("encoder","xlmr")
    scaler_path = os.path.join(MODEL_DIR, f"{encoder}_global_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    # embeddings
    if encoder == "minilm":
        embeds = embed_minilm(texts)
    else:
        embeds = embed_xlmr(texts)

    X = np.hstack([perps.reshape(-1,1), divs.reshape(-1,1), embeds])
    if scaler is not None:
        X[:, :2] = scaler.transform(X[:, :2])

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[:,1]
    else:
        preds = clf.predict(X)
        proba = np.array([1.0 if p==1 else 0.0 for p in preds])

    preds_bin = (proba >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds_bin)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"Confusion: {name}")
    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha='center',va='center', color='white' if cm.max()>0 else 'black')
    plt.tight_layout()
    cm_file = os.path.join(CM_DIR, f"{name}_en.png")
    fig.savefig(cm_file)
    plt.close(fig)

    try:
        fpr, tpr, _ = roc_curve(labels, proba)
        roc_auc = auc(fpr,tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr,tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0,1],[0,1], linestyle="--", color="grey")
        ax.set_title(f"ROC: {name}")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend()
        roc_file = os.path.join(ROC_DIR, f"{name}_en.png")
        fig.savefig(roc_file)
        plt.close(fig)
    except Exception as e:
        print("ROC failed for", name, e)

    print("Saved plots for", name)
