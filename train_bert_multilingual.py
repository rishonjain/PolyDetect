# train_bert_multilingual.py — FINAL
import os, json, joblib, numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

DATASET = "data/multitude_v3_clean.csv"
MODEL_DIR = "models"
ENCODER_KEY = "bert"
ENCODER_PATH = os.path.join(MODEL_DIR, "bert-base-multilingual-cased")
SCALER_PATH = os.path.join(MODEL_DIR, f"{ENCODER_KEY}_global_scaler.pkl")

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

tok = AutoTokenizer.from_pretrained(ENCODER_PATH)
mdl = AutoModel.from_pretrained(ENCODER_PATH).to(device).eval()
emb_dim = mdl.config.hidden_size

print("Loading GPT-2 (CPU)...")
gen_pipe = pipeline("text-generation",
                    model=os.path.join(MODEL_DIR,"distilgpt2"),
                    device=-1)
gen_tok = gen_pipe.tokenizer
gen_mdl = gen_pipe.model.cpu().eval()
if gen_tok.pad_token is None:
    gen_tok.pad_token = gen_tok.eos_token
gen_mdl.config.pad_token_id = gen_tok.pad_token_id

df = pd.read_csv(DATASET).dropna()
df = df[df.language.isin(["en","es","de","ru","zh"])]
Xtr_txt, Xte_txt, ytr, yte = train_test_split(df["text"], df["label"],
                                             stratify=df["label"], test_size=0.2, random_state=42)

BATCH=16

def get_perp(xs):
    vals=[]
    for i in tqdm(range(0,len(xs),BATCH), desc="Perplexity"):
        batch=xs[i:i+BATCH]
        enc = gen_tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
        with torch.no_grad(): out = gen_mdl(**enc, labels=enc["input_ids"])
        vals += torch.exp(out.loss).repeat(len(batch)).cpu().tolist()
    return np.array(vals)

def get_div(xs): return np.array([len(set(t.split()))/len(t.split()) if t.split() else 0 for t in xs])

def get_emb(xs):
    outs=[]
    for i in tqdm(range(0,len(xs),BATCH), desc="BERT embeddings"):
        batch=xs[i:i+BATCH]
        enc=tok(batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256).to(device)
        with torch.no_grad(): out = mdl(**enc)
        outs.append(out.last_hidden_state[:,0,:].cpu().numpy())
        if USE_GPU: torch.cuda.empty_cache()
    return np.vstack(outs)

train_p, test_p = get_perp(Xtr_txt.tolist()), get_perp(Xte_txt.tolist())
train_d, test_d = get_div(Xtr_txt.tolist()), get_div(Xte_txt.tolist())
train_e, test_e = get_emb(Xtr_txt.tolist()), get_emb(Xte_txt.tolist())

Xtr = np.hstack([train_p[:,None], train_d[:,None], train_e])
Xte = np.hstack([test_p[:,None],  test_d[:,None],  test_e])

scaler=StandardScaler(); scaler.fit(Xtr[:, :2])
Xtr[:, :2], Xte[:, :2] = scaler.transform(Xtr[:, :2]), scaler.transform(Xte[:, :2])
joblib.dump(scaler, SCALER_PATH)

xgb_args={"objective":"binary:logistic","eval_metric":"logloss"}
if USE_GPU: xgb_args.update({"tree_method":"gpu_hist","predictor":"gpu_predictor"})

models={
    "xgboost": xgb.XGBClassifier(**xgb_args),
    "logreg": LogisticRegression(max_iter=2000),
    "randomforest": RandomForestClassifier(n_estimators=200, max_depth=15),
    "svm": SVC(probability=True),
    "ffnn": MLPClassifier(hidden_layer_sizes=(256,128), max_iter=100)
}

def save_model(clf, name):
    fn = f"polydetect_{ENCODER_KEY}_{name}"
    joblib.dump(clf, f"{MODEL_DIR}/{fn}.joblib")
    meta = {"encoder":ENCODER_KEY,
            "embedding_dim":emb_dim,
            "feature_order":["perplexity","diversity","embedding"],
            "feature_dim":Xtr.shape[1]}
    json.dump(meta, open(f"{MODEL_DIR}/{fn}.meta.json","w"), indent=4)

results=[]
for name, clf in models.items():
    print("Training:", name)
    clf.fit(Xtr,ytr)
    preds=clf.predict(Xte)
    rep=classification_report(yte,preds,output_dict=True)
    results.append({"Model":name,"Accuracy":rep["accuracy"],
                    "F1":rep["weighted avg"]["f1-score"]})
    save_model(clf,name)

print(pd.DataFrame(results))
