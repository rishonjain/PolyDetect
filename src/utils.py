# src/utils.py
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# XLM-RoBERTa for embeddings
# -----------------------------
def load_xlm_roberta(model_dir="./models/xlm-roberta-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    return tokenizer, model

# -----------------------------
# DistilGPT2 for perplexity
# -----------------------------
def load_gpt2(model_dir="./models/distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

def get_perplexity(texts, tokenizer, model, batch_size=8):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing Perplexity"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        batch_perplexity = torch.exp(outputs.loss).detach().cpu().tolist()
        results.extend(batch_perplexity)
    return np.array(results)

# -----------------------------
# Embeddings + diversity helpers
# -----------------------------
def get_embeddings(texts, tokenizer, model, batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)

def get_lexical_diversity(texts):
    return np.array([
        len(set(t.lower().split())) / len(t.split()) if len(t.split()) > 0 else 0.0
        for t in texts
    ])

# -----------------------------
# Feature caching
# -----------------------------
def save_feature(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)

def load_feature(path):
    return np.load(path) if os.path.exists(path) else None