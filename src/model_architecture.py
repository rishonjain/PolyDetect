# src/model_architecture.py
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

def build_xgboost_model(device_type="cpu"):
    params = {"tree_method": "gpu_hist", "device": "cuda"} if device_type == "cuda" else {}
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        **params
    )

def build_logistic_regression():
    return LogisticRegression(max_iter=2000, solver="lbfgs")

def build_random_forest():
    return RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

def build_svm():
    return SVC(kernel="linear", probability=True, random_state=42)

# PyTorch Feedforward Neural Network
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(FFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)
