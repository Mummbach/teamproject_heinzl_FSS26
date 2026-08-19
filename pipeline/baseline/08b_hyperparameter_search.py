"""
Hyperparameter Search — GRU Model (Optuna)
Searches for optimal hyperparameters using Bayesian optimization (Optuna).
Optimizes validation F1 score.

Respects the same flags as 07_model_gru.py:
  USE_TEXT  — include CXR text branch
  CXR_ONLY  — restrict cohort to patients with CXR report

Run AFTER:  preprocessing/06_normalize.py
            preprocessing/02b_cxr_features.py  (if USE_TEXT = True)

Output:  prints best hyperparameters and val F1
         output/best_hparams.txt
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Install optuna: pip install optuna")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # pipeline/ -> config.py / multimodal_utils.py
from config import OUTPUT_DIR
from multimodal_utils import ICUDataset as _BaseICUDataset, GRUModel

# Flags (match 07_model_gru.py)
USE_TEXT = True
CXR_ONLY = True

# Search settings 
N_TRIALS   = 100     # number of hyperparameter combinations to try
EPOCHS     = 30      # shorter training per trial for speed
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CXR_EMB_DIM = 1536

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}  |  Trials: {N_TRIALS}")


# Dataset
# ICUDataset (ts/static/label) and GRUModel are shared with 07_model_gru.py
# and the explainability scripts — see multimodal_utils.py. Here we only add
# the CXR embedding branch on top of the shared dataset.

class ICUDataset(_BaseICUDataset):
    def __init__(self, X_static, y, ts, ts_features,
                 cxr: Optional[pd.DataFrame] = None):
        super().__init__(X_static, y, ts, ts_features)

        cxr_cols = [c for c in cxr.columns if c.startswith("cxr_")] if cxr is not None else []
        emb_dim  = len(cxr_cols) if cxr_cols else CXR_EMB_DIM
        self.cxr_arr = np.zeros((len(self.stay_ids), emb_dim), dtype=np.float32)
        if cxr is not None and cxr_cols:
            cxr_indexed = cxr.set_index("stay_id")
            for i, sid in enumerate(self.stay_ids):
                if sid in cxr_indexed.index:
                    self.cxr_arr[i] = cxr_indexed.loc[sid, cxr_cols].values

    def __getitem__(self, idx):
        ts_t, static_t, label_t = super().__getitem__(idx)
        return ts_t, static_t, torch.tensor(self.cxr_arr[idx]), label_t


# Load data

print("Loading data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
X_val   = pd.read_parquet(OUTPUT_DIR / "X_val_scaled.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
y_val   = pd.read_parquet(OUTPUT_DIR / "y_val.parquet")
ts      = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

cxr_path = OUTPUT_DIR / "cxr_bert_embeddings.parquet"
cxr = pd.read_parquet(cxr_path) if (USE_TEXT and cxr_path.exists()) else None
use_text_actual = USE_TEXT and cxr is not None

if CXR_ONLY and cxr is not None:
    cxr_ids = set(cxr["stay_id"].values)
    X_train = X_train[X_train["stay_id"].isin(cxr_ids)]
    X_val   = X_val[X_val["stay_id"].isin(cxr_ids)]
    y_train = y_train[y_train["stay_id"].isin(cxr_ids)]
    y_val   = y_val[y_val["stay_id"].isin(cxr_ids)]
    print(f"  CXR_ONLY: {len(X_train):,} train / {len(X_val):,} val stays")

TS_FEATURES      = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_INPUT_SIZE = X_train.shape[1] - 1

n_pos = int(y_train["los_gt7"].sum())
n_neg = len(y_train) - n_pos
if n_pos == 0:
    raise ValueError("Training set has no positive examples")
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"  pos_weight = {pos_weight.item():.2f}")


# Objective 

def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers  = trial.suggest_int("num_layers", 1, 2)
    static_dim  = trial.suggest_categorical("static_dim", [32, 64, 128])
    if use_text_actual:
        text_dim = trial.suggest_categorical("text_dim", [32, 64, 128])
    else:
        text_dim = 0
    dropout     = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES, cxr)
    val_ds   = ICUDataset(X_val,   y_val,   ts, TS_FEATURES, cxr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = GRUModel(
        ts_input_size     = len(TS_FEATURES),
        static_input_size = STATIC_INPUT_SIZE,
        hidden_size       = hidden_size,
        num_layers        = num_layers,
        static_dim        = static_dim,
        dropout           = dropout,
        use_text          = use_text_actual,
        text_dim          = text_dim,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for ts_b, static_b, text_b, labels_b in train_loader:
            ts_b, static_b, text_b, labels_b = (
                ts_b.to(DEVICE), static_b.to(DEVICE),
                text_b.to(DEVICE), labels_b.to(DEVICE)
            )
            optimizer.zero_grad()
            loss = criterion(model(ts_b, static_b, text_b), labels_b)
            loss.backward()
            optimizer.step()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for ts_b, static_b, text_b, labels_b in val_loader:
                logits = model(ts_b.to(DEVICE), static_b.to(DEVICE), text_b.to(DEVICE))
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels_b.numpy())

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        probs  = 1 / (1 + np.exp(-logits))
        preds  = (probs >= 0.5).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) > 0 else 0.0
        best_val_f1 = max(best_val_f1, f1)

        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1


# Run search

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_trial
print(f"\nBest val F1 : {best.value:.4f}")
print(f"Best params :")
for k, v in best.params.items():
    print(f"  {k:<20} {v}")

result = f"Best val F1: {best.value:.4f}\n"
result += "\n".join(f"{k}: {v}" for k, v in best.params.items())
(OUTPUT_DIR / "best_hparams.txt").write_text(result)
print(f"\nSaved: output/best_hparams.txt")
