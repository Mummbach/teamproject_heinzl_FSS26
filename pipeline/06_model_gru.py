"""
GRU Model — ICU Prolonged Stay Prediction
Predicts whether a patient's ICU stay exceeds 7 days (los_gt7).

Architecture:
  - GRU encoder:   processes 48h × 12 vital features (time-series)
  - Static branch: processes static features (demographics, ICD, ATC)
  - Text branch:   projects 1536-dim BioClinicalBERT CXR embeddings (optional)
  - Fusion:        concatenates all active branch outputs
  - Output:        single sigmoid neuron (binary classification)

Flags:
- USE_HOURLY_TIMESERIES:    True, GRU processes 48h × 12 vital signs
                            False, GRU disabled, static features only (ablation)
- USE_TEXT:                 True, BioClinicalBERT CXR embeddings added as third branch
                            False, text branch disabled
- CXR_ONLY:                 True, cohort restricted to patients with a CXR report (~17.5%)
                            False, full cohort — patients without report get zero vector

Run AFTER:  05_normalize.py  (scaled static features)

Input:   output/X_train_scaled.parquet  /  X_val_scaled  /  X_test_scaled
         output/y_train.parquet         /  y_val         /  y_test
         output/timeseries.parquet
         output/scaler_params.parquet
         output/cxr_bert_embeddings.parquet  (optional, only if USE_TEXT = True)

Output:  output/best_gru_model.pt       — best checkpoint (by val F1)
         output/training_log.csv        — per-epoch metrics
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from pathlib import Path
from typing import Optional
from config import OUTPUT_DIR

# Reproducibility 
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hourly time-series branch
USE_HOURLY_TIMESERIES = True

# CXR text branch
USE_TEXT = True
TEXT_DIM = 64         # projection size for CXR embeddings

# CXR-only cohort
# True  — train/val/test restricted to patients with a CXR report
# False — full cohort, patients without report get zero vector (default)
CXR_ONLY = False

# Hyperparameters 
BATCH_SIZE    = 64
EPOCHS        = 30
LEARNING_RATE = 6e-3
HIDDEN_SIZE   = 64
NUM_LAYERS    = 1
DROPOUT       = 0.1
STATIC_DIM    = 32
TEXT_DIM      = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")



# DATASET

class ICUDataset(Dataset):
    """
    Returns one sample per ICU stay:
      ts     — (48, 12)  float32 tensor  — hourly vitals
      static — (F,)      float32 tensor  — scaled static features
      text   — (1536,)   float32 tensor  — CXR embedding; zero vector if no report
      label  — scalar    float32         — los_gt7 (0 or 1)
    """

    def __init__(self, X_static: pd.DataFrame, y: pd.DataFrame,
                 ts: pd.DataFrame, ts_features: list[str],
                 cxr: Optional[pd.DataFrame] = None):
        self.stay_ids   = X_static["stay_id"].values
        self.static_arr = X_static.drop(columns=["stay_id"]).values.astype(np.float32)
        self.labels     = y.set_index("stay_id").loc[self.stay_ids, "los_gt7"].values.astype(np.float32)
        self.ts_features = ts_features

        # Build (stays, 48, 12) array from long-format timeseries
        # Fill remaining NaN (vitals with no data at all) with 0
        ts_pivot = (
            ts[ts["stay_id"].isin(self.stay_ids)]
            .sort_values(["stay_id", "hour"])
            .set_index(["stay_id", "hour"])[ts_features]
            .fillna(0.0)
        )
        # Pivot to (stay, hour, feature) — shape (N, 48, 12)
        stays_ordered = list(self.stay_ids)
        self.ts_arr = np.zeros(
            (len(stays_ordered), 48, len(ts_features)), dtype=np.float32
        )
        for i, sid in enumerate(stays_ordered):
            if sid in ts_pivot.index.get_level_values("stay_id"):
                self.ts_arr[i] = ts_pivot.loc[sid].values

        # Build (stays, 1536) CXR embedding array.
        # Patients without a report get a zero vector — the model receives
        # has_cxr=0 via the static features so it can learn to down-weight absent text.
        cxr_cols = [c for c in cxr.columns if c.startswith("cxr_")] if cxr is not None else []
        emb_dim  = len(cxr_cols) if cxr_cols else 1536
        self.cxr_arr = np.zeros((len(stays_ordered), emb_dim), dtype=np.float32)
        if cxr is not None and cxr_cols:
            cxr_indexed = cxr.set_index("stay_id")
            for i, sid in enumerate(stays_ordered):
                if sid in cxr_indexed.index:
                    self.cxr_arr[i] = cxr_indexed.loc[sid, cxr_cols].values

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ts_arr[idx]),      # (48, 12)
            torch.tensor(self.static_arr[idx]),  # (F,)
            torch.tensor(self.cxr_arr[idx]),     # (1536,)
            torch.tensor(self.labels[idx]),      # scalar
        )



# MODEL

class GRUModel(nn.Module):
    """
    GRU encoder for time-series + linear branch for static features
    + optional text branch for BioClinicalBERT CXR embeddings.

    Args:
        ts_input_size    : number of time-series features (12)
        static_input_size: number of static features
        hidden_size      : GRU hidden state dimension
        num_layers       : number of stacked GRU layers
        static_dim       : static branch embedding size
        dropout          : dropout probability (applied between layers)
        use_gru          : enable/disable GRU branch (ablation)
        use_text         : enable/disable CXR text branch
        text_dim         : projection size for CXR embeddings
    """

    def __init__(self, ts_input_size: int, static_input_size: int,
                 hidden_size: int, num_layers: int,
                 static_dim: int, dropout: float,
                 use_gru: bool = True,
                 use_text: bool = False, text_dim: int = 32):
        super().__init__()
        self.use_gru  = use_gru
        self.use_text = use_text

        if use_gru:
            self.gru = nn.GRU(
                input_size=ts_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.static_branch = nn.Sequential(
            nn.Linear(static_input_size, static_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_text:
            self.text_branch = nn.Sequential(
                nn.Linear(1536, text_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        fusion_input = (hidden_size if use_gru else 0) + static_dim + (text_dim if use_text else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_input, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, ts, static, text):
        parts = [self.static_branch(static)]
        if self.use_gru:
            _, h_n = self.gru(ts)
            parts.append(h_n[-1])
        if self.use_text:
            parts.append(self.text_branch(text))
        return self.classifier(torch.cat(parts, dim=1)).squeeze(1)


# METRICS

def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    probs = 1 / (1 + np.exp(-logits))   # sigmoid
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auroc":     roc_auc_score(labels, probs),
        "auprc":     average_precision_score(labels, probs),
    }



# TRAINING LOOP

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for ts, static, text, labels in loader:
        ts, static, text, labels = ts.to(DEVICE), static.to(DEVICE), text.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ts, static, text)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for ts, static, text, labels in loader:
        ts, static, text = ts.to(DEVICE), static.to(DEVICE), text.to(DEVICE)
        logits = model(ts, static, text)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return compute_metrics(labels, logits)


# MAIN

# Load data
print("Loading data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
X_val   = pd.read_parquet(OUTPUT_DIR / "X_val_scaled.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")

y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
y_val   = pd.read_parquet(OUTPUT_DIR / "y_val.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")

ts = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

cxr_path = OUTPUT_DIR / "cxr_bert_embeddings.parquet"
if USE_TEXT and cxr_path.exists():
    cxr = pd.read_parquet(cxr_path)
    print(f"  CXR embeddings loaded: {len(cxr):,} stays with report")
else:
    cxr = None
    if USE_TEXT:
        print("  WARNING: USE_TEXT=True but cxr_bert_embeddings.parquet not found — text branch disabled")

if CXR_ONLY and cxr is not None:
    cxr_ids = set(cxr["stay_id"].values)
    X_train = X_train[X_train["stay_id"].isin(cxr_ids)]
    X_val   = X_val[X_val["stay_id"].isin(cxr_ids)]
    X_test  = X_test[X_test["stay_id"].isin(cxr_ids)]
    y_train = y_train[y_train["stay_id"].isin(cxr_ids)]
    y_val   = y_val[y_val["stay_id"].isin(cxr_ids)]
    y_test  = y_test[y_test["stay_id"].isin(cxr_ids)]
    print(f"  CXR_ONLY=True — cohort restricted to {len(X_train):,} train / {len(X_val):,} val / {len(X_test):,} test stays")

TS_FEATURES = [c for c in ts.columns if c not in ["stay_id", "hour"]]
print(f"  Time-series features : {TS_FEATURES}")
print(f"  Static features      : {X_train.shape[1] - 1}")
print(f"  Train stays          : {len(X_train):,}")
print(f"  Val stays            : {len(X_val):,}")
print(f"  Test stays           : {len(X_test):,}")

# Class imbalance → pos_weight
# BCEWithLogitsLoss(pos_weight=w) upweights the positive class.
# w = n_negative / n_positive tells the loss to treat each positive
# sample as if it were w negative samples.
n_pos = int(y_train["los_gt7"].sum())
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"\nClass balance (train): {n_pos:,} positive / {n_neg:,} negative")
print(f"  pos_weight = {pos_weight.item():.2f}")

# Datasets & DataLoaders
train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES, cxr)
val_ds   = ICUDataset(X_val,   y_val,   ts, TS_FEATURES, cxr)
test_ds  = ICUDataset(X_test,  y_test,  ts, TS_FEATURES, cxr)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, optimizer
static_input_size = X_train.shape[1] - 1   # exclude stay_id

use_text_actual = USE_TEXT and cxr is not None

model = GRUModel(
    ts_input_size    = len(TS_FEATURES),
    static_input_size= static_input_size,
    hidden_size      = HIDDEN_SIZE,
    num_layers       = NUM_LAYERS,
    static_dim       = STATIC_DIM,
    dropout          = DROPOUT,
    use_gru          = USE_HOURLY_TIMESERIES,
    use_text         = use_text_actual,
    text_dim         = TEXT_DIM,
).to(DEVICE)
print(f"GRU branch  : {'enabled' if USE_HOURLY_TIMESERIES else 'disabled (static only)'}")
print(f"Text branch : {'enabled' if use_text_actual else 'disabled'}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
print(f"\n{'Epoch':<6} {'Loss':<10} {'F1':<8} {'Prec':<8} {'Rec':<8} {'AUROC':<8}")
print("─" * 52)

best_val_f1   = 0.0
best_epoch    = 0
log_rows      = []

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_metrics = evaluate(model, val_loader)

    f1    = val_metrics["f1"]
    prec  = val_metrics["precision"]
    rec   = val_metrics["recall"]
    auroc = val_metrics["auroc"]

    print(f"{epoch:<6} {train_loss:<10.4f} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f} {auroc:<8.4f}")

    log_rows.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

    # Save best checkpoint based on validation F1
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_epoch  = epoch
        torch.save(model.state_dict(), OUTPUT_DIR / "best_gru_model.pt")

print(f"\nBest checkpoint: epoch {best_epoch}  (val F1 = {best_val_f1:.4f})")

# Test evaluation
print("\nLoading best checkpoint for test evaluation...")
model.load_state_dict(torch.load(OUTPUT_DIR / "best_gru_model.pt", weights_only=True))
test_metrics = evaluate(model, test_loader)

print("\nTest results:")
print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
print(f"  Precision : {test_metrics['precision']:.4f}")
print(f"  Recall    : {test_metrics['recall']:.4f}")
print(f"  F1        : {test_metrics['f1']:.4f}")
print(f"  AUROC     : {test_metrics['auroc']:.4f}")
print(f"  AUPRC     : {test_metrics['auprc']:.4f}")

# Save training log
log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUTPUT_DIR / "training_log.csv", index=False)
print(f"\nSaved: output/best_gru_model.pt")
print(f"Saved: output/training_log.csv")
