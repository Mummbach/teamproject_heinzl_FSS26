"""
GRU Model — ICU Prolonged Stay Prediction
==========================================
Predicts whether a patient's ICU stay exceeds 7 days (los_gt7).

Architecture:
  - GRU encoder:  processes 48h × 12 vital features (time-series)
  - Static branch: processes 168 static features (demographics, ICD, ATC)
  - Fusion:        concatenates GRU final hidden state + static embedding
  - Output:        single sigmoid neuron (binary classification)

Run AFTER:  06_normalize.py  (scaled static features)
            02_features.py   (timeseries.parquet must exist)

Input:   output/X_train_scaled.parquet  /  X_val_scaled  /  X_test_scaled
         output/y_train.parquet         /  y_val         /  y_test
         output/timeseries.parquet
         output/scaler_params.parquet

Output:  output/best_gru_model.pt       — best checkpoint (by val F1)
         output/training_log.csv        — per-epoch metrics
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))  # pipeline/ -> config.py / multimodal_utils.py
from config import OUTPUT_DIR
from multimodal_utils import ICUDataset, GRUModel

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hourly time-series branch ─────────────────────────────────────────
# True  — GRU processes 48h × 12 vital features (hourly time series)
# False — static branch only; GRU is disabled for ablation comparison
USE_HOURLY_TIMESERIES = True

# ── Hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE   = 64    # GRU hidden state size
NUM_LAYERS    = 2     # GRU depth (more layers = more capacity, more regularization needed)
DROPOUT       = 0.3   # applied between GRU layers and before output
STATIC_DIM    = 64    # static branch embedding size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ICUDataset and GRUModel are shared with 08b_hyperparameter_search.py and the
# explainability scripts (09_shap.py, 10_explainability.py, 11_timeshap.py) —
# see multimodal_utils.py for the single source of truth.

# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    probs = 1 / (1 + np.exp(-logits))   # sigmoid
    # NOTE: threshold fixed at 0.5 — may be suboptimal given pos_weight training; tune on val set for deployment
    preds = (probs >= 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(labels, probs)
    if len(np.unique(labels)) < 2:
        auprc = float("nan")
    else:
        auprc = average_precision_score(labels, probs)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auroc":     auroc,
        "auprc":     auprc,
    }


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for ts, static, labels in loader:
        ts, static, labels = ts.to(DEVICE), static.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ts, static)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for ts, static, labels in loader:
        ts, static = ts.to(DEVICE), static.to(DEVICE)
        logits = model(ts, static)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return compute_metrics(labels, logits)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
X_val   = pd.read_parquet(OUTPUT_DIR / "X_val_scaled.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")

y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
y_val   = pd.read_parquet(OUTPUT_DIR / "y_val.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")

ts = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

TS_FEATURES = [c for c in ts.columns if c not in ["stay_id", "hour"]]
print(f"  Time-series features : {TS_FEATURES}")
print(f"  Static features      : {X_train.shape[1] - 1}")
print(f"  Train stays          : {len(X_train):,}")
print(f"  Val stays            : {len(X_val):,}")
print(f"  Test stays           : {len(X_test):,}")

# ── Class imbalance → pos_weight ───────────────────────────────────────
# BCEWithLogitsLoss(pos_weight=w) upweights the positive class.
# w = n_negative / n_positive tells the loss to treat each positive
# sample as if it were w negative samples.
n_pos = int(y_train["los_gt7"].sum())
n_neg = len(y_train) - n_pos
if n_pos == 0:
    raise ValueError("Training set has no positive examples — check y_train column and split logic")
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"\nClass balance (train): {n_pos:,} positive / {n_neg:,} negative")
print(f"  pos_weight = {pos_weight.item():.2f}")

# ── Datasets & DataLoaders ──────────────────────────────────────────────
train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES)
val_ds   = ICUDataset(X_val,   y_val,   ts, TS_FEATURES)
test_ds  = ICUDataset(X_test,  y_test,  ts, TS_FEATURES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model, loss, optimizer ─────────────────────────────────────────────
static_input_size = X_train.shape[1] - 1   # exclude stay_id

model = GRUModel(
    ts_input_size    = len(TS_FEATURES),
    static_input_size= static_input_size,
    hidden_size      = HIDDEN_SIZE,
    num_layers       = NUM_LAYERS,
    static_dim       = STATIC_DIM,
    dropout          = DROPOUT,
    use_gru          = USE_HOURLY_TIMESERIES,
).to(DEVICE)
print(f"GRU branch: {'enabled' if USE_HOURLY_TIMESERIES else 'disabled (static only)'}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training ───────────────────────────────────────────────────────────
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

# ── Test evaluation ────────────────────────────────────────────────────
print("\nLoading best checkpoint for test evaluation...")
model.load_state_dict(torch.load(OUTPUT_DIR / "best_gru_model.pt", weights_only=True, map_location=DEVICE))
test_metrics = evaluate(model, test_loader)

print("\nTest results:")
print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
print(f"  Precision : {test_metrics['precision']:.4f}")
print(f"  Recall    : {test_metrics['recall']:.4f}")
print(f"  F1        : {test_metrics['f1']:.4f}")
print(f"  AUROC     : {test_metrics['auroc']:.4f}")
print(f"  AUPRC     : {test_metrics['auprc']:.4f}")

# ── Save training log ──────────────────────────────────────────────────
log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUTPUT_DIR / "training_log.csv", index=False)
print(f"\nSaved: output/best_gru_model.pt")
print(f"Saved: output/training_log.csv")
