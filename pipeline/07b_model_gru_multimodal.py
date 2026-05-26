"""
GRU Multimodal Model — Baseline + CXR Features
===============================================
Identical architecture to 07_model_gru.py but loads the multimodal feature
set produced by 11_multimodal_fusion.py (168 baseline + 23 CXR structured +
64 BERT PCA + 1 has_cxr flag = 256 static features).

Saves results alongside the baseline for direct comparison.

Run AFTER:  11_multimodal_fusion.py

Input:   output/X_train_multimodal.parquet  /  X_val_multimodal  /  X_test_multimodal
         output/y_train.parquet             /  y_val             /  y_test
         output/timeseries.parquet

Output:  output/best_gru_multimodal.pt
         output/training_log_multimodal.csv
         output/predictions_multimodal.parquet
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
from config import OUTPUT_DIR

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE    = 64
EPOCHS        = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
STATIC_DIM    = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class ICUDataset(Dataset):
    def __init__(self, X_static, y, ts, ts_features):
        self.stay_ids    = X_static["stay_id"].values
        self.static_arr  = X_static.drop(columns=["stay_id"]).values.astype(np.float32)
        self.labels      = y.set_index("stay_id").loc[self.stay_ids, "los_gt7"].values.astype(np.float32)
        self.ts_features = ts_features

        ts_pivot = (
            ts[ts["stay_id"].isin(self.stay_ids)]
            .sort_values(["stay_id", "hour"])
            .set_index(["stay_id", "hour"])[ts_features]
            .fillna(0.0)
        )
        stays_ordered = list(self.stay_ids)
        self.ts_arr = np.zeros((len(stays_ordered), 48, len(ts_features)), dtype=np.float32)
        for i, sid in enumerate(stays_ordered):
            if sid in ts_pivot.index.get_level_values("stay_id"):
                self.ts_arr[i] = ts_pivot.loc[sid].values

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ts_arr[idx]),
            torch.tensor(self.static_arr[idx]),
            torch.tensor(self.labels[idx]),
        )


class GRUModel(nn.Module):
    def __init__(self, ts_input_size, static_input_size,
                 hidden_size, num_layers, static_dim, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=ts_input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_branch = nn.Sequential(
            nn.Linear(static_input_size, static_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + static_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, ts, static):
        _, h_n = self.gru(ts)
        gru_out    = h_n[-1]
        static_out = self.static_branch(static)
        fused      = torch.cat([gru_out, static_out], dim=1)
        return self.classifier(fused).squeeze(1)


def compute_metrics(labels, logits):
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auroc":     roc_auc_score(labels, probs),
        "auprc":     average_precision_score(labels, probs),
    }


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for ts, static, labels in loader:
        ts, static, labels = ts.to(DEVICE), static.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(ts, static), labels)
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
        all_logits.append(model(ts, static).cpu().numpy())
        all_labels.append(labels.numpy())
    return compute_metrics(np.concatenate(all_labels), np.concatenate(all_logits))


# ── Load data ──────────────────────────────────────────────────────────
print("Loading multimodal data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_multimodal.parquet")
X_val   = pd.read_parquet(OUTPUT_DIR / "X_val_multimodal.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_multimodal.parquet")

y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
y_val   = pd.read_parquet(OUTPUT_DIR / "y_val.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")

ts          = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")
TS_FEATURES = [c for c in ts.columns if c not in ["stay_id", "hour"]]

print(f"  Static features (multimodal): {X_train.shape[1] - 1}")
print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

n_pos      = int(y_train["los_gt7"].sum())
n_neg      = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"\nClass balance: {n_pos:,} pos / {n_neg:,} neg  (pos_weight={pos_weight.item():.2f})")

train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES)
val_ds   = ICUDataset(X_val,   y_val,   ts, TS_FEATURES)
test_ds  = ICUDataset(X_test,  y_test,  ts, TS_FEATURES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

model = GRUModel(
    ts_input_size     = len(TS_FEATURES),
    static_input_size = X_train.shape[1] - 1,
    hidden_size       = HIDDEN_SIZE,
    num_layers        = NUM_LAYERS,
    static_dim        = STATIC_DIM,
    dropout           = DROPOUT,
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training ───────────────────────────────────────────────────────────
print(f"\n{'Epoch':<6} {'Loss':<10} {'F1':<8} {'Prec':<8} {'Rec':<8} {'AUROC':<8}")
print("─" * 52)

best_val_f1, best_epoch, log_rows = 0.0, 0, []

for epoch in range(1, EPOCHS + 1):
    train_loss  = train_epoch(model, train_loader, optimizer, criterion)
    val_metrics = evaluate(model, val_loader)
    f1, prec, rec, auroc = (val_metrics[k] for k in ["f1", "precision", "recall", "auroc"])
    print(f"{epoch:<6} {train_loss:<10.4f} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f} {auroc:<8.4f}")
    log_rows.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
    if f1 > best_val_f1:
        best_val_f1, best_epoch = f1, epoch
        torch.save(model.state_dict(), OUTPUT_DIR / "best_gru_multimodal.pt")

print(f"\nBest checkpoint: epoch {best_epoch}  (val F1 = {best_val_f1:.4f})")

model.load_state_dict(torch.load(OUTPUT_DIR / "best_gru_multimodal.pt", weights_only=True))
test_metrics = evaluate(model, test_loader)

print("\n── Test Results (Multimodal) ──────────────────────────────────")
for k, v in test_metrics.items():
    print(f"  {k:<12}: {v:.4f}")

pd.DataFrame(log_rows).to_csv(OUTPUT_DIR / "training_log_multimodal.csv", index=False)

# ── Save predictions ───────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader, stay_ids):
    model.eval()
    all_logits = []
    for ts, static, _ in loader:
        ts, static = ts.to(DEVICE), static.to(DEVICE)
        all_logits.append(model(ts, static).cpu().numpy())
    logits = np.concatenate(all_logits)
    probs  = 1 / (1 + np.exp(-logits))
    return pd.DataFrame({"stay_id": stay_ids, "y_prob": probs, "y_pred": (probs >= 0.5).astype(int)})

pred_rows = []
for name, X_split, loader in [("train", X_train, train_loader), ("val", X_val, val_loader), ("test", X_test, test_loader)]:
    df = get_predictions(model, loader, X_split["stay_id"].values)
    df["split"] = name
    pred_rows.append(df)

pd.concat(pred_rows).to_parquet(OUTPUT_DIR / "predictions_multimodal.parquet", index=False)
print(f"\nSaved: output/best_gru_multimodal.pt")
print(f"Saved: output/training_log_multimodal.csv")
print(f"Saved: output/predictions_multimodal.parquet")
