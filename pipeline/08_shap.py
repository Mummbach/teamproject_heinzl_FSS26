"""
SHAP Explainability — GRU+MLP Baseline Model
=============================================
Computes SHAP attributions for the trained GRU+MLP model using
shap.GradientExplainer, which supports multi-input PyTorch models.

Two sets of attributions are produced:
  • Static features  — direct SHAP values (one per feature per stay)
  • Time-series      — mean |SHAP| across 48h per vital sign (12 values per stay)
    stored as ts_shap_<vital> columns in explanations.parquet

Run AFTER:  07_model_gru.py  (best_gru_model.pt must exist)

Input:   output/X_train_scaled.parquet  (background reference set)
         output/X_test_scaled.parquet
         output/y_train.parquet / y_test.parquet
         output/timeseries.parquet
         output/best_gru_model.pt
         output/scaler_params.parquet

Output:  output/explanations.parquet   — stay_id, split, shap per feature
         output/shap_summary.png       — beeswarm summary plot (static features)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    raise ImportError("Install shap: pip install shap")

from config import OUTPUT_DIR

SEED       = 42
BG_SIZE    = 200    # background samples for GradientExplainer
BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════
# RE-USE DATASET + MODEL FROM 07_model_gru.py
# ═══════════════════════════════════════════════════════════════════════

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
        self.ts_arr = np.zeros((len(self.stay_ids), 48, len(ts_features)), dtype=np.float32)
        for i, sid in enumerate(self.stay_ids):
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
    def __init__(self, ts_input_size, static_input_size, hidden_size,
                 num_layers, static_dim, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=ts_input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_branch = nn.Sequential(
            nn.Linear(static_input_size, static_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size + static_dim, 32),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, ts, static):
        _, h_n    = self.gru(ts)
        gru_out   = h_n[-1]
        static_out = self.static_branch(static)
        return self.classifier(torch.cat([gru_out, static_out], dim=1)).squeeze(1)


class SHAPWrapper(nn.Module):
    """Wraps GRUModel and adds sigmoid so outputs are probabilities in [0,1]."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ts, static):
        return torch.sigmoid(self.model(ts, static)).unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

print("Loading data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
ts      = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

TS_FEATURES    = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_FEATURES = [c for c in X_train.columns if c != "stay_id"]

print(f"  Time-series features : {len(TS_FEATURES)}")
print(f"  Static features      : {len(STATIC_FEATURES)}")

train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES)
test_ds  = ICUDataset(X_test,  y_test,  ts, TS_FEATURES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════

model = GRUModel(
    ts_input_size     = len(TS_FEATURES),
    static_input_size = len(STATIC_FEATURES),
    hidden_size       = 64,
    num_layers        = 2,
    static_dim        = 64,
    dropout           = 0.3,
).to(DEVICE)
model.load_state_dict(torch.load(OUTPUT_DIR / "best_gru_model.pt",
                                  map_location=DEVICE, weights_only=True))
model.eval()
print("Model loaded from best_gru_model.pt")

shap_model = SHAPWrapper(model).to(DEVICE)
shap_model.eval()


# ═══════════════════════════════════════════════════════════════════════
# BUILD BACKGROUND DATASET
# ═══════════════════════════════════════════════════════════════════════

print(f"\nBuilding background set (n={BG_SIZE})...")
rng = np.random.default_rng(SEED)
bg_idx     = rng.choice(len(train_ds), size=BG_SIZE, replace=False)
bg_ts      = torch.tensor(train_ds.ts_arr[bg_idx]).to(DEVICE)
bg_static  = torch.tensor(train_ds.static_arr[bg_idx]).to(DEVICE)

explainer = shap.GradientExplainer(shap_model, [bg_ts, bg_static])
print("GradientExplainer created.")


# ═══════════════════════════════════════════════════════════════════════
# COMPUTE SHAP VALUES
# ═══════════════════════════════════════════════════════════════════════

def compute_shap_for_loader(loader, dataset_name):
    """Returns DataFrames with static SHAP values and ts SHAP (mean |shap| per vital)."""
    all_static_shap = []
    all_ts_shap     = []
    all_stay_ids    = []

    print(f"\nComputing SHAP for {dataset_name} set...")
    offset = 0
    for batch_idx, (batch_ts, batch_static, _) in enumerate(loader):
        batch_ts     = batch_ts.to(DEVICE)
        batch_static = batch_static.to(DEVICE)

        # shap_vals[0]: ts SHAP   shape (batch, 48, n_ts_feats)
        # shap_vals[1]: static SHAP shape (batch, n_static_feats)
        shap_vals = explainer.shap_values([batch_ts, batch_static])

        # Static: keep raw per-feature SHAP
        all_static_shap.append(shap_vals[1])

        # Time-series: aggregate mean |SHAP| across 48 time steps per vital
        ts_mean_abs = np.abs(shap_vals[0]).mean(axis=1)   # (batch, n_ts_feats)
        all_ts_shap.append(ts_mean_abs)

        n = batch_ts.shape[0]
        all_stay_ids.extend(loader.dataset.stay_ids[offset: offset + n].tolist())
        offset += n
        print(f"  batch {batch_idx+1}/{len(loader)}", end="\r")

    print()

    static_shap_arr = np.vstack(all_static_shap)   # (N, n_static_feats)
    ts_shap_arr     = np.vstack(all_ts_shap)        # (N, n_ts_feats)

    df_static = pd.DataFrame(static_shap_arr, columns=STATIC_FEATURES)
    df_ts     = pd.DataFrame(ts_shap_arr,
                             columns=[f"ts_shap_{f}" for f in TS_FEATURES])

    df = pd.concat([
        pd.DataFrame({"stay_id": all_stay_ids, "split": dataset_name}),
        df_static,
        df_ts,
    ], axis=1)
    return df


df_test  = compute_shap_for_loader(test_loader,  "test")
df_train = compute_shap_for_loader(train_loader, "train")

explanations = pd.concat([df_test, df_train], ignore_index=True)
explanations.to_parquet(OUTPUT_DIR / "explanations.parquet", index=False)
print(f"\nSaved: output/explanations.parquet  ({len(explanations):,} stays × "
      f"{len(explanations.columns)-2} SHAP columns)")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY PLOT — top 20 static features by mean |SHAP|
# ═══════════════════════════════════════════════════════════════════════

print("\nGenerating SHAP summary plot...")

test_static_shap = df_test[STATIC_FEATURES].values
test_static_vals = X_test.drop(columns=["stay_id"]).values

# shap.summary_plot expects (n_samples, n_features)
shap.summary_plot(
    test_static_shap,
    test_static_vals,
    feature_names=STATIC_FEATURES,
    max_display=20,
    show=False,
    plot_size=(12, 8),
)
plt.title("SHAP Feature Importance — Static Features (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/shap_summary.png")

# Per-stay mean |SHAP| ranking (top 10 for quick inspection)
mean_abs = pd.Series(
    np.abs(test_static_shap).mean(axis=0), index=STATIC_FEATURES
).sort_values(ascending=False)
print("\nTop 10 static features by mean |SHAP| on test set:")
for feat, val in mean_abs.head(10).items():
    print(f"  {feat:<35} {val:.4f}")
