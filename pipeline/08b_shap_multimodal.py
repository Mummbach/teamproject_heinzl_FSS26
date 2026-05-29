"""
SHAP Explainability — GRU+MLP Multimodal Model
===============================================
Identical to 08_shap.py but operates on the multimodal model and feature set
produced by 11_multimodal_fusion.py + 07b_model_gru_multimodal.py.

Extra CXR feature groups surfaced in the SHAP output:
  • 23 structured CXR features  (pneumonia, edema, ventilator, ...)
  • up to 64 BERT PCA dims      (bert_pca_0 … bert_pca_N)
  • 1  has_cxr flag

Run AFTER:  07b_model_gru_multimodal.py  (best_gru_multimodal.pt must exist)
            11_multimodal_fusion.py      (X_*_multimodal.parquet must exist)

Input:   output/X_train_multimodal.parquet  (background reference set)
         output/X_test_multimodal.parquet
         output/y_train.parquet / y_test.parquet
         output/timeseries.parquet
         output/best_gru_multimodal.pt
         output/scaler_params.parquet

Output:  output/explanations_multimodal.parquet
         output/shap_summary_multimodal.png
         output/shap_summary_cxr.png          — zoomed view: CXR features only
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    raise ImportError("Install shap: pip install shap")

from config import OUTPUT_DIR
from multimodal_utils import (
    ICUDataset, GRUModel, SHAPWrapper, load_multimodal_model,
    CXR_STRUCT_FEATURES, get_cxr_feature_groups,
)

SEED       = 42
BG_SIZE    = 200
BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

print("Loading multimodal data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_multimodal.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_multimodal.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
ts      = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

TS_FEATURES     = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_FEATURES = [c for c in X_train.columns if c != "stay_id"]

# Identify CXR feature subsets present in the multimodal set
groups               = get_cxr_feature_groups(STATIC_FEATURES)
BERT_PCA_FEATURES    = groups["bert_pca"]
CXR_FEATURES_PRESENT = groups["cxr_all"]
BASELINE_FEATURES    = groups["baseline"]

print(f"  Time-series features : {len(TS_FEATURES)}")
print(f"  Static features total: {len(STATIC_FEATURES)}")
print(f"    Baseline           : {len(BASELINE_FEATURES)}")
print(f"    CXR structured     : {len([c for c in CXR_STRUCT_FEATURES if c in STATIC_FEATURES])}")
print(f"    BERT PCA dims      : {len(BERT_PCA_FEATURES)}")
print(f"    has_cxr flag       : {'has_cxr' in STATIC_FEATURES}")

train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES)
test_ds  = ICUDataset(X_test,  y_test,  ts, TS_FEATURES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════

model = load_multimodal_model(
    OUTPUT_DIR / "best_gru_multimodal.pt",
    ts_input_size=len(TS_FEATURES),
    static_input_size=len(STATIC_FEATURES),
    device=DEVICE,
)
print("Model loaded from best_gru_multimodal.pt")

shap_model = SHAPWrapper(model).to(DEVICE)
shap_model.eval()


# ═══════════════════════════════════════════════════════════════════════
# BACKGROUND
# ═══════════════════════════════════════════════════════════════════════

print(f"\nBuilding background set (n={BG_SIZE})...")
rng       = np.random.default_rng(SEED)
bg_idx    = rng.choice(len(train_ds), size=min(BG_SIZE, len(train_ds)), replace=False)
bg_ts     = torch.tensor(train_ds.ts_arr[bg_idx]).to(DEVICE)
bg_static = torch.tensor(train_ds.static_arr[bg_idx]).to(DEVICE)

explainer = shap.GradientExplainer(shap_model, [bg_ts, bg_static])
print("GradientExplainer created.")


# ═══════════════════════════════════════════════════════════════════════
# COMPUTE SHAP VALUES
# ═══════════════════════════════════════════════════════════════════════

def compute_shap_for_loader(loader, dataset_name):
    all_static_shap, all_ts_shap, all_stay_ids = [], [], []
    print(f"\nComputing SHAP for {dataset_name} set...")
    offset = 0
    for batch_idx, (batch_ts, batch_static, _) in enumerate(loader):
        batch_ts     = batch_ts.to(DEVICE)
        batch_static = batch_static.to(DEVICE)
        shap_vals    = explainer.shap_values([batch_ts, batch_static])

        # GradientExplainer returns list of arrays; squeeze output dim added by SHAPWrapper
        static_shap = shap_vals[1]
        if static_shap.ndim == 3:
            static_shap = static_shap.squeeze(-1)
        ts_shap = shap_vals[0]
        if ts_shap.ndim == 4:
            ts_shap = ts_shap.squeeze(-1)

        all_static_shap.append(static_shap)
        ts_mean_abs = np.abs(ts_shap).mean(axis=1)
        all_ts_shap.append(ts_mean_abs)

        n = batch_ts.shape[0]
        all_stay_ids.extend(loader.dataset.stay_ids[offset: offset + n].tolist())
        offset += n
        print(f"  batch {batch_idx+1}/{len(loader)}", end="\r")
    print()

    static_shap_arr = np.vstack(all_static_shap)
    ts_shap_arr     = np.vstack(all_ts_shap)

    df_static = pd.DataFrame(static_shap_arr, columns=STATIC_FEATURES)
    df_ts     = pd.DataFrame(ts_shap_arr, columns=[f"ts_shap_{f}" for f in TS_FEATURES])

    return pd.concat([
        pd.DataFrame({"stay_id": all_stay_ids, "split": dataset_name}),
        df_static,
        df_ts,
    ], axis=1)


df_test  = compute_shap_for_loader(test_loader,  "test")
df_train = compute_shap_for_loader(train_loader, "train")

explanations = pd.concat([df_test, df_train], ignore_index=True)
explanations.to_parquet(OUTPUT_DIR / "explanations_multimodal.parquet", index=False)
print(f"\nSaved: output/explanations_multimodal.parquet  ({len(explanations):,} rows)")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1 — Full summary (top 20 static features incl. CXR)
# ═══════════════════════════════════════════════════════════════════════

print("\nGenerating SHAP summary plot (all features)...")
test_static_shap = df_test[STATIC_FEATURES].values
test_static_vals = X_test.drop(columns=["stay_id"]).values

shap.summary_plot(
    test_static_shap,
    test_static_vals,
    feature_names=STATIC_FEATURES,
    max_display=20,
    show=False,
    plot_size=(12, 8),
)
plt.title("SHAP Feature Importance — Multimodal Static Features (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary_multimodal.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/shap_summary_multimodal.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2 — CXR features only (struct + BERT PCA + has_cxr)
# ═══════════════════════════════════════════════════════════════════════

if CXR_FEATURES_PRESENT:
    print("\nGenerating CXR-only SHAP summary plot...")
    cxr_idx  = [STATIC_FEATURES.index(c) for c in CXR_FEATURES_PRESENT]
    cxr_shap = test_static_shap[:, cxr_idx]
    cxr_vals = test_static_vals[:, cxr_idx]

    shap.summary_plot(
        cxr_shap,
        cxr_vals,
        feature_names=CXR_FEATURES_PRESENT,
        max_display=30,
        show=False,
        plot_size=(12, 8),
    )
    plt.title("SHAP Feature Importance — CXR Features Only (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary_cxr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/shap_summary_cxr.png")


# ═══════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════

mean_abs = pd.Series(
    np.abs(test_static_shap).mean(axis=0), index=STATIC_FEATURES
).sort_values(ascending=False)

print("\nTop 10 overall features (mean |SHAP|):")
for feat, val in mean_abs.head(10).items():
    print(f"  {feat:<40} {val:.4f}")

if CXR_FEATURES_PRESENT:
    cxr_mean_abs = mean_abs[mean_abs.index.isin(CXR_FEATURES_PRESENT)]
    print(f"\nTop 10 CXR features (mean |SHAP|):")
    for feat, val in cxr_mean_abs.head(10).items():
        print(f"  {feat:<40} {val:.4f}")
