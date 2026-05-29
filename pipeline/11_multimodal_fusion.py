"""
Multimodal Fusion — CXR Features + BERT Embeddings
====================================================
Merges Julia's radiology features into the existing baseline splits using
early fusion (all features combined before training).

Two sets of CXR features are added:
  1. Structured (23 cols) — binary/numeric clinical features extracted from
     radiology reports (pneumonia, edema, ventilator, severity scores, etc.)
  2. BERT embeddings (3072 raw dims) — PCA-reduced to 64 dims, PCA fitted on
     train only to prevent leakage.

Patients without a CXR record (~83% of cohort) receive zeros for all CXR
features plus has_cxr=0.

Run AFTER:  06_normalize.py + cxr_03_extract_features.py + cxr_04_extract_embeddings.py
Run BEFORE: 07b_model_gru_multimodal.py

Input:   output/X_train_scaled.parquet  /  X_val_scaled  /  X_test_scaled
         output/cxr_structured_features.csv
         output/cxr_bert_embeddings.parquet

Output:  output/X_train_multimodal.parquet
         output/X_val_multimodal.parquet
         output/X_test_multimodal.parquet
         output/pca_bert.pkl              — fitted PCA for inference
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from config import OUTPUT_DIR

# ── Paths ──────────────────────────────────────────────────────────────
CXR_STRUCT = OUTPUT_DIR / "cxr_structured_features.csv"
CXR_BERT   = OUTPUT_DIR / "cxr_bert_embeddings.parquet"

BERT_N_COMPONENTS = 64

# ── Load baseline splits ───────────────────────────────────────────────
print("Loading baseline splits...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
X_val   = pd.read_parquet(OUTPUT_DIR / "X_val_scaled.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")
print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

BASELINE_COLS = set(X_train.columns)

# ── Load Julia's files ─────────────────────────────────────────────────
print("\nLoading CXR structured features...")
struct = pd.read_csv(CXR_STRUCT)
struct_cols = [c for c in struct.columns
               if c not in ["subject_id", "hadm_id", "los_gt7"]]
struct = struct[struct_cols]   # keep stay_id + 23 feature cols
print(f"  Rows: {len(struct):,}  Feature cols: {len(struct_cols)-1}")

print("Loading BERT embeddings...")
bert = pd.read_parquet(CXR_BERT)
bert_embed_cols = [c for c in bert.columns
                   if c not in ["subject_id", "hadm_id", "stay_id", "los_gt7"]]
bert = bert[["stay_id"] + bert_embed_cols]
print(f"  Rows: {len(bert):,}  Embedding dims: {len(bert_embed_cols)}")

STRUCT_FEAT_COLS = [c for c in struct.columns if c != "stay_id"]

# ── PCA on BERT embeddings (fit on train only) ─────────────────────────
train_stay_ids = set(X_train["stay_id"])
bert_train = bert[bert["stay_id"].isin(train_stay_ids)][bert_embed_cols].values

n_components = min(BERT_N_COMPONENTS, len(bert_train) - 1, len(bert_embed_cols))
print(f"\nFitting PCA ({n_components} components) on train BERT embeddings (n_train_cxr={len(bert_train)})...")
pca = PCA(n_components=n_components, random_state=42)
pca.fit(bert_train)
explained = pca.explained_variance_ratio_.sum()
print(f"  Explained variance: {explained:.1%}")

# Transform all rows
bert_pca = pca.transform(bert[bert_embed_cols].values)
bert_reduced = pd.DataFrame(
    bert_pca,
    columns=[f"bert_pca_{i}" for i in range(n_components)]
)
bert_reduced.insert(0, "stay_id", bert["stay_id"].values)

# Save PCA for inference
with open(OUTPUT_DIR / "pca_bert.pkl", "wb") as f:
    pickle.dump(pca, f)
print(f"  Saved: output/pca_bert.pkl")


# ── Merge helper ───────────────────────────────────────────────────────
def merge_cxr(X: pd.DataFrame, struct: pd.DataFrame,
              bert_reduced: pd.DataFrame) -> pd.DataFrame:
    """Left-join CXR features onto X; fill missing with 0, add has_cxr flag."""
    X = X.merge(struct,       on="stay_id", how="left")
    X = X.merge(bert_reduced, on="stay_id", how="left")

    cxr_feat_cols = [c for c in X.columns if c not in BASELINE_COLS]

    # Presence is detected via the first structured CXR column — NaN means no CXR record
    X["has_cxr"] = (~X[STRUCT_FEAT_COLS[0]].isna()).astype(int)
    X[cxr_feat_cols] = X[cxr_feat_cols].fillna(0.0)

    return X


# ── Apply fusion ───────────────────────────────────────────────────────
print("\nMerging CXR features into splits...")
X_train_mm = merge_cxr(X_train, struct, bert_reduced)
X_val_mm   = merge_cxr(X_val,   struct, bert_reduced)
X_test_mm  = merge_cxr(X_test,  struct, bert_reduced)

for name, X_orig, X_mm in [
    ("train", X_train, X_train_mm),
    ("val",   X_val,   X_val_mm),
    ("test",  X_test,  X_test_mm),
]:
    n_with_cxr = X_mm["has_cxr"].sum()
    pct = n_with_cxr / len(X_mm) * 100
    added = X_mm.shape[1] - X_orig.shape[1]
    print(f"  {name:<6}: {len(X_mm):>7,} stays | {n_with_cxr:,} with CXR ({pct:.1f}%) | +{added} new cols")

# ── Save ───────────────────────────────────────────────────────────────
print("\nSaving multimodal splits...")
X_train_mm.to_parquet(OUTPUT_DIR / "X_train_multimodal.parquet", index=False)
X_val_mm.to_parquet(  OUTPUT_DIR / "X_val_multimodal.parquet",   index=False)
X_test_mm.to_parquet( OUTPUT_DIR / "X_test_multimodal.parquet",  index=False)

total_features = X_train_mm.shape[1] - 1  # exclude stay_id
print(f"\nDone. Total features in multimodal set: {total_features}")
print(f"  Baseline features : {X_train.shape[1] - 1}")
print(f"  CXR structured    : 23")
print(f"  BERT PCA          : {BERT_N_COMPONENTS}")
print(f"  has_cxr flag      : 1")
print(f"Saved: output/X_train_multimodal.parquet")
print(f"Saved: output/X_val_multimodal.parquet")
print(f"Saved: output/X_test_multimodal.parquet")
