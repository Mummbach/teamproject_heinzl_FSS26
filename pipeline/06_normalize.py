"""
Feature Normalization
=====================
Applies StandardScaler to continuous features, fitted exclusively on
the training split to prevent data leakage into validation and test sets.

Binary features (0/1) are left untouched — scaling them would distort
their meaning and is not needed for neural networks.

Steps:
  1. Detect continuous vs. binary columns from X_train
  2. Fit StandardScaler on X_train continuous columns only
  3. Apply to train / val / test
  4. Save scaled splits + scaler parameters for inference

Run AFTER:  04_preprocessing.py
Run BEFORE: model training

Input:   output/X_train.parquet
         output/X_val.parquet
         output/X_test.parquet

Output:  output/X_train_scaled.parquet
         output/X_val_scaled.parquet
         output/X_test_scaled.parquet
         output/scaler_params.parquet   — feature, mean, std (for inference)
"""

import pandas as pd
import numpy as np
from config import OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True)


# LOAD

X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
X_val   = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")

print(f"Loaded splits:")
print(f"  train : {X_train.shape[0]:>7,} stays × {X_train.shape[1]} cols")
print(f"  val   : {X_val.shape[0]:>7,} stays × {X_val.shape[1]} cols")
print(f"  test  : {X_test.shape[0]:>7,} stays × {X_test.shape[1]} cols")


# IDENTIFY COLUMN TYPES

feature_cols = [c for c in X_train.columns if c != "stay_id"]

# A column is treated as binary if every non-NaN value in X_train is 0 or 1.
# This reliably captures all ICD, ATC, demographic flags, and _missing indicators.
binary_cols     = [c for c in feature_cols if X_train[c].dropna().isin([0, 1]).all()]
continuous_cols = [c for c in feature_cols if c not in binary_cols]

print(f"\nColumn classification:")
print(f"  continuous : {len(continuous_cols):>4}  (will be scaled)")
print(f"  binary     : {len(binary_cols):>4}  (left as-is)")


# FIT SCALER ON TRAIN ONLY

train_mean = X_train[continuous_cols].mean()
train_std  = X_train[continuous_cols].std(ddof=0)

# Replace zero std with 1 to avoid division by zero for constant columns.
# A constant feature carries no information; scaling it to 0/NaN would lose
# the zero-variance signal — keeping it as-is (divide by 1) is the safest fallback.
zero_std_cols = train_std[train_std == 0].index.tolist()
if zero_std_cols:
    print(f"\n  WARNING: {len(zero_std_cols)} constant column(s) in train (std=0) — left unchanged:")
    for c in zero_std_cols:
        print(f"    {c}")
train_std = train_std.replace(0, 1)


# APPLY TO ALL SPLITS

splits = {"train": X_train, "val": X_val, "test": X_test}
scaled = {}

for name, X_split in splits.items():
    X_sc = X_split.copy()
    X_sc[continuous_cols] = (X_split[continuous_cols] - train_mean) / train_std
    scaled[name] = X_sc

    # Verify: binary columns must be unchanged
    assert (X_sc[binary_cols].values == X_split[binary_cols].values).all(), \
        f"{name}: binary columns were modified"

    cont_mean_after = X_sc[continuous_cols].mean().abs().mean()
    cont_std_after  = X_sc[continuous_cols].std().mean()
    print(f"\n  {name}:")
    print(f"    mean of |feature means| after scaling : {cont_mean_after:.4f}  (target ≈ 0)")
    print(f"    mean of feature stds  after scaling   : {cont_std_after:.4f}  (target ≈ 1)")


# SANITY CHECK: no NaN introduced by scaling

for name, X_sc in scaled.items():
    nan_count = X_sc[continuous_cols].isna().sum().sum()
    assert nan_count == 0, f"{name}: {nan_count} NaN values found after scaling"
print("\n  No NaN introduced by scaling ✓")


# SAVE

scaler_params = pd.DataFrame({
    "feature": continuous_cols,
    "mean":    train_mean.values,
    "std":     train_std.values,
})
scaler_params.to_parquet(OUTPUT_DIR / "scaler_params.parquet", index=False)

for name, X_sc in scaled.items():
    path = OUTPUT_DIR / f"X_{name}_scaled.parquet"
    X_sc.to_parquet(path, index=False)
    print(f"\n  Saved: output/X_{name}_scaled.parquet  ({len(X_sc):,} rows × {len(X_sc.columns)} cols)")

print(f"\n  Saved: output/scaler_params.parquet  ({len(scaler_params)} continuous features)")
print(f"\n  Unscaled splits (output/X_*.parquet) are preserved for tree-based models.")
