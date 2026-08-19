"""
Cross-Validation (Random Forest)
=================================
Stratified K-Fold CV on the non-test data to validate model stability.
Reports mean ± std metrics across folds.

Why Random Forest instead of GRU:
  RF trains in seconds per fold vs. hours for GRU.
  The purpose of CV here is to show that performance estimates are stable
  and not the result of a lucky train/val split — not to tune GRU hyperparameters.

Steps:
  1. Load X_train + X_val (combined non-test data)
  2. Run StratifiedKFold(n_splits=K_FOLDS)
  3. Per fold: train RF, evaluate on held-out fold
  4. Report mean ± std across folds
  5. Train final RF on full train+val, evaluate on fixed test set

Run AFTER:  preprocessing/04_preprocessing.py

Input:   output/X_train.parquet  output/y_train.parquet
         output/X_val.parquet    output/y_val.parquet
         output/X_test.parquet   output/y_test.parquet

Output:  output/cv_results.csv   — per-fold metrics
         output/analysis/cv_summary.txt — mean ± std report
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))  # pipeline/ -> config.py / multimodal_utils.py
from config import OUTPUT_DIR

ANALYSIS_DIR = OUTPUT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
K_FOLDS    = 5
RANDOM_STATE = 42
RF_PARAMS  = {
    "n_estimators":  200,
    "max_depth":     None,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
X_val   = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")

y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
y_val   = pd.read_parquet(OUTPUT_DIR / "y_val.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")

# Combine train + val as full non-test set for CV
X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)

feature_cols = [c for c in X_tv.columns if c != "stay_id"]
X_tv_arr = X_tv[feature_cols].values
y_tv_arr = y_tv["los_gt7"].values

X_test_arr = X_test[feature_cols].values
y_test_arr = y_test["los_gt7"].values

print(f"  Non-test set : {len(X_tv_arr):,} stays")
print(f"  Test set     : {len(X_test_arr):,} stays")
print(f"  Features     : {len(feature_cols)}")
print(f"  K-Folds      : {K_FOLDS}")

# ── K-Fold CV ─────────────────────────────────────────────────────────────────
print(f"\nRunning {K_FOLDS}-Fold Stratified CV...")
print(f"{'Fold':<6} {'F1':<8} {'AUROC':<8} {'AUPRC':<8} {'Precision':<10} {'Recall':<8}")
print("─" * 52)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tv_arr, y_tv_arr), start=1):
    X_fold_train, X_fold_val = X_tv_arr[train_idx], X_tv_arr[val_idx]
    y_fold_train, y_fold_val = y_tv_arr[train_idx], y_tv_arr[val_idx]

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_fold_train, y_fold_train)

    probs = rf.predict_proba(X_fold_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    if len(np.unique(y_fold_val)) < 2:
        fold_auc = float("nan")
    else:
        fold_auc = roc_auc_score(y_fold_val, probs)

    metrics = {
        "fold":      fold,
        # NOTE: threshold fixed at 0.5 — suboptimal for class-weighted RF; use AUPRC for comparisons
        "f1":        f1_score(y_fold_val, preds, zero_division=0),
        "auroc":     fold_auc,
        "auprc":     average_precision_score(y_fold_val, probs),
        "precision": precision_score(y_fold_val, preds, zero_division=0),
        "recall":    recall_score(y_fold_val, preds, zero_division=0),
        "accuracy":  accuracy_score(y_fold_val, preds),
    }
    fold_results.append(metrics)
    print(f"  {fold:<4} {metrics['f1']:<8.4f} {metrics['auroc']:<8.4f} "
          f"{metrics['auprc']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
cv_df = pd.DataFrame(fold_results)
metric_cols = ["f1", "auroc", "auprc", "precision", "recall", "accuracy"]

print(f"\n{'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("─" * 48)
for col in metric_cols:
    print(f"  {col:<10} {cv_df[col].mean():>8.4f} {cv_df[col].std():>8.4f} "
          f"{cv_df[col].min():>8.4f} {cv_df[col].max():>8.4f}")

# ── Final model on full train+val → test ──────────────────────────────────────
print(f"\nTraining final RF on full non-test set ({len(X_tv_arr):,} stays)...")
rf_final = RandomForestClassifier(**RF_PARAMS)
rf_final.fit(X_tv_arr, y_tv_arr)

probs_test = rf_final.predict_proba(X_test_arr)[:, 1]
preds_test = (probs_test >= 0.5).astype(int)

print("\nFinal test results (RF trained on full train+val):")
print(f"  Accuracy  : {accuracy_score(y_test_arr, preds_test):.4f}")
print(f"  Precision : {precision_score(y_test_arr, preds_test, zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_test_arr, preds_test, zero_division=0):.4f}")
print(f"  F1        : {f1_score(y_test_arr, preds_test, zero_division=0):.4f}")
if len(np.unique(y_test_arr)) < 2:
    test_auc = float("nan")
else:
    test_auc = roc_auc_score(y_test_arr, probs_test)
print(f"  AUROC     : {test_auc:.4f}")
print(f"  AUPRC     : {average_precision_score(y_test_arr, probs_test):.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
cv_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

summary_lines = [f"Cross-Validation Summary ({K_FOLDS}-Fold Stratified, Random Forest)\n"]
summary_lines.append(f"{'Metric':<12} {'Mean':>8} {'Std':>8}\n")
for col in metric_cols:
    summary_lines.append(f"  {col:<10} {cv_df[col].mean():>8.4f} ± {cv_df[col].std():>8.4f}\n")
(ANALYSIS_DIR / "cv_summary.txt").write_text("".join(summary_lines))

print(f"\nSaved: output/cv_results.csv")
print(f"Saved: output/analysis/cv_summary.txt")
