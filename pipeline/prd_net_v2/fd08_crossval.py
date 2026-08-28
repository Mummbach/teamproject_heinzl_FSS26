"""
PRD-Net v2 — fd08: Cross-Validation
====================================
Stratified K-Fold CV of the feature-diff model, to check that the single-split
test metrics in fd_metrics_{w}.json are stable and not a lucky train/val/test
split. Mirrors the purpose of baseline/08_crossval.py.

Why this can't just resample the feature matrix (unlike baseline's CV):
  Every patient's prediction depends on prototypes built from THEIR peers
  (see fd02_feature-prototypes.py). So each fold must rebuild its own
  prototypes from scratch, using only that fold's training patients as the
  peer pool — never the peers used by the real fd02/fd04 run. We reuse fd02's
  build_prototypes_filtered() for this, since it already takes an arbitrary
  peer pool + query set as plain arguments.

  Only supported under RETRIEVAL_SPACE="feature" (the default). "embedding"
  mode reuses v1's GRU-embedding peer cache, which was built once against the
  fixed official train split — recomputing it per fold would mean retraining
  v1's GRU per fold too, which this script does not do.

Unlike baseline (which CVs a fast Random Forest as a proxy for the too-slow-
to-retrain GRU), the real v2 model is already linear and fast, so this CVs the
actual model architecture (sklearn LogisticRegression, matches the torch
LinearDiffModel per fd04's own sanity check) — no proxy needed.

Note: rebuilding prototypes 2x per fold (once for the fold's own training
patients, once for the held-out fold) costs roughly K_FOLDS times what a
single fd02 run costs. Expect this to take noticeably longer than fd02 alone.

Steps:
  1. Load train + val as one CV pool; test stays held out (untouched), exactly
     like baseline.
  2. StratifiedKFold(K_FOLDS) over the pool.
  3. Per fold: rebuild prototypes for the fold's own training patients (peer
     pool = themselves, self-excluded) and for the held-out fold (peer pool =
     the other folds), fit LogisticRegression, evaluate at threshold 0.5.
  4. Report mean +/- std across folds, and compare against the single-split
     test numbers already in fd_metrics_{w}.json.

Run AFTER: fd01_feature-matrix.py.
(fd02/fd04 do not need to have been run first — this script builds its own
 fold-specific prototypes independently of their cached artifacts.)

Output: exports/fd_cv_results_{w}.csv  — per-fold metrics
        exports/fd_cv_summary_{w}.txt  — mean +/- std report
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from config import OUTPUT_DIR
import config_fd as C


def _load_module(name, filename):
    """fd0N filenames start with a digit and contain a hyphen -> importlib,
    same pattern fd04_diff-train.py already uses to pull in fd03."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fd02 = _load_module("fd02", "fd02_feature-prototypes.py")
fd03 = _load_module("fd03", "fd03_diff-model.py")
fd04 = _load_module("fd04", "fd04_diff-train.py")

if C.RETRIEVAL_SPACE != "feature":
    raise SystemExit(
        "fd08_crossval only supports RETRIEVAL_SPACE='feature' (the default) — "
        "'embedding' mode reuses v1's fixed GRU peer cache, which can't be "
        "rebuilt per fold. Set RETRIEVAL_SPACE='feature' in config_fd.py and rerun."
    )

K_FOLDS      = 5
RANDOM_STATE = 42
W            = C.WINDOW_HOURS
METRIC_COLS  = ["f1", "auroc", "auprc", "precision", "recall", "accuracy"]


# ── Load one split's worth of data (mirrors fd02's own loading block) ─────────
def load_split(split):
    Xdf = pd.read_parquet(OUTPUT_DIR / f"X_{split}.parquet")
    y   = pd.read_parquet(OUTPUT_DIR / f"y_{split}.parquet").set_index("stay_id")
    ids = Xdf["stay_id"].values
    labels = y.loc[ids, "los_gt7"].values
    # Hard filter needs the true primary diagnosis (seq_num==1, from labels.parquet),
    # not the multi-label icd_* columns — see fd02.build_prototypes_filtered's docstring.
    Xdf["primary_diag"] = y.loc[ids, "primary_diag"].values

    scaled = pd.read_parquet(C.feature_matrix_path(split, scaled=True)).set_index("stay_id")
    assert list(scaled.index) == list(ids), f"{split}: scaled matrix misaligned with X_{split}"
    M = scaled[C.feature_names()].values.astype(np.float32)
    absolute = scaled[C.ABSOLUTE_FEATURES].values.astype(np.float32) if C.USE_ABSOLUTE_FEATURES else None

    has_report = None
    if C.USE_CXR_FEATURES:
        raw = pd.read_parquet(C.feature_matrix_path(split, scaled=False),
                              columns=["stay_id", "has_cxr_report"]).set_index("stay_id")
        has_report = (raw.loc[ids, "has_cxr_report"].values > 0)

    return Xdf.reset_index(drop=True), ids, labels, M, absolute, has_report


print(f"fd08 — {K_FOLDS}-fold cross-validation (window={W}h, retrieval={C.RETRIEVAL_SPACE})")
print("Loading train + val (CV pool); test stays held out...")

parts = [load_split(s) for s in ("train", "val")]
pool_Xdf    = pd.concat([p[0] for p in parts], ignore_index=True)
pool_ids    = np.concatenate([p[1] for p in parts])
pool_labels = np.concatenate([p[2] for p in parts])
pool_M      = np.concatenate([p[3] for p in parts])
pool_abs    = np.concatenate([p[4] for p in parts]) if C.USE_ABSOLUTE_FEATURES else None
pool_hasrep = np.concatenate([p[5] for p in parts]) if C.USE_CXR_FEATURES else None

print(f"  CV pool (train+val): {len(pool_ids):,} stays")
print(f"  K-Folds             : {K_FOLDS}")

# CXR content columns get a class-conditional reference from report-holders
# in the peer pool (see fd02._condition_cxr) — recomputed per fold below.
feats_all = C.feature_names()
cxr_idx = None
if C.USE_CXR_FEATURES and "has_cxr_report" in feats_all:
    content_cols = [f for f in C.CXR_FEATURES if f != "has_cxr_report"]
    cxr_idx = np.array([feats_all.index(f) for f in content_cols])


def cxr_refs(labels, M, has_report):
    if cxr_idx is None:
        return None, None
    pos_ref = M[has_report & (labels == 1)][:, cxr_idx].mean(axis=0)
    neg_ref = M[has_report & (labels == 0)][:, cxr_idx].mean(axis=0)
    return pos_ref, neg_ref


def build_bundle(query_idx, peer_idx, exclude_self):
    """One call to fd02.build_prototypes_filtered, with this fold's peer pool
    (peer_idx into the CV pool) standing in for "the training set"."""
    peer_Xdf    = pool_Xdf.iloc[peer_idx].reset_index(drop=True)
    peer_ids    = pool_ids[peer_idx]
    peer_labels = pool_labels[peer_idx]
    peer_M      = pool_M[peer_idx]
    peer_hasrep = pool_hasrep[peer_idx] if pool_hasrep is not None else None
    cxr_pos_ref, cxr_neg_ref = cxr_refs(peer_labels, peer_M, peer_hasrep)

    return fd02.build_prototypes_filtered(
        query_df=pool_Xdf.iloc[query_idx].reset_index(drop=True),
        query_ids=pool_ids[query_idx], query_labels=pool_labels[query_idx], M_query=pool_M[query_idx],
        X_train_df=peer_Xdf, all_train_ids=peer_ids, train_labels=peer_labels, M_train=peer_M,
        emb_cache=None, space="feature", exclude_self=exclude_self,
        cxr_idx=cxr_idx, cxr_pos_ref=cxr_pos_ref, cxr_neg_ref=cxr_neg_ref,
        has_report=pool_hasrep[query_idx] if pool_hasrep is not None else None,
    )


def diff_and_labels(bundle, idx):
    absolute = pool_abs[idx] if pool_abs is not None else None
    X = fd03.assemble_diff(bundle["X"], bundle["pos_proto"], bundle["neg_proto"], absolute=absolute)
    return X.astype(np.float32), bundle["labels"].astype(np.float32)


# ── K-Fold CV ───────────────────────────────────────────────────────────────
print(f"\nRunning {K_FOLDS}-fold CV (prototypes rebuilt per fold, this takes a while)...")
print(f"{'Fold':<6} {'F1':<8} {'AUROC':<8} {'AUPRC':<8} {'Precision':<10} {'Recall':<8}")
print("─" * 52)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_results = []

for fold, (train_idx, query_idx) in enumerate(skf.split(pool_ids, pool_labels), start=1):
    train_bundle = build_bundle(query_idx=train_idx, peer_idx=train_idx, exclude_self=True)
    query_bundle = build_bundle(query_idx=query_idx, peer_idx=train_idx, exclude_self=False)

    Xtr, ytr = diff_and_labels(train_bundle, train_idx)
    Xq, yq   = diff_and_labels(query_bundle, query_idx)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        lr = fd03.build_logreg().fit(Xtr.astype(np.float64), ytr.astype(int))
        probs = lr.predict_proba(Xq.astype(np.float64))[:, 1]

    # NOTE: threshold fixed at 0.5 (no nested val fold to tune on) — same
    # simplification baseline/08_crossval.py makes; use AUPRC for comparisons.
    metrics = fd04.evaluate(probs, yq, threshold=0.5)
    metrics["fold"] = fold
    fold_results.append(metrics)
    print(f"  {fold:<4} {metrics['f1']:<8.4f} {metrics['auroc']:<8.4f} "
          f"{metrics['auprc']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
cv_df = pd.DataFrame(fold_results)

print(f"\n{'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("─" * 48)
for col in METRIC_COLS:
    print(f"  {col:<10} {cv_df[col].mean():>8.4f} {cv_df[col].std():>8.4f} "
          f"{cv_df[col].min():>8.4f} {cv_df[col].max():>8.4f}")

# ── Compare against the existing single train+val -> test split ───────────────
existing = C.metrics_path()
single_split = None
if existing.exists():
    single_split = json.load(open(existing))["test_logreg"]
    print(f"\nFor comparison — single train+val -> test split ({existing.name}):")
    for col in METRIC_COLS:
        print(f"  {col:<10} {single_split[col]:>8.4f}")
else:
    print(f"\n(No {existing.name} found — run fd04_diff-train.py for a single-split comparison.)")

# ── Verdict ───────────────────────────────────────────────────────────────────
# Simple heuristic on AUPRC (the headline metric for this imbalanced task, per
# the README): folds within +/- RELIABILITY_AUPRC_STD of each other -> stable.
RELIABILITY_AUPRC_STD = 0.03
auprc_std = cv_df["auprc"].std()
if auprc_std < RELIABILITY_AUPRC_STD:
    verdict = f"RELIABLE — AUPRC only varies {auprc_std:.4f} across folds, so the test-set result is not a lucky split."
else:
    verdict = f"NOT RELIABLE — AUPRC varies {auprc_std:.4f} across folds, so the test-set result may depend on the split."
print(f"\n{verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
C.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
cv_csv = C.EXPORT_DIR / f"fd_cv_results_{W}h.csv"
cv_txt = C.EXPORT_DIR / f"fd_cv_summary_{W}h.txt"
cv_df.to_csv(cv_csv, index=False)

summary_lines = [f"Cross-Validation Summary ({K_FOLDS}-Fold Stratified, LogisticRegression, window={W}h)\n\n"]
summary_lines.append(f"{'Metric':<12} {'Mean':>8} {'Std':>8}\n")
for col in METRIC_COLS:
    summary_lines.append(f"  {col:<10} {cv_df[col].mean():>8.4f} +/- {cv_df[col].std():>8.4f}\n")
if single_split is not None:
    summary_lines.append("\nFor comparison, single train+val -> test split:\n")
    for col in METRIC_COLS:
        summary_lines.append(f"  {col:<10} {single_split[col]:>8.4f}\n")
summary_lines.append(f"\n{verdict}\n")
cv_txt.write_text("".join(summary_lines))

print(f"\nSaved: {cv_csv.relative_to(C.FD_DIR)}")
print(f"Saved: {cv_txt.relative_to(C.FD_DIR)}")
