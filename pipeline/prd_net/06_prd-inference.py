"""
PRD-Net — Step 6: Clinically Filtered Inference
=================================================
Runs the trained PRD-Net on the test set using the same prototype strategy
as training: for each test patient, apply the ICD+ICU+age clinical filter
against the training set, then take the mean of the K=20 nearest matches
per outcome class as the prototype.

Why clinical filtering?
  During training, peers are filtered by same ICD chapter, same ICU type,
  and age within ±10 years before K-nearest selection. Using raw K-nearest
  (no filter) at test time gives the model structurally different prototypes
  than it was trained on, degrading F1. Applying the same filter end-to-end
  closes that gap (0.55 → 0.59 F1).

Prototype strategy (per test patient):
  1. Hard filter : keep training patients with same primary ICD chapter + ICU type
  2. Age filter  : keep candidates within ±AGE_TOLERANCE years
  3. K-nearest   : rank filtered candidates by L2 distance in embedding space
  4. Mean        : average the K=20 nearest embeddings → pos_proto / neg_proto
  Falls back to unfiltered K-nearest if the filtered pool is empty.

Output:
  - Full test metrics (accuracy, precision, recall, F1, AUROC, AUPRC)
  - Comparison table vs 05_prd-sanity.py global-mean approach
  - Top-3 most confident long/short stay predictions with closest matched prototype ID
"""

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)

sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, ICD_CATEGORIES
from prd_net.config_prd import EMBEDDING_CACHE_PATH, HIDDEN_DIM, K_PEERS, AGE_TOLERANCE

ICU_COLS = ["icu_micu", "icu_sicu", "icu_ccu", "icu_cvicu",
            "icu_micu_sicu", "icu_tsicu", "icu_neuro_sicu"]
ICD_COLS = [f"icd_{cat}" for cat in ICD_CATEGORIES]

# Load PRDNet via importlib (filename starts with digit and contains hyphen)
_spec = importlib.util.spec_from_file_location(
    "prd_model", Path(__file__).parent / "03_prd-model.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PRDNet = _mod.PRDNet

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "prd_net_v1.pt"
INPUT_DIM = 128


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading PRDNet checkpoint...")
model = PRDNet(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
model.eval()
print(f"  Loaded from : {CHECKPOINT_PATH}")

# ── Load embeddings ───────────────────────────────────────────────────────────
print("Loading embeddings...")
with open(EMBEDDING_CACHE_PATH, "rb") as f:
    emb_dict = pickle.load(f)

# ── Load training data ────────────────────────────────────────────────────────
print("Loading training data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")

train_ids    = X_train["stay_id"].values
train_labels = y_train.set_index("stay_id").loc[train_ids, "los_gt7"].values

# Split training patients by outcome class
print(f"  Positive training patients : {(train_labels == 1).sum():,}")
print(f"  Negative training patients : {(train_labels == 0).sum():,}")

# ── Load test set ─────────────────────────────────────────────────────────────
print("Loading test data...")
X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
y_test = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")

test_ids    = X_test["stay_id"].values
test_labels = y_test.set_index("stay_id").loc[test_ids, "los_gt7"].values
N_test = len(test_ids)

test_emb_np = np.stack([emb_dict[int(sid)] for sid in test_ids])  # (N_test, 128)
print(f"  Test patients : {N_test:,}")


# ── Build clinically filtered K=20 prototypes per test patient ───────────────
# Applies the same ICD+ICU+age filter as training (02_peer-groups.py) so the
# model receives the same prototype structure it was trained and validated on.
print(f"Building clinically filtered K={K_PEERS} prototypes per test patient...")

# Full training embedding matrix aligned row-for-row with X_train
all_train_stay_ids = X_train["stay_id"].values
all_train_labels   = train_labels  # already loaded above
all_train_emb      = np.stack([emb_dict[int(sid)] for sid in all_train_stay_ids])

train_icd = X_train[ICD_COLS].values   # (N_train, n_icd)
train_icu = X_train[ICU_COLS].values   # (N_train, n_icu)
train_age = X_train["age"].values       # (N_train,)

test_icd  = X_test[ICD_COLS].values
test_icu  = X_test[ICU_COLS].values
test_age  = X_test["age"].values

sid_to_test_row = {int(sid): i for i, sid in enumerate(test_ids)}

pos_proto_np    = np.zeros((N_test, INPUT_DIM), dtype=np.float32)
neg_proto_np    = np.zeros((N_test, INPUT_DIM), dtype=np.float32)
nearest_pos_ids = []
nearest_neg_ids = []

pos_global_idx = np.where(all_train_labels == 1)[0]
neg_global_idx = np.where(all_train_labels == 0)[0]

def _knn_mean(candidates, target_emb):
    dists = np.linalg.norm(all_train_emb[candidates] - target_emb, axis=1)
    top   = candidates[np.argsort(dists)[:min(K_PEERS, len(candidates))]]
    return all_train_emb[top].mean(axis=0), int(all_train_stay_ids[top[0]])

for i, sid in enumerate(tqdm(test_ids, desc="Filtered prototypes", leave=False)):
    q      = sid_to_test_row[int(sid)]
    target = emb_dict[int(sid)]

    q_icd = int(np.argmax(test_icd[q])) if test_icd[q].max() == 1 else None
    q_icu = int(np.argmax(test_icu[q])) if test_icu[q].max() == 1 else None

    mask = np.ones(len(X_train), dtype=bool)
    if q_icd is not None:
        mask &= train_icd[:, q_icd] == 1
    if q_icu is not None:
        mask &= train_icu[:, q_icu] == 1
    mask &= np.abs(train_age - test_age[q]) <= AGE_TOLERANCE

    candidates = np.where(mask)[0]
    pos_cands  = candidates[all_train_labels[candidates] == 1]
    neg_cands  = candidates[all_train_labels[candidates] == 0]
    if len(pos_cands) == 0: pos_cands = pos_global_idx
    if len(neg_cands) == 0: neg_cands = neg_global_idx

    pos_emb, pid = _knn_mean(pos_cands, target)
    neg_emb, nid = _knn_mean(neg_cands, target)
    pos_proto_np[i] = pos_emb
    neg_proto_np[i] = neg_emb
    nearest_pos_ids.append(pid)
    nearest_neg_ids.append(nid)

pos_proto_raw = torch.tensor(pos_proto_np)
neg_proto_raw = torch.tensor(neg_proto_np)
test_emb_t    = torch.tensor(test_emb_np, dtype=torch.float32)
print("  Done.")


# ── Run PRDNet ────────────────────────────────────────────────────────────────
print("Running PRDNet with K=20 mean prototypes...")
with torch.no_grad():
    pos_proto            = model.encode(pos_proto_raw)   # (N_test, hidden_dim)
    neg_proto            = model.encode(neg_proto_raw)   # (N_test, hidden_dim)
    logits, delta_pos_t, delta_neg_t = model(test_emb_t, pos_proto, neg_proto)

logits_np    = logits.numpy()
delta_pos_np = delta_pos_t.numpy()
delta_neg_np = delta_neg_t.numpy()
print(f"  Predictions computed for {N_test:,} test patients")


# ── Full evaluation metrics ───────────────────────────────────────────────────
probs = 1 / (1 + np.exp(-logits_np))
preds = (probs >= 0.5).astype(int)

metrics = {
    "accuracy"  : accuracy_score(test_labels, preds),
    "precision" : precision_score(test_labels, preds, zero_division=0),
    "recall"    : recall_score(test_labels, preds, zero_division=0),
    "f1"        : f1_score(test_labels, preds, zero_division=0),
    "auroc"     : roc_auc_score(test_labels, probs),
    "auprc"     : average_precision_score(test_labels, probs),
}

print("\nTest metrics (nearest-neighbour prototypes):")
for name, val in metrics.items():
    print(f"  {name:<12}: {val:.4f}")

# Reference values from 05_prd-sanity.py (global mean prototypes)
global_mean_metrics = {
    "accuracy"  : None,
    "precision" : None,
    "recall"    : None,
    "f1"        : None,
    "auroc"     : None,
    "auprc"     : None,
}

print("\nComparison: nearest-neighbour prototype vs global mean (05_prd-sanity.py):")
print(f"  {'metric':<12}  {'nearest-1':>10}  {'global mean':>12}  {'diff':>8}")
print("  " + "─" * 46)
for name, val in metrics.items():
    ref = global_mean_metrics[name]
    if ref is not None:
        diff_str = f"{val - ref:+.4f}"
    else:
        diff_str = "  (run 05 first)"
    ref_str = f"{ref:.4f}" if ref is not None else "      n/a"
    print(f"  {name:<12}  {val:>10.4f}  {ref_str:>12}  {diff_str:>8}")


# ── Top-3 most confident predictions ─────────────────────────────────────────
top3_long  = np.argsort(logits_np)[-3:][::-1]
top3_short = np.argsort(logits_np)[:3]

print("\nTop-3 most confident LONG-STAY predictions (highest logit):")
print(f"  {'stay_id':<12}  {'logit':>8}  {'prob':>6}  {'true':>10}  {'matched pos proto':<18}  {'matched neg proto'}")
print(f"  {'-'*80}")
for i in top3_long:
    label_str = "long  (1)" if test_labels[i] == 1 else "short (0)"
    print(f"  {int(test_ids[i]):<12}  {logits_np[i]:>8.4f}  {probs[i]:>6.3f}  "
          f"{label_str:>10}  {nearest_pos_ids[i]:<18}  {nearest_neg_ids[i]}")

print("\nTop-3 most confident SHORT-STAY predictions (lowest logit):")
print(f"  {'stay_id':<12}  {'logit':>8}  {'prob':>6}  {'true':>10}  {'matched pos proto':<18}  {'matched neg proto'}")
print(f"  {'-'*80}")
for i in top3_short:
    label_str = "long  (1)" if test_labels[i] == 1 else "short (0)"
    print(f"  {int(test_ids[i]):<12}  {logits_np[i]:>8.4f}  {probs[i]:>6.3f}  "
          f"{label_str:>10}  {nearest_pos_ids[i]:<18}  {nearest_neg_ids[i]}")


# ── Delta norm check (same as Step 25 in 05_prd-sanity.py) ───────────────────
print("\nPrototype distance check (nearest-neighbour prototypes):")
print(f"  {'stay_id':<12}  {'prob':>6}  {'‖Δpos‖':>8}  {'‖Δneg‖':>8}  {'expected':>14}  result")
print(f"  {'-'*70}")

n_pass = 0
for label, indices in [("long", top3_long), ("short", top3_short)]:
    for i in indices:
        prob     = float(torch.sigmoid(torch.tensor(logits_np[i])))
        norm_pos = float(np.linalg.norm(delta_pos_np[i]))
        norm_neg = float(np.linalg.norm(delta_neg_np[i]))
        true_str = "long  (1)" if test_labels[i] == 1 else "short (0)"

        if label == "long":
            passed   = norm_pos < norm_neg
            expected = "‖Δpos‖ < ‖Δneg‖"
        else:
            passed   = norm_neg < norm_pos
            expected = "‖Δneg‖ < ‖Δpos‖"

        result = "PASS ✓" if passed else "FAIL ✗"
        if passed:
            n_pass += 1

        print(f"  {int(test_ids[i]):<12}  {prob:>6.3f}  {norm_pos:>8.4f}  {norm_neg:>8.4f}  "
              f"{expected:>14}  [{true_str}]  {result}")

print(f"\n  {n_pass}/6 passed")
if n_pass < 5:
    print("  WARNING: fewer than 5/6 passed — nearest-neighbour prototypes are not "
          "separating as expected.")
else:
    print("  OK — prototypes are separating as expected.")
