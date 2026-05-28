"""
PRD-Net — Step 6: K-Nearest-Neighbour Inference
=================================================
Runs the trained PRD-Net on the test set using the mean of the K=20 nearest
training patients per class as prototypes — consistent with how prototypes are
built during training and validation.

Why K=20 mean instead of K=1 single patient?
  Training uses the mean of K=20 peer embeddings as the prototype signal.
  Using K=1 at test time gives the model a structurally different input
  (single spiky embedding vs smooth average), which degrades performance.
  K=20 mean keeps the prototype distribution the same as at training time.

Prototype strategy:
  For each test patient:
    1. Find the K=20 nearest positive training patients by L2 distance.
       → pos_proto = mean of their embeddings
    2. Same for negative training patients.
       → neg_proto = mean of their embeddings
  Prototypes are encoded via model.encode() before the delta is computed.

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)

sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from prd_net.config_prd import EMBEDDING_CACHE_PATH, HIDDEN_DIM, K_PEERS

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
pos_train_ids = train_ids[train_labels == 1]
neg_train_ids = train_ids[train_labels == 0]

# Stack into (N_pos, 128) and (N_neg, 128) matrices for fast L2 search
pos_train_emb = np.stack([emb_dict[int(sid)] for sid in pos_train_ids])  # (N_pos, 128)
neg_train_emb = np.stack([emb_dict[int(sid)] for sid in neg_train_ids])  # (N_neg, 128)

print(f"  Positive training patients : {len(pos_train_ids):,}")
print(f"  Negative training patients : {len(neg_train_ids):,}")

# ── Load test set ─────────────────────────────────────────────────────────────
print("Loading test data...")
X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
y_test = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")

test_ids    = X_test["stay_id"].values
test_labels = y_test.set_index("stay_id").loc[test_ids, "los_gt7"].values
N_test = len(test_ids)

test_emb_np = np.stack([emb_dict[int(sid)] for sid in test_ids])  # (N_test, 128)
print(f"  Test patients : {N_test:,}")


# ── Build K=20 mean prototypes per test patient ───────────────────────────────
print(f"Building K={K_PEERS} mean prototypes per test patient...")

test_emb_t    = torch.tensor(test_emb_np,    dtype=torch.float32)
pos_train_t   = torch.tensor(pos_train_emb,  dtype=torch.float32)
neg_train_t   = torch.tensor(neg_train_emb,  dtype=torch.float32)

k_pos = min(K_PEERS, pos_train_t.shape[0])
k_neg = min(K_PEERS, neg_train_t.shape[0])

with torch.no_grad():
    pos_top_k = torch.cdist(test_emb_t, pos_train_t).topk(k_pos, dim=1, largest=False).indices
    neg_top_k = torch.cdist(test_emb_t, neg_train_t).topk(k_neg, dim=1, largest=False).indices

pos_proto_raw = pos_train_t[pos_top_k].mean(dim=1)  # (N_test, 128)
neg_proto_raw = neg_train_t[neg_top_k].mean(dim=1)  # (N_test, 128)

# Keep the single nearest ID per patient for the top-3 display
nearest_pos_ids = [int(pos_train_ids[pos_top_k[i, 0]]) for i in range(N_test)]
nearest_neg_ids = [int(neg_train_ids[neg_top_k[i, 0]]) for i in range(N_test)]

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
