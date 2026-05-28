"""
PRD-Net — Step 5: Sanity Check
================================
Loads the trained PRD-Net and runs it on the test set to verify the model
produces meaningful predictions.

Prototype strategy:
  For this sanity check we use global prototypes — the mean embedding of
  ALL positive training patients and ALL negative training patients.
  Every test patient gets the same pos_proto and neg_proto.
  This is simpler than rebuilding per-patient peer groups for the test set
  and is sufficient to verify the model learned something useful.

Output:
  - 3 most confident long-stay predictions  (highest logit)
  - 3 most confident short-stay predictions (lowest logit)
"""

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from prd_net.config_prd import EMBEDDING_CACHE_PATH, HIDDEN_DIM

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

# ── Load training data — build global prototypes ──────────────────────────────
print("Building global prototypes from training set...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")

train_ids    = X_train["stay_id"].values
train_labels = y_train.set_index("stay_id").loc[train_ids, "los_gt7"].values

# Mean embedding of all positive / negative training patients
pos_ids    = train_ids[train_labels == 1]
neg_ids    = train_ids[train_labels == 0]
pos_global = np.mean([emb_dict[int(sid)] for sid in pos_ids], axis=0)  # (128,)
neg_global = np.mean([emb_dict[int(sid)] for sid in neg_ids], axis=0)  # (128,)

print(f"  Positive training patients : {len(pos_ids):,}")
print(f"  Negative training patients : {len(neg_ids):,}")

# ── Load test set ─────────────────────────────────────────────────────────────
print("Running predictions on test set...")
X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
y_test = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")

test_ids    = X_test["stay_id"].values
test_labels = y_test.set_index("stay_id").loc[test_ids, "los_gt7"].values

# Stack test embeddings into a (N_test, 128) tensor
test_emb = torch.tensor(
    np.stack([emb_dict[int(sid)] for sid in test_ids]), dtype=torch.float32
)

# Broadcast global prototypes to match batch size
pos_proto_raw = torch.tensor(pos_global, dtype=torch.float32).unsqueeze(0).expand(len(test_ids), -1)
neg_proto_raw = torch.tensor(neg_global, dtype=torch.float32).unsqueeze(0).expand(len(test_ids), -1)

# Encode prototypes to hidden_dim (same as in train_step)
with torch.no_grad():
    pos_proto      = model.encode(pos_proto_raw)       # (N_test, hidden_dim)
    neg_proto      = model.encode(neg_proto_raw)       # (N_test, hidden_dim)
    logits, _, _   = model(test_emb, pos_proto, neg_proto)  # (N_test,)

logits_np = logits.numpy()
print(f"  Predictions computed for {len(logits_np):,} test patients")

# ── Top-3 most confident long-stay (highest logit) ────────────────────────────
top3_long = np.argsort(logits_np)[-3:][::-1]
print("\nTop-3 most confident LONG-STAY predictions (highest logit):")
print(f"  {'stay_id':<12}  {'logit':>8}  {'true_label':>12}")
print(f"  {'-'*38}")
for i in top3_long:
    label_str = "long  (1)" if test_labels[i] == 1 else "short (0)"
    print(f"  {int(test_ids[i]):<12}  {logits_np[i]:>8.4f}  {label_str:>12}")

# ── Top-3 most confident short-stay (lowest logit) ───────────────────────────
top3_short = np.argsort(logits_np)[:3]
print("\nTop-3 most confident SHORT-STAY predictions (lowest logit):")
print(f"  {'stay_id':<12}  {'logit':>8}  {'true_label':>12}")
print(f"  {'-'*38}")
for i in top3_short:
    label_str = "long  (1)" if test_labels[i] == 1 else "short (0)"
    print(f"  {int(test_ids[i]):<12}  {logits_np[i]:>8.4f}  {label_str:>12}")

# ── Quick summary ─────────────────────────────────────────────────────────────
preds   = (logits_np > 0).astype(int)
correct = (preds == test_labels).mean()
print(f"\nSimple accuracy (threshold=0) : {correct*100:.1f}%")
