"""
PRD-Net — Step 1 & 2: Extract GRU Embeddings
=============================================
Purpose:
  The trained GRU model (baseline/07_model_gru.py) learned a rich internal representation
  of each patient. Before the classifier head, it produces a fusion vector that
  combines the GRU's final hidden state (time-series) and the static branch output.
  This script extracts that fusion vector for every patient and caches it to disk.
  Those embeddings are the input to the peer-retrieval network (Steps 3–4).

Why not import GRUModel directly from baseline/07_model_gru.py?
  That file runs training at module level, so importing it would re-run the full
  training loop. Instead we copy only GRUModel and ICUDataset here. The architecture
  must match exactly so the checkpoint weights load without errors.

Steps in this file:
  1. load_trained_gru(checkpoint_path)
       — Instantiate GRUModel, load checkpoint weights, set eval mode.
         Architecture dimensions are inferred from the weight shapes in the
         checkpoint itself, so this stays robust if hyper-params ever change.
  2. extract_embeddings(model, dataloader, stay_ids)
       — Register a forward hook on the classifier head to capture the fusion
         vector (GRU hidden state + static branch) for every patient.
         Returns {stay_id: embedding_array}.
  3. main block
       — Load all splits, build one combined dataloader, call steps 1–2,
         save result to EMBEDDING_CACHE_PATH.

Run AFTER: baseline/07_model_gru.py has produced output/best_gru_model.pt
Output:    output/prd_net_embeddings.pkl  (dict: stay_id → np.array)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add pipeline/ to path so we can reach config.py and config_prd.py
sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from prd_net.config_prd import EMBEDDING_CACHE_PATH

CHECKPOINT_PATH = OUTPUT_DIR / "best_gru_model.pt"
MAX_TS_HOURS = 48


# ══════════════════════════════════════════════════════════════════════════════
# GRU MODEL — architecture copy from baseline/07_model_gru.py
# Must stay in sync with the original so state_dict loads without key errors.
# ══════════════════════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    """GRU encoder + static branch, fused before a binary classifier head."""

    def __init__(self, ts_input_size: int, static_input_size: int,
                 hidden_size: int, num_layers: int,
                 static_dim: int, dropout: float,
                 use_gru: bool = True):
        super().__init__()
        self.use_gru = use_gru

        if use_gru:
            self.gru = nn.GRU(
                input_size=ts_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.static_branch = nn.Sequential(
            nn.Linear(static_input_size, static_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fusion_input = (hidden_size if use_gru else 0) + static_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_input, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, ts, static):
        static_out = self.static_branch(static)
        if self.use_gru:
            _, h_n  = self.gru(ts)
            gru_out = h_n[-1]
            fused   = torch.cat([gru_out, static_out], dim=1)
        else:
            fused = static_out
        return self.classifier(fused).squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load trained GRU
# ══════════════════════════════════════════════════════════════════════════════

def load_trained_gru(checkpoint_path: Path) -> GRUModel:
    """
    Load the trained GRU model from a .pt checkpoint file.

    Architecture dimensions are inferred from the weight tensor shapes stored
    in the checkpoint, so no hardcoded sizes are needed here.

    Args:
        checkpoint_path: path to the .pt file produced by baseline/07_model_gru.py

    Returns:
        GRUModel in eval mode with weights restored, on CPU.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Guard: detect whether the checkpoint contains a GRU branch
    use_gru = "gru.weight_ih_l0" in state

    # Infer dimensions from stored weight shapes
    # gru.weight_ih_l0 shape: (3 * hidden_size, ts_input_size)  — 3 GRU gates
    if use_gru:
        ts_input_size = state["gru.weight_ih_l0"].shape[1]
        hidden_size   = state["gru.weight_ih_l0"].shape[0] // 3
        num_layers    = sum(1 for k in state if k.startswith("gru.weight_ih_l"))
    else:
        ts_input_size, hidden_size, num_layers = 0, 0, 0
    # static_branch.0 is the first Linear: (static_dim, static_input_size)
    static_dim        = state["static_branch.0.weight"].shape[0]
    static_input_size = state["static_branch.0.weight"].shape[1]

    model = GRUModel(
        ts_input_size=ts_input_size,
        static_input_size=static_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        static_dim=static_dim,
        dropout=0.0,  # irrelevant in eval mode; keeps architecture identical
        use_gru=use_gru,
    )
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# DATASET — copied from baseline/07_model_gru.py (same reason as GRUModel above)
# ══════════════════════════════════════════════════════════════════════════════

class ICUDataset(Dataset):
    """One sample per ICU stay: (ts tensor, static tensor, label scalar)."""

    def __init__(self, X_static: pd.DataFrame, y: pd.DataFrame,
                 ts: pd.DataFrame, ts_features: list):
        self.stay_ids    = X_static["stay_id"].values
        self.static_arr  = X_static.drop(columns=["stay_id"]).values.astype(np.float32)
        self.labels      = y.set_index("stay_id").loc[self.stay_ids, "los_gt7"].values.astype(np.float32)
        self.ts_features = ts_features

        # Build (N, 48, n_features) array from long-format timeseries
        ts_pivot = (
            ts[ts["stay_id"].isin(self.stay_ids)]
            .sort_values(["stay_id", "hour"])
            .set_index(["stay_id", "hour"])[ts_features]
            .fillna(0.0)
        )
        self.ts_arr = np.zeros((len(self.stay_ids), MAX_TS_HOURS, len(ts_features)), dtype=np.float32)
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Extract embeddings
# ══════════════════════════════════════════════════════════════════════════════

def extract_embeddings(model: GRUModel, dataloader: DataLoader,
                       stay_ids: np.ndarray) -> dict:
    """
    Capture the fusion vector (pre-classifier) for every patient.

    We attach a forward hook to model.classifier so that each time the
    classifier is called, we intercept its input — that input IS the fused
    vector combining the GRU hidden state and the static branch output.

    Args:
        model      : trained GRUModel in eval mode
        dataloader : yields (ts, static, label) batches for all patients
        stay_ids   : stay_id array in the same order as the dataloader

    Returns:
        dict mapping stay_id (int) → embedding (np.array of shape (embedding_dim,))
    """
    captured = []

    # Hook stores the input to the classifier head (= the fusion vector)
    def _capture(module, inp, out):
        del module, out  # required by PyTorch hook signature, not used here
        captured.append(inp[0].detach().cpu())

    hook = model.classifier.register_forward_hook(_capture)

    with torch.no_grad():
        for ts, static, _ in dataloader:
            model(ts, static)   # forward pass; hook fires and stores the vector

    hook.remove()

    # Stack all batches → (N, embedding_dim)
    all_embeddings = torch.cat(captured, dim=0).numpy()

    return {int(sid): emb for sid, emb in zip(stay_ids, all_embeddings)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Step 1: load model ────────────────────────────────────────────────────
    print("Step 1 — Loading trained GRU model...")
    model = load_trained_gru(CHECKPOINT_PATH)
    print(f"  Loaded from : {CHECKPOINT_PATH}")
    print(f"  Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ── Load data (all three splits combined) ─────────────────────────────────
    print("\nLoading data...")
    ts = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")
    TS_FEATURES = [c for c in ts.columns if c not in ["stay_id", "hour"]]

    # Concatenate train / val / test so we get embeddings for every patient
    X_all = pd.concat([
        pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet"),
        pd.read_parquet(OUTPUT_DIR / "X_val_scaled.parquet"),
        pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet"),
    ], ignore_index=True)

    y_all = pd.concat([
        pd.read_parquet(OUTPUT_DIR / "y_train.parquet"),
        pd.read_parquet(OUTPUT_DIR / "y_val.parquet"),
        pd.read_parquet(OUTPUT_DIR / "y_test.parquet"),
    ], ignore_index=True)

    print(f"  Total patients : {len(X_all):,}")
    print(f"  Static features: {X_all.shape[1] - 1}")
    print(f"  TS features    : {len(TS_FEATURES)}  ({TS_FEATURES})")
    long_stay = y_all["los_gt7"].sum()
    print(f"  Long stay (>7d): {long_stay:,}  ({long_stay/len(y_all)*100:.1f}%)")

    # Build dataset and dataloader (no shuffling — order must match stay_ids)
    print("\nBuilding dataset and dataloader...")
    dataset    = ICUDataset(X_all, y_all, ts, TS_FEATURES)
    print(f"  Dataset size   : {len(dataset):,} samples")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"  Batches        : {len(dataloader):,}  (batch_size=256)")

    # ── Step 2: extract embeddings ────────────────────────────────────────────
    print("\nStep 2 — Extracting embeddings...")
    embeddings = extract_embeddings(model, dataloader, dataset.stay_ids)
    print(f"  Extracted {len(embeddings):,} embeddings  "
          f"(dim = {next(iter(embeddings.values())).shape[0]})")

    # ── Save to disk ──────────────────────────────────────────────────────────
    with open(EMBEDDING_CACHE_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"\nSaved: {EMBEDDING_CACHE_PATH}")
