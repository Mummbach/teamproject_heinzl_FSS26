"""
PRD-Net — Step 1: Extract GRU Embeddings
=========================================
Purpose:
  The trained GRU model (07_model_gru.py) learned a rich internal representation
  of each patient. Before the classifier head, it produces a fusion vector that
  combines the GRU's final hidden state (time-series) and the static branch output.
  This script extracts that fusion vector for every patient and caches it to disk.
  Those embeddings are the input to the peer-retrieval network (Steps 2–4).

Why not import GRUModel directly from 07_model_gru.py?
  That file runs training at module level, so importing it would re-run the full
  training loop. Instead we copy only the GRUModel class here. The architecture
  must match exactly so the checkpoint weights load without errors.

Steps in this file:
  1. load_trained_gru(checkpoint_path)
       — Instantiate GRUModel, load checkpoint weights, set eval mode.
         Architecture dimensions are inferred from the weight shapes in the
         checkpoint itself, so this stays robust if hyper-params ever change.
  2. extract_embeddings(model, dataloader)   [TODO: implement in Step 2]
       — Hook the fusion layer to capture the pre-classifier vector per patient.
  3. main block
       — Load data, build dataloader, call steps 1–2, save to EMBEDDING_CACHE_PATH.

Run AFTER: 07_model_gru.py has produced output/best_gru_model.pt
Output:    output/prd_net_embeddings.pkl  (dict: stay_id → embedding np.array)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add pipeline/ to path so we can reach config.py and config_prd.py
sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from prd_net.config_prd import EMBEDDING_CACHE_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = OUTPUT_DIR / "best_gru_model.pt"


# ══════════════════════════════════════════════════════════════════════════════
# GRU MODEL — architecture copy from 07_model_gru.py
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
        checkpoint_path: path to the .pt file produced by 07_model_gru.py

    Returns:
        GRUModel in eval mode with weights restored, on CPU.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Infer dimensions from stored weight shapes
    # gru.weight_ih_l0 shape: (3 * hidden_size, ts_input_size)  — 3 GRU gates
    ts_input_size     = state["gru.weight_ih_l0"].shape[1]
    hidden_size       = state["gru.weight_ih_l0"].shape[0] // 3
    num_layers        = sum(1 for k in state if k.startswith("gru.weight_ih_l"))
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
    )
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Extract embeddings  [TODO: implement next]
# ══════════════════════════════════════════════════════════════════════════════

# def extract_embeddings(model, dataloader): ...


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Step 1 — Loading trained GRU model...")
    model = load_trained_gru(CHECKPOINT_PATH)
    print(f"  Loaded from : {CHECKPOINT_PATH}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {total_params:,}")
    print("  Model ready for embedding extraction.")

    print(f"  Output will be saved to: {EMBEDDING_CACHE_PATH}")
    # Step 2 (data loading + extraction) will be added here.
