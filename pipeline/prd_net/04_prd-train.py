"""
PRD-Net — Step 4: Training Loop
=================================
Trains PRDNet using the pre-built peer cache as supervision signal.

For each training patient the model receives:
  - the patient's own embedding (x)
  - the mean embedding of its positive peers (pos_proto)
  - the mean embedding of its negative peers (neg_proto)

The logit output is trained with BCEWithLogitsLoss:
  label = 1 if the patient is a long-stay case, 0 otherwise.

Steps in this file:
  1. load_caches()         — load embedding and peer caches from disk
  2. PRDDataset            — Dataset that builds (x, pos_proto, neg_proto, label)
                             per patient using the two caches  [TODO]
  3. train()               — training loop with validation                [TODO]
  4. __main__              — wire everything together                      [TODO]

Run AFTER: 01_extract-embeddings.py  (prd_net_embeddings.pkl)
           02_peer-groups.py         (prd_net_peers.pkl)
Output:    output/best_prd_model.pt
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from prd_net.config_prd import EMBEDDING_CACHE_PATH, PEER_CACHE_PATH, BATCH_SIZE, HIDDEN_DIM, LR, EPOCHS
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("prd_model", Path(__file__).parent / "03_prd-model.py")
_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
PRDNet = _mod.PRDNet


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load caches
# ══════════════════════════════════════════════════════════════════════════════

def load_caches() -> tuple[dict, dict]:
    """
    Load the pre-built embedding and peer caches from disk.

    Returns:
        embeddings : {stay_id (int) -> np.array of shape (128,)}
                     One vector per patient covering all splits.
        peers      : {stay_id (int) -> (pos_list, neg_list)}
                     Peer indices into the training set for each training patient.
    """
    with open(EMBEDDING_CACHE_PATH, "rb") as f:
        embeddings = pickle.load(f)

    with open(PEER_CACHE_PATH, "rb") as f:
        peers = pickle.load(f)

    return embeddings, peers


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Dataset and DataLoader
# ══════════════════════════════════════════════════════════════════════════════

class PRDDataset(Dataset):
    """
    Lightweight dataset for PRD-Net training.
    Works directly from the embedding cache — no raw time-series needed.

    Returns per patient: (embedding tensor, label, stay_id)
    stay_id is needed at training time to look up peers in compute_prototypes.
    """

    def __init__(self, stay_ids: np.ndarray, embedding_cache: dict,
                 labels: np.ndarray):
        self.stay_ids = stay_ids
        self.embeddings = np.stack([embedding_cache[int(sid)] for sid in stay_ids])
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],     dtype=torch.float32),
            int(self.stay_ids[idx]),   # stay_id passed through for peer lookup
        )


def make_dataloader(stay_ids: np.ndarray, embedding_cache: dict,
                    labels: np.ndarray, shuffle: bool = True) -> DataLoader:
    """Build a DataLoader from the embedding cache for a given set of patients."""
    dataset = PRDDataset(stay_ids, embedding_cache, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Compute peer prototypes
# ══════════════════════════════════════════════════════════════════════════════

def compute_prototypes(
    batch_patient_ids: list[int],
    peer_cache: dict,
    embedding_cache: dict,
    train_stay_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each patient in the batch, average the embeddings of their positive
    and negative peers to produce two prototype vectors.

    The peer cache stores row indices (into the training DataFrame). We convert
    those to stay_ids via train_stay_ids before looking up embeddings.

    Args:
        batch_patient_ids : list of stay_ids for the current batch
        peer_cache        : {stay_id -> (pos_row_indices, neg_row_indices)}
        embedding_cache   : {stay_id -> np.array of shape (embedding_dim,)}
        train_stay_ids    : array mapping row index → stay_id for training set

    Returns:
        pos_proto : (batch, embedding_dim) tensor — mean of positive peer embeddings
        neg_proto : (batch, embedding_dim) tensor — mean of negative peer embeddings
        Both are detached from the computation graph (fixed supervision signal).
    """
    pos_protos, neg_protos = [], []

    for sid in batch_patient_ids:
        pos_idxs, neg_idxs = peer_cache[sid]

        # Convert row indices → stay_ids → embeddings, then average
        pos_emb = np.mean([embedding_cache[int(train_stay_ids[i])] for i in pos_idxs], axis=0)
        neg_emb = np.mean([embedding_cache[int(train_stay_ids[i])] for i in neg_idxs], axis=0)

        pos_protos.append(pos_emb)
        neg_protos.append(neg_emb)

    pos_proto = torch.tensor(np.stack(pos_protos), dtype=torch.float32).detach()
    neg_proto = torch.tensor(np.stack(neg_protos), dtype=torch.float32).detach()

    return pos_proto, neg_proto


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Single training step
# ══════════════════════════════════════════════════════════════════════════════

def train_step(model, batch, peer_cache, embedding_cache,
               train_stay_ids, optimizer, loss_fn) -> float:
    """
    Run one gradient update on a single batch.

    Prototypes (128-dim raw embeddings) are encoded through model.encode()
    before the delta is computed — this projects them to hidden_dim (64)
    so they match the encoded patient h inside forward().
    Prototype encoding is wrapped in torch.no_grad() since prototypes are
    fixed supervision and should not accumulate gradients.

    Args:
        model          : PRDNet instance in train mode
        batch          : (x, y, patient_ids) from PRDDataset
        peer_cache     : {stay_id -> (pos_idxs, neg_idxs)}
        embedding_cache: {stay_id -> np.array}
        train_stay_ids : maps row index → stay_id for the training set
        optimizer      : torch optimiser
        loss_fn        : BCEWithLogitsLoss instance

    Returns:
        loss value as a plain float
    """
    x, y, patient_ids = batch

    # Build prototype tensors (128-dim raw embeddings, detached)
    # DataLoader returns patient_ids as a tensor — convert to plain Python ints
    pos_proto_raw, neg_proto_raw = compute_prototypes(
        patient_ids.tolist(), peer_cache, embedding_cache, train_stay_ids
    )

    # Encode prototypes to hidden_dim so shapes match h inside forward()
    with torch.no_grad():
        pos_proto = model.encode(pos_proto_raw)  # (batch, hidden_dim)
        neg_proto = model.encode(neg_proto_raw)  # (batch, hidden_dim)

    optimizer.zero_grad()

    logit, _, _ = model(x, pos_proto, neg_proto)  # (batch,)
    loss = loss_fn(logit, y)

    loss.backward()
    optimizer.step()

    return loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train one epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, peer_cache, embedding_cache,
                train_stay_ids, optimizer, loss_fn) -> float:
    """
    Run train_step for every batch in the dataloader.
    Shows a tqdm progress bar and returns the mean loss over the epoch.

    Args:
        model, dataloader, peer_cache, embedding_cache,
        train_stay_ids, optimizer, loss_fn  — passed straight to train_step

    Returns:
        mean loss over all batches as a float
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", unit="batch", leave=False):
        total_loss += train_step(
            model, batch, peer_cache, embedding_cache,
            train_stay_ids, optimizer, loss_fn
        )

    return total_loss / len(dataloader)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load caches ───────────────────────────────────────────────────────────
    print("Loading caches...")
    embedding_cache, peer_cache = load_caches()
    print(f"  Embeddings : {len(embedding_cache):,} patients")
    print(f"  Peer cache : {len(peer_cache):,} patients")

    # ── Load training labels and stay_id order ────────────────────────────────
    print("Loading training data...")
    X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
    y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")

    # all_train_stay_ids: full row-index → stay_id lookup used by compute_prototypes
    # Must stay unfiltered — peer row indices reference positions in this array
    all_train_stay_ids = X_train["stay_id"].values
    labels_all = y_train.set_index("stay_id").loc[all_train_stay_ids, "los_gt7"].values

    # Filter out the 6 patients with 0 peers on either side — they crash compute_prototypes
    valid_mask = np.array([
        sid in peer_cache and len(peer_cache[sid][0]) > 0 and len(peer_cache[sid][1]) > 0
        for sid in all_train_stay_ids
    ])
    train_stay_ids = all_train_stay_ids[valid_mask]
    labels         = labels_all[valid_mask]
    print(f"  Training patients (after peer filter) : {len(train_stay_ids):,}")

    # ── Build dataloader ──────────────────────────────────────────────────────
    dataloader = make_dataloader(train_stay_ids, embedding_cache, labels, shuffle=True)

    # ── Model, optimizer, loss ────────────────────────────────────────────────
    INPUT_DIM = next(iter(embedding_cache.values())).shape[0]  # 128
    model     = PRDNet(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.BCEWithLogitsLoss()

    print(f"\nPRDNet  |  input={INPUT_DIM}  hidden={HIDDEN_DIM}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Training  |  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, dataloader, peer_cache, embedding_cache,
                           all_train_stay_ids, optimizer, loss_fn)
        print(f"Epoch {epoch:>3}/{EPOCHS}  |  loss = {loss:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "prd_net_v1.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved: {ckpt_path}")
