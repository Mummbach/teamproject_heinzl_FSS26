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
import torch

sys.path.append(str(Path(__file__).parent.parent))
from prd_net.config_prd import EMBEDDING_CACHE_PATH, PEER_CACHE_PATH


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
# STEP 2 — Compute peer prototypes
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
