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
