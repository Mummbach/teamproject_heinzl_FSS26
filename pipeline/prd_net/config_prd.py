from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

# ── Peer-group parameters ─────────────────────────────────────────────────────
K_PEERS       = 20
AGE_TOLERANCE = 10
GCS_TOLERANCE = 3   # soft filter: GCS total at admission within ±3 points

# ── Model hyperparameters ─────────────────────────────────────────────────────
HIDDEN_DIM = 64
BATCH_SIZE = 64
LR         = 1e-3
EPOCHS     = 30
PATIENCE   = 10   # early stopping: stop if val loss doesn't improve for this many epochs

# ── Cache paths ───────────────────────────────────────────────────────────────
EMBEDDING_CACHE_PATH = OUTPUT_DIR / "prd_net_embeddings.pkl"
PEER_CACHE_PATH      = OUTPUT_DIR / "prd_net_peers.pkl"
