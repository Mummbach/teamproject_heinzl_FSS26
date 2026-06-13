"""
PRD-Net v2 (feature-level contrastive difference track) — central config
========================================================================
Imported by all fd* files. Mirrors pipeline/prd_net/config_prd.py conventions
but adds the switches specific to the feature-difference track.

This track is a *parallel* sub-track to pipeline/prd_net/. It reuses the same
peer retrieval (prd_net_embeddings.pkl / prd_net_peers.pkl) but builds the
prototypes and deltas in interpretable FEATURE space rather than embedding space.
Nothing in pipeline/prd_net/ (01-05) is touched.
"""

from pathlib import Path
import sys

# Reach pipeline/config.py (and reuse ICD_CATEGORIES so we never re-list them)
sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, ICD_CATEGORIES

# ── Observation window ────────────────────────────────────────────────────────
# Single switch driving the whole fd01->fd05 chain. 48 is the primary analysis;
# set to 24 for the robustness / comparability run (Section 6 of the brief).
WINDOW_HOURS = 48

# ── Aggregation (D5) ──────────────────────────────────────────────────────────
# DESIGN DECISION D5 — aggregation granularity: summary stats per time-series
# feature (default). Full per-hour deltas (48 x F) are a possible later extension.
AGG_STATS = ["mean", "last", "min", "max", "slope"]

# Time-series features as stored in timeseries.parquet (order is fixed; do not
# reorder — feature_names() depends on it).
TS_FEATURES = [
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
    "glucose", "gcs_eye", "gcs_verbal", "gcs_motor", "urine_output",
]

# Continuous static features included as difference features. The hard-filter
# one-hots (icd_*, icu_*, adm_*) and the other binary indicators are excluded:
# peers share the hard-filter groups by construction (deltas ~= 0) and binary
# indicators have no "X SD above/below the prototype" reading.
STATIC_FEATURES = ["age"]

# ── Peer retrieval (consistent with the existing track) ───────────────────────
K_PEERS       = 20
AGE_TOLERANCE = 5

# ── Design-decision switches ──────────────────────────────────────────────────
# DESIGN DECISION D1 — retrieval space: reuse the embedding-space peers from
# prd_net_peers.pkl (default). "feature" re-retrieves in scaled feature space
# (a fully-interpretable-pipeline ablation).
RETRIEVAL_SPACE = "embedding"          # "embedding" | "feature"

# DESIGN DECISION D2 — prototype aggregation: simple mean of peer feature
# vectors (default, reads as "the average peer"). Distance-weighted (embedding
# distance, 1/(d+eps)) matches the existing track when set True.
USE_PROTOTYPE_WEIGHTING = False

# DESIGN DECISION D3 — diff input fed to the model.
DIFF_INPUT = "both"                    # both | pos_only | neg_only | proto_gap

# DESIGN DECISION D4 — model: pure linear (default, keeps exact attribution).
# A shallow MLP is available as an ablation, only justified if it buys AUPRC.
MODEL = "linear"                       # linear | mlp
MLP_HIDDEN = 32

# ── Training (mirrors prd_net/04_prd-train.py) ────────────────────────────────
# pos_weight = neg/pos ratio (~3.23) to counteract the 23.7% positive imbalance.
POS_WEIGHT = 16560 / 5130
BATCH_SIZE = 64
LR         = 1e-3
EPOCHS     = 50
PATIENCE   = 15

# ── Hard-filter one-hot column groups (centralized — single source of truth) ──
ICU_COLS = [
    "icu_micu", "icu_sicu", "icu_ccu", "icu_cvicu",
    "icu_micu_sicu", "icu_tsicu", "icu_neuro_sicu",
]
ICD_COLS = [f"icd_{cat}" for cat in ICD_CATEGORIES]
ADM_COLS = ["adm_emergency", "adm_urgent", "adm_elective", "adm_observation"]

# ── Reused caches from the existing track (peer retrieval only) ───────────────
EMBEDDING_CACHE_PATH = OUTPUT_DIR / "prd_net_embeddings.pkl"
PEER_CACHE_PATH      = OUTPUT_DIR / "prd_net_peers.pkl"

# ── This track's own directories ──────────────────────────────────────────────
FD_DIR     = Path(__file__).parent
CKPT_DIR   = FD_DIR / "checkpoints"
EXPORT_DIR = FD_DIR / "exports"


# ── Resolved difference-feature list ──────────────────────────────────────────
def feature_names() -> list[str]:
    """Ordered list of the F difference features: TS feature x stat, then statics.

    Order is (TS_FEATURES outer, AGG_STATS inner) followed by STATIC_FEATURES,
    e.g. ['heart_rate_mean', 'heart_rate_last', ..., 'urine_output_slope', 'age'].
    """
    names = [f"{feat}_{stat}" for feat in TS_FEATURES for stat in AGG_STATS]
    names += list(STATIC_FEATURES)
    return names


def n_features() -> int:
    """F = number of difference features (12 TS x 5 stats + age = 61)."""
    return len(feature_names())


# ── Window-tagged output paths (48h and 24h artifacts coexist) ────────────────
def _tag(window: int | None) -> str:
    return f"{window or WINDOW_HOURS}h"


def feature_matrix_path(split: str, scaled: bool, window: int | None = None) -> Path:
    kind = "scaled" if scaled else "raw"
    return OUTPUT_DIR / f"fd_feature_matrix_{split}_{kind}_{_tag(window)}.parquet"


def scaler_bundle_path(window: int | None = None) -> Path:
    """Fitted StandardScaler + train medians + feature names (pickle)."""
    return OUTPUT_DIR / f"fd_scaler_{_tag(window)}.pkl"


def prototypes_path(split: str, window: int | None = None) -> Path:
    """Per-split prototype bundle (pickle): X, protos, peer ids, labels."""
    return OUTPUT_DIR / f"fd_prototypes_{split}_{_tag(window)}.pkl"


def checkpoint_path(window: int | None = None) -> Path:
    return CKPT_DIR / f"fd_diff_v1_{_tag(window)}.pt"


def threshold_path(window: int | None = None) -> Path:
    return CKPT_DIR / f"fd_diff_v1_{_tag(window)}_threshold.pt"


def metrics_path(window: int | None = None) -> Path:
    return OUTPUT_DIR / f"fd_metrics_{_tag(window)}.json"


def export_path(window: int | None = None) -> Path:
    return EXPORT_DIR / f"fd_explanations_test_{_tag(window)}"  # .parquet/.json added by fd06
