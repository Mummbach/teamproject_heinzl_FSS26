"""
PRD-Net v2 (feature-level contrastive difference track) — central config
========================================================================
Imported by all fd* files. Mirrors pipeline/prd_net/config_prd.py conventions
but adds the switches specific to the feature-difference track.

This track is a *parallel* sub-track to pipeline/prd_net/. By default (D1 =
"feature") it finds its own peers directly in interpretable FEATURE space and
never reads pipeline/prd_net/'s output caches at all. It can optionally reuse
v1's embedding-space peer retrieval (D1 = "embedding") for comparison, but
nothing in pipeline/prd_net/ (01-05) is ever touched or modified by this track.
"""

from pathlib import Path
import sys

# Reach pipeline/config.py (and reuse ICD_CATEGORIES so we never re-list them)
sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, ICD_CATEGORIES

# ── Observation window ────────────────────────────────────────────────────────
# Single switch driving the whole fd01->fd05 chain. 48 is the primary analysis;
# set to 24 for the robustness / comparability run (Section 6 of the brief).
WINDOW_HOURS = 24

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

# ── Optional CXR-derived static features ──────────────────────────────────────
# Interpretable per-report flags from preprocessing/01d_extract_radiology_features.py
# (pathology/severity/progression/device mentions), keyed by stay_id. Unlike
# icd_*/icu_*/adm_* these are NOT part of the hard filter, so a peer-group
# delta carries real signal (e.g. "pneumonia mentioned, vs. 15% of long-stay
# peers"). has_cxr_report disambiguates "no report" from "report present,
# nothing found" — both default to 0 for missing stays.
#
# report_length/sentence_count deliberately excluded: they measure report
# verbosity/documentation style, not a clinical finding, yet outranked real
# pathology flags (severity_score, abnormality_count) in feature importance —
# a confound the "why" explanation view should not surface to a clinician.
USE_CXR_FEATURES = True
CXR_FEATURE_PATH = OUTPUT_DIR / "cxr_structured_features.csv"
CXR_FEATURES = [
    "has_cxr_report",
    "pneumonia", "pleural_effusion", "pneumothorax", "edema",
    "atelectasis", "opacity", "cardiomegaly",
    "severity_score", "worsening", "improved", "stable",
    "ventilator", "central_line", "chest_tube",
    "abnormality_count",
]

# ── Peer retrieval (consistent with the existing track) ───────────────────────
K_PEERS       = 20
AGE_TOLERANCE = 5

# ── Design-decision switches ──────────────────────────────────────────────────
# DESIGN DECISION D1 — retrieval space: "feature" (default) re-retrieves the
# K=20 nearest peers directly in this track's own scaled feature space (hard
# filter + age tolerance, same as before) — no dependency on v1 at all. Tested
# 2026-08-05 against "embedding" (reuse prd_net_peers.pkl / v1's GRU-embedding
# K-NN) on the 24h window: feature space matched or slightly beat embedding
# space on every headline metric (AUROC 0.801 vs 0.791, AUPRC 0.556 vs 0.532,
# torch model) while leaving 0 training patients with an empty peer side
# (vs. some under the cached-peer path). "embedding" is kept only as a
# same-peers-as-v1 comparison point, not because it performs better.
# "embedding" was this track's original default (isolate the prototype/model
# change against v1's exact peer set before also varying retrieval); it's now
# an interim/ablation finding, not a maintained parallel mode — see README
# "Retrieval space" for the full argument. Artifacts are only window-tagged,
# so switching back to "embedding" and re-running fd02/fd04/fd06 overwrites
# this window's "feature"-mode prototypes/checkpoint/metrics on purpose.
RETRIEVAL_SPACE = "feature"          # "embedding" | "feature"

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
# Reference ratio from the development cohort (neg=16560, pos=5130).
# fd04_diff-train.py recomputes pos_weight from the actual training labels at
# runtime, so this constant is never used directly — it documents the design intent.
POS_WEIGHT = 16560 / 5130
BATCH_SIZE = 64
LR         = 1e-3
EPOCHS     = 50
PATIENCE   = 15

# L2 penalty (Adam weight_decay) on the linear diff model. 0.0 = current
# default (unregularized). Several diff features are highly collinear (e.g.
# heart_rate_mean vs. heart_rate_median, r~0.99 in train) — with no penalty,
# the model is free to split weight between them arbitrarily (large offsetting
# +/- weights that cancel in the prediction but distort per-feature
# attribution on the explanation dashboard). A small positive value nudges
# the model toward smaller, more evenly-shared weights among correlated
# features without materially changing predictive accuracy. Experimental —
# compare against the wd=0 checkpoint (see checkpoint_path/metrics_path
# weight_decay tagging below) before changing this default.
WEIGHT_DECAY = 0.0

# ── Hard-filter one-hot column groups (centralized — single source of truth) ──
ICU_COLS = [
    "icu_micu", "icu_sicu", "icu_ccu", "icu_cvicu",
    "icu_micu_sicu", "icu_tsicu", "icu_neuro_sicu",
]
ICD_COLS = [f"icd_{cat}" for cat in ICD_CATEGORIES]
ADM_COLS = ["adm_emergency", "adm_urgent", "adm_elective", "adm_observation"]

# Human-readable labels for the above, kept next to the column lists so they
# can't silently drift out of sync (dashboard use: describing peer-group
# filter criteria — coarser than the raw cohort text, since only these
# buckets are recognized by the hard filter).
ICU_LABELS = {"icu_micu": "MICU", "icu_sicu": "SICU", "icu_ccu": "CCU", "icu_cvicu": "CVICU",
              "icu_micu_sicu": "MICU/SICU", "icu_tsicu": "TSICU", "icu_neuro_sicu": "Neuro SICU"}
ADM_LABELS = {"adm_emergency": "Emergency", "adm_urgent": "Urgent",
              "adm_elective": "Elective", "adm_observation": "Observation"}
ICD_LABELS = {c: c[len("icd_"):].replace("_", " ").title() for c in ICD_COLS}

# ── Absolute (non-differenced) hard-filter features ────────────────────────────
# ICU_COLS/ICD_COLS/ADM_COLS above are used to select peers (see fd02), so a
# DIFFERENCE against a matched peer group is ~0 by construction (see the
# STATIC_FEATURES note) — that's why they were originally left out of
# feature_names() entirely. But the patient's own category still carries real
# baseline-risk signal the diff features never see (e.g. a CVICU/circulatory
# stay has a different typical LOS than a MICU/general-medicine stay). Appended
# to the model input UNCHANGED (never diffed against a prototype) via
# assemble_diff(..., absolute=...) in fd03.
USE_ABSOLUTE_FEATURES = True
ABSOLUTE_FEATURES = ICU_COLS + ICD_COLS + ADM_COLS


def absolute_feature_names() -> list[str]:
    """Ordered list of the patient's-own-value (non-diff) input features."""
    return list(ABSOLUTE_FEATURES) if USE_ABSOLUTE_FEATURES else []


# ── v1 caches, only read when RETRIEVAL_SPACE == "embedding" (see D1) ─────────
EMBEDDING_CACHE_PATH = OUTPUT_DIR / "prd_net_embeddings.pkl"
PEER_CACHE_PATH      = OUTPUT_DIR / "prd_net_peers.pkl"

# ── This track's own directories ──────────────────────────────────────────────
FD_DIR     = Path(__file__).parent
CKPT_DIR   = FD_DIR / "checkpoints"
EXPORT_DIR = FD_DIR / "exports"


# ── Resolved difference-feature list ──────────────────────────────────────────
def feature_names() -> list[str]:
    """Ordered list of the F difference features: TS feature x stat, then statics,
    then (optionally) the CXR-derived features.

    Order is (TS_FEATURES outer, AGG_STATS inner), STATIC_FEATURES, CXR_FEATURES,
    e.g. ['heart_rate_mean', ..., 'urine_output_slope', 'age', 'has_cxr_report', ...].
    """
    names = [f"{feat}_{stat}" for feat in TS_FEATURES for stat in AGG_STATS]
    names += list(STATIC_FEATURES)
    if USE_CXR_FEATURES:
        names += list(CXR_FEATURES)
    return names


def n_features() -> int:
    """F = number of difference features (12 TS x 5 stats + age = 61,
    + 16 CXR-derived features if USE_CXR_FEATURES = 77)."""
    return len(feature_names())


# ── Window-tagged output paths (48h and 24h artifacts coexist) ────────────────
def _tag(window: int | None) -> str:
    return f"{window or WINDOW_HOURS}h"


def _wd_suffix(weight_decay: float | None) -> str:
    """Empty at the wd=0.0 default (keeps existing filenames unchanged);
    "_wdX" otherwise, so a regularization experiment can't clobber the
    current checkpoint/metrics."""
    wd = WEIGHT_DECAY if weight_decay is None else weight_decay
    return "" if wd == 0.0 else f"_wd{wd:g}"


def feature_matrix_path(split: str, scaled: bool, window: int | None = None) -> Path:
    kind = "scaled" if scaled else "raw"
    return OUTPUT_DIR / f"fd_feature_matrix_{split}_{kind}_{_tag(window)}.parquet"


def scaler_bundle_path(window: int | None = None) -> Path:
    """Fitted StandardScaler + train medians + feature names (pickle)."""
    return OUTPUT_DIR / f"fd_scaler_{_tag(window)}.pkl"


def prototypes_path(split: str, window: int | None = None) -> Path:
    """Per-split prototype bundle (pickle): X, protos, peer ids, labels."""
    return OUTPUT_DIR / f"fd_prototypes_{split}_{_tag(window)}.pkl"


def checkpoint_path(window: int | None = None, weight_decay: float | None = None) -> Path:
    return CKPT_DIR / f"fd_diff_v1_{_tag(window)}{_wd_suffix(weight_decay)}.pt"


def threshold_path(window: int | None = None, weight_decay: float | None = None) -> Path:
    return CKPT_DIR / f"fd_diff_v1_{_tag(window)}{_wd_suffix(weight_decay)}_threshold.pt"


def metrics_path(window: int | None = None, weight_decay: float | None = None) -> Path:
    return OUTPUT_DIR / f"fd_metrics_{_tag(window)}{_wd_suffix(weight_decay)}.json"


def export_path(window: int | None = None) -> Path:
    return EXPORT_DIR / f"fd_explanations_test_{_tag(window)}"  # .parquet/.json added by fd06
