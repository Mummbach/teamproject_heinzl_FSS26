"""
SHAP Explainability — PRD-Net (Prototype Retrieval Derivation Network)
=======================================================================
Computes SHAP attributions for the PRD-Net using shap.GradientExplainer,
attributing predictions back to the original clinical input features
(time-series vitals + static features) via end-to-end backprop through
the full pipeline: GRU encoder → PRD-Net → logit.

Why end-to-end and not just SHAP on the embedding?
  The embedding is a 128-dim internal vector with no clinical meaning.
  Attributing back to (ts, static) gives interpretable feature importances
  that can be directly compared to the GRU-only SHAP results (shap_global.py).

Architecture of the wrapped pipeline:
  ts (batch, 48, n_ts)  ──┐
                           ├─► GRU encoder ──► fusion embedding (batch, 128)
  static (batch, n_static) ─┘                         │
                                                        ▼
  pos_proto (batch, 64) ──────────────────────► PRD-Net ──► sigmoid ──► prob
  neg_proto (batch, 64) ──────────────────────►

Prototypes (pos_proto, neg_proto) are fixed tensors computed once from
the clinically filtered training set. They are not SHAP inputs — only
(ts, static) are attributed.

Run AFTER:
  08_model_gru.py          → output/best_gru_model.pt
  prd_net/04_prd-train.py  → prd_net/checkpoints/prd_net_v1.pt
  prd_net/02_peer-groups.py → output/prd_net_peers.pkl
  prd_net/01_extract-embeddings.py → output/prd_net_embeddings.pkl

Input:
  pipeline/output/X_train_scaled.parquet
  pipeline/output/X_test_scaled.parquet
  pipeline/output/y_train.parquet / y_test.parquet
  pipeline/output/timeseries.parquet
  pipeline/output/best_gru_model.pt
  pipeline/prd_net/checkpoints/prd_net_v1.pt
  pipeline/output/prd_net_embeddings.pkl
  pipeline/output/X_train.parquet  (unscaled, for prototype filtering)
  pipeline/output/X_val.parquet    (unscaled, for prototype filtering)
  pipeline/output/X_test.parquet   (unscaled, for prototype filtering)

Output:
  explainability/output/shap_prdnet_summary.png
  explainability/output/shap_prdnet_vs_gru.png
  explainability/output/prdnet_explanations.parquet
"""

import sys
import pickle
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

EXPLAIN_OUT = _HERE / "output"
EXPLAIN_OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    raise ImportError("Install shap: pip install shap")

from config import OUTPUT_DIR
from multimodal_utils import (
    ICUDataset, load_multimodal_model, get_cxr_feature_groups,
)

SEED       = 42
BG_SIZE    = 200
BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════
# PRD-NET ARCHITECTURE (must match prd_net/03_prd-model.py exactly)
# ═══════════════════════════════════════════════════════════════════════

class PRDNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, pos_proto, neg_proto):
        h         = self.encode(x)
        delta_pos = h - pos_proto
        delta_neg = h - neg_proto
        combined  = torch.cat([delta_pos, delta_neg], dim=-1)
        logit     = self.head(combined).squeeze(-1)
        return logit, delta_pos, delta_neg


# ═══════════════════════════════════════════════════════════════════════
# END-TO-END SHAP WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class PRDNetSHAPWrapper(nn.Module):
    """
    Wraps GRU encoder + PRD-Net into a single differentiable forward pass
    that takes (ts, static) and returns a probability in [0, 1].

    pos_proto and neg_proto are registered as fixed buffers (not parameters)
    so they are not SHAP inputs but are available during backprop.
    """

    def __init__(self, gru_model, prd_model, pos_proto: torch.Tensor,
                 neg_proto: torch.Tensor):
        super().__init__()
        self.gru = gru_model
        self.prd = prd_model
        # Fixed buffers — not attributed by SHAP
        self.register_buffer("pos_proto", pos_proto)
        self.register_buffer("neg_proto", neg_proto)

    def forward(self, ts, static):
        # GRU produces the fusion embedding via the forward hook path.
        # We use gru_embedding() which returns the GRU hidden state only;
        # to get the full fusion vector we need to also pass through static_branch.
        gru_hidden  = self.gru.gru(ts)[1][-1]              # (batch, hidden_size)
        static_out  = self.gru.static_branch(static)        # (batch, static_dim)
        embedding   = torch.cat([gru_hidden, static_out], dim=1)  # (batch, 128)

        logit, _, _ = self.prd(embedding, self.pos_proto, self.neg_proto)
        return torch.sigmoid(logit).unsqueeze(1)            # (batch, 1)


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

print("Loading data...")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
ts      = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

TS_FEATURES     = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_FEATURES = [c for c in X_train.columns if c != "stay_id"]

groups           = get_cxr_feature_groups(STATIC_FEATURES)
HAS_CXR          = len(groups["cxr_all"]) > 0

print(f"  Time-series features : {len(TS_FEATURES)}")
print(f"  Static features      : {len(STATIC_FEATURES)}")

train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES)
test_ds  = ICUDataset(X_test,  y_test,  ts, TS_FEATURES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


# ═══════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════

print("Loading GRU model...")
gru_model = load_multimodal_model(
    OUTPUT_DIR / "best_gru_model.pt",
    ts_input_size=len(TS_FEATURES),
    static_input_size=len(STATIC_FEATURES),
    device=DEVICE,
)

PRD_CKPT = _HERE.parent / "prd_net" / "checkpoints" / "prd_net_v1.pt"
if not PRD_CKPT.exists():
    raise FileNotFoundError(
        f"PRD-Net checkpoint not found at {PRD_CKPT}\n"
        "Run prd_net/04_prd-train.py first, or get the checkpoint from Johannes."
    )

print("Loading PRD-Net...")
EMBEDDING_DIM = gru_model.gru.hidden_size + gru_model.static_branch[0].out_features
HIDDEN_DIM    = 64

prd_model = PRDNet(input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
prd_model.load_state_dict(torch.load(PRD_CKPT, map_location=DEVICE, weights_only=True))
prd_model.eval()
prd_model.to(DEVICE)
print(f"PRD-Net loaded  (input_dim={EMBEDDING_DIM}, hidden_dim={HIDDEN_DIM})")


# ═══════════════════════════════════════════════════════════════════════
# BUILD POPULATION-LEVEL PROTOTYPES
# ═══════════════════════════════════════════════════════════════════════
# For SHAP we use fixed global prototypes (mean embedding of all
# positive / negative training patients) rather than per-patient filtered
# prototypes. This is consistent across all test patients and makes SHAP
# attributions comparable — per-patient prototypes would change the
# baseline for each sample, making attributions incomparable.

print("Building global population prototypes...")

EMBED_CACHE_PATH = OUTPUT_DIR / "prd_net_embeddings.pkl"
if not EMBED_CACHE_PATH.exists():
    raise FileNotFoundError(
        f"Embedding cache not found at {EMBED_CACHE_PATH}\n"
        "Run prd_net/01_extract-embeddings.py first."
    )

with open(EMBED_CACHE_PATH, "rb") as f:
    embedding_cache = pickle.load(f)

train_sids   = X_train["stay_id"].values
train_labels = y_train.set_index("stay_id").loc[train_sids, "los_gt7"].values

pos_sids = train_sids[train_labels == 1]
neg_sids = train_sids[train_labels == 0]

pos_embs = np.stack([embedding_cache[int(s)] for s in pos_sids if int(s) in embedding_cache])
neg_embs = np.stack([embedding_cache[int(s)] for s in neg_sids if int(s) in embedding_cache])

pos_mean = torch.tensor(pos_embs.mean(axis=0), dtype=torch.float32).to(DEVICE)
neg_mean = torch.tensor(neg_embs.mean(axis=0), dtype=torch.float32).to(DEVICE)

# Encode prototypes to hidden_dim (64) — PRD-Net expects encoded protos
with torch.no_grad():
    pos_proto_enc = prd_model.encode(pos_mean.unsqueeze(0)).squeeze(0)  # (hidden_dim,)
    neg_proto_enc = prd_model.encode(neg_mean.unsqueeze(0)).squeeze(0)

print(f"  Positive prototype from {len(pos_sids):,} long-stay training patients")
print(f"  Negative prototype from {len(neg_sids):,} short-stay training patients")


# ═══════════════════════════════════════════════════════════════════════
# BUILD END-TO-END WRAPPER
# ═══════════════════════════════════════════════════════════════════════

# Keep prototypes at shape (1, hidden_dim) — PRDNet.forward broadcasts
# `h - pos_proto` against any batch size, including GradientExplainer's
# internal interpolation batches (which don't match the DataLoader batch size).
shap_model = PRDNetSHAPWrapper(
    gru_model  = gru_model,
    prd_model  = prd_model,
    pos_proto  = pos_proto_enc.unsqueeze(0).clone(),
    neg_proto  = neg_proto_enc.unsqueeze(0).clone(),
).to(DEVICE)
shap_model.eval()


# ═══════════════════════════════════════════════════════════════════════
# BACKGROUND SET
# ═══════════════════════════════════════════════════════════════════════

print(f"\nBuilding background set (n={min(BG_SIZE, len(train_ds))})...")
rng       = np.random.default_rng(SEED)
bg_idx    = rng.choice(len(train_ds), size=min(BG_SIZE, len(train_ds)), replace=False)
bg_ts     = torch.tensor(train_ds.ts_arr[bg_idx]).to(DEVICE)
bg_static = torch.tensor(train_ds.static_arr[bg_idx]).to(DEVICE)

explainer = shap.GradientExplainer(shap_model, [bg_ts, bg_static])
print("GradientExplainer ready.")


# ═══════════════════════════════════════════════════════════════════════
# COMPUTE SHAP VALUES
# ═══════════════════════════════════════════════════════════════════════

def compute_shap_for_loader(loader, dataset_name):
    all_static_shap, all_ts_shap, all_stay_ids = [], [], []

    print(f"\nComputing PRD-Net SHAP for {dataset_name} set...")
    offset = 0
    for batch_idx, (batch_ts, batch_static, _) in enumerate(loader):
        batch_ts     = batch_ts.to(DEVICE)
        batch_static = batch_static.to(DEVICE)
        n = batch_ts.shape[0]

        shap_vals   = explainer.shap_values([batch_ts, batch_static])

        static_shap = shap_vals[1]
        if static_shap.ndim == 3:
            static_shap = static_shap.squeeze(-1)
        ts_shap = shap_vals[0]
        if ts_shap.ndim == 4:
            ts_shap = ts_shap.squeeze(-1)

        all_static_shap.append(static_shap)
        all_ts_shap.append(np.abs(ts_shap).mean(axis=1))

        all_stay_ids.extend(loader.dataset.stay_ids[offset: offset + n].tolist())
        offset += n
        print(f"  batch {batch_idx+1}/{len(loader)}", end="\r")

    print()

    df_static = pd.DataFrame(np.vstack(all_static_shap), columns=STATIC_FEATURES)
    df_ts     = pd.DataFrame(np.vstack(all_ts_shap),
                             columns=[f"ts_shap_{f}" for f in TS_FEATURES])
    return pd.concat([
        pd.DataFrame({"stay_id": all_stay_ids, "split": dataset_name}),
        df_static, df_ts,
    ], axis=1)


df_test  = compute_shap_for_loader(test_loader,  "test")
df_train = compute_shap_for_loader(train_loader, "train")

explanations = pd.concat([df_test, df_train], ignore_index=True)
explanations.to_parquet(EXPLAIN_OUT / "prdnet_explanations.parquet", index=False)
print(f"\nSaved prdnet_explanations.parquet ({len(explanations):,} rows)")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1 — PRD-Net SHAP summary (top 20 static features)
# ═══════════════════════════════════════════════════════════════════════

print("\nGenerating PRD-Net SHAP summary plot...")
test_static_shap = df_test[STATIC_FEATURES].values
test_static_vals = X_test.drop(columns=["stay_id"]).values

shap.summary_plot(
    test_static_shap, test_static_vals,
    feature_names=STATIC_FEATURES,
    max_display=20, show=False, plot_size=(12, 8),
)
plt.title("SHAP Feature Importance — PRD-Net (Test Set)\nStatic Features", fontsize=13)
plt.tight_layout()
plt.savefig(EXPLAIN_OUT / "shap_prdnet_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_prdnet_summary.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2 — GRU vs PRD-Net feature importance comparison
# ═══════════════════════════════════════════════════════════════════════

gru_expl_path = EXPLAIN_OUT / "explanations.parquet"
if gru_expl_path.exists():
    print("\nGenerating GRU vs PRD-Net comparison plot...")
    gru_expl = pd.read_parquet(gru_expl_path)
    gru_test = gru_expl[gru_expl["split"] == "test"]

    gru_mean_abs = pd.Series(
        np.abs(gru_test[STATIC_FEATURES].values).mean(axis=0),
        index=STATIC_FEATURES,
    ).sort_values(ascending=False)

    prd_mean_abs = pd.Series(
        np.abs(test_static_shap).mean(axis=0),
        index=STATIC_FEATURES,
    ).sort_values(ascending=False)

    TOP_N   = 15
    top_gru = gru_mean_abs.head(TOP_N)
    top_prd = prd_mean_abs.head(TOP_N)

    # Union of top features from both models
    top_features = list(dict.fromkeys(top_gru.index.tolist() + top_prd.index.tolist()))[:TOP_N]

    gru_vals = gru_mean_abs.reindex(top_features).fillna(0)
    prd_vals = prd_mean_abs.reindex(top_features).fillna(0)

    x     = np.arange(len(top_features))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(x - width / 2, gru_vals.values, width, label="GRU Baseline", color="#4575b4", alpha=0.85)
    ax.barh(x + width / 2, prd_vals.values, width, label="PRD-Net",       color="#d73027", alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Mean |SHAP| (test set)", fontsize=11)
    ax.set_title(
        f"Feature Importance: GRU Baseline vs PRD-Net\n"
        f"Top {TOP_N} static features by mean |SHAP|",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(EXPLAIN_OUT / "shap_prdnet_vs_gru.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_prdnet_vs_gru.png")
else:
    print("\nSkipping GRU vs PRD-Net comparison — run shap_global.py first to generate explanations.parquet")


# ═══════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════

mean_abs = pd.Series(
    np.abs(test_static_shap).mean(axis=0), index=STATIC_FEATURES
).sort_values(ascending=False)

print(f"\nTop 10 static features by mean |SHAP| — PRD-Net (test set):")
for feat, val in mean_abs.head(10).items():
    print(f"  {feat:<40} {val:.4f}")

if HAS_CXR:
    cxr_mean = mean_abs[mean_abs.index.isin(groups["cxr_all"])]
    print(f"\nTop 5 CXR features by mean |SHAP|:")
    for feat, val in cxr_mean.head(5).items():
        print(f"  {feat:<40} {val:.4f}")

    print("\nMean |SHAP| by feature group (PRD-Net):")
    print(f"  Baseline clinical : {mean_abs[groups['baseline']].mean():.4f}")
    if groups["cxr_struct"]:
        print(f"  CXR structured    : {mean_abs[groups['cxr_struct']].mean():.4f}")
    if groups["bert_pca"]:
        print(f"  BERT PCA          : {mean_abs[groups['bert_pca']].mean():.4f}")
