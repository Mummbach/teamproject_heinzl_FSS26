"""
LIME Explainability — GRU+MLP Multimodal Model
===============================================
Identical to 12_lime.py but operates on the multimodal model and feature set
(baseline + CXR structured + BERT PCA + has_cxr).

The GRU hidden state is pre-computed and fixed per patient; LIME then
perturbs only the static features — isolating their marginal effect
and allowing a fair comparison with the static SHAP values from 08b_shap_multimodal.py.

An extra plot compares LIME importance across feature groups
(baseline vs CXR structured vs BERT PCA).

Run AFTER:  07b_model_gru_multimodal.py  (best_gru_multimodal.pt must exist)
            08b_shap_multimodal.py       (explanations_multimodal.parquet for comparison)

Input:   output/X_test_multimodal.parquet
         output/y_test.parquet
         output/timeseries.parquet
         output/best_gru_multimodal.pt
         output/explanations_multimodal.parquet

Output:  output/lime_weights_multimodal.parquet
         output/lime_vs_shap_multimodal.png
         output/lime_feature_groups_mm.png
"""

import pandas as pd
import numpy as np
import torch
import re
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    raise ImportError("Install lime: pip install lime")

from config import OUTPUT_DIR
from multimodal_utils import GRUModel, load_multimodal_model, CXR_STRUCT_FEATURES, get_cxr_feature_groups

SEED       = 42
N_SAMPLES  = 100
N_LIME     = 1000
N_FEATURES = 15

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Load data ──────────────────────────────────────────────────────────

print("Loading multimodal data...")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_multimodal.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
ts      = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

TS_FEATURES     = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_FEATURES = [c for c in X_test.columns if c != "stay_id"]

groups             = get_cxr_feature_groups(STATIC_FEATURES)
BERT_PCA_FEATURES  = groups["bert_pca"]
CXR_STRUCT_PRESENT = groups["cxr_struct"]
CXR_ALL            = groups["cxr_all"]
BASELINE_FEATURES  = groups["baseline"]

print(f"  Static features: {len(STATIC_FEATURES)}  (baseline={len(BASELINE_FEATURES)}, CXR={len(CXR_ALL)})")

stay_ids   = X_test["stay_id"].values
static_arr = X_test.drop(columns=["stay_id"]).values.astype(np.float32)
labels     = y_test.set_index("stay_id").loc[stay_ids, "los_gt7"].values

ts_pivot = (
    ts[ts["stay_id"].isin(stay_ids)]
    .sort_values(["stay_id", "hour"])
    .set_index(["stay_id", "hour"])[TS_FEATURES]
    .fillna(0.0)
)
ts_arr = np.zeros((len(stay_ids), 48, len(TS_FEATURES)), dtype=np.float32)
for i, sid in enumerate(stay_ids):
    if sid in ts_pivot.index.get_level_values("stay_id"):
        ts_arr[i] = ts_pivot.loc[sid].values


# ── Load model ─────────────────────────────────────────────────────────

print("Loading model...")
model = load_multimodal_model(
    OUTPUT_DIR / "best_gru_multimodal.pt",
    ts_input_size=len(TS_FEATURES),
    static_input_size=len(STATIC_FEATURES),
    device=DEVICE,
)
print("Model loaded.")

print("Pre-computing GRU embeddings...")
ts_tensor      = torch.tensor(ts_arr).to(DEVICE)
gru_embeddings = model.gru_embedding(ts_tensor).cpu().numpy()
print(f"  GRU embeddings shape: {gru_embeddings.shape}")


# ── LIME prediction wrapper ────────────────────────────────────────────

def _lime_key_to_feat(key: str, feature_set: set) -> Optional[str]:
    for token in re.split(r'\s*(?:<=|>=|!=|[<>=])\s*', key):
        token = token.strip()
        if token in feature_set:
            return token
    return None


def make_predict_fn(gru_emb_i: np.ndarray):
    gru_tensor = torch.tensor(gru_emb_i, dtype=torch.float32).to(DEVICE)

    def predict(static_perturbed: np.ndarray) -> np.ndarray:
        static_t = torch.tensor(static_perturbed, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            static_out = model.static_branch(static_t)
            gru_batch  = gru_tensor.unsqueeze(0).expand(len(static_t), -1)
            fused      = torch.cat([gru_batch, static_out], dim=1)
            logits     = model.classifier(fused).squeeze(1).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])

    return predict


# ── LIME explainer ─────────────────────────────────────────────────────

print("\nCreating LIME explainer...")
explainer = LimeTabularExplainer(
    training_data         = static_arr,
    feature_names         = STATIC_FEATURES,
    class_names           = ["LOS <= 7d", "LOS > 7d"],
    mode                  = "classification",
    discretize_continuous = True,
    random_state          = SEED,
)

rng       = np.random.default_rng(SEED)
n_explain = min(N_SAMPLES, len(stay_ids))
indices   = rng.choice(len(stay_ids), size=n_explain, replace=False)

lime_rows   = []
feature_set = set(STATIC_FEATURES)
print(f"Computing LIME for {n_explain} patients...")

for count, idx in enumerate(indices, 1):
    predict_fn = make_predict_fn(gru_embeddings[idx])
    exp = explainer.explain_instance(
        data_row   = static_arr[idx],
        predict_fn = predict_fn,
        num_features = len(STATIC_FEATURES),
        num_samples  = N_LIME,
        labels       = (1,),
    )
    weights = dict(exp.as_list(label=1))
    row = {"stay_id": stay_ids[idx], "label": labels[idx]}
    feat_weights = {}
    for k, v in weights.items():
        fn = _lime_key_to_feat(k, feature_set)
        if fn is not None and fn not in feat_weights:
            feat_weights[fn] = v
    for feat in STATIC_FEATURES:
        row[feat] = feat_weights.get(feat, 0.0)
    lime_rows.append(row)
    print(f"  {count}/{n_explain}", end="\r")

print()
lime_df = pd.DataFrame(lime_rows)
lime_df.to_parquet(OUTPUT_DIR / "lime_weights_multimodal.parquet", index=False)
print(f"Saved: output/lime_weights_multimodal.parquet  ({len(lime_df)} patients)")


# ── Plot 1: LIME vs SHAP comparison ───────────────────────────────────

print("\nGenerating LIME vs SHAP comparison plot...")

lime_importance = (
    lime_df[STATIC_FEATURES].abs().mean()
    .sort_values(ascending=False)
    .head(N_FEATURES)
)

shap_importance = None
shap_path = OUTPUT_DIR / "explanations_multimodal.parquet"
if shap_path.exists():
    shap_df = pd.read_parquet(shap_path)
    shap_cols = [c for c in shap_df.columns if c in STATIC_FEATURES]
    shap_importance = (
        shap_df[shap_cols].abs().mean()
        .sort_values(ascending=False)
        .head(N_FEATURES)
    )

fig, axes = plt.subplots(1, 2 if shap_importance is not None else 1,
                         figsize=(14 if shap_importance is not None else 7, 6))
if shap_importance is None:
    axes = [axes]

def feat_label(name):
    if name in CXR_STRUCT_PRESENT:
        return f"{name} [CXR-struct]"
    if name in BERT_PCA_FEATURES:
        return f"{name} [BERT]"
    if name == "has_cxr":
        return f"{name} [CXR-flag]"
    return name

axes[0].barh(
    [feat_label(f) for f in lime_importance.index[::-1]],
    lime_importance.values[::-1],
    color="#4C72B0",
)
axes[0].set_title(f"LIME — Top {N_FEATURES} Static Features (Multimodal)\n"
                  f"(mean |weight|, n={n_explain} patients)", fontsize=11)
axes[0].set_xlabel("Mean |LIME weight|")
axes[0].tick_params(axis="y", labelsize=7)

if shap_importance is not None:
    axes[1].barh(
        [feat_label(f) for f in shap_importance.index[::-1]],
        shap_importance.values[::-1],
        color="#DD8452",
    )
    axes[1].set_title(f"SHAP — Top {N_FEATURES} Static Features (Multimodal)\n"
                      f"(mean |SHAP value|, test set)", fontsize=11)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].tick_params(axis="y", labelsize=7)

plt.suptitle("LIME vs SHAP — Multimodal Feature Importance Comparison", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lime_vs_shap_multimodal.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/lime_vs_shap_multimodal.png")


# ── Plot 2: LIME importance by feature group ──────────────────────────

print("\nGenerating LIME feature group comparison plot...")

group_means = {
    "Baseline clinical": lime_df[BASELINE_FEATURES].abs().mean().mean() if BASELINE_FEATURES else 0.0,
    "CXR structured":    lime_df[CXR_STRUCT_PRESENT].abs().mean().mean() if CXR_STRUCT_PRESENT else 0.0,
    "BERT PCA":          lime_df[BERT_PCA_FEATURES].abs().mean().mean() if BERT_PCA_FEATURES else 0.0,
    "has_cxr":           lime_df["has_cxr"].abs().mean() if "has_cxr" in lime_df.columns else 0.0,
}

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#4575b4", "#d73027", "#fc8d59", "#91bfdb"]
bars = ax.bar(group_means.keys(), group_means.values(), color=colors, width=0.5)
ax.bar_label(bars, fmt="%.4f", fontsize=10, padding=3)
ax.set_ylabel("Mean |LIME weight| (group average)", fontsize=11)
ax.set_title("LIME Contribution by Feature Group — Multimodal Model (Test Set)", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lime_feature_groups_mm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/lime_feature_groups_mm.png")

print(f"\nTop 10 features by LIME (mean |weight|):")
for feat, val in lime_importance.head(10).items():
    print(f"  {feat_label(feat):<45} {val:.4f}")

if shap_importance is not None:
    print(f"\nTop 10 features by SHAP (mean |SHAP|):")
    for feat, val in shap_importance.head(10).items():
        print(f"  {feat_label(feat):<45} {val:.4f}")
