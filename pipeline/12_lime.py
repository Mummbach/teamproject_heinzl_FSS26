"""
LIME Explainability — GRU+MLP Baseline Model
=============================================
Computes LIME explanations for the static features of the trained model
and compares the top features with SHAP results from 08_shap.py.

Strategy: for each patient, the GRU hidden state is pre-computed and fixed.
LIME then perturbs only the static features, isolating their marginal effect.
This gives a fair comparison with the static SHAP values from 08_shap.py.

Run AFTER:  07_model_gru.py  (best_gru_model.pt must exist)
            08_shap.py       (explanations.parquet must exist for comparison)

Input:   output/X_test_scaled.parquet
         output/y_test.parquet
         output/timeseries.parquet
         output/best_gru_model.pt
         output/explanations.parquet   (SHAP — for comparison plot)

Output:  output/lime_weights.parquet   — per-patient LIME weights (static features)
         output/lime_vs_shap.png       — side-by-side top-15 feature comparison
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    raise ImportError("Install lime: pip install lime")

from config import OUTPUT_DIR

SEED       = 42
N_SAMPLES  = 100
N_LIME     = 1000
N_FEATURES = 15

# Must match the hyperparameters used in 07_model_gru.py
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
STATIC_DIM  = 64
DROPOUT     = 0.3

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Model definition (same as 07_model_gru.py) ────────────────────────
class GRUModel(nn.Module):
    def __init__(self, ts_input_size, static_input_size,
                 hidden_size, num_layers, static_dim, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=ts_input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_branch = nn.Sequential(
            nn.Linear(static_input_size, static_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size + static_dim, 32),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, ts, static):
        _, h_n     = self.gru(ts)
        gru_out    = h_n[-1]
        static_out = self.static_branch(static)
        return self.classifier(torch.cat([gru_out, static_out], dim=1)).squeeze(1)

    @torch.no_grad()
    def gru_embedding(self, ts_tensor):
        """Returns the GRU final hidden state for a batch of time-series."""
        self.eval()
        _, h_n = self.gru(ts_tensor)
        return h_n[-1]   # (batch, hidden_size)


# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
ts      = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

TS_FEATURES     = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_FEATURES = [c for c in X_test.columns if c != "stay_id"]
print(f"  Static features: {len(STATIC_FEATURES)}  TS features: {len(TS_FEATURES)}")

# Build time-series array for test patients
stay_ids    = X_test["stay_id"].values
static_arr  = X_test.drop(columns=["stay_id"]).values.astype(np.float32)
labels      = y_test.set_index("stay_id").loc[stay_ids, "los_gt7"].values

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
model = GRUModel(
    ts_input_size     = len(TS_FEATURES),
    static_input_size = len(STATIC_FEATURES),
    hidden_size       = HIDDEN_SIZE,
    num_layers        = NUM_LAYERS,
    static_dim        = STATIC_DIM,
    dropout           = DROPOUT,
).to(DEVICE)
model.load_state_dict(torch.load(OUTPUT_DIR / "best_gru_model.pt",
                                  map_location=DEVICE, weights_only=True))
model.eval()
print("Model loaded.")

# ── Pre-compute GRU embeddings for all test patients ───────────────────
print("Pre-computing GRU embeddings...")
ts_tensor    = torch.tensor(ts_arr).to(DEVICE)
gru_embeddings = model.gru_embedding(ts_tensor).cpu().numpy()  # (N, 64)
print(f"  GRU embeddings shape: {gru_embeddings.shape}")


# ── LIME prediction wrapper ────────────────────────────────────────────
# For patient i: GRU embedding is FIXED, only static features are perturbed.

def _lime_key_to_feat(key: str, feature_set: set) -> str | None:
    """Extract original feature name from a LIME discretized label.

    LIME produces keys like "age <= 50.0", "0.5 < age <= 65.0", "gender_male = 1".
    Splitting on comparison operators and checking each token handles all formats.
    Simple substring matching would incorrectly map "age" to keys for "language_english"
    since "age" appears in "language".
    """
    for token in re.split(r'\s*(?:<=|>=|!=|[<>=])\s*', key):
        token = token.strip()
        if token in feature_set:
            return token
    return None

def make_predict_fn(gru_emb_i: np.ndarray):
    """Returns a predict function for a single patient with fixed GRU embedding."""
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


# ── LIME explainer (fitted on full test static array as background) ────
print("\nCreating LIME explainer...")
explainer = LimeTabularExplainer(
    training_data   = static_arr,
    feature_names   = STATIC_FEATURES,
    class_names     = ["LOS <= 7d", "LOS > 7d"],
    mode            = "classification",
    discretize_continuous = True,
    random_state    = SEED,
)

# ── Explain N_SAMPLES test patients ───────────────────────────────────
rng      = np.random.default_rng(SEED)
indices  = rng.choice(len(stay_ids), size=N_SAMPLES, replace=False)

lime_rows = []
feature_set = set(STATIC_FEATURES)
print(f"Computing LIME for {N_SAMPLES} patients...")
for count, idx in enumerate(indices, 1):
    predict_fn = make_predict_fn(gru_embeddings[idx])
    exp = explainer.explain_instance(
        data_row          = static_arr[idx],
        predict_fn        = predict_fn,
        num_features      = len(STATIC_FEATURES),
        num_samples       = N_LIME,
        labels            = (1,),
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
    print(f"  {count}/{N_SAMPLES}", end="\r")

print()
lime_df = pd.DataFrame(lime_rows)
lime_df.to_parquet(OUTPUT_DIR / "lime_weights.parquet", index=False)
print(f"Saved: output/lime_weights.parquet  ({len(lime_df)} patients)")


# ── Comparison plot: LIME vs SHAP ─────────────────────────────────────
print("\nGenerating LIME vs SHAP comparison plot...")

# Mean absolute LIME weight per feature
lime_importance = (
    lime_df[STATIC_FEATURES].abs().mean()
    .sort_values(ascending=False)
    .head(N_FEATURES)
)

# Mean absolute SHAP value per feature (from 08_shap.py output)
shap_importance = None
shap_path = OUTPUT_DIR / "explanations.parquet"
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

# LIME plot
axes[0].barh(lime_importance.index[::-1], lime_importance.values[::-1],
             color="#4C72B0")
axes[0].set_title(f"LIME — Top {N_FEATURES} Static Features\n"
                  f"(mean |weight|, n={N_SAMPLES} patients)", fontsize=11)
axes[0].set_xlabel("Mean |LIME weight|")
axes[0].tick_params(axis="y", labelsize=8)

# SHAP plot (if available)
if shap_importance is not None:
    axes[1].barh(shap_importance.index[::-1], shap_importance.values[::-1],
                 color="#DD8452")
    axes[1].set_title(f"SHAP — Top {N_FEATURES} Static Features\n"
                      f"(mean |SHAP value|, test set)", fontsize=11)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].tick_params(axis="y", labelsize=8)

plt.suptitle("LIME vs SHAP — Feature Importance Comparison", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lime_vs_shap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/lime_vs_shap.png")

# Quick console summary
print(f"\nTop 10 features by LIME (mean |weight|):")
for feat, val in lime_importance.head(10).items():
    print(f"  {feat:<35} {val:.4f}")

if shap_importance is not None:
    print(f"\nTop 10 features by SHAP (mean |SHAP|):")
    for feat, val in shap_importance.head(10).items():
        print(f"  {feat:<35} {val:.4f}")
