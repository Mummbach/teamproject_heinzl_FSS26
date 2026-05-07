"""
Expanded Explainability — GRU+MLP Baseline Model
=================================================
Four complementary explainability expansions on top of the existing
SHAP beeswarm summary produced by 08_shap.py.

  A) Waterfall plots   — per-patient SHAP breakdown for 3 representative
                         patients (highest / median / lowest predicted risk)
  B) Dependence plots  — feature value vs. SHAP value for the 3 most
                         important static features
  C) TS SHAP heatmap   — mean |SHAP| per (vital × hour) over 200 test
                         patients → shows WHICH hour each vital matters most
  D) Calibration       — reliability diagram + Expected Calibration Error (ECE)
                         to assess how well predicted probabilities match
                         actual outcome frequencies

Run AFTER:  08_shap.py  (explanations.parquet must exist)
            07_model_gru.py  (best_gru_model.pt, predictions.parquet)

Input:   output/explanations.parquet
         output/predictions.parquet
         output/X_test_scaled.parquet
         output/y_test.parquet
         output/timeseries.parquet
         output/best_gru_model.pt

Output:  output/waterfall_high_risk.png
         output/waterfall_median_risk.png
         output/waterfall_low_risk.png
         output/dependence_plots.png
         output/ts_shap_heatmap.png
         output/calibration.png
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.calibration import calibration_curve

try:
    import shap
except ImportError:
    raise ImportError("Install shap: pip install shap")

from config import OUTPUT_DIR

SEED       = 42
N_HEATMAP  = 200   # patients to average for the TS SHAP heatmap
BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS  (same as 08_shap.py — must match checkpoint)
# ═══════════════════════════════════════════════════════════════════════

class ICUDataset(Dataset):
    def __init__(self, X_static, y, ts, ts_features):
        self.stay_ids    = X_static["stay_id"].values
        self.static_arr  = X_static.drop(columns=["stay_id"]).values.astype(np.float32)
        self.labels      = y.set_index("stay_id").loc[self.stay_ids, "los_gt7"].values.astype(np.float32)
        self.ts_features = ts_features

        ts_pivot = (
            ts[ts["stay_id"].isin(self.stay_ids)]
            .sort_values(["stay_id", "hour"])
            .set_index(["stay_id", "hour"])[ts_features]
            .fillna(0.0)
        )
        self.ts_arr = np.zeros((len(self.stay_ids), 48, len(ts_features)), dtype=np.float32)
        for i, sid in enumerate(self.stay_ids):
            if sid in ts_pivot.index.get_level_values("stay_id"):
                self.ts_arr[i] = ts_pivot.loc[sid].values

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ts_arr[idx]),
            torch.tensor(self.static_arr[idx]),
            torch.tensor(self.labels[idx]),
        )


class GRUModel(nn.Module):
    def __init__(self, ts_input_size, static_input_size, hidden_size,
                 num_layers, static_dim, dropout):
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


class SHAPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ts, static):
        return torch.sigmoid(self.model(ts, static)).unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

print("Loading data...")
X_test       = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")
X_train      = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
y_test       = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
y_train      = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
ts           = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")
explanations = pd.read_parquet(OUTPUT_DIR / "explanations.parquet")
predictions  = pd.read_parquet(OUTPUT_DIR / "predictions.parquet")

TS_FEATURES     = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_FEATURES = [c for c in X_test.columns if c != "stay_id"]

print(f"  Static features      : {len(STATIC_FEATURES)}")
print(f"  Time-series features : {len(TS_FEATURES)}")
print(f"  Explanations rows    : {len(explanations):,}")

test_expl  = explanations[explanations["split"] == "test"].reset_index(drop=True)
test_preds = predictions[predictions["split"] == "test"].reset_index(drop=True)

# Merge with ground truth
test_preds = test_preds.merge(
    y_test[["stay_id", "los_gt7"]], on="stay_id", how="left"
)

# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL  (needed for sections B and C)
# ═══════════════════════════════════════════════════════════════════════

model = GRUModel(
    ts_input_size     = len(TS_FEATURES),
    static_input_size = len(STATIC_FEATURES),
    hidden_size       = 64,
    num_layers        = 2,
    static_dim        = 64,
    dropout           = 0.3,
).to(DEVICE)
model.load_state_dict(torch.load(
    OUTPUT_DIR / "best_gru_model.pt", map_location=DEVICE, weights_only=True
))
model.eval()

shap_model = SHAPWrapper(model).to(DEVICE)
shap_model.eval()
print("Model loaded.")

# Background set from training data (same as 08_shap.py)
train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES)
rng      = np.random.default_rng(SEED)
bg_idx   = rng.choice(len(train_ds), size=200, replace=False)
bg_ts     = torch.tensor(train_ds.ts_arr[bg_idx]).to(DEVICE)
bg_static = torch.tensor(train_ds.static_arr[bg_idx]).to(DEVICE)

explainer = shap.GradientExplainer(shap_model, [bg_ts, bg_static])
print("GradientExplainer created.")


# ═══════════════════════════════════════════════════════════════════════
# A) WATERFALL PLOTS — 3 representative patients
# ═══════════════════════════════════════════════════════════════════════
# A waterfall plot shows how each feature pushes the prediction UP (red)
# or DOWN (blue) relative to the model's average prediction (base value).
# Starting at E[f(x)] on the left, each bar adds or subtracts its SHAP
# value until we arrive at this patient's prediction f(x) on the right.

print("\n─── A) Waterfall Plots ───────────────────────────────────────────")

# Pick 3 representative patients from the test set
test_preds_sorted = test_preds.sort_values("y_prob").reset_index(drop=True)
n_test            = len(test_preds_sorted)

patient_cases = {
    "high_risk":   test_preds_sorted.iloc[-1],
    "median_risk": test_preds_sorted.iloc[n_test // 2],
    "low_risk":    test_preds_sorted.iloc[0],
}

# Align SHAP rows with test_expl (explanations are in test_expl keyed by stay_id)
test_expl_indexed = test_expl.set_index("stay_id")

# Expected output value = mean(sigmoid) across background → approx base value
# We compute it as mean y_prob over the full test set (reasonable proxy)
base_value = float(test_preds["y_prob"].mean())

TOP_N_WATERFALL = 15   # features shown in each waterfall

for case_name, patient_row in patient_cases.items():
    sid      = patient_row["stay_id"]
    y_prob   = float(patient_row["y_prob"])
    y_true   = int(patient_row["los_gt7"])

    if sid not in test_expl_indexed.index:
        print(f"  WARNING: stay_id {sid} not in explanations — skipping {case_name}")
        continue

    shap_row    = test_expl_indexed.loc[sid, STATIC_FEATURES].values.astype(float)
    feature_row = X_test.set_index("stay_id").loc[sid, STATIC_FEATURES].values.astype(float)

    # Sort features by |SHAP| descending; keep top N
    order      = np.argsort(np.abs(shap_row))[::-1]
    top_idx    = order[:TOP_N_WATERFALL]
    top_shap   = shap_row[top_idx]
    top_names  = [STATIC_FEATURES[i] for i in top_idx]
    top_vals   = feature_row[top_idx]

    # Residual: sum of all shap values NOT shown
    residual = shap_row.sum() - top_shap.sum()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Build cumulative baseline for waterfall bars
    shap_with_residual = np.append(top_shap, residual)
    names_with_residual = top_names + [f"... {len(STATIC_FEATURES) - TOP_N_WATERFALL} others"]

    cumulative   = base_value
    bar_bottoms  = []
    bar_heights  = []
    bar_colors   = []
    y_positions  = list(range(len(shap_with_residual)))

    for sv in shap_with_residual:
        bar_bottoms.append(min(cumulative, cumulative + sv))
        bar_heights.append(abs(sv))
        bar_colors.append("#d73027" if sv >= 0 else "#4575b4")
        cumulative += sv

    ax.barh(y_positions, bar_heights, left=bar_bottoms,
            color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7)

    # Feature labels with value annotation
    labels = []
    for name, val, sv in zip(names_with_residual, np.append(top_vals, [np.nan]), shap_with_residual):
        sign  = "+" if sv >= 0 else "−"
        if np.isnan(val):
            labels.append(f"{name}   ({sign}{abs(sv):.3f})")
        else:
            labels.append(f"{name} = {val:.2f}   ({sign}{abs(sv):.3f})")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(base_value, color="black", linewidth=1.2, linestyle="--", label=f"E[f(x)] = {base_value:.3f}")
    ax.axvline(y_prob,     color="gray",  linewidth=1.2, linestyle=":",  label=f"f(x) = {y_prob:.3f}")

    truth_str = "prolonged (>7d)" if y_true == 1 else "normal (≤7d)"
    ax.set_xlabel("Model output (probability)", fontsize=10)
    ax.set_title(
        f"SHAP Waterfall — {case_name.replace('_', ' ').title()}\n"
        f"stay_id={sid}  |  predicted={y_prob:.3f}  |  actual={truth_str}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"waterfall_{case_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: output/waterfall_{case_name}.png  (stay={sid}, prob={y_prob:.3f}, true={y_true})")


# ═══════════════════════════════════════════════════════════════════════
# B) DEPENDENCE PLOTS — top-3 static features
# ═══════════════════════════════════════════════════════════════════════
# A SHAP dependence plot shows the relationship between a feature's VALUE
# (x-axis) and its SHAP contribution (y-axis) across all test patients.
# Unlike a global bar chart, it reveals non-linear effects and thresholds:
# e.g., "age only matters above 65" or "high heart rate always hurts".

print("\n─── B) Dependence Plots ─────────────────────────────────────────")

test_static_shap = test_expl[STATIC_FEATURES].values         # (N_test, F)
test_static_vals = X_test.drop(columns=["stay_id"]).values   # (N_test, F)

mean_abs_shap = np.abs(test_static_shap).mean(axis=0)
top3_idx      = np.argsort(mean_abs_shap)[::-1][:3]
top3_names    = [STATIC_FEATURES[i] for i in top3_idx]

print(f"  Top-3 static features: {top3_names}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, idx, name in zip(axes, top3_idx, top3_names):
    x_vals  = test_static_vals[:, idx]
    y_shap  = test_static_shap[:, idx]

    # Color points by the feature's own value for extra depth
    sc = ax.scatter(x_vals, y_shap, c=x_vals, cmap="coolwarm",
                    s=12, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Feature value")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(name, fontsize=10)
    ax.set_ylabel("SHAP value", fontsize=10)
    ax.set_title(f"{name}\nmean|SHAP| = {mean_abs_shap[idx]:.4f}", fontsize=10)

fig.suptitle("SHAP Dependence Plots — Top-3 Static Features (Test Set)", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dependence_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/dependence_plots.png")


# ═══════════════════════════════════════════════════════════════════════
# C) TS SHAP HEATMAP — mean |SHAP| per vital × hour
# ═══════════════════════════════════════════════════════════════════════
# The existing 08_shap.py collapses TS SHAP to one number per vital by
# averaging over all 48 hours.  Here we keep the full (48, n_ts) grid
# for N_HEATMAP test patients and visualise it as a heatmap.
#
# Each cell (hour h, vital v) = mean of |SHAP| across all selected
# test patients at that hour for that vital.  This answers: "At which
# hour of the ICU stay did each vital sign matter most?"

print(f"\n─── C) TS SHAP Heatmap ({N_HEATMAP} patients) ──────────────────────")

test_ds = ICUDataset(X_test, y_test, ts, TS_FEATURES)

rng_hm     = np.random.default_rng(SEED + 1)
hm_idx     = rng_hm.choice(len(test_ds), size=min(N_HEATMAP, len(test_ds)), replace=False)
hm_ts      = torch.tensor(test_ds.ts_arr[hm_idx]).to(DEVICE)    # (N, 48, 12)
hm_static  = torch.tensor(test_ds.static_arr[hm_idx]).to(DEVICE)  # (N, F)

print(f"  Running GradientExplainer on {len(hm_idx)} test patients...")

# Process in one batch (200 × 48 × 12 fits in RAM for CPU; split if GPU OOM)
shap_vals = explainer.shap_values([hm_ts, hm_static])
# shap_vals[0]: shape (N, 48, n_ts_feats)

ts_shap_3d = np.abs(shap_vals[0])         # (N, 48, n_ts_feats)
heatmap    = ts_shap_3d.mean(axis=0)      # (48, n_ts_feats)  mean |SHAP| per hour × vital

# Normalise each vital's column to [0, 1] so rare-but-important vitals
# are visible alongside dominant ones.
col_max = heatmap.max(axis=0, keepdims=True)
col_max[col_max == 0] = 1.0
heatmap_norm = heatmap / col_max

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_norm.T, aspect="auto", cmap="YlOrRd", interpolation="nearest",
               vmin=0, vmax=1)

ax.set_xticks(range(0, 48, 4))
ax.set_xticklabels([f"h{h}" for h in range(0, 48, 4)], fontsize=8)
ax.set_yticks(range(len(TS_FEATURES)))
ax.set_yticklabels(TS_FEATURES, fontsize=9)
ax.set_xlabel("Hour of ICU stay (first 48 h)", fontsize=11)
ax.set_ylabel("Vital sign", fontsize=11)
ax.set_title(
    f"TS SHAP Heatmap — mean |SHAP| per vital × hour\n"
    f"(column-normalised, n={len(hm_idx)} test patients)",
    fontsize=12,
)
cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("Relative importance (0=min, 1=max per vital)", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ts_shap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/ts_shap_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════
# D) CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
# A well-calibrated model means: when it says "60% probability", roughly
# 60% of those patients actually have prolonged stays.
# The reliability diagram plots mean predicted probability vs. true
# frequency inside each probability bin.  A perfectly calibrated model
# would lie on the diagonal.  ECE (Expected Calibration Error) is the
# weighted average deviation from perfect calibration across bins.

print("\n─── D) Calibration Analysis ─────────────────────────────────────")

y_prob_test = test_preds["y_prob"].values
y_true_test = test_preds["los_gt7"].values.astype(int)

N_BINS = 10
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true_test, y_prob_test, n_bins=N_BINS, strategy="uniform"
)

# Expected Calibration Error = weighted mean |fraction_pos − mean_pred|
bin_edges  = np.linspace(0, 1, N_BINS + 1)
bin_counts = np.zeros(N_BINS, dtype=int)
for i in range(N_BINS):
    mask = (y_prob_test >= bin_edges[i]) & (y_prob_test < bin_edges[i + 1])
    bin_counts[i] = mask.sum()
# calibration_curve may return fewer bins if some are empty; align lengths
n_returned = len(fraction_of_positives)
valid_counts = np.array([
    bin_counts[i] for i in range(N_BINS)
    if bin_counts[i] > 0
])[:n_returned]
ece = float(
    np.sum(
        valid_counts * np.abs(fraction_of_positives - mean_predicted_value)
    ) / valid_counts.sum()
)

print(f"  ECE (Expected Calibration Error) = {ece:.4f}")
print(f"  Prevalence (test set positive rate) = {y_true_test.mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: reliability diagram
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
ax.plot(mean_predicted_value, fraction_of_positives,
        "o-", color="#d73027", linewidth=2, markersize=6, label="Model")
ax.fill_between(mean_predicted_value, fraction_of_positives,
                mean_predicted_value, alpha=0.15, color="#d73027", label="Calibration gap")
ax.set_xlabel("Mean predicted probability", fontsize=11)
ax.set_ylabel("Fraction of positives", fontsize=11)
ax.set_title(f"Reliability Diagram\nECE = {ece:.4f}", fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Right: predicted probability distribution split by true label
ax2 = axes[1]
pos_probs = y_prob_test[y_true_test == 1]
neg_probs = y_prob_test[y_true_test == 0]
ax2.hist(neg_probs, bins=30, alpha=0.6, color="#4575b4", label="Actual ≤7d (negative)", density=True)
ax2.hist(pos_probs, bins=30, alpha=0.6, color="#d73027", label="Actual >7d (positive)", density=True)
ax2.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision boundary (0.5)")
ax2.set_xlabel("Predicted probability", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Score Distribution by True Label", fontsize=12)
ax2.legend(fontsize=9)

fig.suptitle("Calibration Analysis — GRU+MLP Baseline (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/calibration.png")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("Outputs written to output/:")
print("  A) waterfall_high_risk.png")
print("     waterfall_median_risk.png")
print("     waterfall_low_risk.png")
print("  B) dependence_plots.png")
print("  C) ts_shap_heatmap.png")
print("  D) calibration.png")
print("═" * 60)
