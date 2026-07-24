"""
Expanded Explainability — GRU Model
Produces four complementary analyses building on the SHAP attributions
from shap.py.

- Waterfall plots  — per-patient SHAP breakdown for high / median / low risk
- Dependence plots — feature value vs. SHAP value for the 3 most important features
- TS SHAP heatmap  — mean |SHAP| per vital × hour over 200 test patients
- Calibration      — reliability diagram and Expected Calibration Error (ECE)

Run AFTER:  shap.py  (explanations.parquet must exist)
            06_model_gru.py  (best_gru_model.pt must exist)

Input:   explainability/output/explanations.parquet
         explainability/output/predictions.parquet
         pipeline/output/X_test_scaled.parquet
         pipeline/output/y_test.parquet
         pipeline/output/timeseries.parquet
         pipeline/output/best_gru_model.pt

Output:  explainability/output/waterfall_high_risk.png
         explainability/output/waterfall_median_risk.png
         explainability/output/waterfall_low_risk.png
         explainability/output/dependence_plots.png
         explainability/output/ts_shap_heatmap.png
         explainability/output/calibration.png
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

EXPLAIN_OUT = _HERE / "output"
EXPLAIN_OUT.mkdir(exist_ok=True)

import pandas as pd
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

try:
    import shap
except ImportError:
    raise ImportError("Install shap: pip install shap")

from config import OUTPUT_DIR
from multimodal_utils import (
    ICUDataset, SHAPWrapper, load_multimodal_model,
    get_cxr_feature_groups,
)

SEED = 42
N_HEATMAP = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


print("Loading data...")
X_test = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
y_test = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
ts = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")
explanations = pd.read_parquet(EXPLAIN_OUT / "explanations.parquet")
predictions = pd.read_parquet(EXPLAIN_OUT / "predictions.parquet")

TS_FEATURES = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_FEATURES = [c for c in X_test.columns if c != "stay_id"]
groups = get_cxr_feature_groups(STATIC_FEATURES)
CXR_FEATURES = groups["cxr_all"]

print(f"  Static features      : {len(STATIC_FEATURES)}")
print(f"  Time-series features : {len(TS_FEATURES)}")
print(f"  Explanations rows    : {len(explanations):,}")
if CXR_FEATURES:
    print(f"  CXR features present : {len(CXR_FEATURES)}")

test_expl = explanations[explanations["split"] == "test"].reset_index(drop=True)
test_preds = predictions[predictions["split"] == "test"].reset_index(drop=True)
test_preds = test_preds.merge(y_test[["stay_id", "los_gt7"]], on="stay_id", how="left")


model = load_multimodal_model(
    OUTPUT_DIR / "best_gru_model.pt",
    ts_input_size=len(TS_FEATURES),
    static_input_size=len(STATIC_FEATURES),
    device=DEVICE,
)
shap_model = SHAPWrapper(model).to(DEVICE)
shap_model.eval()
print("Model loaded.")

train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES)
rng = np.random.default_rng(SEED)
bg_idx = rng.choice(len(train_ds), size=min(200, len(train_ds)), replace=False)
bg_ts = torch.tensor(train_ds.ts_arr[bg_idx]).to(DEVICE)
bg_static = torch.tensor(train_ds.static_arr[bg_idx]).to(DEVICE)

explainer = shap.GradientExplainer(shap_model, [bg_ts, bg_static])
print("GradientExplainer ready.")


# waterfall plots — high / median / low risk patient

print("\nWaterfall plots...")

test_preds_sorted = test_preds.sort_values("y_prob").reset_index(drop=True)
n_test = len(test_preds_sorted)

patient_cases = {
    "high_risk":   test_preds_sorted.iloc[-1],
    "median_risk": test_preds_sorted.iloc[n_test // 2],
    "low_risk":    test_preds_sorted.iloc[0],
}

test_expl_indexed = test_expl.set_index("stay_id")
base_value = float(test_preds["y_prob"].mean())
TOP_N_WATERFALL = 15

for case_name, patient_row in patient_cases.items():
    sid = patient_row["stay_id"]
    y_prob = float(patient_row["y_prob"])
    y_true = int(patient_row["los_gt7"])

    if sid not in test_expl_indexed.index:
        print(f"  WARNING: stay_id {sid} not in explanations, skipping {case_name}")
        continue

    shap_row = test_expl_indexed.loc[sid, STATIC_FEATURES].values.astype(float)
    feature_row = X_test.set_index("stay_id").loc[sid, STATIC_FEATURES].values.astype(float)

    order = np.argsort(np.abs(shap_row))[::-1]
    top_idx = order[:TOP_N_WATERFALL]
    top_shap = shap_row[top_idx]
    top_names = [STATIC_FEATURES[i] for i in top_idx]
    top_vals = feature_row[top_idx]
    residual = shap_row.sum() - top_shap.sum()

    fig, ax = plt.subplots(figsize=(9, 6))

    shap_with_residual = np.append(top_shap, residual)
    names_with_residual = top_names + [f"... {len(STATIC_FEATURES) - TOP_N_WATERFALL} others"]

    cumulative = base_value
    bar_bottoms = []
    bar_heights = []
    bar_colors = []
    y_positions = list(range(len(shap_with_residual)))

    for sv in shap_with_residual:
        bar_bottoms.append(min(cumulative, cumulative + sv))
        bar_heights.append(abs(sv))
        bar_colors.append("#d73027" if sv >= 0 else "#4575b4")
        cumulative += sv

    ax.barh(y_positions, bar_heights, left=bar_bottoms,
            color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7)

    labels = []
    for name, val, sv in zip(names_with_residual, np.append(top_vals, [np.nan]), shap_with_residual):
        sign = "+" if sv >= 0 else "−"
        tag = " [CXR]" if name in CXR_FEATURES else ""
        if np.isnan(val):
            labels.append(f"{name}{tag}   ({sign}{abs(sv):.3f})")
        else:
            labels.append(f"{name}{tag} = {val:.2f}   ({sign}{abs(sv):.3f})")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(base_value, color="black", linewidth=1.2, linestyle="--", label=f"E[f(x)] = {base_value:.3f}")
    ax.axvline(y_prob, color="gray", linewidth=1.2, linestyle=":", label=f"f(x) = {y_prob:.3f}")

    truth_str = "prolonged (>7d)" if y_true == 1 else "normal (≤7d)"
    ax.set_xlabel("Model output (probability)", fontsize=10)
    ax.set_title(
        f"SHAP Waterfall — {case_name.replace('_', ' ').title()}\n"
        f"stay_id={sid}  |  predicted={y_prob:.3f}  |  actual={truth_str}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = EXPLAIN_OUT / f"waterfall_{case_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved waterfall_{case_name}.png  (stay={sid}, prob={y_prob:.3f}, true={y_true})")


# dependence plots — top 3 static features

print("\nDependence plots...")

test_static_shap = test_expl[STATIC_FEATURES].values
test_static_vals = X_test.drop(columns=["stay_id"]).values

mean_abs_shap = np.abs(test_static_shap).mean(axis=0)
top3_idx = np.argsort(mean_abs_shap)[::-1][:3]
top3_names = [STATIC_FEATURES[i] for i in top3_idx]

print(f"  Top-3 static features: {top3_names}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, idx, name in zip(axes, top3_idx, top3_names):
    x_vals = test_static_vals[:, idx]
    y_shap = test_static_shap[:, idx]

    sc = ax.scatter(x_vals, y_shap, c=x_vals, cmap="coolwarm",
                    s=12, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Feature value")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(name, fontsize=10)
    ax.set_ylabel("SHAP value", fontsize=10)
    ax.set_title(f"{name}\nmean|SHAP| = {mean_abs_shap[idx]:.4f}", fontsize=10)

fig.suptitle("SHAP Dependence Plots — Top-3 Static Features (Test Set)", fontsize=12)
plt.tight_layout()
plt.savefig(EXPLAIN_OUT / "dependence_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved dependence_plots.png")


# ts shap heatmap — mean |shap| per vital x hour

print(f"\nTS SHAP heatmap ({N_HEATMAP} patients)...")

test_ds = ICUDataset(X_test, y_test, ts, TS_FEATURES)

rng_hm = np.random.default_rng(SEED + 1)
hm_idx = rng_hm.choice(len(test_ds), size=min(N_HEATMAP, len(test_ds)), replace=False)
hm_ts = torch.tensor(test_ds.ts_arr[hm_idx]).to(DEVICE)
hm_static = torch.tensor(test_ds.static_arr[hm_idx]).to(DEVICE)

print(f"  Running GradientExplainer on {len(hm_idx)} test patients...")
shap_vals = explainer.shap_values([hm_ts, hm_static])

ts_raw = shap_vals[0]
if ts_raw.ndim == 4:
    ts_raw = ts_raw.squeeze(-1)
heatmap = np.abs(ts_raw).mean(axis=0)  # (48, n_ts_feats)

# normalize per vital so rare-but-important vitals are visible
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
plt.savefig(EXPLAIN_OUT / "ts_shap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved ts_shap_heatmap.png")


# calibration analysis

print("\nCalibration analysis...")

y_prob_test = test_preds["y_prob"].values
y_true_test = test_preds["los_gt7"].values.astype(int)

N_BINS = 10
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true_test, y_prob_test, n_bins=N_BINS, strategy="uniform"
)

bin_edges = np.linspace(0, 1, N_BINS + 1)
bin_counts = np.zeros(N_BINS, dtype=int)
for i in range(N_BINS):
    mask = (y_prob_test >= bin_edges[i]) & (y_prob_test < bin_edges[i + 1])
    bin_counts[i] = mask.sum()

n_returned = len(fraction_of_positives)
valid_counts = np.array([bin_counts[i] for i in range(N_BINS) if bin_counts[i] > 0])[:n_returned]
ece = float(
    np.sum(valid_counts * np.abs(fraction_of_positives - mean_predicted_value)) / valid_counts.sum()
)

print(f"  ECE = {ece:.4f}")
print(f"  Prevalence (test) = {y_true_test.mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

fig.suptitle("Calibration Analysis — GRU Model (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig(EXPLAIN_OUT / "calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved calibration.png")
