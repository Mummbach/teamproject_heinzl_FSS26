"""
TimeSHAP — Temporal SHAP for GRU+MLP Baseline
===============================================
Computes per-timestep Shapley values for the GRU model's time-series input.

Core idea (manual TimeSHAP without the feedzai library, which is
incompatible with shap>=0.45):

  Each of the 48 ICU hours is treated as a single "player" in the SHAP
  game.  For a given patient and a binary mask m ∈ {0,1}^48:
    - If m[t] = 1 → keep the actual vital values at hour t
    - If m[t] = 0 → replace hour t with the per-feature background mean

  shap.KernelExplainer samples random subsets of the 48 hours, evaluates
  the model on masked sequences, and computes the exact Shapley value for
  each time step.

  This tells us: "At which hour did the time-series information contribute
  most to the prediction for this patient?"

Produces:
  A) Per-patient temporal SHAP line plots (high / median / low risk)
  B) Population heatmap: mean |SHAP| per hour across N patients
  C) Aggregated bar chart: which hours matter most on average

Run AFTER: 08_model_gru.py  (best_gru_model.pt must exist)

Input:  output/X_train_scaled.parquet
        output/X_test_scaled.parquet
        output/y_train.parquet / y_test.parquet
        output/timeseries.parquet
        output/best_gru_model.pt
        output/predictions.parquet

Output: output/timeshap_patient_high.png
        output/timeshap_patient_median.png
        output/timeshap_patient_low.png
        output/timeshap_heatmap.png
        output/timeshap_hourly_importance.png
        output/timeshap_values.parquet
"""

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import OUTPUT_DIR
from multimodal_utils import ICUDataset, load_multimodal_model

SEED          = 42
N_BG          = 50
N_PATIENTS    = 30
N_SHAP_SAMPLE = 128
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════

print("Loading data...")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
ts      = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")
preds   = pd.read_parquet(OUTPUT_DIR / "predictions.parquet")

TS_FEATURES     = [c for c in ts.columns if c not in ["stay_id", "hour"]]
STATIC_FEATURES = [c for c in X_test.columns if c != "stay_id"]

train_ds = ICUDataset(X_train, y_train, ts, TS_FEATURES)
test_ds  = ICUDataset(X_test,  y_test,  ts, TS_FEATURES)

model = load_multimodal_model(
    OUTPUT_DIR / "best_gru_model.pt",
    ts_input_size=len(TS_FEATURES),
    static_input_size=len(STATIC_FEATURES),
    device=DEVICE,
)
print("Model loaded.")


# ═══════════════════════════════════════════════════════════════════════
# BACKGROUND
# ═══════════════════════════════════════════════════════════════════════

rng     = np.random.default_rng(SEED)
bg_idx  = rng.choice(len(train_ds), size=min(N_BG, len(train_ds)), replace=False)
bg_ts   = train_ds.ts_arr[bg_idx]
bg_mean = bg_ts.mean(axis=0)   # (48, 12)


# ═══════════════════════════════════════════════════════════════════════
# TIMESHAP WRAPPER
# ═══════════════════════════════════════════════════════════════════════

def make_ts_predictor(static_vec: np.ndarray, patient_ts_actual: np.ndarray):
    static_batch = torch.tensor(static_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    def predict(masks: np.ndarray) -> np.ndarray:
        n     = masks.shape[0]
        probs = np.empty(n, dtype=np.float32)
        bs    = 64
        for start in range(0, n, bs):
            m_batch  = masks[start:start + bs]
            b        = m_batch.shape[0]
            m_exp    = m_batch[:, :, np.newaxis]
            ts_batch = m_exp * patient_ts_actual + (1 - m_exp) * bg_mean
            ts_t     = torch.tensor(ts_batch, dtype=torch.float32).to(DEVICE)
            s_t      = static_batch.expand(b, -1)
            with torch.no_grad():
                logits = model(ts_t, s_t).cpu().numpy().flatten()
            probs[start:start + b] = 1 / (1 + np.exp(-logits))
        return probs

    return predict


background_mask = np.zeros((1, 48), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
# SELECT PATIENTS
# ═══════════════════════════════════════════════════════════════════════

test_preds = (preds[preds["split"] == "test"]
              .merge(y_test[["stay_id", "los_gt7"]], on="stay_id")
              .sort_values("y_prob")
              .reset_index(drop=True))
n_test = len(test_preds)

# 3 spotlight patients
spotlight = {
    "high_risk":   test_preds.iloc[-1],
    "median_risk": test_preds.iloc[n_test // 2],
    "low_risk":    test_preds.iloc[0],
}

# N_PATIENTS for heatmap (spread evenly across risk spectrum)
heatmap_idx = np.linspace(0, n_test - 1, N_PATIENTS, dtype=int)
heatmap_pids = test_preds.iloc[heatmap_idx]["stay_id"].values

test_sid_to_idx = {sid: i for i, sid in enumerate(test_ds.stay_ids)}


# ═══════════════════════════════════════════════════════════════════════
# HELPER: compute per-hour Shapley values for one patient
# ═══════════════════════════════════════════════════════════════════════

def compute_timeshap(stay_id: int) -> np.ndarray:
    """Returns (48,) array of Shapley values — one per ICU hour."""
    idx = test_sid_to_idx[stay_id]
    patient_ts_actual = test_ds.ts_arr[idx]          # (48, 12)
    static_vec        = test_ds.static_arr[idx]       # (F,)

    predictor = make_ts_predictor(static_vec, patient_ts_actual)

    explainer   = shap.KernelExplainer(predictor, background_mask)
    shap_values = explainer.shap_values(
        np.ones((1, 48), dtype=np.float32),
        nsamples=N_SHAP_SAMPLE,
        silent=True,
    )
    # shap_values: (1, 48) — one Shapley value per hour
    return shap_values[0]


# ═══════════════════════════════════════════════════════════════════════
# A) PER-PATIENT PLOTS  (3 spotlight patients)
# ═══════════════════════════════════════════════════════════════════════

print("\n─── A) Per-patient TimeSHAP plots ──────────────────────────────")

for case_name, row in spotlight.items():
    sid    = row["stay_id"]
    y_prob = float(row["y_prob"])
    y_true = int(row["los_gt7"])

    print(f"  Computing TimeSHAP for {case_name} (stay={sid}, prob={y_prob:.3f})...")
    sv = compute_timeshap(sid)   # (48,)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw={"height_ratios": [2, 1]})

    # Top: Shapley values per hour
    ax = axes[0]
    colors = ["#d73027" if v >= 0 else "#4575b4" for v in sv]
    ax.bar(range(48), sv, color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylabel("SHAP value (contribution to prediction)", fontsize=10)
    truth_str = ">7d (prolonged)" if y_true == 1 else "≤7d (normal)"
    ax.set_title(
        f"TimeSHAP — {case_name.replace('_', ' ').title()}\n"
        f"stay_id={sid}  |  predicted={y_prob:.3f}  |  actual={truth_str}",
        fontsize=11)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([f"h{h}" for h in range(0, 48, 4)], fontsize=8)

    # Bottom: cumulative SHAP (running total)
    ax2 = axes[1]
    cumulative = np.cumsum(sv)
    ax2.plot(range(48), cumulative, color="#555555", linewidth=1.5)
    ax2.fill_between(range(48), 0, cumulative,
                     where=(cumulative >= 0), alpha=0.3, color="#d73027")
    ax2.fill_between(range(48), 0, cumulative,
                     where=(cumulative < 0),  alpha=0.3, color="#4575b4")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlim(-0.5, 47.5)
    ax2.set_ylabel("Cumulative SHAP", fontsize=9)
    ax2.set_xlabel("Hour of ICU stay", fontsize=10)
    ax2.set_xticks(range(0, 48, 4))
    ax2.set_xticklabels([f"h{h}" for h in range(0, 48, 4)], fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / f"timeshap_patient_{case_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: output/timeshap_patient_{case_name}.png")


# ═══════════════════════════════════════════════════════════════════════
# B) POPULATION HEATMAP  (N_PATIENTS across risk spectrum)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n─── B) Population heatmap ({N_PATIENTS} patients) ──────────────────")

all_sv   = np.zeros((N_PATIENTS, 48), dtype=np.float32)
all_prob = []

for i, sid in enumerate(heatmap_pids):
    prob = float(test_preds[test_preds["stay_id"] == sid]["y_prob"].iloc[0])
    print(f"  [{i+1:2d}/{N_PATIENTS}] stay={sid}  prob={prob:.3f}...", end="\r")
    all_sv[i]   = compute_timeshap(sid)
    all_prob.append(prob)

print()

# Save raw values
sv_df = pd.DataFrame(all_sv,
                     columns=[f"h{t}" for t in range(48)])
sv_df.insert(0, "stay_id", heatmap_pids)
sv_df.insert(1, "y_prob",  all_prob)
sv_df.to_parquet(OUTPUT_DIR / "timeshap_values.parquet", index=False)
print("  Saved: output/timeshap_values.parquet")

# Heatmap: rows = patients (sorted by risk), cols = hours
sorted_order = np.argsort(all_prob)
heatmap_data = np.abs(all_sv)[sorted_order]   # (N, 48) — sorted low→high risk
prob_sorted  = np.array(all_prob)[sorted_order]

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
               interpolation="nearest")
ax.set_xticks(range(0, 48, 4))
ax.set_xticklabels([f"h{h}" for h in range(0, 48, 4)], fontsize=8)
ax.set_yticks(range(N_PATIENTS))
ax.set_yticklabels([f"{p:.2f}" for p in prob_sorted], fontsize=7)
ax.set_xlabel("Hour of ICU stay (first 48 h)", fontsize=11)
ax.set_ylabel("Predicted risk (low → high)", fontsize=11)
ax.set_title(
    f"TimeSHAP Heatmap — |Shapley value| per hour per patient\n"
    f"(n={N_PATIENTS} test patients, sorted by predicted risk)",
    fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="|SHAP value|")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "timeshap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/timeshap_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════
# C) HOURLY IMPORTANCE BAR CHART  (mean |SHAP| per hour)
# ═══════════════════════════════════════════════════════════════════════

print("\n─── C) Hourly importance bar chart ─────────────────────────────")

mean_abs_sv = np.abs(all_sv).mean(axis=0)   # (48,)
top5_hours  = np.argsort(mean_abs_sv)[::-1][:5]

fig, ax = plt.subplots(figsize=(13, 4))
bar_colors = ["#d73027" if h in top5_hours else "#7faacc" for h in range(48)]
ax.bar(range(48), mean_abs_sv, color=bar_colors, width=0.8)
ax.set_xticks(range(0, 48, 4))
ax.set_xticklabels([f"h{h}" for h in range(0, 48, 4)], fontsize=9)
ax.set_xlabel("Hour of ICU stay", fontsize=11)
ax.set_ylabel("Mean |SHAP value|", fontsize=11)
ax.set_title(
    f"TimeSHAP — Average Hourly Importance (n={N_PATIENTS} patients)\n"
    f"Top-5 hours highlighted in red: {sorted(top5_hours.tolist())}",
    fontsize=12)
ax.set_xlim(-0.5, 47.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "timeshap_hourly_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved: output/timeshap_hourly_importance.png")
print(f"\n  Top-5 most important ICU hours: {sorted(top5_hours.tolist())}")
print(f"  Mean |SHAP| at those hours:")
for h in sorted(top5_hours.tolist()):
    print(f"    h{h:02d}  {mean_abs_sv[h]:.5f}")

print("\n" + "═"*60)
print("All TimeSHAP outputs saved to output/:")
print("  A) timeshap_patient_high_risk.png")
print("     timeshap_patient_median_risk.png")
print("     timeshap_patient_low_risk.png")
print("  B) timeshap_heatmap.png")
print("  C) timeshap_hourly_importance.png")
print("     timeshap_values.parquet")
print("═"*60)
