"""
TimeSHAP — Temporal Shapley Values for the GRU Model
Computes per-hour Shapley values for the time-series input using
shap.KernelExplainer. Each of the 48 ICU hours is treated as one player:
absent hours are replaced with zeros (consistent with the model's missing-data
handling via ffill/bfill + zero-padding in ICUDataset).

Note: due to ffill/bfill preprocessing, consecutive hours share identical
values. SHAP values therefore reflect time windows rather than isolated hours.

- Per-patient plots — Shapley bars + cumulative curve for high / median / low risk
- Population heatmap — mean |SHAP| per hour across N patients sorted by risk
- Hourly bar chart   — average importance per hour across all selected patients

Run AFTER:  06_model_gru.py  (best_gru_model.pt must exist)

Input:   pipeline/output/X_train_scaled.parquet
         pipeline/output/X_test_scaled.parquet
         pipeline/output/y_train.parquet / y_test.parquet
         pipeline/output/timeseries.parquet
         pipeline/output/best_gru_model.pt

Output:  explainability/output/predictions.parquet   — generated if not present
         explainability/output/timeshap_patient_high_risk.png
         explainability/output/timeshap_patient_median_risk.png
         explainability/output/timeshap_patient_low_risk.png
         explainability/output/timeshap_heatmap.png
         explainability/output/timeshap_hourly_importance.png
         explainability/output/timeshap_values.parquet
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
import shap

from config import OUTPUT_DIR
from multimodal_utils import ICUDataset, load_multimodal_model

SEED          = 42
N_PATIENTS    = 30
N_SHAP_SAMPLE = 512  # 128 samples is too few for stable 48-feature Shapley estimates; 512+ recommended
N_ICU_HOURS   = 48
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


print("Loading data...")
X_test  = pd.read_parquet(OUTPUT_DIR / "X_test_scaled.parquet")
X_train = pd.read_parquet(OUTPUT_DIR / "X_train_scaled.parquet")
y_test  = pd.read_parquet(OUTPUT_DIR / "y_test.parquet")
y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")
ts      = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")

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

preds_path = EXPLAIN_OUT / "predictions.parquet"
if not preds_path.exists():
    print("predictions.parquet not found — generating from model...")
    pred_rows = []
    for ds, split_name in [(train_ds, "train"), (test_ds, "test")]:
        for i in range(len(ds)):
            ts_t   = torch.tensor(ds.ts_arr[i:i+1]).to(DEVICE)
            s_t    = torch.tensor(ds.static_arr[i:i+1]).to(DEVICE)
            text_t = torch.zeros(1, 1536, device=DEVICE) if model.use_text else None
            with torch.no_grad():
                logit = model(ts_t, s_t, text_t).cpu().item()
            pred_rows.append({"stay_id": ds.stay_ids[i], "split": split_name,
                              "y_prob": 1 / (1 + np.exp(-logit))})
    preds = pd.DataFrame(pred_rows)
    preds.to_parquet(preds_path, index=False)
    print(f"  Saved predictions.parquet ({len(preds):,} rows)")
else:
    preds = pd.read_parquet(preds_path)


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
            ts_batch = m_exp * patient_ts_actual  # absent hours → 0
            ts_t     = torch.tensor(ts_batch, dtype=torch.float32).to(DEVICE)
            s_t      = static_batch.expand(b, -1)
            with torch.no_grad():
                text_t = torch.zeros(b, 1536, device=DEVICE) if model.use_text else None
                logits = model(ts_t, s_t, text_t).cpu().numpy()
            probs[start:start + b] = 1 / (1 + np.exp(-logits))
        return probs

    return predict


background_mask = np.zeros((1, N_ICU_HOURS), dtype=np.float32)

test_preds = (preds[preds["split"] == "test"]
              .merge(y_test[["stay_id", "los_gt7"]], on="stay_id")
              .sort_values("y_prob")
              .reset_index(drop=True))
n_test = len(test_preds)

spotlight = {
    "high_risk":   test_preds.iloc[-1],
    "median_risk": test_preds.iloc[n_test // 2],
    "low_risk":    test_preds.iloc[0],
}

heatmap_idx  = np.linspace(0, n_test - 1, N_PATIENTS, dtype=int)
heatmap_pids = test_preds.iloc[heatmap_idx]["stay_id"].values

test_sid_to_idx = {sid: i for i, sid in enumerate(test_ds.stay_ids)}


def compute_timeshap(stay_id: int) -> np.ndarray:
    """Returns (48,) array of Shapley values, one per ICU hour."""
    idx               = test_sid_to_idx[stay_id]
    patient_ts_actual = test_ds.ts_arr[idx]
    static_vec        = test_ds.static_arr[idx]

    predictor   = make_ts_predictor(static_vec, patient_ts_actual)
    explainer   = shap.KernelExplainer(predictor, background_mask)
    shap_values = explainer.shap_values(
        np.ones((1, N_ICU_HOURS), dtype=np.float32),
        nsamples=N_SHAP_SAMPLE,
        silent=True,
    )
    return shap_values[0]


print("\nPer-patient TimeSHAP plots...")

for case_name, row in spotlight.items():
    sid    = row["stay_id"]
    y_prob = float(row["y_prob"])
    y_true = int(row["los_gt7"])

    print(f"  Computing for {case_name} (stay={sid}, prob={y_prob:.3f})...")
    sv = compute_timeshap(sid)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    colors = ["#d73027" if v >= 0 else "#4575b4" for v in sv]
    ax.bar(range(N_ICU_HOURS), sv, color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.5, N_ICU_HOURS - 0.5)
    ax.set_ylabel("SHAP value (contribution to prediction)", fontsize=10)
    truth_str = ">7d (prolonged)" if y_true == 1 else "≤7d (normal)"
    ax.set_title(
        f"TimeSHAP — {case_name.replace('_', ' ').title()}\n"
        f"stay_id={sid}  |  predicted={y_prob:.3f}  |  actual={truth_str}",
        fontsize=11)
    ax.set_xticks(range(0, N_ICU_HOURS, 4))
    ax.set_xticklabels([f"h{h}" for h in range(0, N_ICU_HOURS, 4)], fontsize=8)

    ax2 = axes[1]
    cumulative = np.cumsum(sv)
    ax2.plot(range(N_ICU_HOURS), cumulative, color="#555555", linewidth=1.5)
    ax2.fill_between(range(N_ICU_HOURS), 0, cumulative, where=(cumulative >= 0), alpha=0.3, color="#d73027")
    ax2.fill_between(range(N_ICU_HOURS), 0, cumulative, where=(cumulative < 0),  alpha=0.3, color="#4575b4")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlim(-0.5, N_ICU_HOURS - 0.5)
    ax2.set_ylabel("Cumulative SHAP", fontsize=9)
    ax2.set_xlabel("Hour of ICU stay", fontsize=10)
    ax2.set_xticks(range(0, N_ICU_HOURS, 4))
    ax2.set_xticklabels([f"h{h}" for h in range(0, N_ICU_HOURS, 4)], fontsize=8)

    plt.tight_layout()
    plt.savefig(EXPLAIN_OUT / f"timeshap_patient_{case_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved timeshap_patient_{case_name}.png")


print(f"\nPopulation heatmap ({N_PATIENTS} patients)...")

all_sv   = np.zeros((N_PATIENTS, N_ICU_HOURS), dtype=np.float32)
all_prob = []

for i, sid in enumerate(heatmap_pids):
    prob = float(test_preds[test_preds["stay_id"] == sid]["y_prob"].iloc[0])
    print(f"  [{i+1:2d}/{N_PATIENTS}] stay={sid}  prob={prob:.3f}...", end="\r")
    all_sv[i] = compute_timeshap(sid)
    all_prob.append(prob)

print()

sv_df = pd.DataFrame(all_sv, columns=[f"h{t}" for t in range(N_ICU_HOURS)])
sv_df.insert(0, "stay_id", heatmap_pids)
sv_df.insert(1, "y_prob",  all_prob)
sv_df.to_parquet(EXPLAIN_OUT / "timeshap_values.parquet", index=False)
print("  Saved timeshap_values.parquet")

sorted_order = np.argsort(all_prob)
heatmap_data = np.abs(all_sv)[sorted_order]
prob_sorted  = np.array(all_prob)[sorted_order]

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(0, N_ICU_HOURS, 4))
ax.set_xticklabels([f"h{h}" for h in range(0, N_ICU_HOURS, 4)], fontsize=8)
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
plt.savefig(EXPLAIN_OUT / "timeshap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved timeshap_heatmap.png")


print("\nHourly importance bar chart...")

mean_abs_sv = np.abs(all_sv).mean(axis=0)
top5_hours  = np.argsort(mean_abs_sv)[::-1][:5]

fig, ax = plt.subplots(figsize=(13, 4))
bar_colors = ["#d73027" if h in top5_hours else "#7faacc" for h in range(N_ICU_HOURS)]
ax.bar(range(N_ICU_HOURS), mean_abs_sv, color=bar_colors, width=0.8)
ax.set_xticks(range(0, N_ICU_HOURS, 4))
ax.set_xticklabels([f"h{h}" for h in range(0, N_ICU_HOURS, 4)], fontsize=9)
ax.set_xlabel("Hour of ICU stay", fontsize=11)
ax.set_ylabel("Mean |SHAP value|", fontsize=11)
ax.set_title(
    f"TimeSHAP — Average Hourly Importance (n={N_PATIENTS} patients)\n"
    f"Top-5 hours highlighted in red: {sorted(top5_hours.tolist())}",
    fontsize=12)
ax.set_xlim(-0.5, N_ICU_HOURS - 0.5)
plt.tight_layout()
plt.savefig(EXPLAIN_OUT / "timeshap_hourly_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved timeshap_hourly_importance.png")

print(f"\nTop-5 most important ICU hours: {sorted(top5_hours.tolist())}")
for h in sorted(top5_hours.tolist()):
    print(f"  h{h:02d}  {mean_abs_sv[h]:.5f}")
