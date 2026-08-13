"""
PRD-Net v2 — fd05: Contrastive Feature-Difference Explanations
==============================================================
The deliverable that justifies this track: contrastive, prototype-relative,
feature-level attributions.

For the linear model, logit = w.x + b, so the contribution of input dim i is
exactly w_i * (x_i - E[x_i]) — the SHAP value (interventional LinearExplainer).
We compute both shap.LinearExplainer values and the manual w*(x-mean) and assert
they match, then report:

  - GLOBAL importance: mean |contribution| per feature, split by Δpos / Δneg side.
  - PER-PATIENT attributions rendered in raw clinical units, e.g.
      "glucose_mean 18.0 mg/dL below the short-stay prototype -> +0.40 toward long-stay".
  - A sanity block over true long-stay patients confirming the dominant
    contributions are directionally consistent.

Raw-unit deltas are recovered from the scaled deltas via the StandardScaler:
  raw_delta_i = scaled_delta_i * scaler.scale_[feature_i].

Run AFTER: fd04_diff-train.py.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
import config_fd as C

_spec = importlib.util.spec_from_file_location("fd_model", Path(__file__).parent / "fd03_diff-model.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
load_trained = _m.load_trained


if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd05 — explanations  (window={W}, diff_input={C.DIFF_INPUT})")
    if C.MODEL != "linear":
        print("  NOTE: MODEL != 'linear' — exact w*delta attribution assumes the "
              "linear model. For the MLP ablation, aggregate SHAP back to features.")

    # ── Load model, threshold, scaler, train + test diff vectors ──────────────
    lt = load_trained("test")
    tr, te = lt["train"], lt["query"]
    Xtr, Xte = lt["Xtr"], lt["Xq"]
    model, thr = lt["model"], lt["thr"]
    scaler = lt["scaler"]
    scale, mean_ = scaler.scale_, scaler.mean_            # (F + A,)
    coef, bias = lt["coef"], lt["bias"]                   # (in_dim,), scalar
    labels_in = lt["labels_in"]
    feats = lt["feats"]; F = len(feats)
    abs_feats = lt["abs_feats"]

    # ── SHAP (interventional linear) vs manual w*(x-mean) ─────────────────────
    # Use the full train set as background (max_samples=len) — shap otherwise
    # subsamples it, which would shift the base value off the true train mean.
    masker = shap.maskers.Independent(Xtr, max_samples=len(Xtr))
    explainer = shap.LinearExplainer(
        (coef, bias), masker, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(Xte)               # (N, in_dim)
    manual = lt["contribs"]                               # w*(x-mean), from load_trained
    max_diff = float(np.max(np.abs(shap_vals - manual)))
    print(f"  SHAP vs manual w*(x-mean): max abs diff = {max_diff:.2e}")
    assert max_diff < 1e-4, (
        f"SHAP LinearExplainer diverged from w*(x-mean) by {max_diff:.2e} — "
        "this is the track's core 'exact attribution' claim; treat as a hard failure.")

    probs = torch.sigmoid(model(torch.tensor(Xte))).detach().numpy()
    preds = (probs >= thr).astype(int)

    # ── GLOBAL importance, split by side ──────────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)            # (in_dim,)
    glob = pd.DataFrame({"input": labels_in, "mean_abs_shap": mean_abs})
    glob["side"] = glob["input"].str.split(":").str[0]
    glob["feature"] = glob["input"].str.split(":").str[1]
    print("\nGlobal importance — top 15 input dimensions (mean |SHAP|):")
    for _, r in glob.sort_values("mean_abs_shap", ascending=False).head(15).iterrows():
        print(f"  {r['input']:<28} {r['mean_abs_shap']:.4f}")

    print("\nGlobal importance by side (sum of mean |SHAP|):")
    for side, sub in glob.groupby("side"):
        print(f"  {side:<6}: {sub['mean_abs_shap'].sum():.4f}")

    # ── DISTANCE VIEW vs DIFFERENCE VIEW (Task 4) ─────────────────────────────
    # Distance view  = ONE scalar per prototype: ||Δpos||, ||Δneg|| over all F
    #                  features (the prd_net/05 sanity metric). Tells you which
    #                  prototype is closer overall, but NOT which feature caused it.
    # Difference view= the signed per-feature vector w_i·Δ_i whose sum is the
    #                  logit, so every decision is attributable to named features.
    dp = te["X"] - te["pos_proto"]
    dn = te["X"] - te["neg_proto"]
    norm_pos = np.linalg.norm(dp, axis=1)        # ||Δpos|| per patient (scaled units)
    norm_neg = np.linalg.norm(dn, axis=1)        # ||Δneg||
    dist_pred = (norm_pos < norm_neg).astype(int)   # closer to long-stay proto -> long
    from sklearn.metrics import f1_score
    dist_f1 = f1_score(te["labels"], dist_pred, zero_division=0)
    diff_f1 = f1_score(te["labels"], preds, zero_division=0)
    agree = float((dist_pred == preds).mean())
    print("\nDistance view vs. difference view (Task 4):")
    print("  Distance  = single scalar per prototype (||Δpos|| vs ||Δneg||); no per-feature reason.")
    print("  Difference= signed per-feature vector w_i·Δ_i (sum = logit); fully attributable.")
    print(f"  Distance-only verdict (||Δpos|| < ||Δneg||):  test F1 = {dist_f1:.4f}")
    print(f"  Difference model (learned weights):           test F1 = {diff_f1:.4f}")
    print(f"  The two verdicts agree on {agree*100:.1f}% of test patients, but only the")
    print("  difference view names WHICH features drove each individual decision.")

    # ── PER-PATIENT raw-unit rendering helper ─────────────────────────────────
    def proto_word(side):  # Δpos -> long-stay prototype, Δneg -> short-stay
        return "long-stay" if side == "Δpos" else "short-stay"

    def render(i, top=5):
        order = np.argsort(np.abs(shap_vals[i]))[::-1][:top]
        lines = []
        for j in order:
            side, feat = labels_in[j].split(":")
            contrib = shap_vals[i, j]
            toward = "long-stay" if contrib > 0 else "short-stay"
            if side == "abs":
                # Not a diff — patient's own hard-filter category, recovered
                # from the scaled one-hot via the fitted scaler (raw is 0/1).
                ai = F + abs_feats.index(feat)
                raw_val = Xte[i, j] * scale[ai] + mean_[ai]
                state = "present" if raw_val > 0.5 else "absent"
                lines.append(f"      {feat} ({state}, patient's own category) "
                             f"-> {contrib:+.3f} toward {toward}")
                continue
            fi = feats.index(feat)
            raw_delta = float(Xte[i, j] * scale[fi])     # scaled delta -> raw units
            direction = "above" if raw_delta > 0 else "below"
            lines.append(
                f"      {feat} {abs(raw_delta):.2f} {direction} the {proto_word(side)} "
                f"prototype  -> {contrib:+.3f} toward {toward}")
        return lines

    # ── SANITY BLOCK: most-confident TRUE long-stay test patients ─────────────
    true_long = np.where(te["labels"] == 1)[0]
    conf = true_long[np.argsort(probs[true_long])[::-1][:5]]
    print("\nSanity — top-5 most-confident TRUE long-stay patients:")
    n_consistent = 0
    for i in conf:
        net = shap_vals[i].sum()
        ok = net > 0                                     # net push toward long-stay
        n_consistent += ok
        print(f"\n  stay_id {int(te['stay_ids'][i])}  prob={probs[i]:.3f}  "
              f"pred={'long' if preds[i] else 'short'}  net_shap={net:+.3f}  "
              f"[{'consistent' if ok else 'INCONSISTENT'}]")
        print(f"      distance view: ‖Δpos‖={norm_pos[i]:.2f}  ‖Δneg‖={norm_neg[i]:.2f}  "
              f"(closer to {'long' if norm_pos[i] < norm_neg[i] else 'short'}-stay proto)")
        print("      difference view (top per-feature contributions):")
        for line in render(i):
            print(line)
    print(f"\n  {n_consistent}/5 net-direction consistent with the long-stay label.")

    # ── VISUAL EXPORTS — best practices from GRU baseline ─────────────────────
    # Mirrors shap_global.py / explainability.py: beeswarm summary, waterfall
    # plots for 3 representative patients, dependence plots, and calibration.
    C.EXPORT_DIR.mkdir(exist_ok=True)
    TAG = f"_{W}h"

    # 1. Summary plot (beeswarm) ─────────────────────────────────────────────
    print("\nGenerating SHAP summary plot...")
    shap.summary_plot(
        shap_vals, Xte, feature_names=labels_in,
        max_display=20, show=False, plot_size=(12, 8),
    )
    plt.title(
        f"SHAP Feature Importance — PRD-Net v2 Diff Model (Test Set, {W}h window)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(C.EXPORT_DIR / f"fd_shap_summary{TAG}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: exports/fd_shap_summary{TAG}.png")

    # 2. Waterfall plots — high / median / low predicted-risk ────────────────
    print("\nGenerating waterfall plots...")

    sorted_idx = np.argsort(probs.ravel())
    patient_cases = {
        "high_risk":   int(sorted_idx[-1]),
        "median_risk": int(sorted_idx[len(sorted_idx) // 2]),
        "low_risk":    int(sorted_idx[0]),
    }

    base_value      = float(probs.mean())
    TOP_N_WATERFALL = 15

    for case_name, i in patient_cases.items():
        y_prob   = float(probs[i])
        y_true   = int(te["labels"][i])
        shap_row = shap_vals[i]

        order      = np.argsort(np.abs(shap_row))[::-1]
        top_idx    = order[:TOP_N_WATERFALL]
        top_shap   = shap_row[top_idx]
        top_names  = [labels_in[j] for j in top_idx]
        top_vals   = Xte[i, top_idx]
        residual   = shap_row.sum() - top_shap.sum()

        shap_with_res  = np.append(top_shap, residual)
        names_with_res = top_names + [f"... {len(labels_in) - TOP_N_WATERFALL} others"]

        cumulative  = base_value
        bar_bottoms = []
        bar_heights = []
        bar_colors  = []
        for sv in shap_with_res:
            bar_bottoms.append(min(cumulative, cumulative + sv))
            bar_heights.append(abs(sv))
            bar_colors.append("#d73027" if sv >= 0 else "#4575b4")
            cumulative += sv

        y_positions = list(range(len(shap_with_res)))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(y_positions, bar_heights, left=bar_bottoms,
                color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7)

        tick_labels = []
        for name, val, sv in zip(names_with_res,
                                  np.append(top_vals, [np.nan]), shap_with_res):
            sign = "+" if sv >= 0 else "−"
            if np.isnan(val):
                tick_labels.append(f"{name}   ({sign}{abs(sv):.3f})")
            else:
                tick_labels.append(f"{name} = {val:.2f}   ({sign}{abs(sv):.3f})")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.axvline(base_value, color="black", linewidth=1.2, linestyle="--",
                   label=f"E[f(x)] = {base_value:.3f}")
        ax.axvline(y_prob, color="gray", linewidth=1.2, linestyle=":",
                   label=f"f(x) = {y_prob:.3f}")
        truth_str = "prolonged (>7d)" if y_true == 1 else "normal (≤7d)"
        ax.set_xlabel("SHAP contribution (logit units)", fontsize=10)
        ax.set_title(
            f"SHAP Waterfall — {case_name.replace('_', ' ').title()} [{W}h window]\n"
            f"stay_id={int(te['stay_ids'][i])}  |  prob={y_prob:.3f}  |  actual={truth_str}",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(C.EXPORT_DIR / f"fd_waterfall_{case_name}{TAG}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: exports/fd_waterfall_{case_name}{TAG}.png  "
              f"(stay={int(te['stay_ids'][i])}, prob={y_prob:.3f}, true={y_true})")

    # 3. Dependence plots — top-3 input dimensions ───────────────────────────
    print("\nGenerating dependence plots...")

    mean_abs_dep = np.abs(shap_vals).mean(axis=0)
    top3_j       = np.argsort(mean_abs_dep)[::-1][:3]
    top3_names   = [labels_in[j] for j in top3_j]
    print(f"  Top-3 input dimensions: {top3_names}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, j, name in zip(axes, top3_j, top3_names):
        x_vals = Xte[:, j]
        y_shap = shap_vals[:, j]
        sc = ax.scatter(x_vals, y_shap, c=x_vals, cmap="coolwarm",
                        s=12, alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label="Feature value (scaled)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(f"{name}\nmean|SHAP| = {mean_abs_dep[j]:.4f}", fontsize=9)

    fig.suptitle(
        f"SHAP Dependence Plots — Top-3 Input Dimensions, PRD-Net v2 ({W}h window)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(C.EXPORT_DIR / f"fd_dependence{TAG}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: exports/fd_dependence{TAG}.png")

    # 4. Calibration — reliability diagram + ECE ─────────────────────────────
    print("\nGenerating calibration plot...")
    from sklearn.calibration import calibration_curve

    y_prob_arr = probs.ravel()
    y_true_arr = te["labels"].astype(int)

    N_BINS = 10
    frac_pos, mean_pred = calibration_curve(
        y_true_arr, y_prob_arr, n_bins=N_BINS, strategy="uniform"
    )
    bin_edges  = np.linspace(0, 1, N_BINS + 1)
    bin_counts = np.array([
        ((y_prob_arr >= bin_edges[k]) & (y_prob_arr < bin_edges[k + 1])).sum()
        for k in range(N_BINS)
    ])
    n_ret        = len(frac_pos)
    valid_counts = np.array([bin_counts[k] for k in range(N_BINS) if bin_counts[k] > 0])[:n_ret]
    ece          = float(
        np.sum(valid_counts * np.abs(frac_pos - mean_pred)) / valid_counts.sum()
    )
    print(f"  ECE (Expected Calibration Error) = {ece:.4f}")
    print(f"  Prevalence (test positive rate)  = {y_true_arr.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#d73027",
            linewidth=2, markersize=6, label="Model")
    ax.fill_between(mean_pred, frac_pos, mean_pred,
                    alpha=0.15, color="#d73027", label="Calibration gap")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title(f"Reliability Diagram\nECE = {ece:.4f}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax2 = axes[1]
    pos_probs = y_prob_arr[y_true_arr == 1]
    neg_probs = y_prob_arr[y_true_arr == 0]
    ax2.hist(neg_probs, bins=30, alpha=0.6, color="#4575b4",
             label="Actual ≤7d (negative)", density=True)
    ax2.hist(pos_probs, bins=30, alpha=0.6, color="#d73027",
             label="Actual >7d (positive)", density=True)
    ax2.axvline(thr, color="black", linestyle="--", linewidth=1,
                label=f"Threshold ({thr:.2f})")
    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Score Distribution by True Label", fontsize=12)
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"Calibration Analysis — PRD-Net v2 Diff Model (Test Set, {W}h window)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(C.EXPORT_DIR / f"fd_calibration{TAG}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: exports/fd_calibration{TAG}.png")

    print("\nExports written to exports/:")
    print(f"  fd_shap_summary{TAG}.png")
    print(f"  fd_waterfall_high_risk{TAG}.png / _median_risk / _low_risk")
    print(f"  fd_dependence{TAG}.png")
    print(f"  fd_calibration{TAG}.png")
    print("Done.")
