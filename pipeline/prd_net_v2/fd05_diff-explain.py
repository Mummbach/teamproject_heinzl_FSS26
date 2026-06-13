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
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
import config_fd as C

_spec = importlib.util.spec_from_file_location("fd_model", Path(__file__).parent / "fd03_diff-model.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
assemble_diff, input_dim, build_model, input_feature_labels = (
    _m.assemble_diff, _m.input_dim, _m.build_model, _m.input_feature_labels)


def load_bundle(split):
    with open(C.prototypes_path(split), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd05 — explanations  (window={W}, diff_input={C.DIFF_INPUT})")
    if C.MODEL != "linear":
        print("  NOTE: MODEL != 'linear' — exact w*delta attribution assumes the "
              "linear model. For the MLP ablation, aggregate SHAP back to features.")

    # ── Load model, threshold, scaler, train + test diff vectors ──────────────
    tr, te = load_bundle("train"), load_bundle("test")
    Xtr = assemble_diff(tr["X"], tr["pos_proto"], tr["neg_proto"]).astype(np.float32)
    Xte = assemble_diff(te["X"], te["pos_proto"], te["neg_proto"]).astype(np.float32)

    model = build_model(input_dim()); model.load_state_dict(
        torch.load(C.checkpoint_path(), weights_only=True)); model.eval()
    thr = torch.load(C.threshold_path(), weights_only=True)["threshold"]

    with open(C.scaler_bundle_path(), "rb") as f:
        scale = pickle.load(f)["scaler"].scale_          # (F,)

    coef = model.linear.weight.detach().numpy().ravel()  # (in_dim,)
    bias = float(model.linear.bias.item())
    labels_in = input_feature_labels()
    feats = C.feature_names(); F = len(feats)

    # ── SHAP (interventional linear) vs manual w*(x-mean) ─────────────────────
    # Use the full train set as background (max_samples=len) — shap otherwise
    # subsamples it, which would shift the base value off the true train mean.
    masker = shap.maskers.Independent(Xtr, max_samples=len(Xtr))
    explainer = shap.LinearExplainer(
        (coef, bias), masker, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(Xte)               # (N, in_dim)
    manual = coef[None, :] * (Xte - Xtr.mean(axis=0)[None, :])
    max_diff = float(np.max(np.abs(shap_vals - manual)))
    print(f"  SHAP vs manual w*(x-mean): max abs diff = {max_diff:.2e}  "
          f"({'OK' if max_diff < 1e-4 else 'MISMATCH'})")

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

    # ── PER-PATIENT raw-unit rendering helper ─────────────────────────────────
    def proto_word(side):  # Δpos -> long-stay prototype, Δneg -> short-stay
        return "long-stay" if side == "Δpos" else "short-stay"

    def render(i, top=5):
        order = np.argsort(np.abs(shap_vals[i]))[::-1][:top]
        lines = []
        for j in order:
            side, feat = labels_in[j].split(":")
            fi = feats.index(feat)
            raw_delta = float(Xte[i, j] * scale[fi])     # scaled delta -> raw units
            direction = "above" if raw_delta > 0 else "below"
            contrib = shap_vals[i, j]
            toward = "long-stay" if contrib > 0 else "short-stay"
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
        for line in render(i):
            print(line)
    print(f"\n  {n_consistent}/5 net-direction consistent with the long-stay label.")
    print("Done.")
