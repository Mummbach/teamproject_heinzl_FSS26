"""
PRD-Net v2 — fd06: Dashboard Export (data only)
================================================
Exports one contrastive-explanation record per TEST patient for a later
dashboard. This step writes data only; the dashboard UI (e.g. a small Streamlit
app) is a follow-up step.

Per patient:
  - stay_id, predicted probability + label, true label
  - top-3 most-similar long-stay peers and top-3 short-stay peers (stay_ids)
  - patient raw feature values, pos/neg prototype raw values, signed raw deltas
  - the largest deviations to the long prototype and to the short prototype
  - top model contributions (signed, logit units)

Contributions use the exact linear decomposition w_i * (x_i - E_train[x_i])
(identical to fd05's SHAP values). Raw-unit values are recovered via the scaler.

Outputs: exports/fd_explanations_test_{w}.json   (full nested records)
         exports/fd_explanations_test_{w}.parquet (flat; nested fields as JSON)

Run AFTER: fd04_diff-train.py.
"""

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
import config_fd as C

_spec = importlib.util.spec_from_file_location("fd_model", Path(__file__).parent / "fd03_diff-model.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
assemble_diff, input_dim, build_model, input_feature_labels = (
    _m.assemble_diff, _m.input_dim, _m.build_model, _m.input_feature_labels)

TOP_K = 5      # entries per ranked list in the export


def load_bundle(split):
    with open(C.prototypes_path(split), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd06 — dashboard export  (window={W})")

    tr, te = load_bundle("train"), load_bundle("test")
    Xtr = assemble_diff(tr["X"], tr["pos_proto"], tr["neg_proto"]).astype(np.float32)
    Xte = assemble_diff(te["X"], te["pos_proto"], te["neg_proto"]).astype(np.float32)

    model = build_model(input_dim()); model.load_state_dict(
        torch.load(C.checkpoint_path(), weights_only=True)); model.eval()
    thr = torch.load(C.threshold_path(), weights_only=True)["threshold"]
    probs = torch.sigmoid(model(torch.tensor(Xte))).detach().numpy()
    preds = (probs >= thr).astype(int)

    coef = model.linear.weight.detach().numpy().ravel()
    contribs = coef[None, :] * (Xte - Xtr.mean(axis=0)[None, :])   # (N, in_dim)
    labels_in = input_feature_labels()

    with open(C.scaler_bundle_path(), "rb") as f:
        sb = pickle.load(f)
    scaler = sb["scaler"]; feats = C.feature_names(); F = len(feats)

    # Raw-unit patient values and raw-unit prototypes (invert the scaler).
    raw_te = (pd.read_parquet(C.feature_matrix_path("test", scaled=False))
              .set_index("stay_id").loc[te["stay_ids"], feats])
    pos_raw = te["pos_proto"] * scaler.scale_[None, :] + scaler.mean_[None, :]
    neg_raw = te["neg_proto"] * scaler.scale_[None, :] + scaler.mean_[None, :]
    patient_raw = raw_te.values
    delta_pos_raw = patient_raw - pos_raw          # signed raw deviation vs long proto
    delta_neg_raw = patient_raw - neg_raw          # signed raw deviation vs short proto

    def top_dev(delta_raw_row, scaled_delta_row):
        """Largest |standardized| deviations, reported in raw units."""
        order = np.argsort(np.abs(scaled_delta_row))[::-1][:TOP_K]
        return [{"feature": feats[j], "raw_delta": round(float(delta_raw_row[j]), 3),
                 "sd_delta": round(float(scaled_delta_row[j]), 3)} for j in order]

    records = []
    for i, sid in enumerate(te["stay_ids"]):
        c_order = np.argsort(np.abs(contribs[i]))[::-1][:TOP_K]
        top_contrib = []
        for j in c_order:
            side, feat = labels_in[j].split(":")
            fi = feats.index(feat)
            top_contrib.append({
                "input": labels_in[j], "feature": feat,
                "prototype": "long-stay" if side == "Δpos" else "short-stay",
                "raw_delta": round(float(Xte[i, j] * scaler.scale_[fi]), 3),
                "contribution": round(float(contribs[i, j]), 4),
            })
        records.append({
            "stay_id": int(sid),
            "prob": round(float(probs[i]), 4),
            "pred_label": int(preds[i]),
            "true_label": int(te["labels"][i]),
            "top_long_peers":  [int(x) for x in te["pos_peer_ids"][i][:3]],
            "top_short_peers": [int(x) for x in te["neg_peer_ids"][i][:3]],
            "patient": {f: round(float(patient_raw[i, k]), 3) for k, f in enumerate(feats)},
            "long_prototype":  {f: round(float(pos_raw[i, k]), 3) for k, f in enumerate(feats)},
            "short_prototype": {f: round(float(neg_raw[i, k]), 3) for k, f in enumerate(feats)},
            "largest_dev_vs_long":  top_dev(delta_pos_raw[i], te["X"][i] - te["pos_proto"][i]),
            "largest_dev_vs_short": top_dev(delta_neg_raw[i], te["X"][i] - te["neg_proto"][i]),
            "top_contributions": top_contrib,
        })

    C.EXPORT_DIR.mkdir(exist_ok=True)
    base = C.export_path()
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(records, f, indent=2)

    # Flat parquet: scalars as columns, nested fields JSON-encoded so it loads cleanly.
    flat = pd.DataFrame([{
        "stay_id": r["stay_id"], "prob": r["prob"],
        "pred_label": r["pred_label"], "true_label": r["true_label"],
        "top_long_peers": json.dumps(r["top_long_peers"]),
        "top_short_peers": json.dumps(r["top_short_peers"]),
        "top_contributions": json.dumps(r["top_contributions"]),
        "largest_dev_vs_long": json.dumps(r["largest_dev_vs_long"]),
        "largest_dev_vs_short": json.dumps(r["largest_dev_vs_short"]),
    } for r in records])
    flat.to_parquet(base.with_suffix(".parquet"))

    print(f"  {len(records):,} records -> {base.with_suffix('.json').name} / "
          f"{base.with_suffix('.parquet').name}")
    print("Done.")
