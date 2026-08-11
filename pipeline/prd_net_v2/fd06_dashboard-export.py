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
assemble_diff, input_dim, build_model, input_feature_labels, load_absolute_block = (
    _m.assemble_diff, _m.input_dim, _m.build_model, _m.input_feature_labels, _m.load_absolute_block)

TOP_K = 15     # entries in the largest_dev_* lists. Those rank whole features,
               # so one entry is one feature and no de-duplication applies.

# Distinct FEATURES the contribution list must cover — not entries. A diff
# feature enters the model twice (Δpos and Δneg) and can rank highly on both,
# so "the 15 strongest contributions" is not "15 features": at TOP_K=15 entries
# that fell short of 15 distinct features for 2252 of 4593 test patients, and
# the dashboards then rendered 13 rows under a "top 15" heading. Collecting
# until the 15th distinct feature appears costs a handful of extra entries and
# lets the dashboards mean what their labels say.
TOP_FEATURES = 15

# diagnose()'s dominance ratio is a *heuristic* calibrated against a 0.55
# threshold on the top-5 drivers. It shares no meaning with TOP_K beyond having
# once been the same number — widening the export would otherwise shift the
# ratio for every patient and silently rewrite the wrong_reasons texts.
DOMINANCE_K = 5


def load_bundle(split):
    with open(C.prototypes_path(split), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd06 — dashboard export  (window={W})")

    tr, te = load_bundle("train"), load_bundle("test")
    abs_tr = load_absolute_block("train", tr["stay_ids"])
    abs_te = load_absolute_block("test", te["stay_ids"])
    Xtr = assemble_diff(tr["X"], tr["pos_proto"], tr["neg_proto"], absolute=abs_tr).astype(np.float32)
    Xte = assemble_diff(te["X"], te["pos_proto"], te["neg_proto"], absolute=abs_te).astype(np.float32)

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
    abs_feats = C.absolute_feature_names()
    CXR_SET = set(C.CXR_FEATURES)   # kept out of top_contributions; surfaced separately (cxr_support)

    # Raw-unit patient values and raw-unit prototypes (invert the scaler).
    raw_te = (pd.read_parquet(C.feature_matrix_path("test", scaled=False))
              .set_index("stay_id").loc[te["stay_ids"], feats])
    # te["pos_proto"]/["neg_proto"] are diff-feature-only (F dims) — fd02 never
    # touches the absolute block, so slice the scaler to the first F entries.
    pos_raw = te["pos_proto"] * scaler.scale_[None, :F] + scaler.mean_[None, :F]
    neg_raw = te["neg_proto"] * scaler.scale_[None, :F] + scaler.mean_[None, :F]
    patient_raw = raw_te.values
    delta_pos_raw = patient_raw - pos_raw          # signed raw deviation vs long proto
    delta_neg_raw = patient_raw - neg_raw          # signed raw deviation vs short proto

    def top_dev(delta_raw_row, scaled_delta_row):
        """Largest |standardized| deviations, reported in raw units."""
        order = np.argsort(np.abs(scaled_delta_row))[::-1][:TOP_K]
        return [{"feature": feats[j], "raw_delta": round(float(delta_raw_row[j]), 3),
                 "sd_delta": round(float(scaled_delta_row[j]), 3)} for j in order]

    def cxr_support(i, pred):
        """CXR-derived findings that (a) actually fired for this patient and
        (b) push in the same direction as the predicted class. Kept separate
        from top_contributions: has_cxr_report is a coverage flag, not a
        finding, and CXR only covers ~2% of patients, so it reads as
        corroborating anamnesis rather than a primary driver."""
        has_report = "has_cxr_report" in feats and patient_raw[i, feats.index("has_cxr_report")] > 0
        findings = []
        if has_report:
            for j in range(contribs.shape[1]):
                side, feat = labels_in[j].split(":")
                if feat not in CXR_SET or feat == "has_cxr_report":
                    continue
                fi = feats.index(feat)
                c = float(contribs[i, j])
                same_direction = (c > 0) if pred == 1 else (c < 0)
                if patient_raw[i, fi] > 0 and same_direction:
                    findings.append({"feature": feat, "raw_value": round(float(patient_raw[i, fi]), 3),
                                     "contribution": round(c, 4)})
            findings.sort(key=lambda d: -abs(d["contribution"]))
        return {"has_report": bool(has_report), "findings": findings}

    # Peers are training patients -> attach their true outcome so the dashboard
    # can confirm "here are 3 similar long-stay patients".
    train_lab = (pd.read_parquet(C.OUTPUT_DIR / "y_train.parquet")
                 .set_index("stay_id")["los_gt7"].to_dict())
    def peer_objs(ids, k=3):
        return [{"stay_id": int(s), "true_label": int(train_lab.get(int(s), -1))}
                for s in ids[:k]]

    # ── General patient info + peer group composition/outcome (dashboard) ─────
    # cohort.csv carries the raw demographic/admission text (ground truth) plus
    # continuous LOS, absent from both the 77-dim clinical feature matrix and
    # the binary los_gt7 label. The one-hot icd_*/icu_*/adm_* columns in
    # X_test.parquet are a separate, coarser thing: exactly what fd02's hard
    # filter matched peers on (only 7 recognized ICU buckets, etc.) — used
    # below only to describe the peer-group filter criteria truthfully, not
    # for general display (where the raw text is strictly more informative).
    cohort = (pd.read_csv(C.OUTPUT_DIR / "cohort.csv",
                          usecols=["stay_id", "gender", "first_careunit", "admission_type", "los"])
              .set_index("stay_id"))
    cohort_los = cohort["los"].to_dict()
    demo = (pd.read_parquet(C.OUTPUT_DIR / "X_test.parquet",
                            columns=["stay_id"] + C.ICD_COLS + C.ICU_COLS + C.ADM_COLS)
            .set_index("stay_id"))

    def _active_label(row, cols, labels):
        return next((labels[c] for c in cols if row[c] > 0), None)

    def filter_criteria(sid):
        """One-hot-derived category fd02's hard filter actually matched this
        patient on — can be None when the patient falls outside every
        recognized bucket (the filter then skips that criterion entirely)."""
        row = demo.loc[sid]
        return {
            "icu_unit": _active_label(row, C.ICU_COLS, C.ICU_LABELS),
            "admission_type": _active_label(row, C.ADM_COLS, C.ADM_LABELS),
            "diagnosis_category": _active_label(row, C.ICD_COLS, C.ICD_LABELS),
        }

    def patient_info(sid, fc):
        """General demographics/admission context for display — raw cohort
        text (more complete than the one-hot buckets); diagnosis has no raw-
        text source, so it reuses the filter's one-hot category."""
        row = cohort.loc[sid]
        return {
            "gender": "Male" if row["gender"] == "M" else "Female",
            "icu_unit": row["first_careunit"],
            "admission_type": row["admission_type"],
            "diagnosis_category": fc["diagnosis_category"],
        }

    def peer_group_outcomes(peer_ids, fc):
        """Avg LOS + % long-stay among the class-independent nearest peers
        (same hard filter + age band as this patient) — the actual peer
        group the dashboard shows, not just the top-3 displayed names."""
        los = [cohort_los[s] for s in peer_ids if s in cohort_los]
        long_rate = [train_lab[s] for s in peer_ids if s in train_lab]
        return {
            "n_peers": len(peer_ids),
            "avg_los_days": round(float(np.mean(los)), 2) if los else None,
            "pct_long_stay": round(100 * float(np.mean(long_rate)), 1) if long_rate else None,
            "filter_criteria": fc,
        }

    def diagnose(i):
        """Peer-support stats + heuristic reasons a prediction is probably wrong.

        Combines four signals on the full contribution vector and the filtered
        peer support: empty-filter fallback, low support, single-feature
        dominance, borderline probability, and conflicting counter-evidence.
        """
        n_long, n_short = int(te["n_pos_filtered"][i]), int(te["n_neg_filtered"][i])
        support = {"n_long_filtered": n_long, "n_short_filtered": n_short,
                   "long_fallback": bool(te["pos_fallback"][i]),
                   "short_fallback": bool(te["neg_fallback"][i])}

        ci = contribs[i]
        top_j = int(np.argmax(np.abs(ci)))
        top_feat = labels_in[top_j].split(":")[1]
        topk = np.sort(np.abs(ci))[::-1][:DOMINANCE_K]
        dominance = float(topk[0] / (topk.sum() + 1e-9))     # share among the leading drivers
        pos_sum, neg_sum = float(ci[ci > 0].sum()), float(-ci[ci < 0].sum())
        pred = int(preds[i])
        supporting, opposing = (pos_sum, neg_sum) if pred == 1 else (neg_sum, pos_sum)
        conflict = opposing / (supporting + 1e-9)
        cls = "long" if pred == 1 else "short"
        n_side = n_long if pred == 1 else n_short
        fb_side = support["long_fallback"] if pred == 1 else support["short_fallback"]

        # Reasons are appended in priority order; the generic ones (near-tie /
        # hard-case) only fire when no specific structural reason was found, so a
        # single explanation does not drown every wrong case in boilerplate.
        reasons = []
        if pred != int(te["labels"][i]):                     # only diagnose wrong cases
            if fb_side:
                reasons.append(f"No similar {cls}-stay patients matched the hard filter — "
                               f"the {cls}-stay prototype is a global fallback, so the contrast is unreliable.")
            elif n_side < 5:
                reasons.append(f"Low support: only {n_side} similar {cls}-stay patient(s) "
                               "in the training set — the model extrapolates.")
            if dominance > 0.55:
                reasons.append(f"Prediction hinges on one feature ({top_feat}, "
                               f"{dominance:.0%} of the top drivers) — fragile if that value is atypical.")
            if abs(probs[i] - thr) < 0.10:
                reasons.append(f"Borderline: probability {probs[i]:.2f} is close to the "
                               f"decision threshold {thr:.2f}.")
            if not reasons and conflict > 0.85:
                reasons.append(f"Near-tie: counter-evidence is {conflict:.0%} of the supporting "
                               "signal, so the two prototypes were almost equally close.")
            if not reasons:
                reasons.append(f"No single dominant cause — likely an inherently hard case "
                               f"(LOS driven by events after the {W}h window).")
        return support, reasons

    records = []
    for i, sid in enumerate(te["stay_ids"]):
        support, wrong_reasons = diagnose(i)
        pred = int(preds[i])
        fc = filter_criteria(int(sid))
        c_order = np.argsort(np.abs(contribs[i]))[::-1]
        top_contrib, seen_feats = [], []
        for j in c_order:
            side, feat = labels_in[j].split(":")
            if feat in CXR_SET:      # surfaced separately via cxr_support, not here
                continue
            if side == "abs":
                # Not a diff — patient's own hard-filter category (see
                # filter_criteria/patient_info); raw is 0/1, recovered via the scaler.
                ai = F + abs_feats.index(feat)
                raw_val = Xte[i, j] * scaler.scale_[ai] + scaler.mean_[ai]
                top_contrib.append({
                    "input": labels_in[j], "feature": feat,
                    "prototype": "own category",
                    "raw_value": 1.0 if raw_val > 0.5 else 0.0,  # binary one-hot
                    "contribution": round(float(contribs[i, j]), 4),
                })
            else:
                fi = feats.index(feat)
                top_contrib.append({
                    "input": labels_in[j], "feature": feat,
                    "prototype": "long-stay" if side == "Δpos" else "short-stay",
                    "raw_delta": round(float(Xte[i, j] * scaler.scale_[fi]), 3),
                    "contribution": round(float(contribs[i, j]), 4),
                })
            if feat not in seen_feats:
                seen_feats.append(feat)
            if len(seen_feats) == TOP_FEATURES:
                break
        records.append({
            "stay_id": int(sid),
            "prob": round(float(probs[i]), 4),
            "pred_label": pred,
            "true_label": int(te["labels"][i]),
            "top_similar_peers": peer_objs(te["all_peer_ids"][i]),
            "top_long_peers":  peer_objs(te["pos_peer_ids"][i]),
            "top_short_peers": peer_objs(te["neg_peer_ids"][i]),
            "patient": {f: round(float(patient_raw[i, k]), 3) for k, f in enumerate(feats)},
            "long_prototype":  {f: round(float(pos_raw[i, k]), 3) for k, f in enumerate(feats)},
            "short_prototype": {f: round(float(neg_raw[i, k]), 3) for k, f in enumerate(feats)},
            "largest_dev_vs_long":  top_dev(delta_pos_raw[i], te["X"][i] - te["pos_proto"][i]),
            "largest_dev_vs_short": top_dev(delta_neg_raw[i], te["X"][i] - te["neg_proto"][i]),
            "top_contributions": top_contrib,
            "cxr_support": cxr_support(i, pred),
            "support": support,
            "wrong_reasons": wrong_reasons,
            "patient_info": patient_info(int(sid), fc),
            "peer_group": peer_group_outcomes(te["all_peer_ids"][i], fc),
        })

    C.EXPORT_DIR.mkdir(exist_ok=True)
    base = C.export_path()
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(records, f, indent=2)

    # Flat parquet: scalars as columns, nested fields JSON-encoded so it loads cleanly.
    flat = pd.DataFrame([{
        "stay_id": r["stay_id"], "prob": r["prob"],
        "pred_label": r["pred_label"], "true_label": r["true_label"],
        "top_similar_peers": json.dumps(r["top_similar_peers"]),
        "top_long_peers": json.dumps(r["top_long_peers"]),
        "top_short_peers": json.dumps(r["top_short_peers"]),
        "top_contributions": json.dumps(r["top_contributions"]),
        "cxr_support": json.dumps(r["cxr_support"]),
        "largest_dev_vs_long": json.dumps(r["largest_dev_vs_long"]),
        "largest_dev_vs_short": json.dumps(r["largest_dev_vs_short"]),
        "support": json.dumps(r["support"]),
        "wrong_reasons": json.dumps(r["wrong_reasons"]),
        "patient_info": json.dumps(r["patient_info"]),
        "peer_group": json.dumps(r["peer_group"]),
    } for r in records])
    flat.to_parquet(base.with_suffix(".parquet"))

    # ── Global summary for the dashboard's overview page ──────────────────────
    import collections
    from sklearn.metrics import f1_score
    mean_abs = np.abs(contribs).mean(axis=0)                 # (in_dim,)
    feat_imp = collections.defaultdict(float)
    for j, lab in enumerate(labels_in):
        feat_imp[lab.split(":")[1]] += float(mean_abs[j])    # sum Δpos+Δneg per feature
    norm_pos = np.linalg.norm(te["X"] - te["pos_proto"], axis=1)
    norm_neg = np.linalg.norm(te["X"] - te["neg_proto"], axis=1)
    with open(C.metrics_path()) as f:
        metrics = json.load(f)
    global_summary = {
        "window_hours": W,
        "n_test": int(len(records)),
        "prevalence": round(float(te["labels"].mean()), 4),
        "threshold": float(thr),
        "metrics": metrics["test_torch"],
        "feature_importance": sorted(
            [{"feature": k, "importance": round(v, 4)} for k, v in feat_imp.items()],
            key=lambda d: -d["importance"]),
        "input_importance": sorted(
            [{"input": labels_in[j], "mean_abs_contribution": round(float(mean_abs[j]), 4)}
             for j in range(len(labels_in))], key=lambda d: -d["mean_abs_contribution"]),
        "distance_view_f1": round(float(f1_score(te["labels"], (norm_pos < norm_neg).astype(int))), 4),
        "difference_view_f1": round(float(f1_score(te["labels"], preds)), 4),
    }
    with open(C.EXPORT_DIR / f"fd_global_{W}h.json", "w") as f:
        json.dump(global_summary, f, indent=2)

    print(f"  {len(records):,} records -> {base.with_suffix('.json').name} / "
          f"{base.with_suffix('.parquet').name}")
    print(f"  global summary -> fd_global_{W}h.json")
    print("Done.")
