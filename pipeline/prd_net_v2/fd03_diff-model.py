"""
PRD-Net v2 — fd03: Difference Model
====================================
Assembles the signed per-feature difference vector and defines the classifier.

  delta_pos = patient_scaled - pos_proto      (signed, per feature)
  delta_neg = patient_scaled - neg_proto

Model input (D3, DIFF_INPUT):
  both       -> concat([delta_pos, delta_neg])            dim 2F   (default)
  pos_only   -> delta_pos                                  dim F
  neg_only   -> delta_neg                                  dim F
  proto_gap  -> concat([delta_pos, delta_neg, neg-pos])    dim 3F

Models (D4, MODEL):
  linear -> nn.Linear(in_dim, 1)   (default; logit = w.x + b -> exact attribution)
  mlp    -> one shallow hidden layer (ablation; interpret via aggregated SHAP)

A sklearn LogisticRegression reference fit on the same diff vectors is also
provided — it pairs with shap.LinearExplainer for an exact decomposition that
must match w.delta from the linear torch model.

The deltas are NEVER collapsed to a scalar norm: the explanation is contrastive
and directional ("feature is above/below the prototype"), not a distance.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).parent))
import config_fd as C


# ══════════════════════════════════════════════════════════════════════════════
# DIFF ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble_diff(patient: np.ndarray, pos_proto: np.ndarray,
                  neg_proto: np.ndarray, diff_input: str | None = None,
                  absolute: np.ndarray | None = None) -> np.ndarray:
    """Build the model input from patient/prototype matrices (all (N, F)).

    `absolute` (N, A), if given, is the patient's own hard-filter one-hot
    block (see config_fd.absolute_feature_names()) — concatenated UNCHANGED,
    never diffed against a prototype (a diff would be ~0 by construction
    since peers are matched on it). Load it with `load_absolute_block`.
    """
    diff_input = diff_input or C.DIFF_INPUT
    delta_pos = patient - pos_proto
    delta_neg = patient - neg_proto
    # DESIGN DECISION D3 — diff input assembled here.
    if diff_input == "both":
        out = np.concatenate([delta_pos, delta_neg], axis=1)
    elif diff_input == "pos_only":
        out = delta_pos
    elif diff_input == "neg_only":
        out = delta_neg
    elif diff_input == "proto_gap":
        out = np.concatenate([delta_pos, delta_neg, neg_proto - pos_proto], axis=1)
    else:
        raise ValueError(f"unknown DIFF_INPUT: {diff_input}")
    if absolute is not None:
        out = np.concatenate([out, absolute], axis=1)
    return out


def input_dim(diff_input: str | None = None) -> int:
    diff_input = diff_input or C.DIFF_INPUT
    F = C.n_features()
    base = {"both": 2 * F, "pos_only": F, "neg_only": F, "proto_gap": 3 * F}[diff_input]
    return base + len(C.absolute_feature_names())


def input_feature_labels(diff_input: str | None = None) -> list[str]:
    """Human-readable label per input dimension (for attribution reports)."""
    diff_input = diff_input or C.DIFF_INPUT
    feats = C.feature_names()
    dp = [f"Δpos:{f}" for f in feats]
    dn = [f"Δneg:{f}" for f in feats]
    gap = [f"gap:{f}" for f in feats]
    base = {"both": dp + dn, "pos_only": dp, "neg_only": dn,
            "proto_gap": dp + dn + gap}[diff_input]
    return base + [f"abs:{f}" for f in C.absolute_feature_names()]


def load_bundle(split: str) -> dict:
    """Load fd02's per-split prototype bundle (stay_ids, labels, X, protos, peer ids)."""
    with open(C.prototypes_path(split), "rb") as f:
        return pickle.load(f)


def load_scaler():
    with open(C.scaler_bundle_path(), "rb") as f:
        return pickle.load(f)["scaler"]


def load_trained(split: str, train_split: str = "train") -> dict:
    """Load everything fd05/fd06 need to explain a trained model's predictions
    on `split`: bundles, assembled diff vectors, the checkpointed model +
    threshold, and the exact linear decomposition contribs = w*(x - E_train[x])
    (identical to the SHAP values fd05 verifies). Centralized here so the two
    scripts don't each re-derive the same formula from disk.
    """
    tr = load_bundle(train_split)
    qy = load_bundle(split)
    Xtr = assemble_diff(tr["X"], tr["pos_proto"], tr["neg_proto"],
                        absolute=load_absolute_block(train_split, tr["stay_ids"])).astype(np.float32)
    Xq = assemble_diff(qy["X"], qy["pos_proto"], qy["neg_proto"],
                       absolute=load_absolute_block(split, qy["stay_ids"])).astype(np.float32)

    model = build_model(input_dim())
    model.load_state_dict(torch.load(C.checkpoint_path(), weights_only=True))
    model.eval()
    thr = torch.load(C.threshold_path(), weights_only=True)["threshold"]

    if C.MODEL == "linear":
        coef = model.linear.weight.detach().numpy().ravel()
        bias = float(model.linear.bias.item())
        contribs = coef[None, :] * (Xq - Xtr.mean(axis=0)[None, :])
    else:
        raise NotImplementedError(
            "MLP model does not support exact w*(x-mean) attribution. "
            "Use aggregate SHAP instead, or set MODEL='linear'."
        )

    return {
        "train": tr, "query": qy, "Xtr": Xtr, "Xq": Xq,
        "model": model, "thr": thr, "coef": coef, "bias": bias, "contribs": contribs,
        "scaler": load_scaler(), "labels_in": input_feature_labels(),
        "feats": C.feature_names(), "abs_feats": C.absolute_feature_names(),
    }


def load_absolute_block(split: str, stay_ids) -> np.ndarray | None:
    """Patient's own SCALED absolute hard-filter block, aligned to `stay_ids`
    order. None if USE_ABSOLUTE_FEATURES is off.

    Read straight from fd01's scaled feature matrix — those columns sit
    outside feature_names(), so fd02's prototype building never touches them.
    """
    if not C.USE_ABSOLUTE_FEATURES:
        return None
    m = pd.read_parquet(C.feature_matrix_path(split, scaled=True)).set_index("stay_id")
    return m.loc[list(stay_ids), C.ABSOLUTE_FEATURES].to_numpy(dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class LinearDiffModel(nn.Module):
    """Logistic regression in DL form: a single linear layer -> one logit.

    DESIGN DECISION D4 — pure linear by default so logit = w.x + b and the
    per-feature contribution w_i * x_i is the exact (SHAP) decomposition.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPDiffModel(nn.Module):
    """Shallow one-hidden-layer MLP (D4 ablation). Kept small to stay readable."""

    def __init__(self, in_dim: int, hidden: int = C.MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(in_dim: int, model: str | None = None) -> nn.Module:
    model = model or C.MODEL
    return LinearDiffModel(in_dim) if model == "linear" else MLPDiffModel(in_dim)


def build_logreg():
    """sklearn LogisticRegression reference (class-balanced to mirror pos_weight)."""
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N, F, A = 8, C.n_features(), len(C.absolute_feature_names())
    patient = np.random.randn(N, F).astype(np.float32)
    pos = np.random.randn(N, F).astype(np.float32)
    neg = np.random.randn(N, F).astype(np.float32)
    absolute = np.random.randint(0, 2, size=(N, A)).astype(np.float32)

    for di in ["both", "pos_only", "neg_only", "proto_gap"]:
        x = assemble_diff(patient, pos, neg, di, absolute=absolute)
        labels = input_feature_labels(di)
        assert x.shape == (N, input_dim(di)), di
        assert len(labels) == input_dim(di), di
        m = build_model(input_dim(di), "linear")
        out = m(torch.tensor(x))
        assert out.shape == (N,), di
        print(f"  {di:<10} in_dim={input_dim(di):<4} out={tuple(out.shape)}  OK")

    print(f"F = {F}, default DIFF_INPUT={C.DIFF_INPUT} -> in_dim={input_dim()}")
    print("All shape assertions passed.")
