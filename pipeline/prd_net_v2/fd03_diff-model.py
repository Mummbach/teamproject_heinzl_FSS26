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

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))
import config_fd as C


# ══════════════════════════════════════════════════════════════════════════════
# DIFF ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble_diff(patient: np.ndarray, pos_proto: np.ndarray,
                  neg_proto: np.ndarray, diff_input: str | None = None) -> np.ndarray:
    """Build the model input from patient/prototype matrices (all (N, F))."""
    diff_input = diff_input or C.DIFF_INPUT
    delta_pos = patient - pos_proto
    delta_neg = patient - neg_proto
    # DESIGN DECISION D3 — diff input assembled here.
    if diff_input == "both":
        return np.concatenate([delta_pos, delta_neg], axis=1)
    if diff_input == "pos_only":
        return delta_pos
    if diff_input == "neg_only":
        return delta_neg
    if diff_input == "proto_gap":
        return np.concatenate([delta_pos, delta_neg, neg_proto - pos_proto], axis=1)
    raise ValueError(f"unknown DIFF_INPUT: {diff_input}")


def input_dim(diff_input: str | None = None) -> int:
    diff_input = diff_input or C.DIFF_INPUT
    F = C.n_features()
    return {"both": 2 * F, "pos_only": F, "neg_only": F, "proto_gap": 3 * F}[diff_input]


def input_feature_labels(diff_input: str | None = None) -> list[str]:
    """Human-readable label per input dimension (for attribution reports)."""
    diff_input = diff_input or C.DIFF_INPUT
    feats = C.feature_names()
    dp = [f"Δpos:{f}" for f in feats]
    dn = [f"Δneg:{f}" for f in feats]
    gap = [f"gap:{f}" for f in feats]
    return {"both": dp + dn, "pos_only": dp, "neg_only": dn,
            "proto_gap": dp + dn + gap}[diff_input]


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
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N, F = 8, C.n_features()
    patient = np.random.randn(N, F).astype(np.float32)
    pos = np.random.randn(N, F).astype(np.float32)
    neg = np.random.randn(N, F).astype(np.float32)

    for di in ["both", "pos_only", "neg_only", "proto_gap"]:
        x = assemble_diff(patient, pos, neg, di)
        labels = input_feature_labels(di)
        assert x.shape == (N, input_dim(di)), di
        assert len(labels) == input_dim(di), di
        m = build_model(input_dim(di), "linear")
        out = m(torch.tensor(x))
        assert out.shape == (N,), di
        print(f"  {di:<10} in_dim={input_dim(di):<4} out={tuple(out.shape)}  OK")

    print(f"F = {F}, default DIFF_INPUT={C.DIFF_INPUT} -> in_dim={input_dim()}")
    print("All shape assertions passed.")
