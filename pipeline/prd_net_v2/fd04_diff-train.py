"""
PRD-Net v2 — fd04: Train the Difference Model
==============================================
Trains LinearDiffModel on the assembled diff vectors, mirroring the existing
train harness: BCEWithLogitsLoss(pos_weight), early stopping on val F1, then a
val-only threshold scan (0.10-0.90, maximize F1).

Reports test metrics with AUPRC as the headline number for this imbalanced task,
plus a sklearn LogisticRegression reference fit on the same diff vectors.

Outputs (window-tagged):
  checkpoints/fd_diff_v1_{w}.pt            best torch state_dict
  checkpoints/fd_diff_v1_{w}_threshold.pt  tuned decision threshold
  fd_metrics_{w}.json                      metrics for the side-by-side table

Run AFTER: fd02_feature-prototypes.py.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
import config_fd as C

# fd03 has a digit + hyphen in its name -> importlib (same pattern as prd_net/)
_spec = importlib.util.spec_from_file_location("fd_model", Path(__file__).parent / "fd03_diff-model.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
assemble_diff, input_dim, build_model, build_logreg, load_absolute_block, load_bundle = (
    _m.assemble_diff, _m.input_dim, _m.build_model, _m.build_logreg, _m.load_absolute_block,
    _m.load_bundle)


def make_xy(bundle, split):
    absolute = load_absolute_block(split, bundle["stay_ids"])
    X = assemble_diff(bundle["X"], bundle["pos_proto"], bundle["neg_proto"], absolute=absolute)
    return X.astype(np.float32), bundle["labels"].astype(np.float32)


def find_best_threshold(probs, labels):
    """Scan thresholds in [0.10, 0.90] on val, return (threshold, F1)."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, round(float(t), 2)
    return best_t, best_f1


def evaluate(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auroc":     roc_auc_score(labels, probs),
        "auprc":     average_precision_score(labels, probs),
    }


if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd04 — train difference model  (window={W}, model={C.MODEL}, "
          f"diff_input={C.DIFF_INPUT})")

    Xtr, ytr = make_xy(load_bundle("train"), "train")
    Xva, yva = make_xy(load_bundle("val"), "val")
    Xte, yte = make_xy(load_bundle("test"), "test")
    in_dim = input_dim()
    print(f"  train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  in_dim={in_dim}")

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva)
    Xte_t = torch.tensor(Xte)

    torch.manual_seed(C.SEED)
    model = build_model(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([C.POS_WEIGHT]))

    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True)

    C.CKPT_DIR.mkdir(exist_ok=True)
    best_f1, no_improve = -1.0, 0
    ckpt = C.checkpoint_path()

    print(f"\n{'Epoch':<8}{'TrainLoss':<12}{'ValF1':<10}Best")
    print("─" * 38)
    for epoch in range(1, C.EPOCHS + 1):
        model.train(); total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            vprob = torch.sigmoid(model(Xva_t)).numpy()
        vf1 = f1_score(yva, (vprob >= 0.5).astype(int), zero_division=0)

        is_best = vf1 > best_f1
        if is_best:
            best_f1, no_improve = vf1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
        mark = " ✓" if is_best else f" ({no_improve}/{C.PATIENCE})"
        print(f"{epoch:>3}/{C.EPOCHS} {total/len(loader):<12.4f}{vf1:<10.4f}{mark}")
        if no_improve >= C.PATIENCE:
            print(f"\nEarly stopping — no val-F1 improvement for {C.PATIENCE} epochs.")
            break

    # ── Threshold tuning on val (best checkpoint) ─────────────────────────────
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"No checkpoint written at {ckpt} — the model never improved F1 above 0.0. "
            "Check data, class balance, and pos_weight."
        )
    model.load_state_dict(torch.load(ckpt, weights_only=True)); model.eval()
    with torch.no_grad():
        vprob = torch.sigmoid(model(Xva_t)).numpy()
        tprob = torch.sigmoid(model(Xte_t)).numpy()
    thr, tuned = find_best_threshold(vprob, yva)
    torch.save({"threshold": thr}, C.threshold_path())
    print(f"\nBest val F1 @0.50 = {best_f1:.4f}   |   tuned thr={thr:.2f} -> val F1 {tuned:.4f}")

    # ── Test metrics ──────────────────────────────────────────────────────────
    test_metrics = evaluate(tprob, yte, thr)
    print("\nTest metrics (torch linear model):")
    for k, v in test_metrics.items():
        print(f"  {k:<10}: {v:.4f}")

    # ── sklearn LogisticRegression reference (same diff vectors) ──────────────
    # np.errstate guards a spurious NumPy 1.26 matmul FP-warning ("divide by zero
    # encountered in matmul") that fires on some CPUs even for a plain `A @ w`.
    # It is cosmetic: the fit converges and matches the torch model. Use float64.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        lr = build_logreg().fit(Xtr.astype(np.float64), ytr.astype(int))
        lr_prob  = lr.predict_proba(Xte.astype(np.float64))[:, 1]
        lr_vprob = lr.predict_proba(Xva.astype(np.float64))[:, 1]
    lr_thr, _ = find_best_threshold(lr_vprob, yva)
    lr_metrics = evaluate(lr_prob, yte, lr_thr)
    print("\nTest metrics (sklearn LogisticRegression reference):")
    for k, v in lr_metrics.items():
        print(f"  {k:<10}: {v:.4f}")

    with open(C.metrics_path(), "w") as f:
        json.dump({
            "window_hours": W, "model": C.MODEL, "diff_input": C.DIFF_INPUT,
            "retrieval_space": C.RETRIEVAL_SPACE, "weighting": C.USE_PROTOTYPE_WEIGHTING,
            "weight_decay": C.WEIGHT_DECAY,
            "threshold": thr, "val_f1_best": best_f1,
            "test_torch": test_metrics, "test_logreg": lr_metrics,
        }, f, indent=2)
    print(f"\nSaved metrics -> {C.metrics_path().name}")
