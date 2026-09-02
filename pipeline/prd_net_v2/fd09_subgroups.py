"""
PRD-Net v2 — fd09: Subgroup and Fairness Breakdown
===================================================
Splits the test-set predictions of the reported feature-difference model by the
demographic and administrative attributes recorded in the static feature table
and reports, per subgroup, the threshold-free discrimination (AUROC, AUPRC) and
the operating point the model actually runs at (recall, precision, selection
rate at the tuned threshold).

Why these measures:
  AUROC/AUPRC say whether the model ranks equally well inside each group.
  Recall gap  = equal-opportunity difference: does the model find prolonged
                stays as often for one group as for another?
  Precision   = predictive parity: is a flag worth the same in each group?
  Selection   = share flagged; a group flagged far more often bears more of
                the review burden regardless of whether the flags are correct.

Prevalence differs sharply between these groups, so accuracy is not comparable
across rows and is not reported. Groups smaller than MIN_N are listed with
their size but their metrics are suppressed, because a rate estimated on a few
dozen patients carries a confidence interval wider than any gap it would show.

None of the attributes below is an input to PRD-Net: the difference block holds
only vital-sign statistics, age and the radiograph indicators, and the absolute
block only diagnosis, unit and admission type. Sex, ethnicity, insurance and
language are read here purely as grouping variables.

Run AFTER: fd06_dashboard-export.py (needs the per-patient export).

Input:   prd_net_v2/exports/fd_explanations_test_{w}h.parquet
         output/X_test.parquet
Output:  prd_net_v2/exports/fd_subgroups_{w}h.json
         prd_net_v2/exports/fd_subgroups_{w}h.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, roc_auc_score,
)

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
import config_fd as C

# Groups below this size get a row with n only; rates are not estimated.
MIN_N = 100

# Percentile bootstrap over the widest contrast on each axis. Without it a gap
# of a few points cannot be told apart from resampling noise, and the subgroup
# table would invite conclusions it does not support.
N_BOOT    = 2000
BOOT_SEED = 0

# (axis, label, boolean mask builder) — one-hot columns are read as indicators,
# age is bucketed, and gender_male is expanded to both levels.
AXES = {
    "Sex": [
        ("Female", lambda X: X["gender_male"] == 0),
        ("Male",   lambda X: X["gender_male"] == 1),
    ],
    "Age band": [
        ("18-49", lambda X: X["age"] < 50),
        ("50-64", lambda X: (X["age"] >= 50) & (X["age"] < 65)),
        ("65-79", lambda X: (X["age"] >= 65) & (X["age"] < 80)),
        ("80+",   lambda X: X["age"] >= 80),
    ],
    "Ethnicity": [
        ("White",    lambda X: X["eth_white"] == 1),
        ("Black",    lambda X: X["eth_black"] == 1),
        ("Hispanic", lambda X: X["eth_hispanic"] == 1),
        ("Asian",    lambda X: X["eth_asian"] == 1),
        ("Other",    lambda X: X["eth_other"] == 1),
    ],
    "Insurance": [
        ("Medicare", lambda X: X["ins_medicare"] == 1),
        ("Medicaid", lambda X: X["ins_medicaid"] == 1),
        ("Other",    lambda X: X["ins_other"] == 1),
    ],
    "Language": [
        ("English",     lambda X: X["language_english"] == 1),
        ("Non-English", lambda X: X["language_english"] == 0),
    ],
    "Admission type": [
        ("Emergency",   lambda X: X["adm_emergency"] == 1),
        ("Urgent",      lambda X: X["adm_urgent"] == 1),
        ("Elective",    lambda X: X["adm_elective"] == 1),
        ("Observation", lambda X: X["adm_observation"] == 1),
    ],
    "ICU type": [
        ("MICU",       lambda X: X["icu_micu"] == 1),
        ("SICU",       lambda X: X["icu_sicu"] == 1),
        ("CCU",        lambda X: X["icu_ccu"] == 1),
        ("CVICU",      lambda X: X["icu_cvicu"] == 1),
        ("MICU/SICU",  lambda X: X["icu_micu_sicu"] == 1),
        ("TSICU",      lambda X: X["icu_tsicu"] == 1),
        ("Neuro SICU", lambda X: X["icu_neuro_sicu"] == 1),
        ("Other",      lambda X: X[C.ICU_COLS].sum(axis=1) == 0),
    ],
}


def group_metrics(y, p, yhat):
    """Threshold-free discrimination plus the realised operating point."""
    out = {
        "n":         int(len(y)),
        "prevalence": float(y.mean()),
        "selection":  float(yhat.mean()),
    }
    if len(y) < MIN_N or y.nunique() < 2:
        out.update(auroc=None, auprc=None, recall=None, precision=None)
        return out
    out.update(
        auroc     = float(roc_auc_score(y, p)),
        auprc     = float(average_precision_score(y, p)),
        recall    = float(recall_score(y, yhat, zero_division=0)),
        precision = float(precision_score(y, yhat, zero_division=0)),
    )
    return out


METRIC_FN = {
    "auroc":     lambda y, p, h: roc_auc_score(y, p),
    "auprc":     lambda y, p, h: average_precision_score(y, p),
    "recall":    lambda y, p, h: recall_score(y, h, zero_division=0),
    "precision": lambda y, p, h: precision_score(y, h, zero_division=0),
}


def bootstrap_gap(a, b, metric, rng):
    """Percentile CI for metric(a) - metric(b), resampling each group separately."""
    f = METRIC_FN[metric]
    ya, pa, ha = a.true_label.values, a.prob.values, a.pred_label.values
    yb, pb, hb = b.true_label.values, b.prob.values, b.pred_label.values
    observed = f(ya, pa, ha) - f(yb, pb, hb)
    draws = []
    for _ in range(N_BOOT):
        ia = rng.integers(0, len(ya), len(ya))
        ib = rng.integers(0, len(yb), len(yb))
        try:
            draws.append(f(ya[ia], pa[ia], ha[ia]) - f(yb[ib], pb[ib], hb[ib]))
        except ValueError:      # a resample with one class only
            continue
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"observed": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_draws": len(draws)}


if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd09 — subgroup breakdown  (window={W})")

    export = Path(str(C.export_path()) + ".parquet")
    if not export.exists():
        raise FileNotFoundError(f"{export} missing — run fd06_dashboard-export.py first.")
    pred = pd.read_parquet(export)

    X = pd.read_parquet(C.OUTPUT_DIR / "X_test.parquet") if hasattr(C, "OUTPUT_DIR") \
        else pd.read_parquet(Path(__file__).parent.parent / "output" / "X_test.parquet")
    df = pred.merge(X, on="stay_id", how="left", validate="one_to_one")
    if len(df) != len(pred):
        raise ValueError("static table did not cover every exported test stay")

    thr = float(json.loads((C.metrics_path()).read_text())["threshold"])
    print(f"  test stays {len(df):,}   threshold {thr:.2f}   "
          f"overall prevalence {df.true_label.mean():.4f}")

    overall = group_metrics(df.true_label, df.prob.values, df.pred_label.values)
    rows = [{"axis": "All", "group": "All test stays", **overall}]

    for axis, levels in AXES.items():
        for label, mask_fn in levels:
            m = mask_fn(df).values
            if m.sum() == 0:
                continue
            s = df[m]
            rows.append({"axis": axis, "group": label,
                         **group_metrics(s.true_label, s.prob.values, s.pred_label.values)})

    table = pd.DataFrame(rows)

    # ── Largest gap per axis, over the groups that cleared MIN_N ──────────────
    gaps = {}
    for axis in AXES:
        sub = table[(table.axis == axis) & table.auroc.notna()]
        if len(sub) < 2:
            continue
        gaps[axis] = {
            metric: {
                "spread": float(sub[metric].max() - sub[metric].min()),
                "best":   sub.loc[sub[metric].idxmax(), "group"],
                "worst":  sub.loc[sub[metric].idxmin(), "group"],
            }
            for metric in ("auroc", "recall", "precision", "selection")
        }

    # ── Bootstrap the widest AUROC contrast on each axis ──────────────────────
    rng = np.random.default_rng(BOOT_SEED)
    boot = {}
    for axis, g in gaps.items():
        hi_name, lo_name = g["auroc"]["best"], g["auroc"]["worst"]
        hi_mask = dict(AXES[axis])[hi_name](df).values
        lo_mask = dict(AXES[axis])[lo_name](df).values
        boot[axis] = {
            "contrast": f"{lo_name} minus {hi_name}",
            **{metric: bootstrap_gap(df[lo_mask], df[hi_mask], metric, rng)
               for metric in ("auroc", "auprc", "recall", "precision")},
        }

    C.EXPORT_DIR.mkdir(exist_ok=True)
    csv_path  = C.EXPORT_DIR / f"fd_subgroups_{W}h.csv"
    json_path = C.EXPORT_DIR / f"fd_subgroups_{W}h.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(
        {"window_hours": W, "threshold": thr, "min_n": MIN_N,
         "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
         "n_test": int(len(df)), "overall": overall,
         "rows": rows, "gaps": gaps, "bootstrap": boot}, indent=2))

    # ── Report ────────────────────────────────────────────────────────────────
    fmt = lambda v: "  --  " if v is None else f"{v:.3f}"
    print(f"\n{'axis':<15}{'group':<14}{'n':>6}{'prev':>8}{'AUROC':>8}"
          f"{'AUPRC':>8}{'recall':>8}{'prec':>8}{'flag%':>8}")
    print("─" * 83)
    last = None
    for r in rows:
        axis = "" if r["axis"] == last else r["axis"]
        last = r["axis"]
        print(f"{axis:<15}{r['group']:<14}{r['n']:>6}{r['prevalence']:>8.3f}"
              f"{fmt(r['auroc']):>8}{fmt(r['auprc']):>8}{fmt(r['recall']):>8}"
              f"{fmt(r['precision']):>8}{r['selection']:>8.3f}")

    print(f"\nWidest AUROC contrast per axis, {N_BOOT} bootstrap resamples "
          f"(95 % percentile CI; * = excludes zero):")
    for axis, b in boot.items():
        print(f"  {axis} — {b['contrast']}")
        for metric in ("auroc", "auprc", "recall", "precision"):
            r = b[metric]
            print(f"      {metric:<10}{r['observed']:+.4f}  "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  f"{'  *' if r['excludes_zero'] else ''}")

    print(f"\nSaved: {csv_path.name}, {json_path.name}")
