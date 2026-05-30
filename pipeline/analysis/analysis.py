"""
Data Analysis
Exploratory analysis of the cohort and prepared features.
Run on training data only (where statistics are involved) to avoid leakage.

Sections:
  1. Cohort overview        — raw MIMIC-IV table shapes, missing values, distributions
  2. Label distribution     — LOS buckets, positive rate per split
  3. Feature analysis       — missing values, binary rates, numeric distributions
  4. Correlations           — top feature correlations with los_gt7 (train only)
  5. Planned vs. Emergency  — LOS distribution and positive rate by admission type

Output:     output/analysis/   — txt reports + PNG plots
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config import OUTPUT_DIR, HOSP_DIR, ICU_DIR

ANALYSIS_DIR = Path(__file__).parent / "output"
ANALYSIS_DIR.mkdir(exist_ok=True)

SEP  = "=" * 70
SEP2 = "-" * 50


# COHORT OVERVIEW (raw tables)
print("Cohort overview")

cohort = pd.read_csv(OUTPUT_DIR / "cohort.csv", parse_dates=["intime", "outtime"])
print(f"\nCohort: {len(cohort):,} stays\n")

# LOS distribution
los = cohort["los"]
print("LOS (days):")
print(f"  min    : {los.min():.1f}")
print(f"  median : {los.median():.1f}")
print(f"  mean   : {los.mean():.1f}")
print(f"  max    : {los.max():.1f}")
print(f"  std    : {los.std():.1f}")

buckets = [(0, 3, "<3d"), (3, 5, "3-5d"), (5, 7, "5-7d"),
           (7, 14, "7-14d"), (14, 30, "14-30d"), (30, 90, "30-90d")]
print(f"\nLOS buckets:")
for lo, hi, label in buckets:
    n = ((los >= lo) & (los < hi)).sum()
    print(f"  {label:<10} {n:>7,}  ({n/len(los)*100:.1f}%)")

# Plot LOS distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(cohort["los"].clip(upper=30), bins=60, color="#6fa8d4", edgecolor="white")
ax.axvline(7, color="red", linestyle="--", linewidth=1.5, label="7-day threshold")
ax.set_title("ICU Length of Stay distribution (clipped at 30d)")
ax.set_xlabel("LOS (days)")
ax.set_ylabel("Stays")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
fig.savefig(ANALYSIS_DIR / "los_distribution.png", dpi=120)
plt.close(fig)
print(f"\n  Plot saved: analysis/los_distribution.png")

# Missing values in cohort
missing = cohort.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\nMissing values in cohort.csv ({len(missing)} columns affected):")
if missing.empty:
    print("  None")
else:
    for col, cnt in missing.items():
        print(f"  {col:<35} {cnt:>7,}  ({cnt/len(cohort)*100:.1f}%)")

# Demographic breakdowns
print(f"\nGender distribution:")
for val, cnt in cohort["gender"].value_counts().items():
    print(f"  {val:<10} {cnt:>7,}  ({cnt/len(cohort)*100:.1f}%)")

print(f"\nICU unit distribution:")
for val, cnt in cohort["first_careunit"].value_counts().items():
    print(f"  {val:<50} {cnt:>7,}  ({cnt/len(cohort)*100:.1f}%)")

print(f"\nAdmission type distribution:")
for val, cnt in cohort["admission_type"].value_counts().items():
    print(f"  {val:<40} {cnt:>7,}  ({cnt/len(cohort)*100:.1f}%)")


# LABEL DISTRIBUTION
print("Label distribution")

labels = pd.read_parquet(OUTPUT_DIR / "labels.parquet")
split_ids = pd.read_parquet(OUTPUT_DIR / "split_ids.parquet")

n_pos = int(labels["los_gt7"].sum())
n_neg = int((labels["los_gt7"] == 0).sum())
n_tot = len(labels)
ratio = n_neg / n_pos if n_pos > 0 else float("inf")

print(f"\n  los_gt7 = 1 (> 7 days)  : {n_pos:>7,}  ({n_pos/n_tot*100:.1f}%)")
print(f"  los_gt7 = 0 (<= 7 days) : {n_neg:>7,}  ({n_neg/n_tot*100:.1f}%)")
print(f"  Negative / Positive ratio: {ratio:.1f}:1")

if ratio < 1.5:
    print("\n  Balance: roughly balanced — no special handling needed.")
elif ratio < 4:
    print("\n  Balance: moderately imbalanced — recommended:")
    print("    class_weight='balanced' + stratified split + AUC-ROC metric")
else:
    print("\n  Balance: strongly imbalanced — recommended:")
    print("    class_weight='balanced' + SMOTE (train only) + PR-AUC metric")

print(f"\n  Per-split positive rates:")
for s in ["train", "val", "test"]:
    ids = set(split_ids[split_ids["split"] == s]["stay_id"])
    y   = labels[labels["stay_id"].isin(ids)]["los_gt7"]
    print(f"  {s:<6}: {len(y):>7,} stays  positive rate: {y.mean()*100:.1f}%")

print(f"\n  Primary diagnosis (top 10):")
for diag, cnt in labels["primary_diag"].value_counts().head(10).items():
    print(f"  {diag:<25} {cnt:>7,}  ({cnt/n_tot*100:.1f}%)")


# FEATURE ANALYSIS (training data only)
print("Feature analysis (X_train)")

X_train_path = OUTPUT_DIR / "X_train.parquet"
if not X_train_path.exists():
    print("  X_train.parquet not found — run 04_preprocessing.py first.")
else:
    X_train = pd.read_parquet(X_train_path)
    y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")

    feature_cols = [c for c in X_train.columns if c != "stay_id"]
    print(f"\n  {len(X_train):,} stays × {len(feature_cols)} features")

    # Missing values (should be 0 after imputation)
    missing_train = X_train[feature_cols].isnull().sum()
    missing_train = missing_train[missing_train > 0]
    if missing_train.empty:
        print("  Missing values: none ✓")
    else:
        print(f"  WARNING — {len(missing_train)} columns still have NaN:")
        for col, cnt in missing_train.items():
            print(f"    {col:<40} {cnt:,}")

    # Binary features
    binary_cols = [c for c in feature_cols
                   if set(X_train[c].dropna().unique()).issubset({0, 1})]
    print(f"\n  Binary features ({len(binary_cols)}):")
    print(f"  {'Feature':<40} {'Positive %':>10}")
    print(f"  {SEP2}")
    for col in sorted(binary_cols):
        rate = X_train[col].mean() * 100
        print(f"  {col:<40} {rate:>9.1f}%")

    # Numeric features
    numeric_cols = [c for c in feature_cols if c not in binary_cols]
    if numeric_cols:
        print(f"\n  Numeric features ({len(numeric_cols)}):")
        desc = X_train[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
        desc.columns = ["mean", "std", "min", "median", "max"]
        print(desc.round(2).to_string())

    # Plot binary feature positive rates (grouped)
    demo_binary  = [c for c in binary_cols if any(c.startswith(p) for p in
                    ["gender", "eth_", "ins_", "adm_", "loc_", "icu_",
                     "marital_", "language_"])]
    icd_binary   = [c for c in binary_cols if c.startswith("icd_")]
    atc_binary   = [c for c in binary_cols if c.startswith("atc_")]

    for group_name, cols in [("demographics", demo_binary),
                              ("icd_categories", icd_binary),
                              ("atc_classes", atc_binary)]:
        if not cols:
            continue
        rates = X_train[cols].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(8, max(3, len(rates) * 0.35)))
        ax.barh(rates.index, rates.values, color="#82c28e")
        ax.set_title(f"Positive rates — {group_name} (train)")
        ax.set_xlabel("% of stays with feature = 1")
        ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        fig.savefig(ANALYSIS_DIR / f"rates_{group_name}.png", dpi=120)
        plt.close(fig)
        print(f"\n  Plot saved: analysis/rates_{group_name}.png")


# CORRELATIONS WITH LABEL (training data only)
print("Feature correlations with los_gt7 (X_train)")

X_train_path = OUTPUT_DIR / "X_train.parquet"
if X_train_path.exists():
    X_train = pd.read_parquet(X_train_path)
    y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")

    feature_cols = [c for c in X_train.columns if c != "stay_id"]
    merged = X_train.merge(y_train[["stay_id", "los_gt7"]], on="stay_id")

    corr = merged[feature_cols + ["los_gt7"]].corr()["los_gt7"].drop("los_gt7")
    corr_abs = corr.abs().sort_values(ascending=False)

    print(f"\n  Top 20 features by |correlation| with los_gt7:")
    print(f"  {'Feature':<40} {'Correlation':>12}")
    print(f"  {SEP2}")
    for feat, val in corr_abs.head(20).items():
        direction = "+" if corr[feat] > 0 else "-"
        print(f"  {feat:<40} {direction}{abs(val):>10.3f}")

    # Plot top 20 correlations
    top20 = corr.reindex(corr_abs.head(20).index).sort_values()
    colors = ["#e06c75" if v < 0 else "#61afef" for v in top20.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top20.index, top20.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 20 feature correlations with los_gt7 (train)")
    ax.set_xlabel("Pearson correlation")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "correlations_top20.png", dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved: analysis/correlations_top20.png")

print(f"\n{SEP}")
print("  Done. All outputs in output/analysis/")
print(SEP)


# PLANNED vs. EMERGENCY ADMISSIONS

print("Planned vs. Emergency Admissions")

cohort5 = pd.read_csv(OUTPUT_DIR / "cohort.csv")

# Group raw admission_type values into three categories
emergency_types  = {"EW EMER.", "DIRECT EMER."}
urgent_types     = {"URGENT"}
planned_types    = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
observation_types = {"OBSERVATION ADMIT", "EU OBSERVATION",
                     "DIRECT OBSERVATION", "AMBULATORY OBSERVATION"}

def map_adm_group(val):
    if val in emergency_types:  return "Emergency"
    if val in urgent_types:     return "Urgent"
    if val in planned_types:    return "Planned"
    if val in observation_types: return "Observation"
    return "Other"

cohort5["adm_group"] = cohort5["admission_type"].map(map_adm_group)

# Text summary
print(f"\n  {'Type':<40} {'N':>7}  {'%':>6}  {'los_gt7 rate':>13}  {'Median LOS':>10}")
print(f"  {SEP2}")
for atype, grp in cohort5.groupby("admission_type", sort=False):
    n    = len(grp)
    pct  = n / len(cohort5) * 100
    rate = grp["los_gt7"].mean() * 100
    med  = grp["los"].median()
    print(f"  {atype:<40} {n:>7,}  {pct:>5.1f}%  {rate:>12.1f}%  {med:>9.1f}d")

print(f"\n  Grouped summary:")
print(f"  {'Group':<15} {'N':>7}  {'%':>6}  {'los_gt7 rate':>13}  {'Median LOS':>10}")
print(f"  {SEP2}")
group_order = ["Emergency", "Urgent", "Observation", "Planned"]
for g in group_order:
    grp  = cohort5[cohort5["adm_group"] == g]
    n    = len(grp)
    pct  = n / len(cohort5) * 100
    rate = grp["los_gt7"].mean() * 100
    med  = grp["los"].median()
    print(f"  {g:<15} {n:>7,}  {pct:>5.1f}%  {rate:>12.1f}%  {med:>9.1f}d")

# Hypothesis check
planned = cohort5[cohort5["adm_group"] == "Planned"]
emergency = cohort5[cohort5["adm_group"] == "Emergency"]
urgent = cohort5[cohort5["adm_group"] == "Urgent"]
planned_rate   = planned["los_gt7"].mean() * 100
emergency_rate = emergency["los_gt7"].mean() * 100
urgent_rate    = urgent["los_gt7"].mean() * 100

print(f"\n  Hypothesis: 'few planned admissions have LOS > 7d'")
print(f"  Planned   los_gt7 rate: {planned_rate:.1f}%")
print(f"  Emergency los_gt7 rate: {emergency_rate:.1f}%")
print(f"  Urgent    los_gt7 rate: {urgent_rate:.1f}%")
if planned_rate < emergency_rate and planned_rate < urgent_rate:
    print(f"  → CONFIRMED: planned admissions have lowest positive rate")
    print(f"  → Recommendation: keep admission_type as feature (clear signal)")
    print(f"    Restricting to Emergency/Urgent only would remove {len(planned):,} stays ({len(planned)/len(cohort5)*100:.1f}%)")
    print(f"    and discard a useful feature — not recommended")
else:
    print(f"  → NOT confirmed: planned admissions do not have lower positive rate")

# Plot 1: Positive rate by group
rates = {g: cohort5[cohort5["adm_group"] == g]["los_gt7"].mean() * 100
         for g in group_order}
colors = ["#e06c75", "#e5c07b", "#56b6c2", "#61afef"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(rates.keys(), rates.values(), color=colors, width=0.5)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_ylabel("los_gt7 rate (%)")
ax.set_title("Positive rate (LOS > 7d) by admission group")
ax.set_ylim(0, max(rates.values()) * 1.25)
plt.tight_layout()
fig.savefig(ANALYSIS_DIR / "admission_positive_rate.png", dpi=120)
plt.close(fig)
print(f"\n  Plot saved: analysis/admission_positive_rate.png")

# Plot 2: LOS distribution by group
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
axes = axes.flatten()
for i, (g, color) in enumerate(zip(group_order, colors)):
    grp = cohort5[cohort5["adm_group"] == g]["los"].clip(upper=30)
    axes[i].hist(grp, bins=40, color=color, edgecolor="white", alpha=0.85)
    axes[i].axvline(7, color="red", linestyle="--", linewidth=1.2, label="7d threshold")
    rate = cohort5[cohort5["adm_group"] == g]["los_gt7"].mean() * 100
    axes[i].set_title(f"{g}  (n={len(grp):,}, pos={rate:.1f}%)")
    axes[i].set_xlabel("LOS (days, clipped 30d)")
    axes[i].set_ylabel("Count")
    axes[i].legend(fontsize=8)
plt.suptitle("LOS distribution by admission group", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(ANALYSIS_DIR / "admission_los_distribution.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved: analysis/admission_los_distribution.png")


# AGE ANALYSIS
print(f"\n{SEP}")
print("Age analysis")
print(SEP)

cohort_age = pd.read_csv(OUTPUT_DIR / "cohort.csv")
labels_age = pd.read_parquet(OUTPUT_DIR / "labels.parquet")
cohort_age = cohort_age.merge(labels_age[["stay_id", "los_gt7"]], on="stay_id")

print(f"\n  Age distribution:")
for label, name in [(0, "LOS <= 7d"), (1, "LOS > 7d")]:
    grp = cohort_age[cohort_age["los_gt7"] == label]["age_at_icu"]
    print(f"  {name}: mean={grp.mean():.1f}  median={grp.median():.1f}  std={grp.std():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for label, name, color in [(0, "LOS ≤ 7d", "#61afef"), (1, "LOS > 7d", "#e06c75")]:
    grp = cohort_age[cohort_age["los_gt7"] == label]["age_at_icu"]
    axes[0].hist(grp, bins=30, alpha=0.6, label=name, color=color)
axes[0].set_title("Age distribution by LOS label")
axes[0].set_xlabel("Age at ICU admission")
axes[0].set_ylabel("Stays")
axes[0].legend()

cohort_age.boxplot(column="age_at_icu", by="los_gt7", ax=axes[1], grid=False,
                   boxprops=dict(color="#6fa8d4"), medianprops=dict(color="red"))
axes[1].set_title("Age by LOS label")
axes[1].set_xlabel("los_gt7")
axes[1].set_ylabel("Age")
plt.suptitle("")
plt.tight_layout()
fig.savefig(ANALYSIS_DIR / "age_distribution.png", dpi=120)
plt.close(fig)
print(f"  Plot saved: analysis/age_distribution.png")


# ICU UNIT POSITIVE RATES
print(f"\n{SEP}")
print("ICU unit analysis")
print(SEP)

print(f"\n  {'ICU Unit':<55} {'N':>6}  {'LOS > 7d':>9}  {'Median LOS':>10}")
print(f"  {SEP2}")
unit_stats = cohort_age.groupby("first_careunit").agg(
    n=("stay_id", "count"),
    pos_rate=("los_gt7", "mean"),
    median_los=("los", "median"),
).sort_values("pos_rate", ascending=False)
for unit, row in unit_stats.iterrows():
    print(f"  {unit:<55} {int(row['n']):>6,}  {row['pos_rate']*100:>8.1f}%  {row['median_los']:>9.1f}d")

rates = unit_stats["pos_rate"] * 100
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(range(len(rates)), rates.values, color="#82c28e")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
ax.set_xticks(range(len(rates)))
ax.set_xticklabels([u.split("(")[0].strip() for u in rates.index], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("LOS > 7d rate (%)")
ax.set_title("Positive rate by ICU unit")
plt.tight_layout()
fig.savefig(ANALYSIS_DIR / "icu_unit_positive_rates.png", dpi=120)
plt.close(fig)
print(f"  Plot saved: analysis/icu_unit_positive_rates.png")


# MORTALITY
print(f"\n{SEP}")
print("Mortality in cohort")
print(SEP)

cohort_raw = pd.read_csv(OUTPUT_DIR / "cohort.csv")
n_total = len(cohort_raw)
print(f"\n  Total stays : {n_total:,}")
print(f"  Note: patients dying within 48h excluded during cohort selection.")


# FEATURE COMPLETENESS TABLE
print(f"\n{SEP}")
print("Feature completeness (X_train)")
print(SEP)

X_train_path = OUTPUT_DIR / "X_train.parquet"
if X_train_path.exists():
    X_train = pd.read_parquet(X_train_path)
    feature_cols = [c for c in X_train.columns if c != "stay_id"]

    groups = {
        "Demographics": [c for c in feature_cols if any(c.startswith(p) for p in
                         ["age", "gender", "eth_", "ins_", "adm_", "loc_", "icu_", "marital_", "language_", "year_"])],
        "ICD":          [c for c in feature_cols if c.startswith("icd_")],
        "ATC":          [c for c in feature_cols if c.startswith("atc_")],
        "Vitals":       [c for c in feature_cols if not any(c.startswith(p) for p in
                         ["age", "gender", "eth_", "ins_", "adm_", "loc_", "icu_", "marital_", "language_", "year_", "icd_", "atc_"])],
    }
    print(f"\n  {'Group':<15} {'Features':>9}  {'Avg missing %':>14}  {'Near-zero (>95% zeros)':>22}")
    print(f"  {SEP2}")
    all_near_zero = []
    for group, cols in groups.items():
        if not cols:
            continue
        avg_missing = X_train[cols].isnull().mean().mean() * 100
        binary = [c for c in cols if set(X_train[c].dropna().unique()).issubset({0, 1})]
        near_zero = [c for c in binary if X_train[c].mean() < 0.05]
        all_near_zero.extend(near_zero)
        print(f"  {group:<15} {len(cols):>9,}  {avg_missing:>13.1f}%  {len(near_zero):>22,}")

    if all_near_zero:
        print(f"\n  Near-zero variance features (<5% positive rate):")
        for c in sorted(all_near_zero):
            print(f"    {c:<40} {X_train[c].mean()*100:.2f}%")


# SPLIT DISTRIBUTION COMPARISON
print(f"\n{SEP}")
print("Split distribution comparison")
print(SEP)

if X_train_path.exists():
    split_ids = pd.read_parquet(OUTPUT_DIR / "split_ids.parquet")
    cohort_split = pd.read_csv(OUTPUT_DIR / "cohort.csv").merge(split_ids, on="stay_id")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"train": "#61afef", "val": "#e5c07b", "test": "#e06c75"}
    for split, color in colors.items():
        grp = cohort_split[cohort_split["split"] == split]
        axes[0].hist(grp["age_at_icu"], bins=30, alpha=0.5, label=split, color=color)
        axes[1].hist(grp["los"].clip(upper=30), bins=40, alpha=0.5, label=split, color=color)
    axes[0].set_title("Age distribution across splits")
    axes[0].set_xlabel("Age")
    axes[0].legend()
    axes[1].set_title("LOS distribution across splits (clipped 30d)")
    axes[1].set_xlabel("LOS (days)")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "split_distributions.png", dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved: analysis/split_distributions.png")


# TIME SERIES COVERAGE
print(f"\n{SEP}")
print("Time series coverage")
print(SEP)

ts_path = OUTPUT_DIR / "timeseries.parquet"
if ts_path.exists():
    ts = pd.read_parquet(ts_path)
    labels_ts = pd.read_parquet(OUTPUT_DIR / "labels.parquet")
    ts = ts.merge(labels_ts[["stay_id", "los_gt7"]], on="stay_id")

    vital_cols = [c for c in ts.columns if c not in ["stay_id", "hour", "los_gt7"]]

    coverage = (ts[vital_cols].notna().mean() * 100).sort_values(ascending=False)
    print(f"\n  Vital coverage (% of stay-hours with a value):")
    for feat, pct in coverage.items():
        bar = "█" * int(pct / 5)
        print(f"  {feat:<18} {pct:>5.1f}%  {bar}")

    # Mean trajectory per vital: los_gt7=0 vs 1
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, feat in enumerate(vital_cols[:16]):
        ax = axes[i]
        for label, name, color in [(0, "LOS ≤ 7d", "#61afef"), (1, "LOS > 7d", "#e06c75")]:
            grp = ts[ts["los_gt7"] == label].groupby("hour")[feat].mean()
            ax.plot(grp.index, grp.values, label=name, color=color, linewidth=1.2)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("Hour", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(vital_cols), 16):
        axes[j].set_visible(False)
    plt.suptitle("Mean vital trajectory over 48h (los_gt7=0 vs 1)", fontsize=12)
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "timeseries_trajectories.png", dpi=120)
    plt.close(fig)
    print(f"  Plot saved: analysis/timeseries_trajectories.png")

    # Coverage heatmap: hour × vital
    coverage_matrix = ts.groupby("hour")[vital_cols].apply(lambda x: x.notna().mean())
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(coverage_matrix.T.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels(range(0, 48, 4))
    ax.set_yticks(range(len(vital_cols)))
    ax.set_yticklabels(vital_cols, fontsize=8)
    ax.set_xlabel("Hour after ICU admission")
    ax.set_title("Vital sign coverage heatmap (fraction of stays with measurement)")
    plt.colorbar(im, ax=ax, label="Coverage fraction")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "timeseries_coverage_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"  Plot saved: analysis/timeseries_coverage_heatmap.png")


# CXR COVERAGE BY ICU UNIT
print(f"\n{SEP}")
print("CXR report coverage by ICU unit")
print(SEP)

cxr_path = OUTPUT_DIR / "cxr_bert_embeddings.parquet"
if cxr_path.exists():
    cxr = pd.read_parquet(cxr_path)[["stay_id"]]
    cohort_cxr = pd.read_csv(OUTPUT_DIR / "cohort.csv")
    cohort_cxr["has_cxr"] = cohort_cxr["stay_id"].isin(set(cxr["stay_id"])).astype(int)

    total_cxr = cohort_cxr["has_cxr"].sum()
    print(f"\n  Overall CXR coverage: {total_cxr:,} / {len(cohort_cxr):,} ({total_cxr/len(cohort_cxr)*100:.1f}%)")

    print(f"\n  {'ICU Unit':<55} {'N':>6}  {'With CXR':>9}  {'Coverage':>9}")
    print(f"  {SEP2}")
    for unit, grp in cohort_cxr.groupby("first_careunit"):
        n = len(grp)
        n_cxr = grp["has_cxr"].sum()
        print(f"  {unit:<55} {n:>6,}  {n_cxr:>9,}  {n_cxr/n*100:>8.1f}%")
else:
    print("  cxr_bert_embeddings.parquet not found — run 02b_cxr_features.py first.")

print(f"\n{SEP}")
print("  Done. All outputs in pipeline/analysis/")
print(SEP)
