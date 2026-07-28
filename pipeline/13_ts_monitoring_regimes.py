"""
ICU Feature Monitoring Regimes + Visualizations
================================================

Pipeline:

1. Load cohort
2. Extract ICU events (chartevents + outputevents)
3. Compute temporal monitoring regimes
4. Generate summary
5. Save outputs
6. Create visualisations

Input:   output/cohort.csv

Outputs:
    output/feature_monitoring_regimes.parquet
    output/feature_monitoring_summary.csv
    output/figures/coverage_heatmap.png
    output/figures/measurement_gaps.png
    output/figures/monitoring_regimes.png
    output/figures/coverage_gap_scatter.png
    output/figures/monitoring_heatmap.png   

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import ICU_DIR, OUTPUT_DIR, OBS_WINDOW, VITAL_ITEMIDS, URINE_ITEMIDS


# ============================================================
# SETUP
# ============================================================

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


# ============================================================
# LOAD COHORT
# ============================================================

print("Loading cohort...")

cohort = pd.read_csv(
    OUTPUT_DIR / "cohort.csv",
    parse_dates=["intime"]
)

stay_ids = set(cohort["stay_id"])
intime_map = cohort.set_index("stay_id")["intime"].to_dict()

print(f"  {len(cohort):,} ICU stays")


# ============================================================
# EVENT LOADER
# ============================================================

def load_events(file_path, itemids, feature_map=None, default_feature=None):

    reader = pd.read_csv(
        file_path,
        usecols=["stay_id", "charttime", "itemid"],
        dtype={"stay_id": "Int64", "itemid": int},
        compression="gzip",
        chunksize=1_000_000
    )

    chunks = []

    for chunk in reader:

        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        chunk = chunk[chunk["itemid"].isin(itemids)]

        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna()

        chunk["hour"] = (
            (chunk["charttime"] - chunk["stay_id"].map(intime_map))
            .dt.total_seconds() / 3600
        )

        chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] < OBS_WINDOW)]

        if feature_map is not None:
            chunk["feature"] = chunk["itemid"].map(lambda x: feature_map[x][0])
        else:
            chunk["feature"] = default_feature

        chunks.append(chunk[["stay_id", "feature", "hour"]])

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ============================================================
# LOAD EVENTS
# ============================================================

print("\nLoading ICU events...")

vitals = load_events(
    ICU_DIR / "chartevents.csv.gz",
    VITAL_ITEMIDS,
    feature_map=VITAL_ITEMIDS
)

urine = load_events(
    ICU_DIR / "outputevents.csv.gz",
    URINE_ITEMIDS,
    default_feature="urine_output"
)

events = pd.concat([vitals, urine], ignore_index=True)
events = events.dropna()
events["hour"] = events["hour"].astype(int)

print(f"Total events: {len(events):,}")


# ============================================================
# COMPUTE MONITORING REGIMES
# ============================================================

print("\nComputing monitoring regimes...")

results = []

for (sid, feat), g in events.groupby(["stay_id", "feature"]):

    hours = np.sort(g["hour"].unique())

    if len(hours) == 0:
        continue

    coverage = len(hours) / OBS_WINDOW

    if len(hours) > 1:
        gaps = np.diff(hours)
        mean_gap = gaps.mean()
        max_gap = gaps.max()
    else:
        mean_gap = OBS_WINDOW
        max_gap = OBS_WINDOW

    sampling_rate = len(hours) / OBS_WINDOW

    # regime classification
    if mean_gap <= 1.5:
        regime = "continuous"
    elif mean_gap <= 4:
        regime = "frequent"
    elif mean_gap <= 12:
        regime = "intermittent"
    elif mean_gap <= 24:
        regime = "sparse"
    else:
        regime = "episodic"

    results.append([
        sid, feat,
        len(hours),
        coverage,
        mean_gap,
        max_gap,
        sampling_rate,
        regime
    ])

reg = pd.DataFrame(results, columns=[
    "stay_id", "feature",
    "n_observations",
    "coverage",
    "mean_gap_hours",
    "max_gap_hours",
    "sampling_rate",
    "regime"
])


# ============================================================
# SUMMARY
# ============================================================

summary = reg.groupby("feature").agg(
    mean_coverage=("coverage", "mean"),
    mean_gap=("mean_gap_hours", "mean"),
    max_gap=("max_gap_hours", "mean"),
    continuous_rate=("regime", lambda x: (x == "continuous").mean()),
    intermittent_rate=("regime", lambda x: (x == "intermittent").mean()),
    sparse_rate=("regime", lambda x: (x == "sparse").mean()),
    n_stays=("stay_id", "count")
).reset_index()

summary = summary.sort_values("mean_coverage", ascending=False)


# ============================================================
# SAVE OUTPUTS
# ============================================================

print("\nSaving outputs...")

reg.to_parquet(
    OUTPUT_DIR / "feature_monitoring_regimes.parquet",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "feature_monitoring_summary.csv",
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nICU Monitoring Regimes")
print("=" * 80)

for _, r in summary.iterrows():
    print(
        f"{r['feature']:<16} "
        f"cov={r['mean_coverage']:.2f} "
        f"gap={r['mean_gap']:.1f}h "
        f"cont={r['continuous_rate']:.2f} "
        f"sparse={r['sparse_rate']:.2f}"
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

df = summary.copy().sort_values("mean_coverage", ascending=False)


# ------------------------------------------------------------
# Coverage barplot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="mean_coverage",
    y="feature",
    palette="viridis"
)

plt.xlabel("Coverage")
plt.ylabel("")
plt.title("ICU Monitoring Coverage (First 48h)")

plt.xlim(0, 1)

plt.savefig(FIG_DIR / "coverage_heatmap.png")
plt.close()


# ------------------------------------------------------------
# Mean gap plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="mean_gap",
    y="feature",
    palette="magma"
)

plt.xlabel("Mean Gap Between Measurements (hours)")
plt.ylabel("")
plt.title("ICU Temporal Gaps")

plt.savefig(FIG_DIR / "measurement_gaps.png")
plt.close()


# ------------------------------------------------------------
# Continuous vs sparse
# ------------------------------------------------------------

plot_df = df.melt(
    id_vars="feature",
    value_vars=["continuous_rate", "sparse_rate"],
    var_name="metric",
    value_name="value"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=plot_df,
    x="feature",
    y="value",
    hue="metric"
)

plt.xticks(rotation=45, ha="right")
plt.title("Continuous vs Sparse Monitoring")

plt.legend(title="Regime")

plt.savefig(FIG_DIR / "monitoring_regimes.png")
plt.close()


# ------------------------------------------------------------
# Coverage vs gap scatter
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="mean_coverage",
    y="mean_gap",
    s=120
)

for _, r in df.iterrows():
    plt.text(
        r["mean_coverage"] + 0.01,
        r["mean_gap"],
        r["feature"],
        fontsize=9
    )

plt.xlabel("Coverage")
plt.ylabel("Mean Gap (hours)")
plt.title("Monitoring Regime Space")

plt.xlim(0, 1.05)

plt.savefig(FIG_DIR / "coverage_gap_scatter.png")
plt.close()


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------

heatmap_df = df.set_index("feature")[
    ["mean_coverage", "mean_gap", "continuous_rate", "sparse_rate"]
]

plt.figure(figsize=(8, 6))

sns.heatmap(
    heatmap_df,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("ICU Monitoring Structure Heatmap")

plt.savefig(FIG_DIR / "monitoring_heatmap.png")
plt.show()
plt.close()


# ============================================================
# DONE
# ============================================================

print("\nSaved outputs:")
print(" - feature_monitoring_regimes.parquet")
print(" - feature_monitoring_summary.csv")
print(" - figures in output/figures/")