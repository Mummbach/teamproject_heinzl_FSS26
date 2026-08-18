"""
ICU Monitoring Intensity Index (MII)
====================================

Creates feature-level monitoring intensity metrics for time-series features, based on the first 48 hours of ICU stay.

For each stay × feature:
    - frequency      (# measurements)
    - density        (# measurements / 48h)
    - coverage       (% ICU hours containing ≥1 measurement)
    - irregularity   (SD of inter-measurement gaps)

Input:   output/cohort.csv

Outputs:
    output/mii_features.parquet
    output/mii_summary_statistics.csv
    output/mii_heatmap.png
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # pipeline/ -> config.py / multimodal_utils.py
from config import (
    ICU_DIR,
    OUTPUT_DIR,
    OBS_WINDOW,
    VITAL_ITEMIDS,
    URINE_ITEMIDS
)

# ============================================================
# LOAD COHORT
# ============================================================

print("Loading cohort...")

cohort = pd.read_csv(
    OUTPUT_DIR / "cohort.csv",
    parse_dates=["intime"]
)

stay_set = set(cohort["stay_id"])

intime_map = (
    cohort
    .set_index("stay_id")["intime"]
    .to_dict()
)

# ============================================================
# EVENT LOADER
# ============================================================

def load_events(
    file_path,
    itemids,
    feature_map=None,
    default_feature=None
):

    reader = pd.read_csv(
        file_path,
        usecols=["stay_id", "charttime", "itemid"],
        dtype={"stay_id": "Int64", "itemid": int},
        compression="gzip",
        chunksize=5_000_000
    )

    chunks = []

    for chunk in reader:

        chunk = chunk[chunk["stay_id"].isin(stay_set)]
        chunk = chunk[chunk["itemid"].isin(itemids)]

        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(
            chunk["charttime"],
            errors="coerce"
        )

        chunk = chunk.dropna()

        chunk["hour"] = (
            (
                chunk["charttime"]
                - chunk["stay_id"].map(intime_map)
            )
            .dt.total_seconds()
            / 3600
        )

        chunk = chunk[
            (chunk["hour"] >= 0)
            & (chunk["hour"] < OBS_WINDOW)
        ]

        if feature_map is not None:
            chunk["feature"] = chunk["itemid"].map(
                lambda x: feature_map[x][0]
            )
        else:
            chunk["feature"] = default_feature

        chunks.append(
            chunk[["stay_id", "feature", "hour"]]
        )

    if len(chunks) == 0:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)

# ============================================================
# LOAD EVENTS
# ============================================================

print("Loading ICU events...")

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

events = pd.concat(
    [vitals, urine],
    ignore_index=True
)

print(f"Total events: {len(events):,}")

# ============================================================
# COMPUTE MII
# ============================================================

def compute_mii(group):

    # --------------------------------------------------------
    # FREQUENCY 
    # --------------------------------------------------------

    hours = np.sort(group["hour"].values)

    freq = len(hours)

    hour_bins = np.floor(hours).astype(int)

    active_hours = len(np.unique(hour_bins))

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    coverage = active_hours / OBS_WINDOW

    # --------------------------------------------------------
    # DENSITY
    # --------------------------------------------------------

    density = freq / OBS_WINDOW

    # --------------------------------------------------------
    # IRREGULARITY
    # --------------------------------------------------------

    if len(hours) > 1:

        gaps = np.diff(hours)

        irregularity = np.std(gaps)

    else:

        irregularity = 0

    return pd.Series({
        "freq": freq,
        "density": density,
        "coverage": coverage,
        "irregularity": irregularity
    })

print("Computing MII...")

mii = (
    events
    .groupby(["stay_id", "feature"])
    .apply(compute_mii)
    .reset_index()
)

# ============================================================
# WIDE FEATURE MATRIX
# ============================================================

mii_wide = mii.pivot_table(
    index="stay_id",
    columns="feature",
    values=[
        "freq",
        "density",
        "coverage",
        "irregularity"
    ]
)

mii_wide.columns = [
    f"{feat}_{metric}"
    for metric, feat in mii_wide.columns
]

mii_wide = (
    mii_wide
    .reset_index()
    .fillna(0)
)

# ============================================================
# SAVE FEATURE MATRIX
# ============================================================

mii_wide.to_parquet(
    OUTPUT_DIR / "mii_features.parquet",
    index=False
)

print(
    f"Saved: {OUTPUT_DIR / 'mii_features.parquet'}"
)

# ============================================================
# SUMMARY TABLE
# ============================================================

print("Creating summary statistics...")

rows = []

feature_cols = [
    c for c in mii_wide.columns
    if c != "stay_id"
]

for col in feature_cols:

    x = mii_wide[col]

    available = x > 0

    rows.append({

        "feature": col,

        "mean": x.mean(),
        "std": x.std(),

        "median": x.median(),

        "q1": x.quantile(0.25),
        "q3": x.quantile(0.75),

        "min": x.min(),
        "max": x.max(),

        "available_n": available.sum(),
        "available_pct": 100 * available.mean()
    })

summary = pd.DataFrame(rows)

summary["Mean ± SD"] = (
    summary["mean"].round(2).astype(str)
    + " ± "
    + summary["std"].round(2).astype(str)
)

summary["Median (IQR)"] = (
    summary["median"].round(2).astype(str)
    + " ("
    + summary["q1"].round(2).astype(str)
    + "–"
    + summary["q3"].round(2).astype(str)
    + ")"
)

summary["Range"] = (
    summary["min"].round(2).astype(str)
    + "–"
    + summary["max"].round(2).astype(str)
)

summary["Availability"] = (
    summary["available_n"].astype(int).astype(str)
    + " ("
    + summary["available_pct"].round(1).astype(str)
    + "%)"
)

summary = summary[
    [
        "feature",
        "Mean ± SD",
        "Median (IQR)",
        "Range",
        "Availability"
    ]
]

summary = summary.sort_values("feature")

summary.to_csv(
    OUTPUT_DIR / "mii_summary_statistics.csv",
    index=False
)

print(
    f"Saved: {OUTPUT_DIR / 'mii_summary_statistics.csv'}"
)

# ============================================================
# VISUALIZATION
# ============================================================

print("Creating heatmap...")

heatmap_df = pd.DataFrame({

    "freq":
        mii_wide.filter(like="_freq").median(),

    "density":
        mii_wide.filter(like="_density").median(),

    "coverage":
        mii_wide.filter(like="_coverage").median(),

    "irregularity":
        mii_wide.filter(like="_irregularity").median()
})

heatmap_df.index = (
    heatmap_df.index
    .str.replace("_freq", "", regex=False)
    .str.replace("_density", "", regex=False)
    .str.replace("_coverage", "", regex=False)
    .str.replace("_irregularity", "", regex=False)
)

heatmap_df = heatmap_df.groupby(level=0).mean()

plt.figure(figsize=(10, 10))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".2f",
    cmap="viridis"
)

plt.title(
    "ICU Monitoring Intensity Index"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "mii_heatmap.png",
    dpi=300
)

plt.close()

print(
    f"Saved: {OUTPUT_DIR / 'mii_heatmap.png'}"
)

print("\nDone.")