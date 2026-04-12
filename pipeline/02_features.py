"""
Feature Engineering
====================
Transforms raw MIMIC-IV event data into three new feature sets that do
not exist in the source tables — new information is derived from raw codes,
prescriptions, and time-stamped measurements.

Sections:
  1. ICD Diagnoses   — maps ICD-9/10 codes to 18 binary disease categories
  2. Medications     — maps prescriptions (first 48h) to 14 ATC drug classes
  3. Time-series     — aggregates 12 vitals over 48h into summary statistics

Run AFTER:  01_selection.py
Run BEFORE: 03_splitting.py

Input:   output/cohort.csv
         data/hosp/diagnoses_icd.csv.gz
         data/hosp/prescriptions.csv.gz
         data/RXCUI2atc4.csv
         data/icu/chartevents.csv.gz
         data/icu/outputevents.csv.gz

Output:  output/icd_features.parquet    — 18 binary ICD columns per stay
         output/atc_features.parquet    — 14 binary ATC columns per stay
         output/ts_features.parquet     — 12 vitals × summary stats per stay
         output/labels.parquet          — stay_id, los, los_gt7, primary_diag

Notes:
  ATC Level 1 is coarse (1 letter = entire drug class, e.g. N = all neurological).
  ATC Level 2 (2 chars, e.g. N02 = analgesics) gives more granularity at the
  cost of more columns and higher sparsity.

  The .apply() ICD mapping is slow on ~500k rows. Vectorised alternative possible.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    OUTPUT_DIR, HOSP_DIR, ICU_DIR, MAPPING_PATH,
    ICD9_RANGES, ICD10_MAP, ICD_CATEGORIES,
    ATC1_CODES, ATC1_COL_NAMES,
    VITAL_ITEMIDS, URINE_ITEMIDS, RANGE_FILTERS, TS_FEATURES,
    OBS_WINDOW,
)

OUTPUT_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 10_000_000


# HELPERS, ICD code mapping


def map_icd9(code: str) -> str:
    """Map an ICD-9 code to one of 18 disease category names."""
    if not isinstance(code, str) or len(code) == 0:
        return "unknown"
    code = code.strip().upper()
    if code.startswith("V") or code.startswith("E"):
        return "supplementary"
    try:
        num = int(code[:3].replace(".", "").ljust(3, "0")[:3])
    except ValueError:
        return "unknown"
    for lo, hi, label in ICD9_RANGES[:-1]:
        if int(lo) <= num <= int(hi):
            return label
    return "unknown"


def map_icd10(code: str) -> str:
    """Map an ICD-10 code to one of 18 disease category names.

    Tries two-character prefix first to handle D/H codes that span categories.
    """
    if not isinstance(code, str) or len(code) == 0:
        return "unknown"
    code = code.strip().upper()
    if len(code) >= 2 and code[:2] in ICD10_MAP:
        return ICD10_MAP[code[:2]]
    if code[0] in ICD10_MAP:
        return ICD10_MAP[code[0]]
    return "unknown"



# LOAD COHORT (shared across all three sections)

print("Loading cohort...")
cohort          = pd.read_csv(OUTPUT_DIR / "cohort.csv", parse_dates=["intime"])
cohort_stay_set = set(cohort["stay_id"])
hadm_to_stay    = cohort.set_index("hadm_id")["stay_id"].to_dict()
intime_map      = cohort.set_index("stay_id")["intime"].to_dict()
print(f"  {len(cohort):,} stays\n")



# 1 — ICD DIAGNOSES (18 binary categories)

# Raw ICD codes (thousands of distinct values) are transformed into 18 new binary columns.
# The mapping captures clinically meaningful disease groups without creating sparsity.

print("1. ICD Diagnoses → 18 binary categories")

diag = pd.read_csv(HOSP_DIR / "diagnoses_icd.csv.gz", compression="gzip")
print(f"  {len(diag):,} diagnosis rows loaded")

# Map every raw ICD code to a category label
diag["icd_category"] = diag.apply(
    lambda r: map_icd9(r["icd_code"]) if r["icd_version"] == 9
              else map_icd10(r["icd_code"]),
    axis=1,
)

# Filter to cohort stays only
diag["stay_id"] = diag["hadm_id"].map(hadm_to_stay)
diag_cohort = diag.dropna(subset=["stay_id"]).copy()
diag_cohort["stay_id"] = diag_cohort["stay_id"].astype(int)
print(f"  {len(diag_cohort):,} rows after filtering to cohort")

# Build one binary column per category: 1 if the stay has any diagnosis in it
icd_features = cohort[["stay_id"]].copy()
for cat in ICD_CATEGORIES:
    stays_with = set(diag_cohort.loc[diag_cohort["icd_category"] == cat, "stay_id"])
    icd_features[f"icd_{cat}"] = icd_features["stay_id"].isin(stays_with).astype(int)

icd_cols = [c for c in icd_features.columns if c.startswith("icd_")]
icd_sum  = icd_features[icd_cols].sum(axis=1)
print(f"\n  {len(icd_cols)} ICD binary features")
print(f"  Stays with >= 1 category : {(icd_sum > 0).sum():,}")
print(f"  Mean categories per stay : {icd_sum.mean():.1f}")

# Primary diagnosis (seq_num == 1) → stored in labels for reporting
primary = diag[diag["seq_num"] == 1].copy()
primary["primary_diag"] = primary.apply(
    lambda r: map_icd9(r["icd_code"]) if r["icd_version"] == 9
              else map_icd10(r["icd_code"]),
    axis=1,
)
primary_map = (
    primary.drop_duplicates("hadm_id")
    .set_index("hadm_id")["primary_diag"]
    .to_dict()
)
cohort["primary_diag"] = cohort["hadm_id"].map(primary_map).fillna("unknown")
labels = cohort[["stay_id", "los", "los_gt7", "primary_diag"]].copy()

icd_features.to_parquet(OUTPUT_DIR / "icd_features.parquet", index=False)
labels.to_parquet(OUTPUT_DIR / "labels.parquet", index=False)
print(f"\n  Saved: output/icd_features.parquet")
print(f"  Saved: output/labels.parquet")


# 2 — MEDICATIONS / ATC LEVEL 1 (14 binary categories)
# Raw prescription rows (NDC drug codes) are aggregated and mapped onto 14 new binary
# drug-class indicators. Only the first 48h after ICU admission are used to avoid
# leaking information about the outcome.

print("2. Medications → 14 ATC Level-1 binary categories")

# Load NDC → ATC mapping (links prescription drug codes to drug classes)
print(f"  Loading NDC→ATC mapping...")
mapping = pd.read_csv(MAPPING_PATH, dtype=str)
atc_col = next(
    (c for c in ["ATC4", "ATC", *mapping.columns] if "atc" in c.lower()), None
)
assert atc_col is not None, f"No ATC column found. Columns: {mapping.columns.tolist()}"

mapping = mapping.dropna(subset=["NDC", atc_col])
mapping["ndc_clean"] = mapping["NDC"].str.strip().str.replace("-", "").str.zfill(11)
mapping["atc1"]      = mapping[atc_col].str.strip().str[0].str.upper()
mapping              = mapping[mapping["atc1"].isin(ATC1_CODES)]
ndc_to_atc1          = mapping.groupby("ndc_clean")["atc1"].apply(set).to_dict()
print(f"  {len(ndc_to_atc1):,} unique NDC codes mapped")

# Load prescriptions and filter to first 48h after ICU admission
rx = pd.read_csv(
    HOSP_DIR / "prescriptions.csv.gz",
    usecols=["hadm_id", "starttime", "ndc"],
    dtype={"ndc": str},
    compression="gzip",
)
rx = rx.merge(cohort[["hadm_id", "stay_id", "intime"]], on="hadm_id", how="inner")
rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")
rx = rx.dropna(subset=["starttime"])
rx["hours_in"] = (rx["starttime"] - rx["intime"]).dt.total_seconds() / 3600
rx = rx[(rx["hours_in"] >= 0) & (rx["hours_in"] < OBS_WINDOW)]
print(f"  {len(rx):,} prescriptions within first {OBS_WINDOW}h")

# Map NDC codes to ATC Level-1 letters
rx["ndc_clean"] = rx["ndc"].astype(str).str.strip().str.replace("-", "").str.zfill(11)
rx = rx[~rx["ndc_clean"].isin(["00000000000", "0".zfill(11)])]
rx["atc1_set"]  = rx["ndc_clean"].map(ndc_to_atc1)
n_mapped = rx["atc1_set"].notna().sum()
print(f"  Mapped: {n_mapped:,} / {len(rx):,} ({n_mapped/max(len(rx),1)*100:.1f}%)")

rx_exploded = rx.dropna(subset=["atc1_set"]).explode("atc1_set")

# Build one binary column per ATC class
atc_features = cohort[["stay_id"]].copy()
for code in ATC1_CODES:
    stays_with = set(rx_exploded.loc[rx_exploded["atc1_set"] == code, "stay_id"])
    atc_features[ATC1_COL_NAMES[code]] = atc_features["stay_id"].isin(stays_with).astype(int)

atc_cols = [ATC1_COL_NAMES[c] for c in ATC1_CODES]
atc_sum  = atc_features[atc_cols].sum(axis=1)
print(f"\n  {len(atc_cols)} ATC binary features")
print(f"  Stays with >= 1 ATC class : {(atc_sum > 0).sum():,}")
print(f"  Mean classes per stay     : {atc_sum.mean():.1f}")

atc_features.to_parquet(OUTPUT_DIR / "atc_features.parquet", index=False)
print(f"\n  Saved: output/atc_features.parquet")



# 3 — TIME-SERIES VITALS (12 features × 48h aggregated)
# Thousands of timestamped measurements are collapsed into summary statistics
# (mean, median, std, min, max, first, last, slope) per vital per stay.
# This flat representation can be fed directly into tree-based or MLP models.

print("3. Time-series vitals (12 features × 48h)")

itemid_list = list(VITAL_ITEMIDS.keys())

# Read chartevents in chunks — this table is ~30GB uncompressed
print(f"  Reading chartevents in {CHUNK_SIZE:,}-row chunks...")
ts_rows = []

reader = pd.read_csv(
    ICU_DIR / "chartevents.csv.gz",
    usecols=["stay_id", "charttime", "itemid", "valuenum"],
    dtype={"stay_id": "Int64", "itemid": int, "valuenum": float},
    chunksize=CHUNK_SIZE,
    compression="gzip",
)

for i, chunk in enumerate(reader):
    chunk = chunk[
        chunk["stay_id"].isin(cohort_stay_set) &
        chunk["itemid"].isin(itemid_list)
    ].dropna(subset=["valuenum", "stay_id"])
    if len(chunk) == 0:
        continue

    chunk["charttime"] = pd.to_datetime(chunk["charttime"])
    chunk["intime"]    = chunk["stay_id"].map(intime_map)
    chunk["hour"]      = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600
    chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] < OBS_WINDOW)]
    if len(chunk) == 0:
        continue

    chunk["feature"]   = chunk["itemid"].map(lambda x: VITAL_ITEMIDS[x][0])
    chunk["unit_flag"] = chunk["itemid"].map(lambda x: VITAL_ITEMIDS[x][1])

    # Convert Fahrenheit → Celsius so both temperature itemids unify
    f_mask = chunk["unit_flag"] == "F"
    chunk.loc[f_mask, "valuenum"] = (chunk.loc[f_mask, "valuenum"] - 32) * 5 / 9
    chunk.loc[chunk["feature"].isin(["temperature_f", "temperature_c"]), "feature"] = "temperature"

    chunk["hour"] = chunk["hour"].astype(int)
    ts_rows.append(chunk[["stay_id", "hour", "feature", "valuenum"]])
    print(f"  Chunk {i+1:>3}: {sum(len(r) for r in ts_rows):>10,} rows", end="\r")

print()

ts_all = pd.concat(ts_rows, ignore_index=True) if ts_rows else \
         pd.DataFrame(columns=["stay_id", "hour", "feature", "valuenum"])
print(f"  Total rows after filtering: {len(ts_all):,}")

# Remove physiologically implausible values (data entry errors / unit issues)
n_before = len(ts_all)
for feat, (lo, hi) in RANGE_FILTERS.items():
    if feat == "urine_output":
        continue
    bad = (ts_all["feature"] == feat) & ((ts_all["valuenum"] < lo) | (ts_all["valuenum"] > hi))
    ts_all = ts_all[~bad]
print(f"  After range filters: {len(ts_all):,}  (removed {n_before - len(ts_all):,})")


def compute_slope(group):
    """Linear trend (slope) of a vital over the observation window.

    Uses closed-form OLS: slope = cov(hours, values) / var(hours)
    """
    hours  = group["hour"].values.astype(float)
    values = group["valuenum"].values.astype(float)
    mask   = np.isfinite(hours) & np.isfinite(values)
    hours  = hours[mask]
    values = values[mask]
    if len(hours) < 2:
        return 0.0
    dh    = hours - hours.mean()
    var_h = (dh ** 2).sum()
    if var_h == 0:
        return 0.0
    return (dh * (values - values.mean())).sum() / var_h


# Summary statistics per stay × feature
print("  Computing aggregate statistics per stay...")
ts_summary = (
    ts_all.groupby(["stay_id", "feature"])["valuenum"]
    .agg(["mean", "median", "std", "min", "max", "first", "last", "count"])
    .reset_index()
)

# Slope per stay × feature
ts_slopes = (
    ts_all.groupby(["stay_id", "feature"])
    .apply(compute_slope, include_groups=False)
    .reset_index()
    .rename(columns={0: "slope"})
)
ts_summary = ts_summary.merge(ts_slopes, on=["stay_id", "feature"], how="left")
ts_summary["slope"] = ts_summary["slope"].fillna(0.0)

# Pivot to wide format: one column per (feature, statistic)
agg_stats = ["mean", "median", "std", "min", "max", "first", "last", "slope", "count"]
ts_wide = ts_summary.pivot_table(
    index="stay_id", columns="feature", values=agg_stats,
)
ts_wide.columns = [f"{feat}_{stat}" for stat, feat in ts_wide.columns]
ts_wide = ts_wide.reset_index()

# Add binary missingness flags (1 = no measurements at all in 48h)
for feat in TS_FEATURES:
    count_col   = f"{feat}_count"
    missing_col = f"{feat}_missing"
    if count_col in ts_wide.columns:
        ts_wide[missing_col] = (ts_wide[count_col].isna() | (ts_wide[count_col] == 0)).astype(int)
    else:
        ts_wide[missing_col] = 1
# Drop count columns (served only to build missingness flags)
count_cols = [c for c in ts_wide.columns if c.endswith("_count")]
ts_wide = ts_wide.drop(columns=count_cols)

# Merge with full cohort (ensures every stay has a row, even with no vitals)
ts_features = cohort[["stay_id"]].merge(ts_wide, on="stay_id", how="left")

# Urine output (from outputevents, not chartevents)
# Summed per stay (total volume) — multiple output events are additive.
print(f"\n  Reading outputevents for urine output...")
uo_chunks = []
uo_reader = pd.read_csv(
    ICU_DIR / "outputevents.csv.gz",
    usecols=["stay_id", "charttime", "itemid", "value"],
    dtype={"stay_id": "Int64", "itemid": int, "value": float},
    chunksize=CHUNK_SIZE,
    compression="gzip",
)
lo_uo, hi_uo = RANGE_FILTERS["urine_output"]

for j, uo_chunk in enumerate(uo_reader):
    uo_chunk = uo_chunk[
        uo_chunk["stay_id"].isin(cohort_stay_set) &
        uo_chunk["itemid"].isin(URINE_ITEMIDS)
    ].dropna(subset=["value", "stay_id"])
    if len(uo_chunk) == 0:
        continue
    uo_chunk["charttime"] = pd.to_datetime(uo_chunk["charttime"])
    uo_chunk["intime"]    = uo_chunk["stay_id"].map(intime_map)
    uo_chunk["hour"]      = (uo_chunk["charttime"] - uo_chunk["intime"]).dt.total_seconds() / 3600
    uo_chunk = uo_chunk[(uo_chunk["hour"] >= 0) & (uo_chunk["hour"] < OBS_WINDOW)]
    uo_chunk = uo_chunk[(uo_chunk["value"] >= lo_uo) & (uo_chunk["value"] <= hi_uo)]
    uo_chunk["hour"] = uo_chunk["hour"].astype(int)
    uo_chunks.append(uo_chunk[["stay_id", "hour", "value"]])
    print(f"  Chunk {j+1:>3}: {sum(len(c) for c in uo_chunks):>10,} rows", end="\r")

print()

if uo_chunks:
    uo_all = pd.concat(uo_chunks, ignore_index=True)
    uo_summary = uo_all.groupby("stay_id")["value"].agg(
        urine_total="sum", urine_mean="mean", urine_max="max",
    ).reset_index()
    ts_features = ts_features.merge(uo_summary, on="stay_id", how="left")
    ts_features["urine_missing"] = ts_features["urine_total"].isna().astype(int)
    ts_features[["urine_total", "urine_mean", "urine_max"]] = (
        ts_features[["urine_total", "urine_mean", "urine_max"]].fillna(0.0)
    )
else:
    ts_features["urine_total"]   = 0.0
    ts_features["urine_mean"]    = 0.0
    ts_features["urine_max"]     = 0.0
    ts_features["urine_missing"] = 1
    print("  WARNING: No urine output data found.")

# Report coverage
vital_cols = [c for c in ts_features.columns if c.endswith("_missing")]
print(f"\n  Feature coverage (% of stays with any data):")
for mc in sorted(vital_cols):
    feat_name = mc.replace("_missing", "")
    pct = (1 - ts_features[mc].mean()) * 100
    print(f"    {feat_name:<18} {pct:.1f}%")

n_feature_cols = len([c for c in ts_features.columns if c != "stay_id"])
print(f"\n  Total aggregate features: {n_feature_cols}")

ts_features.to_parquet(OUTPUT_DIR / "ts_features.parquet", index=False)
print(f"\n  Saved: output/ts_features.parquet")
print(f"  {ts_features['stay_id'].nunique():,} stays × {n_feature_cols} features")



# 4 — HOURLY TIME-SERIES (stay_id × hour × 12 features)
# Stores the full temporal structure needed for sequence models (GRU, LSTM).
# Each stay has exactly 48 rows (one per hour). Missing hours are forward-filled
# then backward-filled; vitals with zero data in the entire 48h remain NaN
# (the model Dataset should fill these, e.g. with 0 after normalization).

print("\n4. Hourly time-series (48h × 12 features per stay)")

vital_feature_cols = [f for f in TS_FEATURES if f != "urine_output"]

# Mean per (stay, hour, feature) — multiple measurements within same hour are averaged
ts_hourly_agg = (
    ts_all.groupby(["stay_id", "hour", "feature"])["valuenum"]
    .mean()
    .reset_index()
)

# Pivot to wide: one column per vital
ts_hourly_wide = ts_hourly_agg.pivot_table(
    index=["stay_id", "hour"], columns="feature", values="valuenum",
).reset_index()
ts_hourly_wide.columns.name = None

# Ensure all vital columns exist (absent if no data at all for that vital)
for feat in vital_feature_cols:
    if feat not in ts_hourly_wide.columns:
        ts_hourly_wide[feat] = np.nan

# Build complete grid: every stay × every hour 0–47
all_hours = pd.DataFrame(
    [(sid, h) for sid in cohort["stay_id"] for h in range(OBS_WINDOW)],
    columns=["stay_id", "hour"],
)
ts_hourly = all_hours.merge(ts_hourly_wide, on=["stay_id", "hour"], how="left")

# Forward-fill then backward-fill per stay (fills short measurement gaps)
ts_hourly = ts_hourly.sort_values(["stay_id", "hour"])
ts_hourly[vital_feature_cols] = ts_hourly.groupby("stay_id")[vital_feature_cols].ffill()
ts_hourly[vital_feature_cols] = ts_hourly.groupby("stay_id")[vital_feature_cols].bfill()

# Add urine output: hourly sum (0 for hours with no output event — physiologically correct)
if uo_chunks:
    uo_hourly = (
        uo_all.groupby(["stay_id", "hour"])["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "urine_output"})
    )
    ts_hourly = ts_hourly.merge(uo_hourly, on=["stay_id", "hour"], how="left")
else:
    ts_hourly["urine_output"] = 0.0
ts_hourly["urine_output"] = ts_hourly["urine_output"].fillna(0.0)

ts_hourly = ts_hourly[["stay_id", "hour"] + TS_FEATURES]

n_stays_ts = ts_hourly["stay_id"].nunique()
print(f"  {n_stays_ts:,} stays × {OBS_WINDOW}h × {len(TS_FEATURES)} features")
print(f"  NaN remaining (vitals with zero coverage): "
      f"{ts_hourly[TS_FEATURES].isna().sum().sum():,}")

ts_hourly.to_parquet(OUTPUT_DIR / "timeseries.parquet", index=False)
print(f"  Saved: output/timeseries.parquet")
