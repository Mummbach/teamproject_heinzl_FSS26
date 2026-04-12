"""
Preprocessing
=============
Encodes categorical features, merges all feature files, and applies
median imputation fitted exclusively on the training split.

Steps:
  0. Data quality checks    — duplicate stay_id / hadm_id
  1. Encode demographics    — categorical → binary/numeric (rule-based, no leakage)
  2. Merge features         — demographics + ICD + ATC + time-series
  3. Impute                 — median fitted on train only, applied to all splits
  4. Save                   — X_train/val/test + y_train/val/test as parquet

No statistics are computed before the split. All imputed values are derived
solely from training data to prevent data leakage into validation and test sets.

Run AFTER:  03_splitting.py
Run BEFORE: 05_analysis.py  /  model training

Input:   output/cohort.csv
         output/split_ids.parquet
         output/icd_features.parquet
         output/atc_features.parquet
         output/ts_features.parquet
         output/labels.parquet

Output:  output/X_train.parquet    output/y_train.parquet
         output/X_val.parquet      output/y_val.parquet
         output/X_test.parquet     output/y_test.parquet
         output/imputer_medians.parquet  — train medians for inference
"""

import pandas as pd
import numpy as np
from config import OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True)


# LOAD

cohort    = pd.read_csv(OUTPUT_DIR / "cohort.csv", parse_dates=["intime", "outtime"])
split_ids = pd.read_parquet(OUTPUT_DIR / "split_ids.parquet")
icd       = pd.read_parquet(OUTPUT_DIR / "icd_features.parquet")
atc       = pd.read_parquet(OUTPUT_DIR / "atc_features.parquet")
ts_flat   = pd.read_parquet(OUTPUT_DIR / "ts_features.parquet")
labels    = pd.read_parquet(OUTPUT_DIR / "labels.parquet")

print(f"Cohort loaded   : {len(cohort):,} stays")
print(f"Split IDs loaded: {len(split_ids):,} stays")
for s in ["train", "val", "test"]:
    print(f"  {s:<6}: {(split_ids['split']==s).sum():,}")


# STEP 0 — DATA QUALITY CHECKS

print("\n" + "=" * 60)
print("STEP 0 — Data quality checks")
print("=" * 60)

dup_stays = cohort["stay_id"].duplicated().sum()
assert dup_stays == 0, f"{dup_stays} duplicate stay_ids found"
print(f"  stay_id duplicates  : 0 ✓")

dup_hadm = cohort["hadm_id"].duplicated().sum()
if dup_hadm > 0:
    print(f"  WARNING: {dup_hadm} duplicate hadm_ids — review groupby logic in 01_selection.py")
else:
    print(f"  hadm_id duplicates  : 0 ✓")

print(f"  Total rows          : {len(cohort):,}")


# STEP 1 — ENCODE DEMOGRAPHICS
# All transformations here are rule-based (fixed mappings, no statistics).
# They can safely run on the full cohort without leakage.

print("\n" + "=" * 60)
print("STEP 1 — Encoding demographics")
print("=" * 60)

static = cohort[["stay_id"]].copy()

# Age: continuous. NaN possible if anchor fields are missing; keep as NaN here —
# imputation happens in Step 3, fitted on train only.
static["age"] = cohort["age_at_icu"].values

# Binary: 1 = male, 0 = female.
static["gender_male"] = (cohort["gender"].fillna("F").values == "M").astype(int)

# Race grouped into 5 clinical categories to avoid sparse columns.
RACE_MAP = {
    "WHITE":              "white",
    "BLACK":              "black",    # catches BLACK/AFRICAN AMERICAN, BLACK/AFRICAN, BLACK/CAPE VERDEAN, etc.
    "HISPANIC/LATINO":    "hispanic",
    "HISPANIC OR LATINO": "hispanic",
    "ASIAN":              "asian",
}
eth_raw     = cohort["race"].fillna("UNKNOWN").str.upper().str.strip()
eth_grouped = eth_raw.map(
    lambda x: next((v for k, v in RACE_MAP.items() if k in x), "other")
)
for cat in ["white", "black", "hispanic", "asian", "other"]:
    static[f"eth_{cat}"] = (eth_grouped.values == cat).astype(int)

# Language: 1 = English, 0 = non-English / unknown.
static["language_english"] = (
    cohort["language"].fillna("Unknown").str.strip().str.lower() == "english"
).astype(int)

# Insurance grouped: Medicare, Medicaid, Other.
ins = cohort["insurance"].fillna("Other").str.lower().str.strip()
static["ins_medicare"] = ins.str.contains("medicare").astype(int)
static["ins_medicaid"] = ins.str.contains("medicaid").astype(int)
static["ins_other"]    = (~ins.str.contains("medicare|medicaid")).astype(int)

# Admission type grouped into 4 categories.
adm = cohort["admission_type"].fillna("OTHER").str.upper().str.strip()
static["adm_emergency"]   = adm.isin(["EW EMER.", "DIRECT EMER."]).astype(int)
static["adm_urgent"]      = (adm == "URGENT").astype(int)
static["adm_elective"]    = adm.isin(["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]).astype(int)
static["adm_observation"] = adm.str.contains("OBSERVATION").astype(int)

# Admission location grouped into 5 categories.
loc = cohort["admission_location"].fillna("UNKNOWN").str.upper().str.strip()
static["loc_emergency_room"] = (loc == "EMERGENCY ROOM").astype(int)
static["loc_transfer"]       = loc.str.contains("TRANSFER").astype(int)
static["loc_referral"]       = loc.str.contains("REFERRAL").astype(int)
static["loc_walk_in"]        = (loc == "WALK-IN/SELF REFERRAL").astype(int)
static["loc_other"]          = (
    ~loc.isin(["EMERGENCY ROOM", "WALK-IN/SELF REFERRAL"]) &
    ~loc.str.contains("TRANSFER|REFERRAL")
).astype(int)

# NOTE: discharge_location excluded — only known at discharge (future leakage).

# Marital status: 4 binary flags; NaN/unknown → all 0.
mar = cohort["marital_status"].fillna("UNKNOWN").str.upper().str.strip()
static["marital_married"]  = mar.str.contains("MARRIED").astype(int)
static["marital_single"]   = mar.str.contains("SINGLE|NEVER").astype(int)
static["marital_widowed"]  = mar.str.contains("WIDOWED").astype(int)
static["marital_divorced"] = mar.str.contains("DIVORCED|SEPARATED").astype(int)

# ICU unit type: 7 binary flags.
ICU_DUMMIES = {
    "micu":       "Medical Intensive Care Unit (MICU)",
    "sicu":       "Surgical Intensive Care Unit (SICU)",
    "ccu":        "Coronary Care Unit (CCU)",
    "cvicu":      "Cardiac Vascular Intensive Care Unit (CVICU)",
    "micu_sicu":  "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "tsicu":      "Trauma SICU (TSICU)",
    "neuro_sicu": "Neuro Surgical Intensive Care Unit (Neuro SICU)",
}
icu_type = cohort["first_careunit"].fillna("Other").str.strip()
for col_suffix, full_name in ICU_DUMMIES.items():
    static[f"icu_{col_suffix}"] = (icu_type == full_name).astype(int)

# Admission era: ordinal encoding of anchor_year_group.
# Captures temporal trends in LOS across the MIMIC-IV collection period.
YEAR_GROUP_ORDER = {
    "2008 - 2010": 0,
    "2011 - 2013": 1,
    "2014 - 2016": 2,
    "2017 - 2019": 3,
    "2020 - 2022": 4,
}
static["year_group"] = (
    cohort["anchor_year_group"].map(YEAR_GROUP_ORDER).fillna(-1).astype(int)
)

feature_cols = [c for c in static.columns if c != "stay_id"]
print(f"\n  {len(feature_cols)} demographic features encoded")


# STEP 2 — MERGE ALL FEATURES

print("\n" + "=" * 60)
print("STEP 2 — Merging all features")
print("=" * 60)

features_all = (
    labels[["stay_id"]]
    .merge(static,  on="stay_id", how="left")
    .merge(icd,     on="stay_id", how="left")
    .merge(atc,     on="stay_id", how="left")
    .merge(ts_flat, on="stay_id", how="left")
)

print(f"  demographics : {len(static.columns)-1} features")
print(f"  ICD          : {len(icd.columns)-1} features")
print(f"  ATC          : {len(atc.columns)-1} features")
print(f"  time-series  : {len(ts_flat.columns)-1} features")
all_feature_cols = [c for c in features_all.columns if c != "stay_id"]
print(f"  total        : {len(all_feature_cols)} features")


# STEP 3 — SPLIT + IMPUTE

print("\n" + "=" * 60)
print("STEP 3 — Split + imputation (train median only)")
print("=" * 60)

train_ids = set(split_ids[split_ids["split"] == "train"]["stay_id"])
val_ids   = set(split_ids[split_ids["split"] == "val"]["stay_id"])
test_ids  = set(split_ids[split_ids["split"] == "test"]["stay_id"])

X_train = features_all[features_all["stay_id"].isin(train_ids)].copy()
X_val   = features_all[features_all["stay_id"].isin(val_ids)].copy()
X_test  = features_all[features_all["stay_id"].isin(test_ids)].copy()

y_train = labels[labels["stay_id"].isin(train_ids)].copy()
y_val   = labels[labels["stay_id"].isin(val_ids)].copy()
y_test  = labels[labels["stay_id"].isin(test_ids)].copy()

print(f"  train : {len(X_train):>7,} stays")
print(f"  val   : {len(X_val):>7,} stays")
print(f"  test  : {len(X_test):>7,} stays")

# Identify columns that need imputation (any NaN present in train)
impute_cols = [
    c for c in all_feature_cols
    if not c.endswith("_missing") and X_train[c].isna().any()
]
print(f"\n  Columns to impute: {len(impute_cols)}")

# Fit median on training data only
# fillna(0): fallback for columns where all train values are NaN (median would be NaN)
train_medians = X_train[impute_cols].median().fillna(0)

# Apply to all three splits
for name, X_split in [("train", X_train), ("val", X_val), ("test", X_test)]:
    n_before = X_split[impute_cols].isna().sum().sum()
    X_split[impute_cols] = X_split[impute_cols].fillna(train_medians)
    n_after = X_split[impute_cols].isna().sum().sum()
    print(f"  {name:<6}: {n_before:>7,} NaN filled  ({n_after} remaining in impute_cols)")

# _missing flag columns are excluded from impute_cols (they are binary indicators,
# not continuous features), but stays with zero chartevents rows get NaN for these
# columns after the left-merge in 02_features.py. Fill with 1 = "measurement absent".
missing_flag_cols = [c for c in all_feature_cols if c.endswith("_missing")]
for X_split in [X_train, X_val, X_test]:
    X_split[missing_flag_cols] = X_split[missing_flag_cols].fillna(1)

# Save imputer values for inference time
imputer_df = train_medians.reset_index()
imputer_df.columns = ["feature", "median"]
imputer_df.to_parquet(OUTPUT_DIR / "imputer_medians.parquet", index=False)

# Sanity checks
for name, X_split in [("train", X_train), ("val", X_val), ("test", X_test)]:
    remaining = X_split[all_feature_cols].isna().sum().sum()
    assert remaining == 0, f"{name}: {remaining} NaN values remain after imputation"

assert set(X_train["stay_id"]).isdisjoint(set(X_val["stay_id"])),  "train/val overlap"
assert set(X_train["stay_id"]).isdisjoint(set(X_test["stay_id"])), "train/test overlap"
assert set(X_val["stay_id"]).isdisjoint(set(X_test["stay_id"])),   "val/test overlap"
print("\n  No NaN remaining ✓")
print("  No stay_id overlap between splits ✓")

# Class balance per split
print("\n  Label distribution:")
for name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
    pos = y_split["los_gt7"].mean() * 100
    n   = len(y_split)
    print(f"  {name:<6}: {n:>7,} stays  positive rate: {pos:.1f}%")


# STEP 4 — SAVE

print("\n" + "=" * 60)
print("STEP 4 — Saving splits")
print("=" * 60)

splits = {
    "X_train": X_train, "y_train": y_train,
    "X_val":   X_val,   "y_val":   y_val,
    "X_test":  X_test,  "y_test":  y_test,
}
for name, df in splits.items():
    path = OUTPUT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved: output/{name}.parquet  ({len(df):,} rows × {len(df.columns)} cols)")

imputer_path = OUTPUT_DIR / "imputer_medians.parquet"
print(f"  Saved: output/imputer_medians.parquet  ({len(imputer_df)} features)")
