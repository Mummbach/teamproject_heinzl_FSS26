"""
Data Inclusion / Exclusion

Pipeline:
  icustays  → first ICU stay per patient (groupby subject_id → first)
            → merge patients   (age, gender)
            → merge admissions (admission type, insurance, ...)
            → filter: age >= 18
            → filter: LOS >= 2 days
            → filter: death within 48h excluded
            → filter: LOS <= 90 days
            → optional: exclude transfers to acute hospital (EXCLUDE_TRANSFERS)
            → optional: exclude early deaths 48h–7d    (EXCLUDE_EARLY_DEATHS)

Label:
  los_gt7 = 1  if ICU LOS > 7 days
  los_gt7 = 0  otherwise

Flags:
  EXCLUDE_EARLY_DEATHS — exclude patients dying 48h–7d after ICU admission
  EXCLUDE_TRANSFERS    — exclude patients discharged to another acute hospital

Run BEFORE: 02_features.py
Output:     output/cohort.csv
"""

import pandas as pd
from config import HOSP_DIR, ICU_DIR, OUTPUT_DIR

# True — exclude patients dying 48h–7d after ICU admission (bias-reduced cohort)
# False — keep them (default)
EXCLUDE_EARLY_DEATHS = False

# True — exclude patients transferred to another acute hospital
# False — keep them (default)
EXCLUDE_TRANSFERS = True

OUTPUT_DIR.mkdir(exist_ok=True)

# Load raw tables

icustays = pd.read_csv(
    ICU_DIR / "icustays.csv.gz",
    compression="gzip",
    parse_dates=["intime", "outtime"],
)
patients = pd.read_csv(
    HOSP_DIR / "patients.csv.gz",
    compression="gzip",
)
admissions = pd.read_csv(
    HOSP_DIR / "admissions.csv.gz",
    compression="gzip",
    parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
)

print(f"  icustays   : {len(icustays):>7,} rows")
print(f"  patients   : {len(patients):>7,} rows")
print(f"  admissions : {len(admissions):>7,} rows")

# First ICU stay per patient
# Sort by admission earliest stay.
icustays = icustays.sort_values(["subject_id", "intime"])
icustays = icustays.groupby("subject_id", as_index=False).first()


assert icustays["stay_id"].nunique() == len(icustays), "duplicate stay_ids found after first-stay filter"

# Merge icustays, patients, admissions
df = (
    icustays
    .merge(patients,   on="subject_id",             how="left")
    .merge(admissions, on=["subject_id", "hadm_id"], how="left")
)


df["age_at_icu"] = df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])


# Inclusion / exclusion filters
n = len(df)

# Adults only (age >= 18)
df = df[df["age_at_icu"] >= 18]
print(f"  age >= 18:                  {len(df):>7,}  (removed {n - len(df):,})")
n = len(df)

# Minimum ICU LOS >= 2 days
df = df[df["los"] >= 2]
print(f"  LOS >= 2 days:              {len(df):>7,}  (removed {n - len(df):,})")
n = len(df)


# Patients who died within 48h of ICU admission
death_within_48h = (
    df["deathtime"].notna() &
    ((df["deathtime"] - df["intime"]).dt.total_seconds() / 3600 <= 48)
)
df = df[~death_within_48h]
print(f"  excl. death within 48h:     {len(df):>7,}  (removed {n - len(df):,})")
n = len(df)

# Optional: exclude patients who died between 48h and 7d (early deaths bias check).
if EXCLUDE_EARLY_DEATHS:
    death_48h_to_7d = (
        df["deathtime"].notna() &
        ((df["deathtime"] - df["intime"]).dt.total_seconds() / 3600 > 48) &
        ((df["deathtime"] - df["intime"]).dt.total_seconds() / 3600 <= 168)
    )
    df = df[~death_48h_to_7d]
    print(f"  excl. death 48h–7d:         {len(df):>7,}  (removed {n - len(df):,})")
    n = len(df)
else:
    early_deaths = (
        df["deathtime"].notna() &
        ((df["deathtime"] - df["intime"]).dt.total_seconds() / 3600 > 48) &
        ((df["deathtime"] - df["intime"]).dt.total_seconds() / 3600 <= 168)
    )
    print(f"  early deaths 48h–7d (kept): {early_deaths.sum():>7,}  (EXCLUDE_EARLY_DEATHS=False)")


# Extreme Outliers: ICU LOS > 90 days 
df = df[df["los"] <= 90]
print(f"  LOS <= 90 days:             {len(df):>7,}  (removed {n - len(df):,})")
n = len(df)

# Optional: exclude patients transferred to another acute-care facility.
if EXCLUDE_TRANSFERS:
    df = df[df["discharge_location"] != "ACUTE HOSPITAL"]
    print(f"  excl. transfers (ACUTE HOSPITAL): {len(df):>7,}  (removed {n - len(df):,})")
    n = len(df)
else:
    transfers = (df["discharge_location"] == "ACUTE HOSPITAL").sum()
    print(f"  transfers kept (ACUTE HOSPITAL):  {transfers:>7,}  (EXCLUDE_TRANSFERS=False)")

# Create binary label LOS > 7 days
df["los_gt7"] = (df["los"] > 7).astype(int)

label_counts = df["los_gt7"].value_counts().sort_index()
print(f"\nLabel distribution:")
print(f"  LOS <= 7 days (0): {label_counts.get(0, 0):,}")
print(f"  LOS >  7 days (1): {label_counts.get(1, 0):,}")
print(f"  Positive rate    : {df['los_gt7'].mean()*100:.1f}%")

# Build cohort DataFrame
# Dropped: anchor_age/anchor_year (replaced by age_at_icu), icu_admission_year (intermediate),
# dod/edregtime/edouttime (irrelevant), admittime (intime preferred),
# dischtime/deathtime/hospital_expire_flag (leaks outcome), last_careunit (future leakage)
cohort_cols = [
    # Identifiers
    "subject_id", "hadm_id", "stay_id",

    # Timestamps (intime safe for splitting; outtime kept for reference only —
    # never use as model feature: outtime - intime = los leaks the label)
    "intime", "outtime",

    # ICU context
    "first_careunit", "los",

    # Patient demographics
    "gender", "age_at_icu", "anchor_year_group",

    # Admission context
    "admission_type", "admission_location", "discharge_location",
    "insurance", "language", "marital_status", "race",

    # Target label
    "los_gt7",
]


cohort = df[cohort_cols].reset_index(drop=True)

cohort.to_csv(OUTPUT_DIR / "cohort.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'cohort.csv'}")
