"""
Data Inclusion / Exclusion
===========================
Defines the study cohort from MIMIC-IV ICU data.

Pipeline:
  icustays  → first ICU stay per patient (groupby subject_id → first)
            → merge patients   (age, gender)
            → merge admissions (admission type, insurance, ...)
            → filter: age >= 18
            → filter: LOS >= 2 days
            → filter: death within 48h excluded
            → filter: LOS <= 90 days

Label:
  los_gt7 = 1  if ICU LOS > 7 days
  los_gt7 = 0  otherwise

Run BEFORE: 02_features.py
Output:     output/cohort.csv  +  output/cohort_stay_ids.txt
"""

import pandas as pd
from config import HOSP_DIR, ICU_DIR, OUTPUT_DIR, ADMISSIONS_PATH, PATIENTS_PATH

OUTPUT_DIR.mkdir(exist_ok=True)

# Load raw tables

icustays = pd.read_csv(
    ICU_DIR / "icustays.csv.gz",
    compression="gzip",
    parse_dates=["intime", "outtime"],
)
patients = pd.read_csv(
    PATIENTS_PATH,
    compression="gzip",
)
admissions = pd.read_csv(
    ADMISSIONS_PATH,
    compression="gzip",
    parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
)

print(f"  icustays   : {len(icustays):>7,} rows")
print(f"  patients   : {len(patients):>7,} rows")
print(f"  admissions : {len(admissions):>7,} rows")

# First ICU stay per patient
# Sort by admission earliest stay.
# Using only the first stay avoids data leakage from prior hospitalizations
# Group-based splitting
icustays = icustays.sort_values(["subject_id", "intime"])
icustays = icustays.groupby("subject_id", as_index=False).first()

print(f"\nAfter keeping first ICU stay per patient: {len(icustays):,}")

# Sanity check: groupby().first() should guarantee one stay_id per patient.
assert icustays["stay_id"].nunique() == len(icustays), "duplicate stay_ids found after first-stay filter"

# Merge icustays, patients, admissions
df = (
    icustays
    .merge(patients,   on="subject_id",             how="left")
    .merge(admissions, on=["subject_id", "hadm_id"], how="left")
)

# to protect patient privacy, all real dates are shifted by a random offset per patient
# age_at_icu = anchor_age + (icu_admission_year - anchor_year)
df["icu_admission_year"] = df["intime"].dt.year
df["age_at_icu"] = df["anchor_age"] + (df["icu_admission_year"] - df["anchor_year"])


# Inclusion / exclusion filters
print("\nApplying selection criteria ...")
n = len(df)

# Adults only (age >= 18)
df = df[df["age_at_icu"] >= 18]
print(f"  age >= 18:                  {len(df):>7,}  (removed {n - len(df):,})")
n = len(df)

# Minimum ICU LOS >= 2 days
# Stays shorter than 2 days are often observation stays or rapid recoveries;
# predicting >7 days for these is clinically trivial and skews the label distribution.
df = df[df["los"] >= 2]
print(f"  LOS >= 2 days:              {len(df):>7,}  (removed {n - len(df):,})")
n = len(df)

# Patients who died within 48h of ICU admission
# These patients physically cannot stay >7 days — including them adds label noise.
death_within_48h = (
    df["deathtime"].notna() &
    ((df["deathtime"] - df["intime"]).dt.total_seconds() / 3600 <= 48)
)
df = df[~death_within_48h]
print(f"  excl. death within 48h:     {len(df):>7,}  (removed {n - len(df):,})")
n = len(df)

# ICU LOS > 90 days
# Extreme outliers may represent data-quality issues or LTACH transfers.
df = df[df["los"] <= 90]
print(f"  LOS <= 90 days:             {len(df):>7,}  (removed {n - len(df):,})")
n = len(df)

# INACTIVE FILTERS (activate if needed)

# Patients transferred to another acute-care facility on discharge
# df = df[df["discharge_location"] != "TRANSFER TO OTHER FACILITY"]

# planned surgical admissions (focus on unplanned/emergency only)
# df = df[df["admission_type"] != "ELECTIVE"]

# specific care units only (unit-specific model)
# df = df[df["first_careunit"].isin(["Medical Intensive Care Unit (MICU)",
#                                    "Surgical Intensive Care Unit (SICU)"])]


# Create binary label LOS > 7 days
df["los_gt7"] = (df["los"] > 7).astype(int)

label_counts = df["los_gt7"].value_counts().sort_index()
print(f"\nLabel distribution:")
print(f"  LOS <= 7 days (0): {label_counts.get(0, 0):,}")
print(f"  LOS >  7 days (1): {label_counts.get(1, 0):,}")
print(f"  Positive rate    : {df['los_gt7'].mean()*100:.1f}%")

# Build cohort DataFrame
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

# Columns intentionally dropped:
#   anchor_age        → replaced by age_at_icu
#   anchor_year       → only needed to compute age_at_icu
#   dod               → privacy-sensitive, not needed for LOS prediction
#   admittime         → hospital admission time; ICU intime is more relevant
#   dischtime         → leaks outcome indirectly
#   deathtime         → leaks outcome (death = short stay)
#   edregtime/edouttime → not relevant for ICU LOS
#   icu_admission_year  → intermediate column, not meaningful on its own
#   hospital_expire_flag → leaks outcome (died → short stay)
#   last_careunit       → only known after stay (future leakage)

cohort = df[cohort_cols].reset_index(drop=True)

cohort_stay_set = set(cohort["stay_id"])

cohort_path    = OUTPUT_DIR / "cohort.csv"
stay_ids_path  = OUTPUT_DIR / "cohort_stay_ids.txt"

cohort.to_csv(cohort_path, index=False)

with open(stay_ids_path, "w") as f:
    f.write("\n".join(str(sid) for sid in sorted(cohort_stay_set)))

print(f"\nSaved:")
print(f"  {cohort_path}")
print(f"  {stay_ids_path}")
