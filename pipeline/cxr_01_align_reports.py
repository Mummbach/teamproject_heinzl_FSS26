"""
Align MIMIC-CXR radiology reports with the ICU cohort from 01_selection.py.

=======================

Run AFTER:  01_selection.py
Run BEFORE: cxr_02_extract_sections.py


Input:   output/cohort.csv
Output:  output/cohort_with_cxr.csv


For each ICU stay:
    - finds radiology reports from the same patient
    - keeps ONLY reports written within:
            ICU intime <= report_time <= ICU intime + 48h
    - selects ONLY the FIRST valid radiology report
    - preserves the FULL ICU cohort
            -> patients without reports remain included
            -> text modality is optional
    - appends a FINAL LAST COLUMN:
            los_gt7
      where:
            1 = ICU LOS > 7 days
            0 = ICU LOS <= 7 days

"""

from pathlib import Path
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, OBS_WINDOW

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

CXR_DIR = DATA_DIR / "mimic-cxr-reports"

METADATA_PATH = CXR_DIR / "mimic-cxr-2.0.0-metadata.csv.gz"

COHORT_PATH = OUTPUT_DIR / "cohort.csv"

OUTPUT_PATH = OUTPUT_DIR / "cohort_with_cxr.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Load ICU cohort
# ──────────────────────────────────────────────────────────────────────────────

print("Loading ICU cohort ...")

cohort = pd.read_csv(
    COHORT_PATH,
    parse_dates=["intime", "outtime"]
)

print(f"  cohort stays: {len(cohort):,}")

# Observation window end
cohort["obs_end"] = cohort["intime"] + pd.Timedelta(hours=OBS_WINDOW)

# ──────────────────────────────────────────────────────────────────────────────
# Load MIMIC-CXR metadata
# ──────────────────────────────────────────────────────────────────────────────

print("\nLoading MIMIC-CXR metadata ...")

meta = pd.read_csv(
    METADATA_PATH,
    compression="gzip"
)

# Ensure IDs are integers
meta["subject_id"] = meta["subject_id"].astype(int)
meta["study_id"] = meta["study_id"].astype(int)

print(f"  metadata rows: {len(meta):,}")

# Keep only subjects present in ICU cohort
cohort_subjects = set(cohort["subject_id"].unique())

meta = meta[
    meta["subject_id"].isin(cohort_subjects)
].copy()

print(f"  rows after subject filter: {len(meta):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Construct study datetime
# ──────────────────────────────────────────────────────────────────────────────

meta["StudyDate"] = meta["StudyDate"].astype(str)

meta["StudyTime"] = (
    meta["StudyTime"]
    .fillna(0)
    .astype(str)
    .str.split(".")
    .str[0]
    .str.zfill(6)
)

meta["study_datetime"] = pd.to_datetime(
    meta["StudyDate"] + meta["StudyTime"],
    format="%Y%m%d%H%M%S",
    errors="coerce"
)

meta = meta.dropna(subset=["study_datetime"])

print(f"  rows with valid timestamps: {len(meta):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Align reports to ICU stays
# ──────────────────────────────────────────────────────────────────────────────

print("\nAligning reports to ICU stays ...")

merged = cohort.merge(
    meta[
        [
            "subject_id",
            "study_id",
            "study_datetime"
        ]
    ],
    on="subject_id",
    how="left"
)

# Keep ONLY reports within first 48h
within_window = (
    (merged["study_datetime"] >= merged["intime"]) &
    (merged["study_datetime"] <= merged["obs_end"])
)

aligned = merged[within_window].copy()

print(f"  reports within observation window: {len(aligned):,}")

# Keep FIRST report per ICU stay
aligned = aligned.sort_values(
    ["stay_id", "study_datetime"]
)

aligned = aligned.drop_duplicates(
    subset="stay_id",
    keep="first"
)

print(f"  first reports kept: {len(aligned):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Build report file paths
# ──────────────────────────────────────────────────────────────────────────────

def get_report_path(subject_id: int, study_id: int) -> Path:
    """
    Example path:
        p10/p10000032/s50414267.txt
    """

    subject_str = str(int(subject_id))

    prefix = f"p{subject_str[:2]}"
    patient_folder = f"p{subject_str}"

    filename = f"s{int(study_id)}.txt"

    return (
        CXR_DIR /
        prefix /
        patient_folder /
        filename
    )


# ──────────────────────────────────────────────────────────────────────────────
# Load report texts
# ──────────────────────────────────────────────────────────────────────────────

def load_report_text(path: Path):

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if len(text) == 0:
            return None

        return text

    except Exception:
        return None


print("\nLoading report texts ...")

aligned["report_path"] = aligned.apply(
    lambda row: get_report_path(
        row["subject_id"],
        row["study_id"]
    ),
    axis=1
)

# Debug example
example_path = aligned["report_path"].iloc[0]

print(f"  example path: {example_path}")
print(f"  exists: {example_path.exists()}")

aligned["cxr_report"] = aligned["report_path"].apply(
    load_report_text
)

aligned["has_cxr"] = aligned["cxr_report"].notna()

print(f"  valid report texts loaded: {aligned['has_cxr'].sum():,}")

# ──────────────────────────────────────────────────────────────────────────────
# Merge back into FULL cohort
# ──────────────────────────────────────────────────────────────────────────────

aligned_small = aligned[
    [
        "stay_id",
        "study_id",
        "study_datetime",
        "cxr_report",
        "has_cxr"
    ]
].copy()

print("\nBuilding final cohort ...")

final = cohort.merge(
    aligned_small,
    on="stay_id",
    how="left"
)

# Optional text modality
final["has_cxr"] = (
    final["has_cxr"]
    .fillna(False)
    .astype(bool)
)

# ──────────────────────────────────────────────────────────────────────────────
# ADD FINAL BINARY LABEL COLUMN
# ──────────────────────────────────────────────────────────────────────────────

# 1 = ICU LOS > 7 days
# 0 = ICU LOS <= 7 days

final["los_gt7"] = (final["los"] > 7).astype(int)

# Move label column to LAST position
label_col = final.pop("los_gt7")
final["los_gt7"] = label_col

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

print(f"  final cohort size: {len(final):,}")
print(f"  patients WITH CXR : {final['has_cxr'].sum():,}")
print(f"  patients WITHOUT  : {(~final['has_cxr']).sum():,}")

label_counts = final["los_gt7"].value_counts().sort_index()

print("\nLabel distribution:")
print(f"  LOS <= 7 days (0): {label_counts.get(0, 0):,}")
print(f"  LOS >  7 days (1): {label_counts.get(1, 0):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

final.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(f"  {OUTPUT_PATH}")