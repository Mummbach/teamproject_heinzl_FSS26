"""

Extract structured radiology-report sections from cohort_with_cxr.csv.

=======================

Run AFTER:  01b_align_cxr_reports.py
Run BEFORE: 01d_extract_radiology_features.py

Ported from feature/time-series-monitoring-intensity-analysis
(10_extract_cxr_sections.py). Renamed 01c_ to keep the CXR sub-chain together
right after 01b_ (depends only on cohort_with_cxr.csv, not on 02-07).

For each patient / ICU stay:
    - extracts FINDINGS section
    - extracts IMPRESSION section
    - keeps binary ICU LOS label:
            los_gt7
                1 = ICU LOS > 7 days
                0 = ICU LOS <= 7 days


Input:      output/cohort_with_cxr.csv

Output:     output/cxr_sections.csv


Output columns
--------------
subject_id
hadm_id
stay_id
findings
impression
los_gt7

Notes
-----
- Patients WITHOUT radiology reports are removed
  because there is no text to extract sections from.

- Section extraction is robust to common MIMIC-CXR formatting:
      FINDINGS:
      IMPRESSION:
      Findings:
      Impression:

- If a section is missing:
      -> empty string ""
"""

import re
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # pipeline/ -> config.py / multimodal_utils.py
from config import OUTPUT_DIR, extract_section as _extract_section_raw

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

INPUT_PATH = OUTPUT_DIR / "cohort_with_cxr.csv"

OUTPUT_PATH = OUTPUT_DIR / "cxr_sections.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Load cohort with reports
# ──────────────────────────────────────────────────────────────────────────────

print("Loading cohort_with_cxr.csv ...")

df = pd.read_csv(INPUT_PATH)

print(f"  rows: {len(df):,}")

# Keep only patients with report text
df = df[
    df["cxr_report"].notna()
].copy()

print(f"  rows with reports: {len(df):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Section extraction helper
# ──────────────────────────────────────────────────────────────────────────────

def extract_section(text: str, section_name: str) -> str:
    """
    Extract section from radiology report (see config.extract_section),
    with repeated whitespace collapsed for clean CSV output.
    """
    extracted = _extract_section_raw(text, section_name)
    return re.sub(r"\s+", " ", extracted) if extracted else extracted


# ──────────────────────────────────────────────────────────────────────────────
# Extract sections
# ──────────────────────────────────────────────────────────────────────────────

print("\nExtracting FINDINGS and IMPRESSION sections ...")

df["findings"] = df["cxr_report"].apply(
    lambda x: extract_section(x, "FINDINGS")
)

df["impression"] = df["cxr_report"].apply(
    lambda x: extract_section(x, "IMPRESSION")
)

# remove rows where BOTH sections are empty

before = len(df)

df = df[
    (df["findings"] != "") |
    (df["impression"] != "")
].copy()

removed = before - len(df)

print(f"  removed empty-section reports: {removed:,}")

# ──────────────────────────────────────────────────────────────────────────────
# Build final dataset
# ──────────────────────────────────────────────────────────────────────────────

final = df[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "findings",
        "impression",
        "los_gt7"
    ]
].copy()

# Ensure label is integer
final["los_gt7"] = final["los_gt7"].astype(int)

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

print("\nFinal dataset summary")

print(f"  rows: {len(final):,}")

label_counts = final["los_gt7"].value_counts().sort_index()

print(f"  LOS <= 7 days (0): {label_counts.get(0, 0):,}")
print(f"  LOS >  7 days (1): {label_counts.get(1, 0):,}")

# Missing-section statistics
missing_findings = (final["findings"] == "").sum()
missing_impression = (final["impression"] == "").sum()

print(f"  missing findings   : {missing_findings:,}")
print(f"  missing impression : {missing_impression:,}")

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

final.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(f"  {OUTPUT_PATH}")
