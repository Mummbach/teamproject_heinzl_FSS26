"""
01d_extract_radiology_features.py
==================================

Extract structured radiology features from:
    - FINDINGS
    - IMPRESSION

Goal:
Predict ICU LOS > 7 days (los_gt7)

Ported from feature/time-series-monitoring-intensity-analysis
(12_extract_radiology_features.py). Renamed 01d_ to keep the CXR sub-chain
together right after 01c_ (depends only on cxr_sections.csv, not on 02-07).

Run AFTER: 01c_extract_cxr_sections.py

Input:
    output/cxr_sections.csv

Output:
    output/cxr_structured_features.csv
"""

import re
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # pipeline/ -> config.py / multimodal_utils.py
from config import OUTPUT_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

INPUT_PATH = OUTPUT_DIR / "cxr_sections.csv"

OUTPUT_PATH = OUTPUT_DIR / "cxr_structured_features.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

print("Loading radiology sections ...")

df = pd.read_csv(INPUT_PATH)

print(f"  rows: {len(df):,}")

# Combine findings + impression
df["report_text"] = (
    df["findings"].fillna("") + " " +
    df["impression"].fillna("")
).str.lower()

# ──────────────────────────────────────────────────────────────────────────────
# Clinical feature dictionaries
# ──────────────────────────────────────────────────────────────────────────────

PATHOLOGIES = {
    "pneumonia": [
        "pneumonia",
        "consolidation",
        "airspace opacity"
    ],

    "pleural_effusion": [
        "pleural effusion"
    ],

    "pneumothorax": [
        "pneumothorax"
    ],

    "edema": [
        "pulmonary edema",
        "edema"
    ],

    "atelectasis": [
        "atelectasis"
    ],

    "opacity": [
        "opacity",
        "opacities"
    ],

    "cardiomegaly": [
        "cardiomegaly",
        "enlarged cardiac silhouette"
    ]
}

SEVERITY_TERMS = [
    "bilateral",
    "diffuse",
    "multifocal",
    "extensive",
    "severe",
    "marked"
]

PROGRESSION_TERMS = {
    "worsening": [
        "worsening",
        "increased",
        "progression"
    ],

    "improved": [
        "improved",
        "decreased",
        "resolved"
    ],

    "stable": [
        "stable",
        "unchanged"
    ]
}

DEVICE_TERMS = {
    "ventilator": [
        "endotracheal tube",
        "et tube",
        "intubated"
    ],

    "central_line": [
        "central venous catheter",
        "central line",
        "picc"
    ],

    "chest_tube": [
        "chest tube"
    ]
}

NEGATIONS = [
    "no",
    "without",
    "negative for",
    "absence of"
]

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def contains_term(text, terms):
    return any(term in text for term in terms)

def negated(text, keyword):
    """
    Simple negation detection, scoped to the sentence containing the keyword.

    The old pattern matched "(no|without|negative for) ... keyword" over the
    whole report with no clause boundary, so a negation earlier in the report
    could falsely negate an unrelated, confirmed finding later on, e.g.
    "no evidence of pneumothorax and pneumonia is present" marked pneumonia
    as negated. Splitting on sentence punctuation keeps the check local.
    """

    pattern = rf"(no|without|negative for)\s+[\w\s]*{keyword}"

    for sentence in re.split(r"[.;]", text):
        if keyword in sentence and re.search(pattern, sentence):
            return True

    return False

# ──────────────────────────────────────────────────────────────────────────────
# Extract pathology features
# ──────────────────────────────────────────────────────────────────────────────

print("\nExtracting pathology features ...")

for feature, terms in PATHOLOGIES.items():

    values = []

    for text in df["report_text"]:

        found = False

        for term in terms:

            if term in text and not negated(text, term):
                found = True
                break

        values.append(int(found))

    df[feature] = values

# ──────────────────────────────────────────────────────────────────────────────
# Severity features
# ──────────────────────────────────────────────────────────────────────────────

print("Extracting severity features ...")

for term in SEVERITY_TERMS:
    df[f"severity_{term}"] = df["report_text"].apply(
        lambda x: int(term in x)
    )

# Severity score
severity_cols = [f"severity_{t}" for t in SEVERITY_TERMS]

df["severity_score"] = df[severity_cols].sum(axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# Progression features
# ──────────────────────────────────────────────────────────────────────────────

print("Extracting progression features ...")

for feature, terms in PROGRESSION_TERMS.items():

    df[feature] = df["report_text"].apply(
        lambda x: int(contains_term(x, terms))
    )

# ──────────────────────────────────────────────────────────────────────────────
# Device features
# ──────────────────────────────────────────────────────────────────────────────

print("Extracting device features ...")

for feature, terms in DEVICE_TERMS.items():

    df[feature] = df["report_text"].apply(
        lambda x: int(contains_term(x, terms))
    )

# ──────────────────────────────────────────────────────────────────────────────
# Complexity features
# ──────────────────────────────────────────────────────────────────────────────

print("Extracting report complexity features ...")

df["report_length"] = df["report_text"].apply(
    lambda x: len(x.split())
)

df["sentence_count"] = df["report_text"].apply(
    lambda x: len(re.split(r"[.!?]", x))
)

# Total abnormality burden
pathology_cols = list(PATHOLOGIES.keys())

df["abnormality_count"] = df[pathology_cols].sum(axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# Final dataset
# ──────────────────────────────────────────────────────────────────────────────

feature_cols = (
    pathology_cols +
    severity_cols +
    [
        "severity_score",
        "worsening",
        "improved",
        "stable",
        "ventilator",
        "central_line",
        "chest_tube",
        "report_length",
        "sentence_count",
        "abnormality_count"
    ]
)

final = df[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "los_gt7"
    ] + feature_cols
].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

final.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:")
print(f"  {OUTPUT_PATH}")

print("\nFeature summary:")
print(final[feature_cols].mean().sort_values(ascending=False))
