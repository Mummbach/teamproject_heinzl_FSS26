"""
CXR Feature Extraction — BioClinicalBERT Embeddings
=====================================================
Aligns MIMIC-CXR radiology reports with the ICU cohort and extracts
BioClinicalBERT embeddings from the FINDINGS and IMPRESSION sections.

Run AFTER:  01_selection.py
Run BEFORE: 07_model_gru.py

Input:   output/cohort.csv
         data/mimic-cxr-2.0.0-metadata.csv
         data/files/<p##>/<p#######>/<s#######>.txt

Output:  output/cxr_bert_embeddings.parquet

         Each row = one ICU stay with a valid report:
           stay_id, subject_id, hadm_id
           find_0 … find_767    — FINDINGS embedding   (768-dim)
           imp_0  … imp_767     — IMPRESSION embedding (768-dim)
           cxr_0  … cxr_1535   — concatenated         (1536-dim)

Notes:
  - Only reports written within the first 48h of ICU admission are used.
  - One report per stay (the earliest within the window).
  - Patients without a valid report are absent from the output.
    The GRU model handles them via zero-vector fallback.
  - BioClinicalBERT is run in inference-only mode (no fine-tuning).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config import DATA_DIR, OUTPUT_DIR, OBS_WINDOW

# ── Paths ─────────────────────────────────────────────────────────────
CXR_DIR       = DATA_DIR / "files"
METADATA_PATH = DATA_DIR / "mimic-cxr-2.0.0-metadata.csv"
COHORT_PATH   = OUTPUT_DIR / "cohort.csv"
OUTPUT_PATH   = OUTPUT_DIR / "cxr_bert_embeddings.parquet"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


# Align reports to ICU stays
print("Loading ICU cohort ...")
cohort = pd.read_csv(COHORT_PATH, parse_dates=["intime", "outtime"])
print(f"  cohort stays: {len(cohort):,}")

cohort["obs_end"] = cohort["intime"] + pd.Timedelta(hours=OBS_WINDOW)

print("\nLoading MIMIC-CXR metadata ...")
meta = pd.read_csv(METADATA_PATH)
meta["subject_id"] = meta["subject_id"].astype(int)
meta["study_id"]   = meta["study_id"].astype(int)
print(f"  metadata rows: {len(meta):,}")

meta = meta[meta["subject_id"].isin(set(cohort["subject_id"]))].copy()
print(f"  rows after subject filter: {len(meta):,}")

meta["StudyDate"] = meta["StudyDate"].astype(str)
meta["StudyTime"] = (
    meta["StudyTime"].fillna(0).astype(str)
    .str.split(".").str[0].str.zfill(6)
)
meta["study_datetime"] = pd.to_datetime(
    meta["StudyDate"] + meta["StudyTime"],
    format="%Y%m%d%H%M%S",
    errors="coerce",
)
meta = meta.dropna(subset=["study_datetime"])
print(f"  rows with valid timestamps: {len(meta):,}")

print("\nAligning reports to ICU stays ...")
merged = cohort.merge(
    meta[["subject_id", "study_id", "study_datetime"]],
    on="subject_id",
    how="left",
)
within_window = (
    (merged["study_datetime"] >= merged["intime"]) &
    (merged["study_datetime"] <= merged["obs_end"])
)
aligned = (
    merged[within_window]
    .sort_values(["stay_id", "study_datetime"])
    .drop_duplicates(subset="stay_id", keep="first")
    .copy()
)
print(f"  stays with report in window: {len(aligned):,}")


# Load report text files
def get_report_path(subject_id: int, study_id: int) -> Path:
    subject_str = str(int(subject_id))
    return CXR_DIR / f"p{subject_str[:2]}" / f"p{subject_str}" / f"s{int(study_id)}.txt"


def load_report_text(path: Path):
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text if text else None
    except Exception:
        return None


print("\nLoading report texts ...")
aligned["report_path"] = aligned.apply(
    lambda r: get_report_path(r["subject_id"], r["study_id"]), axis=1
)
aligned["cxr_report"] = aligned["report_path"].apply(load_report_text)

df = aligned[aligned["cxr_report"].notna()].copy()
print(f"  usable reports: {len(df):,}")


# BioClinicalBERT embeddings
print(f"\nLoading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert      = AutoModel.from_pretrained(MODEL_NAME)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert      = bert.to(device)
bert.eval()
print(f"  device: {device}")


def extract_section(text: str, section: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", "\n")
    match = re.search(
        rf"{section}\s*:\s*(.*?)(?=\n[A-Z ]+\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


@torch.no_grad()
def embed(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768, dtype=np.float32)
    inputs = tokenizer(
        text, truncation=True, padding=True,
        max_length=256, return_tensors="pt",
    ).to(device)
    outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()


print("\nExtracting embeddings ...")
findings_emb, impression_emb, combined_emb = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    ef = embed(extract_section(row["cxr_report"], "FINDINGS"))
    ei = embed(extract_section(row["cxr_report"], "IMPRESSION"))
    findings_emb.append(ef)
    impression_emb.append(ei)
    combined_emb.append(np.concatenate([ef, ei]))


# Save
out = pd.DataFrame({
    "subject_id": df["subject_id"].values,
    "hadm_id":    df["hadm_id"].values,
    "stay_id":    df["stay_id"].values,
})

find_df = pd.DataFrame(findings_emb, columns=[f"find_{i}" for i in range(768)])
imp_df  = pd.DataFrame(impression_emb, columns=[f"imp_{i}"  for i in range(768)])
cxr_df  = pd.DataFrame(combined_emb,  columns=[f"cxr_{i}"  for i in range(1536)])

out = pd.concat([out, find_df, imp_df, cxr_df], axis=1)

out.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"Shape: {out.shape}")
