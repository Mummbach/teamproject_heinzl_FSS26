"""

Extract BioClinicalBERT embeddings from MIMIC-CXR reports
aligned with ICU cohort. 

========================================
Run AFTER:  10_extract_cxr_sections.py
Run BEFORE: 12_build_multimodal_dataset.py

Input:      output/cohort_with_cxr.csv

Output:     output/cxr_bert_embeddings.parquet

Each row corresponds to one ICU stay:
    - findings embedding (768)
    - impression embedding (768)
    - concatenated embedding (1536)
    - los_gt7 label

Model:
    BioClinicalBERT (HuggingFace)
"""


import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config import OUTPUT_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

INPUT_PATH = OUTPUT_DIR / "cohort_with_cxr.csv"
OUTPUT_PATH = OUTPUT_DIR / "cxr_bert_embeddings.parquet"

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

print("Loading cohort_with_cxr.csv ...")
df = pd.read_csv(INPUT_PATH)

print(f"  rows: {len(df):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Ensure label exists
# ──────────────────────────────────────────────────────────────────────────────

if "los_gt7" not in df.columns:
    print("WARNING: los_gt7 missing → recomputing from los")
    df["los_gt7"] = (df["los"] > 7).astype(int)

df["los_gt7"] = df["los_gt7"].astype(int)

# ──────────────────────────────────────────────────────────────────────────────
# Keep only rows with reports
# ──────────────────────────────────────────────────────────────────────────────

df = df[df["cxr_report"].notna()].copy()
print(f"  usable reports: {len(df):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

print("\nLoading BioClinicalBERT ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# Section extraction
# ──────────────────────────────────────────────────────────────────────────────

import re

def extract_section(text, section):
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", "\n")

    pattern = rf"{section}\s*:\s*(.*?)(?=\n[A-Z ]+\s*:|\Z)"

    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    return match.group(1).strip() if match else ""

# ──────────────────────────────────────────────────────────────────────────────
# Embedding function
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed(text: str):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(768, dtype=np.float32)

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

# ──────────────────────────────────────────────────────────────────────────────
# Compute embeddings
# ──────────────────────────────────────────────────────────────────────────────

findings_emb = []
impression_emb = []
combined_emb = []

print("\nExtracting embeddings ...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    report = row["cxr_report"]

    f = extract_section(report, "FINDINGS")
    i = extract_section(report, "IMPRESSION")

    ef = embed(f)
    ei = embed(i)

    findings_emb.append(ef)
    impression_emb.append(ei)
    combined_emb.append(np.concatenate([ef, ei]))

# ──────────────────────────────────────────────────────────────────────────────
# Build output
# ──────────────────────────────────────────────────────────────────────────────

out = pd.DataFrame({
    "subject_id": df["subject_id"].values,
    "hadm_id": df["hadm_id"].values,
    "stay_id": df["stay_id"].values,
    "los_gt7": df["los_gt7"].values   # now guaranteed to exist
})

find_df = pd.DataFrame(findings_emb)
find_df.columns = [f"find_{i}" for i in range(768)]

imp_df = pd.DataFrame(impression_emb)
imp_df.columns = [f"imp_{i}" for i in range(768)]

cxr_df = pd.DataFrame(combined_emb)
cxr_df.columns = [f"cxr_{i}" for i in range(1536)]

out = pd.concat([out, find_df, imp_df, cxr_df], axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

print("\nSaving embeddings ...")
out.to_parquet(OUTPUT_PATH, index=False)

print(f"Saved: {OUTPUT_PATH}")
print(f"Shape: {out.shape}")