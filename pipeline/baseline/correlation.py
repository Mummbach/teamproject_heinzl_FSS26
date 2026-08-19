"""
Feature Correlation Analysis — ICU Feature Redundancy Check
============================================================
Loads all engineered features, merges them, and computes pairwise Pearson
correlation. Feature pairs above |r| > 0.8 are identified and one of each
pair is flagged for removal to reduce redundancy before model training.

Run AFTER:  preprocessing/02_features.py  (ts_features, icd_features, atc_features must exist)
            preprocessing/01_selection.py (cohort.csv must exist)

Note: This is an exploratory/analysis step — it does not write output files.
      Inspect results to inform feature selection in downstream steps.

Input:   output/ts_features.parquet
         output/icd_features.parquet
         output/atc_features.parquet
         output/cohort.csv

Output:  (none — prints correlation pairs and shows heatmap plot)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # pipeline/ -> config.py / multimodal_utils.py
from config import OUTPUT_DIR

THRESHOLD = 0.8

# ## Load feature data
# Load the parquet files produced by `preprocessing/02_features.py` — these are the actual model features before any scaling or imputation.

ts     = pd.read_parquet(OUTPUT_DIR / "ts_features.parquet")
icd    = pd.read_parquet(OUTPUT_DIR / "icd_features.parquet")
atc    = pd.read_parquet(OUTPUT_DIR / "atc_features.parquet")
cohort = pd.read_csv(OUTPUT_DIR / "cohort.csv",
                     usecols=["stay_id", "age_at_icu", "gender", "admission_type", "insurance"])

print(f"ts_features : {ts.shape}")
print(f"icd_features: {icd.shape}")
print(f"atc_features: {atc.shape}")
print(f"cohort      : {cohort.shape}")

# Encode categorical demographics as binary columns
demo = pd.get_dummies(cohort, columns=["gender", "admission_type", "insurance"]).astype(float)
demo.columns = [c.lower().replace(" ", "_") for c in demo.columns]

# Merge all on stay_id
df = (ts
      .merge(demo, on="stay_id", how="inner")
      .merge(icd,  on="stay_id", how="left")
      .merge(atc,  on="stay_id", how="left"))
df = df.fillna(0)
print(f"Merged: {df.shape[0]:,} stays × {df.shape[1]} columns")

# Filter to training stays only to avoid val/test leakage
train_ids = pd.read_parquet(OUTPUT_DIR / "split_ids.parquet")
train_ids = train_ids[train_ids["split"] == "train"]["stay_id"]
df = df[df["stay_id"].isin(train_ids)]
print(f"After train filter: {df.shape[0]:,} stays × {df.shape[1]} columns")

# ## 3. Prepare features
# Keep only numerical columns (drop `stay_id` (as its just a random identifier with no medical meaning) and any text), then fill missing values with the column median.

# Select numerical columns only, drop stay_id
feat_df = df.drop(columns=["stay_id"])

# Fill missing values with the median of each column
feat_df = feat_df.fillna(feat_df.median())

print(f"Features ready: {feat_df.shape[1]} columns, {feat_df.isnull().sum().sum()} NaNs remaining")
print(feat_df.head(3))

# ## 4. Compute Pearson correlation matrix
# Every feature is compared against every other feature. Values range from -1 to +1.

pcorr_matrix = feat_df.corr(method="pearson")
print(f"Correlation matrix: {pcorr_matrix.shape}")

# ## 5. Find pairs above threshold
# We look at the upper triangle of the matrix (to avoid counting each pair twice) and list all pairs where |r| > 0.8.

# Upper triangle only (avoids A-B and B-A duplicates)
upper = pcorr_matrix.where(np.triu(np.ones(pcorr_matrix.shape), k=1).astype(bool))

high_corr = (
    upper.stack()
    .reset_index()
    .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "r"})
)
high_corr = (
    high_corr[high_corr["r"].abs() > THRESHOLD]
    .assign(abs_r=lambda x: x["r"].abs())       # get absolute value of r to sort by correlation strength
    .sort_values("abs_r", ascending=False)
    .reset_index(drop=True)
)

print(f"{len(high_corr)} pairs with |r| > {THRESHOLD}:\n")
print(high_corr)

# ## 6. Visualise — heatmap of high-correlation features
# Shows only the features involved in at least one high-correlation pair, so the plot stays readable.

involved = pd.unique(high_corr[["feature_a", "feature_b"]].values.ravel())

if len(involved) > 0:
    sub = pcorr_matrix.loc[involved, involved]
    sz  = min(max(len(involved) * 0.5, 6), 30)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
    sns.heatmap(sub, annot=len(involved) <= 30, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title(f"Features with |r| > {THRESHOLD} (n={len(involved)})")
    plt.tight_layout()
    plt.show()
else:
    print("No high-correlation pairs found.")

# ## 7. Remove redundant features
# For each correlated pair we keep `feature_a` and drop `feature_b`. A feature is only dropped once even if it appears in multiple pairs.

to_drop = set()

for _, row in high_corr.iterrows():
    # Only drop feature_b if feature_a is not already being dropped itself
    if row["feature_a"] not in to_drop:
        to_drop.add(row["feature_b"])

print(f"Dropping {len(to_drop)} features:")
for f in sorted(to_drop):
    print(f"  - {f}")

feat_df_clean = feat_df.drop(columns=list(to_drop))

print(f"Before: {feat_df.shape[1]} features")
print(f"After:  {feat_df_clean.shape[1]} features")
print(f"Removed: {feat_df.shape[1] - feat_df_clean.shape[1]} features")
