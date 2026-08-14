"""
Train / Validation / Test Split
================================
Splits the cohort stay_ids into three non-overlapping sets.
Only stay_id and the stratification label (los_gt7) are needed here —
no features are loaded, so no data leakage is possible.

Split:  70 % train  /  15 % validation  /  15 % test
        Stratified on los_gt7 to preserve label ratio in all three splits.
        Random seed fixed for reproducibility.

Run AFTER:  02_features.py
Run BEFORE: 04_preprocessing.py

Input:   output/cohort.csv
Output:  output/split_ids.parquet   — stay_id + split column ("train"/"val"/"test")
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from config import OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True)

VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_STATE = 42

# Load only what is needed for splitting — no features
cohort = pd.read_csv(OUTPUT_DIR / "cohort.csv", usecols=["stay_id", "los_gt7"])
print(f"Cohort loaded: {len(cohort):,} stays")
print(f"  los_gt7 = 1 : {cohort['los_gt7'].sum():,}  ({cohort['los_gt7'].mean()*100:.1f}%)")
print(f"  los_gt7 = 0 : {(cohort['los_gt7']==0).sum():,}  ({(1-cohort['los_gt7'].mean())*100:.1f}%)")

# Step 1: split off test set
trainval, test = train_test_split(
    cohort,
    test_size=TEST_RATIO,
    stratify=cohort["los_gt7"],
    random_state=RANDOM_STATE,
)

# Step 2: split remainder into train + val
# val_ratio must be adjusted relative to the remaining trainval portion
val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)

train, val = train_test_split(
    trainval,
    test_size=val_ratio_adjusted,
    stratify=trainval["los_gt7"],
    random_state=RANDOM_STATE,
)

# Assign split labels
train = train.copy()
train["split"] = "train"
val   = val.copy()
val["split"]   = "val"
test  = test.copy()
test["split"]  = "test"

split_ids = pd.concat([train, val, test], ignore_index=True)[["stay_id", "split"]]

print(f"\nSplit sizes:")
for name in ["train", "val", "test"]:
    subset = split_ids[split_ids["split"] == name]
    n = len(subset)
    pos = cohort.loc[cohort["stay_id"].isin(subset["stay_id"]), "los_gt7"].mean() * 100
    print(f"  {name:<6} {n:>7,}  ({n/len(cohort)*100:.1f}%)  positive rate: {pos:.1f}%")

# Sanity checks
assert split_ids["stay_id"].nunique() == len(split_ids), "duplicate stay_ids in split"
assert len(split_ids) == len(cohort), "stay count mismatch after split"
sets = [set(split_ids[split_ids["split"]==s]["stay_id"]) for s in ["train","val","test"]]
assert sets[0].isdisjoint(sets[1]), "train/val overlap"
assert sets[0].isdisjoint(sets[2]), "train/test overlap"
assert sets[1].isdisjoint(sets[2]), "val/test overlap"
print("\n  No stay_id overlap between splits ✓")

split_ids.to_parquet(OUTPUT_DIR / "split_ids.parquet", index=False)
print(f"  Saved: output/split_ids.parquet")
