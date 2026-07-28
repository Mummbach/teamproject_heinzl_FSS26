"""
Patient Monitoring Intensity Pipeline
=====================================

Aggregates feature-level ICU monitoring intensity into
clinically meaningful patient groups:

1. Load feature-level MII
2. Aggregate to patient-level metrics
3. Compute Patient Monitoring Intensity (PMI)
4. Create monitoring groups
5. Save patient-level outputs
6. Create t-SNE phenotype visualization

Inputs:
    output/mii_features.parquet

Outputs:
    output/patient_mii.parquet
    output/patient_mii_summary.csv
    output/patient_monitoring_embedding.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from config import OUTPUT_DIR


# ============================================================
# LOAD FEATURE-LEVEL MII
# ============================================================

print("Loading MII feature-level data...")

mii = pd.read_parquet(
    OUTPUT_DIR / "mii_features.parquet"
)

# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

metric_cols = [c for c in mii.columns if c != "stay_id"]

freq_cols = [c for c in metric_cols if c.endswith("_freq")]
cov_cols = [c for c in metric_cols if c.endswith("_coverage")]
irr_cols = [c for c in metric_cols if c.endswith("_irregularity")]

# ============================================================
# AGGREGATE TO PATIENT LEVEL
# ============================================================

print("Computing patient-level monitoring metrics...")

patient = pd.DataFrame({
    "stay_id": mii["stay_id"]
})

# overall monitoring burden
patient["total_freq"] = mii[freq_cols].sum(axis=1)

# average temporal coverage
patient["mean_coverage"] = mii[cov_cols].mean(axis=1)

# most intensely monitored feature
patient["max_freq"] = mii[freq_cols].max(axis=1)

# temporal irregularity
patient["mean_irregularity"] = mii[irr_cols].mean(axis=1)

# number of monitored variables
patient["active_features"] = (
    mii[freq_cols] > 0
).sum(axis=1)

# ============================================================
# PATIENT MONITORING INTENSITY SCORE (PMI)
# ============================================================

print("Computing PMI...")


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


patient["PMI"] = (
    0.4 * normalize(patient["total_freq"])
    + 0.2 * normalize(patient["mean_coverage"])
    + 0.2 * normalize(patient["max_freq"])
    + 0.2 * normalize(patient["active_features"])
)

# ============================================================
# CLINICAL MONITORING GROUPS
# ============================================================

print("Creating monitoring groups...")

patient["monitoring_group"] = pd.qcut(
    patient["PMI"],
    q=4,
    labels=[
        "Low",
        "Moderate",
        "High",
        "Critical"
    ]
)

# ============================================================
# SUMMARY TABLE
# ============================================================

summary = patient.groupby(
    "monitoring_group",
    observed=False
).agg(
    n_patients=("stay_id", "count"),
    mean_freq=("total_freq", "mean"),
    mean_coverage=("mean_coverage", "mean"),
    mean_active_features=("active_features", "mean"),
    mean_irregularity=("mean_irregularity", "mean"),
    mean_pmi=("PMI", "mean")
).reset_index()

print("\nPatient Monitoring Groups")
print("=" * 70)
print(summary.to_string(index=False))

# ============================================================
# SAVE TABULAR OUTPUTS
# ============================================================

patient.to_parquet(
    OUTPUT_DIR / "patient_mii.parquet",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "patient_mii_summary.csv",
    index=False
)

print("\nSaved:")
print("  patient_mii.parquet")
print("  patient_mii_summary.csv")

# ============================================================
# PATIENT PHENOTYPE EMBEDDING
# ============================================================

print("\nCreating patient phenotype embedding...")

feature_cols = [
    "total_freq",
    "mean_coverage",
    "max_freq",
    "mean_irregularity",
    "active_features",
    "PMI"
]

X = patient[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# UMAP / t-SNE
# ============================================================

if UMAP_AVAILABLE:

    print("Using UMAP...")

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    embedding = reducer.fit_transform(X_scaled)
    method = "UMAP"

else:

    print("UMAP unavailable → using t-SNE")

    reducer = TSNE(
        n_components=2,
        perplexity=40,
        random_state=42,
        init="pca",
        learning_rate="auto"
    )

    embedding = reducer.fit_transform(X_scaled)
    method = "t-SNE"

# ============================================================
# VISUALIZATION
# ============================================================

colors = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Critical": "#e74c3c"
}

plt.figure(figsize=(10, 7))

for group in patient["monitoring_group"].unique():

    idx = patient["monitoring_group"] == group

    plt.scatter(
        embedding[idx, 0],
        embedding[idx, 1],
        s=8,
        alpha=0.6,
        c=colors.get(group, "gray"),
        label=group
    )

plt.title(
    f"ICU Patient Monitoring Phenotypes ({method})"
)

plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.legend(
    title="Monitoring Group"
)

plt.tight_layout()

out_path = (
    OUTPUT_DIR /
    "patient_monitoring_embedding.png"
)

plt.savefig(
    out_path,
    dpi=300
)

plt.show()

print(f"\nSaved visualization: {out_path}")

# ============================================================
# DONE
# ============================================================

print("\nPipeline complete.")