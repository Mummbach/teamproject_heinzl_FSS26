# 🧠 Case-Based Explanations for Multimodal Clinical Decision-Making

> **Teamproject FSS 2026** · Chair Prof. Heinzl · Universität Mannheim
> Supervised by Florian Rüffer

---

## ⚙️ Pipeline Configuration

The pipeline behaviour is controlled by flags at the top of each script — no code changes needed, just flip the value and re-run.

### `pipeline/baseline/01_selection.py`

| Variable | Values | Description |
|---|---|---|
| `EXCLUDE_EARLY_DEATHS` | `False` (default) / `True` | Exclude patients who died between 48h and 7d after ICU admission (bias check) |

### `pipeline/baseline/04_preprocessing.py`

| Variable | Values | Description |
|---|---|---|
| `IMPUTATION_STRATEGY` | `"median"` (default) / `"mean"` / `"rf"` | Strategy for filling missing feature values |
| `USE_MISSINGNESS_FLAGS` | `True` (default) / `False` | Include binary `_missing` flags as extra features |
| `USE_AGGREGATED_VITALS` | `True` (default) / `False` | Include aggregated vital sign stats (mean, std, slope etc.) in the static feature set |

### `pipeline/baseline/07_model_gru.py`

| Variable | Values | Description |
|---|---|---|
| `USE_HOURLY_TIMESERIES` | `True` (default) / `False` | Enable GRU branch on 48h × 12 vital hourly time-series |

### Typical comparison runs

| Experiment | `USE_AGGREGATED_VITALS` | `USE_HOURLY_TIMESERIES` |
|---|---|---|
| Full model | `True` | `True` |
| No hourly TS (ablation) | `True` | `False` |
| No aggregated vitals (ablation) | `False` | `True` |
| Static only (ablation) | `False` | `False` |

---

## 📌 Research Question

> *How can comparative reasoning against similar patients be leveraged to produce clinically meaningful and faithful explanations for AI-based predictions in critical care?*

We develop **Patient Similarity-Based Graph Neural Networks** that explain AI predictions in critical care by referencing comparable patients — enabling both **factual** ("why this outcome?") and **contrastive** ("what would need to change?") explanations.

---

## 🗂️ Repository Structure

```
teamproject_heinzl_FSS26/
├── pipeline/
│   ├── config.py              # shared paths/constants — imported by all three tracks below
│   ├── multimodal_utils.py    # shared GRU/SHAP dataset + model helpers (baseline + explainability/shap_prdnet.py)
│   ├── data/                  # raw MIMIC-IV / MIMIC-CXR tables (gitignored, PhysioNet-credentialed)
│   ├── output/                # generated features/models/plots, shared across all three tracks (gitignored)
│   │
│   ├── baseline/               # 01–14: cohort → features → GRU baseline (the "core" numbered pipeline)
│   │   ├── 01_selection.py … 01e_extract_bioclinicalbert_embeddings.py   # optional CXR/BERT chain
│   │   ├── 02_features.py, 03_splitting.py, 04_preprocessing.py, 06_normalize.py
│   │   ├── 07_model_gru.py, 08_crossval.py, 08b_hyperparameter_search.py
│   │   ├── 09_shap.py, 10_explainability.py, 11_timeshap.py             # baseline GRU explainability
│   │   ├── 12_ts_monitoring.py … 14_patient_mii_clustering.py           # monitoring-intensity track
│   │   └── correlation.py                                               # feature-redundancy EDA
│   │
│   ├── prd_net/                # v1: latent-space Patient-peer Reference/Difference net (GRU embedding delta)
│   ├── prd_net_v2/             # v2: feature-space contrastive difference model (see its README for the full story)
│   │   └── README.md           # design decisions, reproduction steps, results — start here for PRD-Net
│   │
│   └── explainability/         # shap_prdnet.py — SHAP explanations for prd_net v1 (the only file here; see note below)
│
├── archive/                    # earlier pre-restructure notebooks/figures/results, kept for reference only
├── docs/                       # misc process/spec docs
├── presentations/              # slide decks
└── requirements.txt             # full dev-env freeze; pipeline/requirements.txt is the curated, pinned runtime list
```

> **Note on `explainability/`**: this folder used to hold a second, independently-maintained
> implementation of the SHAP/explainability/TimeSHAP logic that already lives in
> `baseline/09_shap.py`/`10_explainability.py`/`11_timeshap.py` — a leftover from a
> 2026-07-28 five-branch merge that kept a losing branch's rewrite "folder-namespaced"
> instead of deleting it. Those three duplicate scripts (plus their own forked
> `multimodal_utils.py`) were removed on 2026-08-17; only `shap_prdnet.py` remains,
> since it's the unique tool that explains **prd_net v1** and has no baseline
> equivalent.

---

## 📦 Dataset

This project uses **[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)** — a large, freely available database of de-identified Electronic Health Records from the Beth Israel Deaconess Medical Center ICU.

| Property | Detail |
|---|---|
| **Dataset** | MIMIC-IV ICU |
| **Size** | ~70,000 ICU stays |
| **Modalities** | Time series vitals, clinical text reports, tabular features |
| **Extension** | MIMIC-CXR-JPG (lung images) |
| **Access** | Requires PhysioNet credentialing |

### Prediction Tasks

| Task | Type | Label |
|---|---|---|
| **Length of Stay** | Binary classification | > 7 days |
| **48-h Mortality** | Binary classification | Death within 48h |
| **Readmission** | Binary classification | Readmission within 30 days |

> ⚠️ **Data Access**: MIMIC-IV is not included in this repository. You must apply for access via [PhysioNet](https://physionet.org/content/mimiciv/3.1/). Once approved, follow the "Data layout" instructions in [`pipeline/prd_net_v2/README.md`](pipeline/prd_net_v2/README.md) to place the files correctly under `pipeline/data/`.

---

## 🛠️ Local Environment Setup

### 1️⃣ Clone the repository

```bash
git clone git@github.com:Mummbach/teamproject_heinzl_FSS26.git
cd teamproject_heinzl_FSS26
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv venv
```

### 3️⃣ Activate the virtual environment

```bash
source venv/bin/activate
```

Your prompt should change to:

```
(venv) Mummbach@ubuntu:~/teamproject_heinzl_FSS26$
```

To deactivate at any time:

```bash
deactivate
```

### 4️⃣ Install all required dependencies

```bash
pip install -r pipeline/requirements.txt
```

### 5️⃣ Register the environment as a Jupyter kernel

```bash
python3 -m ipykernel install --user --name=cbr_env --display-name "CBR Clinical GNN"
```

### 6️⃣ Select the kernel in VS Code

1. Open the Command Palette → `Ctrl + Shift + P`
2. Search for **Python: Select Interpreter**
3. Choose **CBR Clinical GNN** (or `venv` if you skipped step 5)

You'll see it confirmed in the bottom-right corner of VS Code.

### 7️⃣ Verify the setup

Open `notebooks/01_data_exploration.ipynb` and run:

```python
import torch
import torch_geometric
import pandas as pd
import numpy as np

print("Environment setup successful!")
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
```

If it runs without error — 🎉 you're ready to go!

### 🧠 Notes

- `venv/` is excluded from Git via `.gitignore` — never push it.
- If you install new packages, update the team's dependency list:

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
```

---

## 🔬 Reproducing the PRD-Net results

The implemented pipeline — cohort → features → GRU → PRD-Net → the feature-level
contrastive **difference track** — and a full **step-by-step reproduction from the
raw MIMIC tables** live in
**[`pipeline/prd_net_v2/README.md`](pipeline/prd_net_v2/README.md)** (see *"Full
reproduction from raw MIMIC"*). Dependencies are pinned in
[`pipeline/requirements.txt`](pipeline/requirements.txt); the GRU trains on CPU
(no GPU required), ~15 min end-to-end.

Headline test-set results (length-of-stay > 7 days, 23.6 % positive; full rebuild
2026-07-26):

| model | F1 | AUROC | AUPRC |
|---|--:|--:|--:|
| GRU (baseline, 48h) | 0.611 | 0.848 | 0.644 |
| PRD-Net latent (48h) | 0.621 | 0.835 | 0.607 |
| PRD-Net feature-diff (48h) | 0.601 | 0.826 | 0.600 |
| PRD-Net feature-diff (24h) | 0.552 | 0.792 | 0.533 |

---

## 🧭 Git Workflow Guide

### 1. Update main before branching

```bash
git checkout main
git pull origin main
```

### 2. Create a feature branch

```bash
git checkout -b feature/your-branch-name
```

**Branch naming conventions:**

| Prefix | Use case |
|---|---|
| `feature/` | New functionality |
| `fix/` | Bug fix |
| `exp/` | Experiment run |
| `docs/` | Documentation update |

### 3. Stage and commit

```bash
git add .
git commit -m "Add patient similarity graph construction"
```

### 4. Push your branch

```bash
git push -u origin feature/your-branch-name
```

### 5. Open a Pull Request

1. Go to the repository on GitHub
2. Click **Compare & Pull Request**
3. Target branch: `main`
4. Request at least one review before merging

### 6. Clean up after merge

```bash
# Delete remote branch
git push origin --delete feature/your-branch-name

# Delete local branch
git branch -d feature/your-branch-name
```

---

## 🔬 Methodology Overview

```
MIMIC-IV ICU Data
      │
      ▼
Multimodal Feature Encoding
  ├── Time series vitals  → Temporal encoder (LSTM / Transformer)
  ├── Clinical text       → Text encoder (BioClinicalBERT)
  └── Tabular features    → MLP encoder
      │
      ▼
Patient Similarity Graph Construction
  (k-NN based on encoded patient representations)
      │
      ▼
Graph Neural Network (GNN)
  (Message passing over patient graph)
      │
      ▼
Prediction + Case-Based Explanation
  ├── Factual:     "Similar patients also stayed > 7 days"
  └── Contrastive: "Peers with < 7 days had better SpO2"
```

---

## 📊 Evaluation

We evaluate explanations along two dimensions:

- **Faithfulness**: Do the explanations reflect the model's actual reasoning?
- **Clinical Meaningfulness**: Are the cited patient similarities clinically plausible?

Metrics include: fidelity, explanation stability, nearest-neighbor alignment, and expert evaluation.

---

## 👥 Team

| Name | Role |
|---|---|
| Florian Rüffer | Supervisor |
| Julia Kocharina | |
| Johannes Kramberg | |
| Mika Luu | |
| Ebubekir Günaydin | |
| Maximilian Rumbach | |

---

## 📄 License

This project is for academic use only. MIMIC-IV data usage is governed by the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/3.1/).

---

*Teamproject FSS 2026 · Chair Prof. Heinzl · Universität Mannheim · [Mummbach](https://github.com/Mummbach)*
