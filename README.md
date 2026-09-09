# 🧠 Case-Based Explanations for Multimodal Clinical Decision-Making

> **Teamproject FSS 2026** · Chair Prof. Heinzl · Universität Mannheim
> Supervised by Florian Rüffer

---

## ⚙️ Pipeline Configuration

The pipeline behaviour is controlled by flags at the top of each script — no code changes needed, just flip the value and re-run.

### `pipeline/preprocessing/01_selection.py`

| Variable | Values | Description |
|---|---|---|
| `EXCLUDE_EARLY_DEATHS` | `False` (default) / `True` | Exclude patients who died between 48h and 7d after ICU admission (bias check) |

### `pipeline/preprocessing/04_preprocessing.py`

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
│   ├── preprocessing/           # 01–06: cohort → features → split → impute → normalize (shared by all 3 tracks below)
│   │   ├── 01_selection.py … 01d_extract_radiology_features.py          # optional CXR chain
│   │   ├── 02_features.py, 02b_cxr_features.py, 03_splitting.py
│   │   └── 04_preprocessing.py, 05_analysis.py, 06_normalize.py
│   │
│   ├── baseline/                # 07–14: GRU baseline model + its explainability/monitoring
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
└── presentations/              # slide decks
```

> **Note on `requirements.txt`**: `pipeline/requirements.txt` is the single, curated
> dependency list for this project — pinned to the versions the reported results
> were produced with (see setup below). Other, unmaintained copies used to exist
> elsewhere in the repo (raw `pip freeze` snapshots that had gone stale relative
> to the actual code); those have been removed to avoid drift.

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

This project uses **[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)** — a large, freely available database of de-identified Electronic Health Records from the Beth Israel Deaconess Medical Center ICU — extended with **[MIMIC-CXR v2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/)** for the free-text chest X-ray radiology reports.

| Property | Detail |
|---|---|
| **Dataset** | MIMIC-IV ICU |
| **Size** | ~70,000 ICU stays |
| **Modalities** | Time series vitals, clinical text reports, tabular features |
| **Extension** | [MIMIC-CXR v2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/) (radiology reports) |
| **Access** | Requires PhysioNet credentialing |

### Prediction Tasks

| Task | Type | Label |
|---|---|---|
| **Length of Stay** | Binary classification | > 7 days |
| **48-h Mortality** | Binary classification | Death within 48h |
| **Readmission** | Binary classification | Readmission within 30 days |

> ⚠️ **Data Access**: Neither MIMIC-IV nor MIMIC-CXR is included in this repository. You must apply for access via [PhysioNet](https://physionet.org/content/mimiciv/3.1/) — MIMIC-CXR requires the same credentialing, granted separately via its own [PhysioNet page](https://physionet.org/content/mimic-cxr/2.1.0/). Once approved, follow the data layout below before running anything under `preprocessing/` — every track (`baseline/`, `prd_net/`, `prd_net_v2/`) shares these preprocessing scripts.

### Data Layout

Place the raw files under `pipeline/data/`:

```
pipeline/data/
  hosp/                             MIMIC-IV hosp  (admissions, patients, diagnoses_icd, prescriptions, …)
  icu/                              MIMIC-IV icu   (icustays, chartevents.csv.gz, outputevents, …)
  RXCUI2atc4.csv                    NDC → ATC mapping
  mimic-cxr-2.0.0-metadata.csv.gz
  mimic-cxr-reports/                per-subject radiology report .txt files (p10/, p11/, …)
```

> **Gotcha:** `preprocessing/01b_align_cxr_reports.py` expects the CXR metadata *inside*
> `mimic-cxr-reports/`. If yours sits directly in `data/`, symlink it once:
> ```bash
> ln -s ../mimic-cxr-2.0.0-metadata.csv.gz \
>   pipeline/data/mimic-cxr-reports/mimic-cxr-2.0.0-metadata.csv.gz
> ```

`RXCUI2atc4.csv` is not part of either PhysioNet dataset — it's a separate NDC→ATC
drug-code mapping. `config.py`'s in-code comment points to `MIT-LCP/mimic-code`,
but that link doesn't actually host this file; it originates from
[`sjy1203/GAMENet`](https://github.com/sjy1203/GAMENet/blob/master/data/ndc2atc_level4.csv)
(as `data/ndc2atc_level4.csv`) and is reused under this filename across several
downstream MIMIC medication-mapping projects.

---

## 🛠️ Local Environment Setup

### 1️⃣ Clone the repository

```bash
git clone git@github.com:Mummbach/teamproject_heinzl_FSS26.git
cd teamproject_heinzl_FSS26
```

### 2️⃣ Create a virtual environment

Requires **Python 3.13** (the version the pinned dependencies in
`pipeline/requirements.txt` were tested against).

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

Run this from the terminal (inside the activated venv):

```bash
python3 -c "
import torch, pandas, numpy, sklearn, shap
print('Environment setup successful!')
print(f'PyTorch : {torch.__version__}')
print(f'pandas  : {pandas.__version__}')
print(f'numpy   : {numpy.__version__}')
print(f'sklearn : {sklearn.__version__}')
print(f'shap    : {shap.__version__}')
"
```

If it runs without error — 🎉 you're ready to go!

### 🧠 Notes

- `venv/` is excluded from Git via `.gitignore` — never push it.
- If you install a new package, add it by hand to `pipeline/requirements.txt` with a
  pinned version and a short comment on which script needs it — don't overwrite the
  file with a raw `pip freeze` (that's how it drifted out of sync with the code
  before).

---

## 🚀 Running the Pipeline

Full reproduction from the raw MIMIC-IV / MIMIC-CXR tables, once the venv is set
up (above) and the raw files are placed per "Data Layout" (above). Dependencies
are pinned in [`pipeline/requirements.txt`](pipeline/requirements.txt); the GRU
trains on CPU (no GPU required), ~15 min end-to-end on a laptop (dominated by
the ~3.5 GB `chartevents.csv.gz` parse and GRU training).

This covers everything shared across tracks — cohort → features → GRU baseline
→ PRD-Net v1 (latent delta). Run from `pipeline/`:

```bash
cd pipeline
python3 preprocessing/01_selection.py                     # cohort.csv  (ICU cohort, LOS>7 label)
python3 preprocessing/01b_align_cxr_reports.py            # cohort_with_cxr.csv
python3 preprocessing/01c_extract_cxr_sections.py         # cxr_sections.csv       (FINDINGS/IMPRESSION)
python3 preprocessing/01d_extract_radiology_features.py   # cxr_structured_features.csv (16 CXR flags)
python3 preprocessing/02_features.py                      # timeseries.parquet, X_*, icd/atc/labels  (RANGE_FILTERS vitals fix applied here)
python3 preprocessing/03_splitting.py                     # split_ids.parquet      (70/15/15, seeded)
python3 preprocessing/04_preprocessing.py                 # X_*/y_*                 (median imputation)
python3 preprocessing/06_normalize.py                     # X_*_scaled + scaler_params
python3 baseline/07_model_gru.py                          # best_gru_model.pt       (GRU, 30 epochs, CPU)
python3 prd_net/01_extract-embeddings.py                  # prd_net_embeddings.pkl
python3 prd_net/02_peer-groups.py                         # prd_net_peers.pkl
python3 prd_net/04_prd-train.py                           # prd_net/checkpoints/prd_net_v1.pt (+ threshold)
python3 prd_net/05_prd-inference.py                       # (optional) latent-PRD test metrics
# optional EDA / cross-val: preprocessing/05_analysis.py, baseline/08_crossval.py
```

**PRD-Net v2** (the feature-space contrastive **difference track**, benchmarked
against v1 above) builds on these same upstream artifacts but has its own
config, ablation switches, dashboard and results —
see **[`pipeline/prd_net_v2/README.md`](pipeline/prd_net_v2/README.md)** for its
run commands and full design-decision writeup.

Headline test-set results (length-of-stay > 7 days, 23.6 % positive; full rebuild
2026-08-25, after fixing the ICD hard-filter — see *"ICD hard-filter matched the
wrong column"* below):

| model | F1 | AUROC | AUPRC |
|---|--:|--:|--:|
| GRU (baseline, 48h) | 0.611 | 0.848 | 0.644 |
| PRD-Net latent (48h) | 0.617 | 0.833 | 0.598 |
| PRD-Net feature-diff (48h) | 0.601 | 0.834 | 0.607 |
| PRD-Net feature-diff (24h) | 0.562 | 0.800 | 0.543 |

> **ICD hard-filter matched the wrong column (fixed 2026-08-25):** the peer-group
> hard filter (`prd_net/02_peer-groups.py`, `prd_net/04_prd-train.py`,
> `prd_net/05_prd-inference.py`, `prd_net_v2/fd02_feature-prototypes.py`) picked
> a patient's "primary ICD chapter" via `argmax`/first-`1` over the `icd_*`
> columns in `X_*.parquet`. Those columns are multi-label (patients carry ~8.3
> chapters on average), so this actually picked the first chapter in
> `config.ICD_CATEGORIES` order — not the true primary diagnosis. Fixed to
> match on `primary_diag` (`seq_num==1`, from `labels.parquet`/`y_*.parquet`)
> instead. `prd_net/04_prd-train.py` also had no random seed (GRU DataLoader
> shuffle was nondeterministic); it now calls `torch.manual_seed(0)`, matching
> `prd_net_v2/fd04_diff-train.py`. All peer/prototype caches and models were
> rebuilt after the fix; headline numbers above reflect the corrected filter.

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
