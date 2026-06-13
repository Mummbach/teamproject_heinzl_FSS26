# PRD-Net v2 — Feature-Level Contrastive Difference Track

A parallel sub-track to `pipeline/prd_net/`. It reuses the same peer retrieval
but builds prototypes and deltas in **interpretable feature space** instead of
embedding space, so attributions map directly onto clinical features.

The existing PRD-Net track (`pipeline/prd_net/01`–`05`) is **untouched** and
remains the latent-delta baseline for comparison.

## What's different from the latent track

The latent track encodes a patient to a 64-dim hidden vector and forms deltas
against **peer-embedding** prototypes — distance reasoning that is not
interpretable per feature. This track instead:

- builds a **positive prototype** = average clinical feature values of the
  long-stay peers, and a **negative prototype** = same for short-stay peers;
- feeds the model **signed, per-feature differences** `patient − prototype`
  against both prototypes (never collapsed to a scalar norm);
- uses a **linear** model so `weight × delta` is the exact (SHAP) attribution.

The explanation is contrastive and directional: *"predicted long-stay because
mean GCS is below the short-stay prototype,"* not *"GCS = 6 → long-stay."*

## Feature list and F

Difference features are continuous only. Per time-series feature we take
`AGG_STATS = [mean, last, min, max, slope]` over `[0, WINDOW_HOURS)`
(`slope` = OLS slope vs hour over non-missing samples), plus the continuous
static `age`:

- 12 time-series features: `heart_rate, sbp, dbp, map, resp_rate, spo2,
  temperature, glucose, gcs_eye, gcs_verbal, gcs_motor, urine_output`
- × 5 stats = 60, **+ age = F = 61**.
- Model input (`DIFF_INPUT="both"`) = `concat([delta_pos, delta_neg])` = **2F = 122**.

Excluded from difference features (used for filtering only): the hard-filter
one-hots `icd_*` (18), `icu_*` (7), `adm_*` (4) — peers share these by
construction so their deltas are ≈0 — and the other binary indicators
(`atc_*`, `eth_*`, `ins_*`, `marital_*`, `loc_*`, `gender_male`, `year_group`),
which have no "X SD above/below the prototype" reading.

Missing cells (feature never measured in the window) are imputed with **train
medians**, then standardized with **train** `StandardScaler` stats. A raw-unit
copy is kept for the dashboard; raw deltas are recovered as
`raw_delta = scaled_delta × scaler.scale_`.

> Note: `X_*.parquet` already contains 48h aggregates, but we re-aggregate from
> `timeseries.parquet` so `WINDOW_HOURS` is a single switch (48h ↔ 24h); the 48h
> matrix also cross-checks against those existing columns.

## Design-decision values (defaults)

| | Decision | Value |
|---|---|---|
| D1 | Retrieval space | `embedding` (reuse `prd_net_peers.pkl`); `feature` = ablation |
| D2 | Prototype aggregation | simple **mean** (`USE_PROTOTYPE_WEIGHTING=False`) |
| D3 | Diff input | `both` = `[delta_pos, delta_neg]` (`pos_only`/`neg_only`/`proto_gap` available) |
| D4 | Model | `linear` (`mlp` ablation behind a switch) |
| D5 | Aggregation granularity | summary stats (`AGG_STATS`) |

All switches live in `config_fd.py` and are marked with `# DESIGN DECISION:`
comments at each choice point.

## Results (test set)

Headline metrics are AUROC and **AUPRC** (23.7% positive imbalance). 48h is the
primary analysis; 24h is the robustness / comparability check.

| window | accuracy | precision | recall | F1 | AUROC | **AUPRC** |
|--------|---------:|----------:|-------:|---:|------:|----------:|
| **48h** (primary) | 0.742 | 0.469 | 0.696 | 0.560 | **0.783** | **0.513** |
| 24h (robustness)  | 0.703 | 0.418 | 0.654 | 0.510 | 0.737 | 0.446 |

Reference — Wu et al. GBDT: AUROC 0.747 / AUPRC 0.536. The 48h linear
difference model exceeds the GBDT on AUROC and is close on AUPRC, while staying
fully interpretable. A sklearn `LogisticRegression` fit on the same diff vectors
matches the torch model (sanity check), and `shap.LinearExplainer` reproduces
`w·(x − E[x])` exactly (max abs diff 0.0).

## Notes / gotchas

- Peer-cache row indices reference `X_train` order; fd01/fd02 assert the feature
  matrix is in the same order.
- **312** training patients have an empty peer side (after the admission-type
  hard filter added upstream) and are skipped, exactly as in
  `prd_net/04_prd-train.py`. (The original brief's "~6" predates that filter.)
- Files are loaded via `importlib` where the name starts with a digit / contains
  a hyphen, matching the existing track.
- All artifacts are window-tagged (`_48h` / `_24h`) so both runs coexist.

## How to run

From `pipeline/prd_net_v2/`, with the upstream outputs and the existing track's
caches (`prd_net_embeddings.pkl`, `prd_net_peers.pkl`) already present:

```bash
python3 fd01_feature-matrix.py      # feature matrices + scaler   (output/fd_feature_matrix_*, fd_scaler_*)
python3 fd02_feature-prototypes.py  # feature-space prototypes     (output/fd_prototypes_*)
python3 fd04_diff-train.py          # train + threshold + metrics  (checkpoints/, output/fd_metrics_*)
python3 fd05_diff-explain.py        # global + per-patient SHAP explanations (stdout)
python3 fd06_dashboard-export.py    # per-patient records          (exports/fd_explanations_test_*)
```

`fd03_diff-model.py` is a module (model + diff assembly); run it directly only
for its smoke test.

For the **24h robustness run**, set `WINDOW_HOURS = 24` in `config_fd.py` and
re-run `fd01 → fd06`, then set it back to `48` (primary). Artifacts are
window-tagged, so the 24h run does not overwrite the 48h run.

## Files

| file | role |
|------|------|
| `config_fd.py` | hyperparameters, centralized column groups, window-tagged paths, `feature_names()` |
| `fd01_feature-matrix.py` | aggregate `timeseries.parquet` → F=61, impute, scale, persist raw+scaled+scaler |
| `fd02_feature-prototypes.py` | feature-space prototypes: train via peer cache, val/test via filtered K-NN |
| `fd03_diff-model.py` | diff assembly, `LinearDiffModel` (+ MLP), sklearn LogisticRegression reference |
| `fd04_diff-train.py` | train, val-F1 early stop, threshold tune, test metrics (incl. AUPRC) |
| `fd05_diff-explain.py` | `w·delta` == SHAP check; global + per-patient raw-unit explanations |
| `fd06_dashboard-export.py` | per-patient explanation records (JSON + flat parquet) |
