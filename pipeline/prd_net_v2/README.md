# PRD-Net v2 — Feature-Level Contrastive Difference Track

A parallel sub-track to `pipeline/prd_net/` ("v1"). By default it is fully
standalone — it finds its own nearest-peer patients and builds prototypes and
deltas in **interpretable feature space** instead of v1's embedding space, so
attributions map directly onto clinical features. It can optionally reuse v1's
peer retrieval instead (see "Retrieval space" below), but never touches or
modifies v1 itself.

## What's different from v1

Compared to `pipeline/prd_net/` ("v1", the latent-delta model this track is
benchmarked against — see "v1, in short" below for the full picture), this
track:

- builds a **positive prototype** = average clinical feature values of the
  long-stay peers, and a **negative prototype** = same for short-stay peers;
- feeds the model **signed, per-feature differences** `patient − prototype`
  against both prototypes (never collapsed to a scalar norm);
- uses a **linear** model so `weight × delta` is the exact (SHAP) attribution —
  no sampling, no approximation, no separate explainability step required.

The explanation is contrastive and directional: *"predicted long-stay because
mean GCS is below the short-stay prototype,"* not *"GCS = 6 → long-stay."*

## Files

| file | role |
|------|------|
| `config_fd.py` | hyperparameters, centralized column groups, window-tagged + ablation-suffixed paths, `feature_names()`; `FD_WINDOW`/`FD_NO_ICD`/`FD_MODEL`/`FD_SEED` env overrides |
| `fd01_feature-matrix.py` | aggregate `timeseries.parquet` → F=77 (+ CXR flags, + 29 absolute one-hots), impute, scale, persist raw+scaled+scaler |
| `fd02_feature-prototypes.py` | feature-space prototypes: train via peer cache, val/test via filtered K-NN |
| `fd03_diff-model.py` | diff assembly, `LinearDiffModel` (+ MLP), sklearn LogisticRegression reference |
| `fd04_diff-train.py` | train, val-F1 early stop, threshold tune, test metrics (incl. AUPRC) |
| `fd05_diff-explain.py` | `w·delta` == SHAP check; global + per-patient raw-unit explanations |
| `fd06_dashboard-export.py` | per-patient records + global summary (JSON + flat parquet) |
| `fd08_model-compare.py` | per-patient test predictions + metrics for GRU / old PRD / new PRD (48h+24h) |
| `fd07_report.py` | standalone interactive Plotly HTML dashboard (curated cases, no install) |
| `fd07_dashboard.py` | full Streamlit dashboard over all patients (needs `pip install streamlit`) |

## Running this track

This is the v2-specific part only. Environment setup, data layout, and the
shared upstream pipeline (cohort → features → GRU → latent PRD v1) are covered
once, for every track, in the root
**[`README.md`](../../README.md#-running-the-pipeline)** — do that first; this
track builds on those artifacts (`timeseries.parquet`, `X_*`, and, only under
`RETRIEVAL_SPACE="embedding"`, v1's `prd_net_embeddings.pkl`/`prd_net_peers.pkl`).

**Both windows** (run from `pipeline/prd_net_v2/`)

```bash
cd prd_net_v2
# 48h (primary, default — no env var needed):
python3 fd01_feature-matrix.py && python3 fd02_feature-prototypes.py \
  && python3 fd04_diff-train.py && python3 fd06_dashboard-export.py
python3 fd07_report.py                       # -> exports/fd_dashboard_48h.html
# 24h (robustness): prefix the same four + fd07 with FD_WINDOW=24
FD_WINDOW=24 python3 fd01_feature-matrix.py && FD_WINDOW=24 python3 fd02_feature-prototypes.py \
  && FD_WINDOW=24 python3 fd04_diff-train.py && FD_WINDOW=24 python3 fd06_dashboard-export.py
FD_WINDOW=24 python3 fd07_report.py          # -> exports/fd_dashboard_24h.html
```

`fd05_diff-explain.py` is an optional stdout sanity check (`w·Δ == SHAP`).
`fd03_diff-model.py` is a module (run directly only for its smoke test).
For the interactive dashboard: `streamlit run fd07_dashboard.py` (window is a
sidebar toggle, no config edit needed).

**Ablation switches (env vars, no file edit needed)**

| var | effect | default |
|-----|--------|---------|
| `FD_WINDOW` | observation window in hours | `48` |
| `FD_NO_ICD=1` | drops the 18 `icd_*` columns from the absolute block | off |
| `FD_MODEL=mlp` | MLP instead of the linear model (D4 ablation) | `linear` |
| `FD_SEED` | training seed for `fd04`'s `torch.manual_seed` | `0` |

These exist so an ablation run never has to hand-edit `config_fd.py` — that's
how `WINDOW_HOURS` once got left at `24` and silently became the default for
every subsequent run. At the default configuration (no env vars set), all
artifact paths are unchanged from before. Departing from any of the four
appends a suffix to the checkpoint/threshold/metrics filenames (e.g.
`fd_diff_v1_48h_mlp_noicd_s3.pt`, `fd_metrics_48h_mlp_noicd_s3.json`) so an
ablation run can never overwrite the reported checkpoint or metrics — same
mechanism as the existing `weight_decay` suffix. `RETRIEVAL_SPACE` and
`USE_CXR_FEATURES` are **not** covered by this guard yet: they still require a
direct edit to `config_fd.py` and can still overwrite the current window's
artifacts (see "Retrieval space" below).

**Notes**
- **Determinism:** splits and training use fixed seeds (`FD_SEED` overrides
  fd04's, default `0`); the GRU `DataLoader(shuffle=True)` on CPU is close but
  not bit-identical run to run, so downstream metrics can wobble by ~0.01.
- **Vitals fix:** `config.py` `RANGE_FILTERS` reject probe-disconnect zeros
  (SpO₂/BP = 0, etc.); this only takes effect when `preprocessing/02_features.py` re-parses the
  raw data — a v2-only re-run reuses the existing `timeseries.parquet`.
- **24h upstream:** GRU / latent-PRD only see the first 48h; genuinely comparable
  24h GRU/old-PRD rows stay "n/a" until the upstream scripts are
  window-parameterized (see below). The v2 track's own 24h artifacts *are* produced.

## Re-run just this track (upstream already present)

If `timeseries.parquet` and `X_*` already exist, the commands under "Running
this track" above are all you need — `fd01 → fd02 → fd04 → fd06 → fd07` per
window. Artifacts are window-tagged (`_48h`/`_24h`), so the two windows never
overwrite each other. `best_gru_model.pt` and the peer caches
(`prd_net_embeddings.pkl`, `prd_net_peers.pkl`) are **not** required under the
default `feature` retrieval space — only needed if
`RETRIEVAL_SPACE = "embedding"` in `config_fd.py`.

## Results (test set)

Headline metrics are AUROC and **AUPRC** (23.6% positive imbalance). 48h is the
primary analysis; 24h is the robustness / comparability check. Both rows are
current: rebuilt 2026-08-25 end-to-end (`fd02`→`fd04`→`fd06`) under the
`feature`-retrieval default (`RETRIEVAL_SPACE = "feature"`, see "Retrieval
space" below) after fixing the ICD hard-filter bug (it was matching the first
ICD chapter in column order, not the patient's actual primary diagnosis — see
the top-level README's *"ICD hard-filter matched the wrong column"* note),
on top of the 2026-07-26 full rebuild from raw MIMIC (probe-disconnect vitals
fix + absolute hard-filter features + CXR flags) — see "Running this track"
above. Source: `fd_metrics_48h.json` / `fd_metrics_24h.json`.

| window | accuracy | precision | recall | F1 | AUROC | **AUPRC** |
|--------|---------:|----------:|-------:|---:|------:|----------:|
| **48h** (primary) | 0.793 | 0.552 | 0.659 | 0.601 | **0.834** | **0.607** |
| 24h (robustness)  | 0.737 | 0.463 | 0.715 | 0.562 | 0.800 | 0.543 |

Reference — Wu et al. GBDT: AUROC 0.747 / AUPRC 0.536. The 48h linear difference
model exceeds the GBDT on **both** AUROC and AUPRC while staying fully
interpretable. A sklearn `LogisticRegression` fit on the same diff vectors
matches the torch model (sanity check), and `shap.LinearExplainer` reproduces
`w·(x − E[x])` exactly (max abs diff 0.0, enforced by an assertion in `fd05`).

## Model comparison (`fd08`)

`fd08_model-compare.py` is documented as evaluating every model on the same
fixed test set and writing `exports/fd_model_comparison.json` (metrics) +
`fd_model_predictions.parquet` (per-patient probabilities for the ROC/PR
overlays and the per-patient cross-model panel), with both dashboards showing
a "Model comparison" view (metrics table, grouped bars, ROC + PR overlays).
**As of 2026-08-25 this script and its output files are not present in the
repo** — the numbers below are compiled by hand from each track's own metrics
output, not from a `fd08` run. 24h exists only for the new feature-diff track;
GRU, old PRD and (their) baselines are 48h-only. Test-set results:

| model | window | F1 | AUROC | AUPRC |
|-------|:------:|---:|------:|------:|
| GRU (baseline) | 48h | 0.611 | 0.848 | 0.644 |
| Old PRD (latent delta) | 48h | 0.617 | 0.833 | 0.598 |
| New PRD (feature-diff) | 48h | 0.601 | 0.834 | 0.607 |
| New PRD (feature-diff) | 24h | 0.562 | 0.800 | 0.543 |

(GRU: full-rebuild numbers, 2026-07-26, from `baseline/07_model_gru.py` —
unaffected by the ICD-filter fix below, so still current. Old PRD and New PRD:
refreshed 2026-08-25, from `prd_net/05_prd-inference.py` and
`fd_metrics_{w}h.json` respectively, after fixing the ICD hard-filter bug (see
the top-level README's *"ICD hard-filter matched the wrong column"* note) —
contrary to the previous note here, Old PRD's numbers were **not** independent
of that bug: `prd_net/02_peer-groups.py`'s hard filter had the same argmax/
first-`1` issue and has been rebuilt too.) The three models sit within
~0.01–0.02 AUROC of each
other, so the feature-diff track buys an exactly attributable, contrastive,
per-feature explanation at essentially no accuracy cost.

> **fd08 note:** `fd08_model-compare.py` predates the absolute-features change
> (`d48fe19`) and rebuilds the new-PRD model without the 29 absolute inputs, so it
> currently raises a shape mismatch on the 183-dim checkpoint. It needs the same
> `load_absolute_block(...) → assemble_diff(..., absolute=...)` step `fd04`/`fd06`
> use before its JSON can be regenerated.

### Window-parameterized comparison & comparable 24h artifacts

`fd08` is a MODEL × WINDOW registry. Each (model, window) entry resolves to
window-tagged artifacts and is only evaluated if they exist; otherwise it is
recorded `available: false` and shown as **n/a (not trained)** in both dashboards.
The 48h artifacts are the original unsuffixed files; the 24h slots light up
automatically once these window-tagged files are produced:

| model | 48h artifact | 24h artifact (to produce) |
|-------|--------------|---------------------------|
| GRU | `output/best_gru_model.pt` + `X_*_scaled.parquet` + `timeseries.parquet` | `best_gru_model_24h.pt` + `X_*_scaled_24h.parquet` + `timeseries_24h.parquet` |
| Old PRD | `prd_net/checkpoints/prd_net_v1.pt` + `prd_net_embeddings.pkl` | `prd_net_v1_24h.pt` (+ `_threshold`) + `prd_net_embeddings_24h.pkl` |
| New PRD | `fd_diff_v1_48h.pt` + `fd_prototypes_test_48h.pkl` | already present (`*_24h`) |

**Producing comparable 24h artifacts (fair retrain).** A genuinely comparable
24h GRU/old-PRD must see only the first 24h, which means re-aggregating the
*static* features too (not just truncating the time series). Recipe:

1. Re-run upstream feature engineering with a 24h window (`OBS_WINDOW=24` in
   `config.py`, write to `*_24h` outputs) → `X_*_24h`, `timeseries_24h.parquet`.
2. Retrain the GRU on the 24h inputs → `best_gru_model_24h.pt`.
3. Rebuild embeddings + peers from the 24h GRU → `prd_net_embeddings_24h.pkl`,
   `prd_net_peers_24h.pkl`.
4. Retrain old PRD on the 24h embeddings → `prd_net_v1_24h.pt` (+ threshold).
5. Re-run `fd08_model-compare.py` — the 24h GRU/old-PRD rows fill in automatically.

Steps 1–4 require window-parameterizing the upstream scripts (so the 48h
artifacts are not overwritten); that change is pending confirmation.

## Dashboard

Two visualizations on top of the fd06 exports (data-only; they load no model):

- **`fd07_report.py`** — a self-contained `exports/fd_dashboard_{w}.html` that opens
  in any browser (uses plotly, already installed; no server). It shows the global
  overview (metrics, global feature importance, distance-vs-difference) and a
  dropdown over a curated set of illustrative patients with: prediction vs. truth,
  per-feature contributions to the logit (red → long-stay, blue → short-stay), the
  patient's position *between* the two prototypes per feature, and the 3 most-similar
  / 3 long-stay / 3 short-stay peers with their outcomes. For **misclassified**
  patients it also shows a heuristic "why this prediction is probably wrong" box
  (empty-filter fallback, low peer support, single-feature dominance, borderline
  probability, or near-tie), plus the per-class peer-support count.
- **`fd07_dashboard.py`** — the full interactive Streamlit app over all ~4 600 test
  patients (filters by correct/wrong/false-neg/false-pos, patient picker). Same
  components as the static report. Run with `streamlit run fd07_dashboard.py`.

Both carry a **Top features** control (5 / 10 / 15) driving the global importance
chart, the raw-value table, the contribution chart and the prototype-position chart.
The count is in *distinct features*, which is not the same as contributions: a diff
feature enters the model twice (Δpos and Δneg) and can rank highly on both, so one
feature may own two bars in the contribution chart. fd06's `TOP_FEATURES` is the
ceiling — it collects contributions until that many distinct features are covered, so
offering a level above 15 means re-running fd06.

The raw-value table and the prototype-position chart show only the *measured*
features of that set; categorical drivers (`icd_*`/`icu_*`/`adm_*`) have no prototype
to sit between and are called out beneath the panel instead of silently missing.

In the Streamlit app each peer is a **button** (`stay_id · outcome · LOS`) — clicking
one shows that peer's demographics, actual LOS in days and observed outcome, plus a
feature-by-feature comparison against the patient on screen. Buttons rather than a
selectable table because `st.dataframe` registers a selection only from its checkbox
column, never from the id cell a reader actually aims at. Peers are training
patients, so the export names them without their values; the app pulls those from
`fd_feature_matrix_train_raw_{w}h.parquet`, `cohort.csv` and `y_train.parquet` via
`fd07_report.load_peer_source()`. No prediction is shown for a peer — the model was
fitted on them, so a probability would be meaningless.

Where *every* leading driver is categorical, those two panels say so instead of
rendering empty; raising the feature count brings clinical measurements into view.

## v1, in short (what this track is compared against)

`pipeline/prd_net/` (v1, `01`–`05b`) is the original, untouched latent-delta
model this track is benchmarked against:

1. `01_extract-embeddings.py` — a GRU encodes each patient's time series +
   statics into a 128-dim learned embedding.
2. `02_peer-groups.py` — for each patient, finds the K=20 nearest *other*
   patients by L2 distance in that embedding space (same age band, split by
   outcome) → `prd_net_peers.pkl`.
3. `04_prd-train.py` — averages the peers' embeddings into `pos_proto` /
   `neg_proto`, then trains a small nonlinear net on `h − proto` to a logit.

v1's predictions are a black box: the "explanation" for any prediction is an
abstract 128-dim distance, not a clinical feature. Getting a feature-level
explanation for a v1 prediction requires a separate, bolted-on tool
(`explainability/shap_prdnet.py`) that *approximates* feature importance by
sampling/backpropagating through the whole network after the fact — v1 itself
has no built-in notion of "which feature mattered."

## Retrieval space: two modes (D1)

Independent of the model change above, this track can find peers two ways:

| | **`feature`** (default) | **`embedding`** |
|---|---|---|
| Peers found by | plain L2 distance in this track's own scaled clinical feature vector (same hard filter: primary ICD diagnosis + ICU type + admission type + age tolerance) | L2 distance in v1's learned GRU embedding — reuses `prd_net_peers.pkl` / `prd_net_embeddings.pkl` as-is |
| Depends on v1 running first? | **No** — fully standalone | Yes — v1's `01`/`02` must have produced those caches |
| Why it exists | the standalone, self-contained mode | isolates *one* variable at a time in the v1-vs-v2 comparison: same peers as v1, only the prototype/model change |

Tested 2026-08-05 (24h window): `feature` matched or slightly **beat**
`embedding` on every headline test metric (torch model: AUROC 0.801 vs 0.791,
AUPRC 0.556 vs 0.532; sklearn logreg: AUROC 0.808 vs 0.789, AUPRC 0.570 vs
0.525) and left 0 training patients with an empty peer side, vs. some skipped
under the cached-peer path. So `embedding` mode isn't kept because it performs
better — only as a controlled comparison point back to v1's exact peer set.
Switch via `RETRIEVAL_SPACE` in `config_fd.py`.

> **Stale as of the 2026-08-25 ICD-filter fix.** The comparison above predates
> the fix described in the top-level README (*"ICD hard-filter matched the
> wrong column"*) — both `feature` and `embedding` mode shared the same buggy
> hard filter at the time, so the relative comparison may still hold, but
> neither number has been reproduced against the corrected filter. Re-run with
> `RETRIEVAL_SPACE = "embedding"` before citing this ablation again.

**Why `embedding` came first.** This track's first implementation shipped with
`RETRIEVAL_SPACE="embedding"` as the *only* mode — deliberately, so the very
first test of "does a feature-level contrastive difference model work at all"
held the peer set fixed at exactly what v1 already used. That isolates a
single variable (prototype/model design) against a known-good baseline instead
of changing two things (peer retrieval *and* the model) at once and not being
able to tell which one moved the numbers. Once that comparison confirmed the
feature-difference model itself was sound, the natural follow-up question was
whether v1's GRU embedding was even necessary for finding good peers, or
whether plain L2 distance in this track's own interpretable feature space
would do just as well — which is what `feature` mode tests. It did (see
numbers above), and as a bonus drops the `pipeline/prd_net/01`/`02`
dependency entirely, so it was promoted to default in `c41734b`.

Given that history, `embedding` is intentionally an **interim/ablation
finding, not a maintained second mode**: it was the scaffolding that let the
model-design question be answered cleanly, not a deployment path meant to
stay in lock-step with `feature` going forward. That's also why the generated
artifacts (`fd_prototypes_*`, `fd_diff_v1_*.pt`, `fd_metrics_*.json`) are only
window-tagged, not retrieval-space-tagged (see `config_fd.py`'s path
builders) — re-running with `RETRIEVAL_SPACE="embedding"` overwrites the
current window's `feature`-mode artifacts on purpose. If you need the
`embedding` numbers again (e.g. to re-verify the comparison after a feature-set
change), rerun `fd02`/`fd04` with the switch flipped, note the printed
metrics, then flip back and rerun to restore the default artifacts — there is
no dual-mode artifact retention, by design.

## Feature list and F

**Difference features** (continuous, differenced against the prototypes). Per
time-series feature we take `AGG_STATS = [mean, last, min, max, slope]` over
`[0, WINDOW_HOURS)` (`slope` = OLS slope vs hour over non-missing samples), plus
the continuous static `age` and the CXR-derived flags:

- 12 time-series features: `heart_rate, sbp, dbp, map, resp_rate, spo2,
  temperature, glucose, gcs_eye, gcs_verbal, gcs_motor, urine_output`
  × 5 stats = 60, **+ age = 61**.
- **+ 16 CXR-derived flags** (`USE_CXR_FEATURES=True`, from
  `preprocessing/01d_extract_radiology_features.py`): `has_cxr_report`, 7 pathology flags
  (pneumonia, pleural_effusion, pneumothorax, edema, atelectasis, opacity,
  cardiomegaly), `severity_score`, progression (worsening/improved/stable),
  device mentions (ventilator/central_line/chest_tube), `abnormality_count`.
  Stays without a usable report get 0 for every CXR flag. → **F = 77**.

**Absolute features** (`USE_ABSOLUTE_FEATURES=True`, appended UNCHANGED — never
differenced): the hard-filter one-hots `icd_*` (18), `icu_*` (7), `adm_*` (4) =
**29** columns. Peers are matched on these, so a *delta* would be ≈0 by
construction — but the patient's own category still carries baseline-risk signal
a delta never sees (a CVICU / circulatory stay has a different typical LOS than a
MICU / general-medicine stay), so they are fed to the model as-is.

- Model input (`DIFF_INPUT="both"`) =
  `concat([delta_pos, delta_neg, absolute])` = **2F + 29 = 183**.

Still excluded entirely: the other binary indicators (`atc_*`, `eth_*`, `ins_*`,
`marital_*`, `loc_*`, `gender_male`, `year_group`) — not part of the hard filter
and with no "X SD above/below the prototype" reading.

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
| D1 | Retrieval space | `feature` (standalone K-NN, no v1 dependency); `embedding` = reuse `prd_net_peers.pkl` for a same-peers-as-v1 comparison — see "Retrieval space" above |
| D2 | Prototype aggregation | simple **mean** (`USE_PROTOTYPE_WEIGHTING=False`) |
| D3 | Diff input | `both` = `[delta_pos, delta_neg]` (`pos_only`/`neg_only`/`proto_gap` available) |
| D4 | Model | `linear` (`mlp` ablation via `FD_MODEL=mlp`, no file edit) |
| D5 | Aggregation granularity | summary stats (`AGG_STATS`) |

All switches live in `config_fd.py` and are marked with `# DESIGN DECISION:`
comments at each choice point. `WINDOW_HOURS`, `USE_ICD_ABSOLUTE` and the fd04
training seed are additionally overridable per run via `FD_WINDOW`,
`FD_NO_ICD` and `FD_SEED` — see the ablation-switches table above.

## Implementation Notes

- Peer-cache row indices reference `X_train` row order; `fd01`/`fd02` assert
  the feature matrix matches that order.
- Under `RETRIEVAL_SPACE="embedding"` only, 312 training patients have an
  empty peer side (after the admission-type hard filter added upstream) and
  are skipped, matching `prd_net/04_prd-train.py`. The default `feature` mode
  never skips a training patient — an empty side falls back to the unfiltered
  class pool instead, same as val/test (see "Retrieval space" above).
- Scripts are loaded via `importlib`, since filenames start with a digit or
  contain a hyphen.
- All artifacts are window-tagged (`_48h` / `_24h`), so both windows coexist
  without overwriting each other.
