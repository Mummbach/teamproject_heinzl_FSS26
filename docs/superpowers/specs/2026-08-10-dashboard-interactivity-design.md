# Dashboard Interactivity — Design

Date: 2026-08-10
Scope: `pipeline/prd_net_v2/` — fd06 export, fd07 Streamlit dashboard, fd07 static report

## Goal

Two additions to the existing PRD-Net v2 dashboard:

1. **Clickable peers** — selecting one of the displayed similar patients shows that
   patient's values next to the index patient's.
2. **Switchable top-N** — the number of top features shown is selectable between
   5, 10 and 15 instead of being fixed at 5.

Target is `fd07_dashboard.py` (Streamlit), the only currently interactive dashboard.
`fd07_report.py` (static HTML) gets the top-N switch only. `pipeline/dashboard.html`
is an unrelated static wireframe and is out of scope.

## Current state

`fd06_dashboard-export.py` writes one record per test patient to
`exports/fd_explanations_test_{w}h.json` (4593 records, ~47 MB at 48h). Each record
carries the prediction, raw feature values for the patient and both prototypes,
`top_contributions`, and three peer lists of three entries each.

`fd07_report.py` holds the loaders, figure builders and text formatters.
`fd07_dashboard.py` imports it as `R` and adds Streamlit widgets. `fd07_report.py`
must stay free of any streamlit import.

### Constraints discovered while reading the code

- **Peers are training patients** (`fd06_dashboard-export.py:116`). Each exported peer
  carries only `{stay_id, true_label}` — no feature values.
- **`TOP_K = 5` truncates at export time** (`fd06_dashboard-export.py:46`, applied at
  lines 251 and 261). The JSON physically contains five contributions per patient, so
  no dashboard-side change can display ten. fd06 must be re-run.
- The feature set splits into **77 diff features** and **29 absolute one-hot
  categories** (`icd_*`, `icu_*`, `adm_*`). Only diff features exist in
  `rec["patient"]`; absolute ones appear in `top_contributions` with
  `prototype == "own category"`.
- Consequently the raw-value table — which filters on `d["feature"] in rec["patient"]`
  — is **empty for 552 of 4593 patients (12%)**, because all five top contributions are
  absolute categories. Distribution of diff features among the top 5:
  `{0: 552, 1: 1061, 2: 1746, 3: 921, 4: 301, 5: 12}`. Raising `TOP_K` fixes this as a
  side effect.
- The `[:8]` slice in the raw-value table and the `[:6]` slice in
  `fig_prototype_position` can never bind today, since `TOP_K = 5` caps them first.

### Verified data availability

All 10,097 distinct peer `stay_id`s referenced across the 48h export resolve at 100%
against each of:

| Source | Provides | Size |
| --- | --- | --- |
| `output/fd_feature_matrix_train_raw_{w}h.parquet` | all 106 raw feature values | 3.6 MB, 21,429 × 107 |
| `output/cohort.csv` | `gender`, `first_careunit`, `admission_type`, continuous `los` | 6.6 MB |
| `output/y_train.parquet` | `los_gt7` outcome label | 0.3 MB |

The train matrix carries **both** the 77 diff and the 29 absolute features, so it is
the single feature source for peers — `X_train.parquet` is not needed.

`streamlit==1.58.0` is already pinned in `pipeline/requirements.txt:21`; it is merely
not installed in the local `.venv`. Version 1.58 supports `st.dataframe(on_select=…)`.
`exports/` and `pipeline/output/` are both gitignored, so re-running fd06 does not
affect repository size.

## Design

### 1. Peer data source

Add `load_peer_source(window)` to `fd07_report.py`, pandas-only so the module stays
streamlit-free. It joins the three sources above into a `stay_id`-indexed structure
exposing raw feature values, demographics, actual LOS in days, and the outcome label.
The full 3.6 MB matrix is loaded at once; lazy row-group reads are unnecessary at this
size. In `fd07_dashboard.py` the call is wrapped in `@st.cache_data`, matching the
existing `load()`.

**Rejected alternative:** exporting peer raw values from fd06. Nine peers × 106
features × 4593 patients would inflate the JSON severalfold, and the peer values would
again be frozen against whatever `TOP_K` was chosen at export time — reintroducing the
very blocker this change removes.

### 2. Clickable peers

The three peer tables become `st.dataframe(..., on_select="rerun",
selection_mode="single-row")`, each with its own `key`, and gain a column with the
peer's actual LOS in days alongside the existing outcome label.

Selection state lives in `st.session_state`, holding both the selected peer's
`stay_id` and which of the three tables it came from. Three rules govern it:

- Selecting a row in one table clears the other two, so exactly one peer is ever
  active.
- The selection resets when the index patient changes. Without this the dashboard
  would show a peer belonging to the previously selected patient.
- The peer lists overlap — for `stay_id 32610785` the "most similar" and "most similar
  short-stay" lists are identical — so the same `stay_id` can be selectable from two
  tables. Storing the source table alongside the id keeps the highlight unambiguous.

Below the tables, a comparison block renders when a peer is selected:

- A header line describing the peer in the same format `patient_info_line` uses for the
  index patient, extended with actual LOS in days and the outcome label.
- A table over the currently selected top-N features with columns
  `Feature | Index patient | Peer | Difference`, difference signed and in raw units.

Absolute one-hot features are included and rendered as their category value rather than
a numeric difference, since a signed delta between two one-hot flags carries no
clinical meaning.

### 3. Top-N switch

`TOP_K` in `fd06_dashboard-export.py` goes from 5 to 15. A sidebar
`st.radio("Top features", [5, 10, 15], horizontal=True)` drives the raw-value table,
`fig_contributions`, `fig_prototype_position`, the peer comparison table, and the global
importance chart.

Both figure builders take an `n` parameter defaulting to 5, keeping `fd07_report.py`'s
existing call sites valid. Their fixed heights (360 px and 320 px) become a function of
`n`, otherwise fifteen bars are squeezed into space laid out for five.

**Side effect to contain:** `diagnose()` at `fd06_dashboard-export.py:194` reuses the
same `TOP_K` for `dominance = topk[0] / topk.sum()`, tested against a 0.55 threshold.
Raising `TOP_K` to 15 would shift that ratio for every patient and silently rewrite the
`wrong_reasons` text — a behavioural change unrelated to this feature. A separate
`DOMINANCE_K = 5` constant keeps the diagnostic heuristic on its current basis.

### 4. Static report

`fd07_report.py` gains the 5/10/15 switch using the show/hide pattern the file already
applies in `showPatient()`: render one chart variant per level and toggle `display`.
The HTML grows from roughly 232 KB to roughly 700 KB, which is immaterial for a local
file and avoids introducing a second, `Plotly.react()`-based rendering mechanism
alongside the existing one.

Clickable peers are not mirrored here — they require server-side lookup of the training
matrix.

## Verification

- Re-run fd06 for **both** windows. `WINDOW_HOURS = 24` is the config default, so the
  48h run must be triggered explicitly.
- Regression check for `DOMINANCE_K`: `wrong_reasons` in the regenerated export must be
  identical to the current export for every patient. This is the guard that the
  `TOP_K` change did not alter the diagnostic heuristic.
- Confirm each record carries up to 15 contributions, and that the count of patients
  with an empty raw-value table drops from 552.
- Rebuild the static report and confirm the switch drives all three levels.
- Install streamlit into `.venv`, run the dashboard, and confirm: peer click renders the
  comparison, switching the index patient clears the peer selection, and the top-N
  switch drives every affected element.
