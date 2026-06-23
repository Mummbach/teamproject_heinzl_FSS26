"""
PRD-Net v2 — fd01: Interpretable Feature Matrix
================================================
Builds a per-patient, feature-level matrix aligned to X_{split} row order.

Why re-aggregate from timeseries.parquet instead of reusing the X_* columns?
  X_* already contains 48h aggregates, but WINDOW_HOURS must be a single switch
  (48h primary, 24h robustness). The 24h aggregates do not exist anywhere, so we
  re-aggregate from the raw long-format series over [0, WINDOW_HOURS) for BOTH
  windows. This keeps the two windows on identical footing; the 48h matrix also
  doubles as a consistency check against X_*'s pre-computed columns.

Per time-series feature we compute AGG_STATS = mean / last / min / max / slope
over [0, WINDOW_HOURS). 'slope' is the OLS slope of the feature vs hour using the
non-missing samples. The continuous static feature `age` is appended.
=> F = 12 TS features x 5 stats + age = 61 difference features, + 18 CXR-derived
features (01d_extract_radiology_features.py output) if USE_CXR_FEATURES = 79.
Stays without a usable CXR report get 0 for every CXR feature, incl. has_cxr_report.

The hard-filter one-hots (icd_*, icu_*, adm_*) and other binary indicators are
NOT difference features (see config_fd.py) — they are used for filtering only.

Outputs (window-tagged so 48h/24h coexist):
  fd_feature_matrix_{split}_raw_{w}.parquet     interpretable units (dashboard)
  fd_feature_matrix_{split}_scaled_{w}.parquet  z-scored with TRAIN stats (model)
  fd_scaler_{w}.pkl                             scaler + train medians + names

Run AFTER: upstream preprocessing (X_*/y_*/timeseries.parquet exist),
           01d_extract_radiology_features.py (if USE_CXR_FEATURES).
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))   # pipeline/  -> config.py
sys.path.append(str(Path(__file__).parent))          # this dir   -> config_fd.py
from config import OUTPUT_DIR
import config_fd as C


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_window_array(ts: pd.DataFrame, stay_ids: np.ndarray,
                        window_hours: int) -> np.ndarray:
    """Build a (N, window_hours, n_ts) array of raw values, NaN where missing.

    Rows are in the exact order of `stay_ids` (= X_{split} order), so downstream
    row indices line up with the peer cache.
    """
    n_ts = len(C.TS_FEATURES)
    arr  = np.full((len(stay_ids), window_hours, n_ts), np.nan, dtype=np.float32)

    sub = ts[(ts["hour"] >= 0) & (ts["hour"] < window_hours)]
    sub = sub[sub["stay_id"].isin(set(stay_ids.tolist()))]

    row_of = {int(sid): i for i, sid in enumerate(stay_ids)}
    sid_arr  = sub["stay_id"].to_numpy()
    hour_arr = sub["hour"].to_numpy().astype(int)
    vals     = sub[C.TS_FEATURES].to_numpy(dtype=np.float32)

    for sid, hour, v in zip(sid_arr, hour_arr, vals):
        r = row_of.get(int(sid))
        if r is not None:
            arr[r, hour, :] = v
    return arr


def _aggregate(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Compute mean/last/min/max/slope along the time axis, ignoring NaN.

    Returns dict stat -> (N, n_ts). NaN-only series yield NaN for
    mean/last/min/max (imputed later) and 0.0 for slope (no trend information).
    """
    N, T, F = arr.shape
    valid = ~np.isnan(arr)                       # (N, T, F)

    # All-NaN series (feature never measured in the window) yield NaN here by
    # design; they are imputed with train medians in main(). Silence the numpy
    # "empty slice" warnings that this intentionally produces.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_ = np.nanmean(arr, axis=1)
        min_  = np.nanmin (arr, axis=1)
        max_  = np.nanmax (arr, axis=1)

    # last: value at the largest hour index that is non-missing
    hour_idx = np.arange(T)[None, :, None]                       # (1, T, 1)
    idx_when_valid = np.where(valid, hour_idx, -1)
    last_idx = idx_when_valid.max(axis=1)                        # (N, F), -1 if none
    safe_idx = np.where(last_idx < 0, 0, last_idx)
    last_ = np.take_along_axis(arr, safe_idx[:, None, :], axis=1)[:, 0, :]
    last_[last_idx < 0] = np.nan

    # slope: OLS slope of value vs hour over the valid points (vectorized)
    x  = np.arange(T, dtype=np.float64)[None, :, None]           # (1, T, 1)
    m  = valid.astype(np.float64)
    y0 = np.where(valid, arr, 0.0).astype(np.float64)
    n   = m.sum(axis=1)
    sx  = (m * x).sum(axis=1)
    sy  = y0.sum(axis=1)
    sxx = (m * x * x).sum(axis=1)
    sxy = (x * y0).sum(axis=1)
    denom = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        slope_ = (n * sxy - sx * sy) / denom
    slope_[(denom == 0) | (n < 2)] = 0.0

    return {"mean": mean_, "last": last_, "min": min_, "max": max_, "slope": slope_}


def _attach_cxr_features(mat: pd.DataFrame, cxr_df: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join CXR-derived features by stay_id; missing stays -> 0 (no report).

    has_cxr_report is derived from row presence in cxr_df, not read from it, so
    it stays 1 even if every individual flag for that report happens to be 0.
    """
    if cxr_df is None:
        return mat
    aligned = cxr_df.reindex(mat.index)
    present = aligned.notna().any(axis=1)
    flag_cols = [c for c in C.CXR_FEATURES if c != "has_cxr_report"]
    for c in flag_cols:
        mat[c] = aligned[c].fillna(0.0).astype(np.float32)
    mat["has_cxr_report"] = present.astype(np.float32)
    return mat


def build_raw_matrix(ts: pd.DataFrame, X_split: pd.DataFrame,
                     window_hours: int, cxr_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble the raw-unit feature matrix for one split, in X_split row order."""
    stay_ids = X_split["stay_id"].to_numpy()
    arr   = _build_window_array(ts, stay_ids, window_hours)
    stats = _aggregate(arr)

    cols = {}
    for f_i, feat in enumerate(C.TS_FEATURES):
        for stat in C.AGG_STATS:
            cols[f"{feat}_{stat}"] = stats[stat][:, f_i]
    for stat_feat in C.STATIC_FEATURES:
        cols[stat_feat] = X_split[stat_feat].to_numpy(dtype=np.float32)

    mat = pd.DataFrame(cols, index=stay_ids)
    mat.index.name = "stay_id"
    if C.USE_CXR_FEATURES:
        mat = _attach_cxr_features(mat, cxr_df)

    # Critical: column order must equal feature_names(); row order must equal X_split.
    mat = mat[C.feature_names()]
    assert list(mat.index) == list(stay_ids), "feature matrix row order != X_split"
    return mat


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd01 — feature matrix  (WINDOW_HOURS={W}, F={C.n_features()})")

    print("Loading timeseries + splits...")
    ts = pd.read_parquet(OUTPUT_DIR / "timeseries.parquet")
    splits = {
        "train": pd.read_parquet(OUTPUT_DIR / "X_train.parquet"),
        "val":   pd.read_parquet(OUTPUT_DIR / "X_val.parquet"),
        "test":  pd.read_parquet(OUTPUT_DIR / "X_test.parquet"),
    }

    cxr_df = None
    if C.USE_CXR_FEATURES:
        print("Loading CXR-derived features...")
        cxr_df = (pd.read_csv(C.CXR_FEATURE_PATH)
                  .set_index("stay_id")[[c for c in C.CXR_FEATURES if c != "has_cxr_report"]])
        print(f"  {len(cxr_df):,} stays with a usable CXR report")

    print("Aggregating raw matrices...")
    raw = {s: build_raw_matrix(ts, df, W, cxr_df) for s, df in splits.items()}
    for s, m in raw.items():
        n_nan = int(m.isna().sum().sum())
        print(f"  {s:<5}: {m.shape[0]:,} x {m.shape[1]}  (missing cells before impute: {n_nan:,})")

    # Impute missing cells with TRAIN medians (computed before scaling), then
    # standardize with TRAIN stats. Medians are persisted for reproducibility.
    medians = raw["train"].median(axis=0)
    raw = {s: m.fillna(medians) for s, m in raw.items()}

    scaler = StandardScaler().fit(raw["train"].values)
    scaled = {
        s: pd.DataFrame(scaler.transform(m.values), index=m.index, columns=m.columns)
        for s, m in raw.items()
    }

    print("Saving matrices + scaler bundle...")
    for s in splits:
        raw[s].reset_index().to_parquet(C.feature_matrix_path(s, scaled=False))
        scaled[s].reset_index().to_parquet(C.feature_matrix_path(s, scaled=True))

    with open(C.scaler_bundle_path(), "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "medians": medians,
            "feature_names": C.feature_names(),
            "window_hours": W,
        }, f)

    print(f"  raw    -> {C.feature_matrix_path('train', False).name} (+ val/test)")
    print(f"  scaled -> {C.feature_matrix_path('train', True).name} (+ val/test)")
    print(f"  scaler -> {C.scaler_bundle_path().name}")
    print("Done.")
