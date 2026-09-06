"""
PRD-Net v2 — fd02: Feature-Space Prototypes
============================================
Builds, per patient, a positive and a negative prototype IN FEATURE SPACE:
  pos_proto[F] = (weighted) mean of the long-stay  peers' scaled feature vectors
  neg_proto[F] = (weighted) mean of the short-stay peers' scaled feature vectors

Peer membership (D1):
  - Default (RETRIEVAL_SPACE="feature"): train, val AND test patients all
    re-apply the same hard filter (primary ICD diagnosis chapter AND ICU type
    AND admission type) + soft age filter against the training set, rank
    candidates by L2 distance in this track's own scaled feature space, and
    average the K nearest per class. Falls back to the unfiltered class pool
    if a side is empty — no patient is ever skipped under this mode,
    including at train time.
  - Ablation (RETRIEVAL_SPACE="embedding"): train patients instead reuse
    prd_net_peers.pkl (v1's embedding-space K-NN cache, identical to the
    existing track) directly — mapping the cached row indices into the scaled
    feature matrix and averaging those FEATURE vectors. The ~312 train
    patients with an empty peer side THERE are skipped, exactly as in
    prd_net/04_prd-train.py (this cache-based path has no fallback of its
    own). Val/test patients still re-apply the hard+age filter, but rank by
    distance in RETRIEVAL_SPACE (i.e. the v1 embedding) instead, with the
    same fallback-to-unfiltered-pool behaviour as the default mode.

Prototypes are built in SCALED (z-scored) feature space so the later deltas are
in comparable SD units. Raw-unit values are recovered downstream via the scaler.

Outputs (window-tagged): fd_prototypes_{split}_{w}.pkl with, per split,
  stay_ids, labels, X (scaled patient matrix), pos_proto, neg_proto,
  pos_peer_ids, neg_peer_ids (resolved stay_ids for the dashboard).

Run AFTER: fd01_feature-matrix.py and the existing 01/02 caches.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from config import OUTPUT_DIR
import config_fd as C

EPS = 1e-6


def _mean(vectors: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Simple or weighted mean of a (k, F) block of peer feature vectors (D2)."""
    if weights is None:
        return vectors.mean(axis=0)
    w = weights / weights.sum()
    return (vectors * w[:, None]).sum(axis=0)


def _condition_cxr(pos_proto, neg_proto, own_vec, has_report, cxr_idx, cxr_pos_ref, cxr_neg_ref):
    """Override the CXR-content columns (pneumonia, ventilator, ... — NOT
    has_cxr_report itself) of a patient's prototypes.

    has_cxr_report covers only ~2% of patients, so the plain peer/population
    mean on these columns is dominated by patients with no report at all
    (defaulted to 0, indistinguishable from "report present, nothing found").
    That inflates the delta for the rare patient who does have a report,
    without it reflecting an actual report-vs-report comparison. Fix:
      - patient HAS a report: compare against the class-conditional mean
        among OTHER report-holding training patients (a real comparison).
      - patient has NO report: copy the patient's own (defaulted) values
        into the prototype so the delta is exactly 0 — there is nothing to
        compare, and treating "no data" as "no finding" would be misleading.
    """
    if cxr_idx is None:
        return pos_proto, neg_proto
    pos_proto = pos_proto.copy(); neg_proto = neg_proto.copy()
    if has_report:
        pos_proto[cxr_idx] = cxr_pos_ref
        neg_proto[cxr_idx] = cxr_neg_ref
    else:
        pos_proto[cxr_idx] = own_vec[cxr_idx]
        neg_proto[cxr_idx] = own_vec[cxr_idx]
    return pos_proto, neg_proto


def build_train_prototypes_from_cache(
    train_ids, train_labels, M_train, emb_cache, peer_cache,
    cxr_idx=None, cxr_pos_ref=None, cxr_neg_ref=None, has_report=None,
) -> dict:
    """Train prototypes via the cached embedding-space peers (D1 ablation
    only — used when RETRIEVAL_SPACE="embedding"; the default "feature" mode
    uses build_prototypes_filtered for train patients too, see below).

    Returns a bundle dict; patients with an empty peer side are skipped, with
    no fallback (unlike build_prototypes_filtered).
    """
    F = M_train.shape[1]
    kept_ids, kept_lab = [], []
    pos_p, neg_p, pos_ids, neg_ids, all_ids = [], [], [], [], []

    for row, sid in enumerate(tqdm(train_ids, desc="train protos", leave=False)):
        pos_idx, neg_idx = peer_cache[int(sid)]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue                                       # skip empty-side patient

        pos_vec, neg_vec = M_train[pos_idx], M_train[neg_idx]
        tgt = emb_cache[int(sid)]

        w_pos = w_neg = None
        if C.USE_PROTOTYPE_WEIGHTING:
            pe  = np.stack([emb_cache[int(train_ids[i])] for i in pos_idx])
            ne  = np.stack([emb_cache[int(train_ids[i])] for i in neg_idx])
            w_pos = 1.0 / (np.linalg.norm(pe - tgt, axis=1) + EPS)
            w_neg = 1.0 / (np.linalg.norm(ne - tgt, axis=1) + EPS)

        # Class-independent nearest peers (overall most-similar patients), ranked
        # by embedding distance over the union of the cached pos/neg peers.
        union = list(pos_idx) + list(neg_idx)
        ud = np.linalg.norm(
            np.stack([emb_cache[int(train_ids[i])] for i in union]) - tgt, axis=1)
        all_ids.append([int(train_ids[union[j]]) for j in np.argsort(ud)[:C.K_PEERS]])

        pp, npv = _mean(pos_vec, w_pos), _mean(neg_vec, w_neg)
        pp, npv = _condition_cxr(pp, npv, M_train[row], has_report[row] if has_report is not None else False,
                                 cxr_idx, cxr_pos_ref, cxr_neg_ref)

        kept_ids.append(int(sid)); kept_lab.append(int(train_labels[row]))
        pos_p.append(pp); neg_p.append(npv)
        pos_ids.append([int(train_ids[i]) for i in pos_idx])
        neg_ids.append([int(train_ids[i]) for i in neg_idx])

    kept_ids = np.array(kept_ids)
    sid_to_row = {int(s): i for i, s in enumerate(train_ids)}
    X = np.stack([M_train[sid_to_row[s]] for s in kept_ids]).astype(np.float32)

    return {
        "stay_ids": kept_ids,
        "labels":   np.array(kept_lab),
        "X":        X,
        "pos_proto": np.stack(pos_p).astype(np.float32),
        "neg_proto": np.stack(neg_p).astype(np.float32),
        "pos_peer_ids": pos_ids,
        "neg_peer_ids": neg_ids,
        "all_peer_ids": all_ids,
        "n_pos_filtered": [len(p) for p in pos_ids],   # cache peers are always filtered
        "n_neg_filtered": [len(n) for n in neg_ids],
        "pos_fallback": [False] * len(pos_ids),
        "neg_fallback": [False] * len(neg_ids),
        "feature_names": C.feature_names(),
        "window_hours": C.WINDOW_HOURS,
    }


def build_prototypes_filtered(
    query_df, query_ids, query_labels, M_query,
    X_train_df, all_train_ids, train_labels, M_train, emb_cache,
    space, exclude_self=False,
    cxr_idx=None, cxr_pos_ref=None, cxr_neg_ref=None, has_report=None,
) -> dict:
    """Filtered K-NN prototypes for a query split (mirrors build_filtered_prototypes).

    Hard filter (primary_diag+ICU+ADM) + age filter against the train set, then
    rank the per-class candidates by distance in `space` ("embedding" | "feature")
    and average the K nearest FEATURE vectors. Falls back to the unfiltered class
    pool when a side is empty.

    Note: matching is on `primary_diag` (the seq_num==1 diagnosis from
    labels.parquet), not on the icd_* columns in X_train_df/query_df — those are
    multi-label (patients average ~8 chapters), so an argmax/first-1 pick over
    them does not recover the patient's actual primary diagnosis.
    """
    F = M_train.shape[1]
    train_pdiag = X_train_df["primary_diag"].values
    train_icu = X_train_df[C.ICU_COLS].values
    train_adm = X_train_df[C.ADM_COLS].values
    train_age = X_train_df["age"].values

    q_pdiag = query_df["primary_diag"].values
    q_icu = query_df[C.ICU_COLS].values
    q_adm = query_df[C.ADM_COLS].values
    q_age = query_df["age"].values
    q_row_of = {int(s): i for i, s in enumerate(query_df["stay_id"].values)}

    if space == "embedding":
        train_repr = np.stack([emb_cache[int(s)] for s in all_train_ids])
    else:
        train_repr = M_train

    pos_global = np.where(train_labels == 1)[0]
    neg_global = np.where(train_labels == 0)[0]

    pos_p, neg_p, pos_ids, neg_ids, all_ids = [], [], [], [], []
    n_pos_f, n_neg_f, pos_fb, neg_fb = [], [], [], []   # peer-support diagnostics

    def _knn(cands, target_repr):
        d = np.linalg.norm(train_repr[cands] - target_repr, axis=1)
        k_use = min(C.K_PEERS, len(cands))
        top = cands[np.argsort(d)[:k_use]]
        w = (1.0 / (np.linalg.norm(train_repr[top] - target_repr, axis=1) + EPS)
             if C.USE_PROTOTYPE_WEIGHTING else None)
        return _mean(M_train[top], w), [int(all_train_ids[i]) for i in top]

    for sid in tqdm(query_ids, desc=f"protos[{space}]", leave=False):
        qr = q_row_of[int(sid)]
        target_repr = emb_cache[int(sid)] if space == "embedding" else M_query[qr]

        mask = np.ones(len(X_train_df), dtype=bool)
        if q_pdiag[qr] != "unknown": mask &= train_pdiag == q_pdiag[qr]
        if q_icu[qr].max() == 1: mask &= train_icu[:, int(np.argmax(q_icu[qr]))] == 1
        if q_adm[qr].max() == 1: mask &= train_adm[:, int(np.argmax(q_adm[qr]))] == 1
        mask &= np.abs(train_age - q_age[qr]) <= C.AGE_TOLERANCE
        if exclude_self:
            self_row = np.where(all_train_ids == int(sid))[0]
            if len(self_row):
                mask[self_row[0]] = False

        cands = np.where(mask)[0]
        pos_c = cands[train_labels[cands] == 1]
        neg_c = cands[train_labels[cands] == 0]
        # Record filtered support BEFORE any fallback (used by the dashboard to
        # flag low-support / extrapolated predictions).
        n_pos_f.append(int(len(pos_c))); n_neg_f.append(int(len(neg_c)))
        pos_fb.append(len(pos_c) == 0);  neg_fb.append(len(neg_c) == 0)
        if len(pos_c) == 0: pos_c = pos_global
        if len(neg_c) == 0: neg_c = neg_global

        pv, pid = _knn(pos_c, target_repr)
        nv, nid = _knn(neg_c, target_repr)
        pv, nv = _condition_cxr(pv, nv, M_query[qr], has_report[qr] if has_report is not None else False,
                                cxr_idx, cxr_pos_ref, cxr_neg_ref)
        pos_p.append(pv); neg_p.append(nv); pos_ids.append(pid); neg_ids.append(nid)

        # Class-independent nearest peers (overall most-similar) within the same
        # hard+age filter; falls back to the full train pool if nothing matched.
        ranked = cands if len(cands) > 0 else np.arange(len(X_train_df))
        d_all = np.linalg.norm(train_repr[ranked] - target_repr, axis=1)
        all_ids.append([int(all_train_ids[i]) for i in ranked[np.argsort(d_all)[:C.K_PEERS]]])

    return {
        "stay_ids": np.asarray(query_ids),
        "labels":   np.asarray(query_labels),
        "X":        M_query.astype(np.float32),
        "pos_proto": np.stack(pos_p).astype(np.float32),
        "neg_proto": np.stack(neg_p).astype(np.float32),
        "pos_peer_ids": pos_ids,
        "neg_peer_ids": neg_ids,
        "all_peer_ids": all_ids,
        "n_pos_filtered": n_pos_f,
        "n_neg_filtered": n_neg_f,
        "pos_fallback": pos_fb,
        "neg_fallback": neg_fb,
        "feature_names": C.feature_names(),
        "window_hours": C.WINDOW_HOURS,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd02 — feature-space prototypes  (window={W}, retrieval={C.RETRIEVAL_SPACE}, "
          f"weighting={C.USE_PROTOTYPE_WEIGHTING})")

    # ── Load caches + scaled matrices + filter columns ────────────────────────
    # emb_cache/peer_cache are v1 (pipeline/prd_net) artifacts, only needed when
    # RETRIEVAL_SPACE == "embedding"; "feature" mode never touches v1 at all.
    print("Loading caches and matrices...")
    emb_cache = peer_cache = None
    if C.RETRIEVAL_SPACE == "embedding":
        with open(C.EMBEDDING_CACHE_PATH, "rb") as f: emb_cache = pickle.load(f)
        with open(C.PEER_CACHE_PATH, "rb")      as f: peer_cache = pickle.load(f)

    def load_scaled(split):
        m = pd.read_parquet(C.feature_matrix_path(split, scaled=True)).set_index("stay_id")
        return m[C.feature_names()]

    M = {s: load_scaled(s) for s in ("train", "val", "test")}
    Xdf = {s: pd.read_parquet(OUTPUT_DIR / f"X_{s}.parquet") for s in ("train", "val", "test")}
    ydf = {s: pd.read_parquet(OUTPUT_DIR / f"y_{s}.parquet") for s in ("train", "val", "test")}
    # Hard filter needs the true primary diagnosis (seq_num==1), which lives in
    # labels.parquet / y_{split}.parquet, not in the multi-label icd_* columns.
    for s in ("train", "val", "test"):
        Xdf[s] = Xdf[s].merge(ydf[s][["stay_id", "primary_diag"]], on="stay_id", how="left")

    # Alignment assertion: scaled matrix row order must equal X_{split} order so
    # the peer-cache row indices resolve to the correct patients.
    for s in ("train", "val", "test"):
        assert list(M[s].index) == list(Xdf[s]["stay_id"].values), f"{s} matrix misaligned with X_{s}"

    all_train_ids = Xdf["train"]["stay_id"].values
    train_labels  = ydf["train"].set_index("stay_id").loc[all_train_ids, "los_gt7"].values
    M_train = M["train"].values.astype(np.float32)

    # ── CXR content-vs-coverage conditioning (see _condition_cxr) ─────────────
    # has_cxr_report covers only ~2% of patients; compute a fixed class-conditional
    # reference for the OTHER CXR columns from training report-holders, and a
    # per-split has-report mask so each query patient gets the right treatment.
    cxr_idx = cxr_pos_ref = cxr_neg_ref = None
    has_report = {"train": None, "val": None, "test": None}
    feats_all = C.feature_names()
    if C.USE_CXR_FEATURES and "has_cxr_report" in feats_all:
        content_cols = [f for f in C.CXR_FEATURES if f != "has_cxr_report"]
        cxr_idx = np.array([feats_all.index(f) for f in content_cols])

        def raw_has_report(split, ids):
            col = pd.read_parquet(C.feature_matrix_path(split, scaled=False),
                                  columns=["stay_id", "has_cxr_report"]).set_index("stay_id")
            return col.loc[ids, "has_cxr_report"].values > 0

        train_has_report = raw_has_report("train", all_train_ids)
        cxr_pos_ref = M_train[train_has_report & (train_labels == 1)][:, cxr_idx].mean(axis=0)
        cxr_neg_ref = M_train[train_has_report & (train_labels == 0)][:, cxr_idx].mean(axis=0)
        has_report["train"] = train_has_report
        for s in ("val", "test"):
            has_report[s] = raw_has_report(s, Xdf[s]["stay_id"].values)

    # ── Train prototypes ──────────────────────────────────────────────────────
    print("Building train prototypes...")
    if C.RETRIEVAL_SPACE == "embedding":
        train_bundle = build_train_prototypes_from_cache(
            all_train_ids, train_labels, M_train, emb_cache, peer_cache,
            cxr_idx=cxr_idx, cxr_pos_ref=cxr_pos_ref, cxr_neg_ref=cxr_neg_ref,
            has_report=has_report["train"])
    else:  # feature-space ablation: re-retrieve, excluding self
        train_bundle = build_prototypes_filtered(
            Xdf["train"], all_train_ids, train_labels, M_train,
            Xdf["train"], all_train_ids, train_labels, M_train, emb_cache,
            space="feature", exclude_self=True,
            cxr_idx=cxr_idx, cxr_pos_ref=cxr_pos_ref, cxr_neg_ref=cxr_neg_ref,
            has_report=has_report["train"])
    print(f"  train: {len(train_bundle['stay_ids']):,} patients "
          f"(skipped {len(all_train_ids) - len(train_bundle['stay_ids']):,} empty-side)")

    # ── Val / test prototypes ─────────────────────────────────────────────────
    bundles = {"train": train_bundle}
    for s in ("val", "test"):
        ids = Xdf[s]["stay_id"].values
        lab = ydf[s].set_index("stay_id").loc[ids, "los_gt7"].values
        print(f"Building {s} prototypes ({len(ids):,})...")
        bundles[s] = build_prototypes_filtered(
            Xdf[s], ids, lab, M[s].values.astype(np.float32),
            Xdf["train"], all_train_ids, train_labels, M_train, emb_cache,
            space=C.RETRIEVAL_SPACE, exclude_self=False,
            cxr_idx=cxr_idx, cxr_pos_ref=cxr_pos_ref, cxr_neg_ref=cxr_neg_ref,
            has_report=has_report[s])

    for s, b in bundles.items():
        with open(C.prototypes_path(s), "wb") as f:
            pickle.dump(b, f)
        print(f"  saved {C.prototypes_path(s).name}")
    print("Done.")
