"""
PRD-Net — Step 2: Build Peer Groups
=====================================
For each target patient, find K peers from the same embedding space that are
clinically comparable (similar age) but split by outcome:
  - positive peers: stayed > 7 days  (same risk group)
  - negative peers: stayed ≤ 7 days  (contrast group)

These peer pairs are the training signal for the peer-retrieval network (Step 3).

Column names (derived from baseline pipeline — do not guess):
  ICD chapter : 18 binary columns icd_<category>  (02_features.py)
                categories defined in config.ICD_CATEGORIES
  ICU type    : 7 binary columns icu_<unit>        (04_preprocessing.py)
                icu_micu, icu_sicu, icu_ccu, icu_cvicu,
                icu_micu_sicu, icu_tsicu, icu_neuro_sicu
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import ICD_CATEGORIES

# Exact column names as created by 04_preprocessing.py
ICU_COLS = [
    "icu_micu", "icu_sicu", "icu_ccu", "icu_cvicu",
    "icu_micu_sicu", "icu_tsicu", "icu_neuro_sicu",
]

# ICD chapter columns derived from config.ICD_CATEGORIES (02_features.py)
ICD_COLS = [f"icd_{cat}" for cat in ICD_CATEGORIES]


# ══════════════════════════════════════════════════════════════════════════════
# HARD FILTER
# ══════════════════════════════════════════════════════════════════════════════

def _hard_filter(target_idx: int, train_df: pd.DataFrame) -> np.ndarray:
    """
    Return row indices of candidates that share the target patient's primary
    ICD diagnosis category AND ICU type. The target itself is excluded.

    Hard filter means binary match — either both columns agree or the candidate
    is dropped entirely. This ensures peers are clinically comparable before
    any distance metric is applied.

    Args:
        target_idx : row index of the target patient in train_df.
        train_df   : DataFrame containing ICD_COLS and ICU_COLS columns.

    Returns:
        np.ndarray of row indices passing the hard filter (target excluded).
    """
    target = train_df.iloc[target_idx]

    # Find which ICD chapter and ICU type the target belongs to
    # Each patient has exactly one 1 in ICD_COLS and one 1 in ICU_COLS
    target_icd = next((c for c in ICD_COLS if target[c] == 1), None)
    target_icu = next((c for c in ICU_COLS if target[c] == 1), None)

    mask = pd.Series(True, index=train_df.index)

    if target_icd is not None:
        mask &= train_df[target_icd] == 1

    if target_icu is not None:
        mask &= train_df[target_icu] == 1

    # Exclude the target patient itself
    mask.iloc[target_idx] = False

    return np.where(mask)[0]


# ══════════════════════════════════════════════════════════════════════════════
# SOFT FILTER (AGE)
# ══════════════════════════════════════════════════════════════════════════════

def _age_filter(target_idx: int, train_df: pd.DataFrame,
                candidates: np.ndarray, age_tol: int) -> np.ndarray:
    """
    From the hard-filtered candidates, keep only those within age_tol years
    of the target patient's age.

    Called soft filter because it uses a tolerance window rather than an
    exact match — patients aged 60 and 68 are still comparable; patients
    aged 25 and 80 are not.

    Args:
        target_idx : row index of the target patient in train_df.
        train_df   : DataFrame containing an 'age' column.
        candidates : row indices surviving the hard filter.
        age_tol    : maximum absolute age difference allowed (years).

    Returns:
        np.ndarray of row indices passing both hard and age filter.
    """
    target_age = train_df.iloc[target_idx]["age"]

    # Keep only candidates whose age is within the tolerance window
    candidate_ages = train_df.iloc[candidates]["age"].values
    within_tol = np.abs(candidate_ages - target_age) <= age_tol

    return candidates[within_tol]


# ══════════════════════════════════════════════════════════════════════════════
# PEER RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def get_peers(
    target_idx: int,
    train_df,
    train_features,
    train_labels,
    k: int = 20,
    age_tol: int = 10,
) -> tuple[list[int], list[int]]:
    """
    Find positive and negative peers for a single target patient.

    A peer is a candidate from the training set whose age is within age_tol
    years of the target. Among those age-matched candidates, the K nearest
    neighbours in embedding space are selected separately for each outcome
    class (positive = prolonged stay, negative = not prolonged).

    Args:
        target_idx      : row index of the target patient in train_df.
        train_df        : DataFrame with all training patients, must contain
                          an 'age' column and a 'stay_id' column.
        train_features  : (N, embedding_dim) array of patient embeddings,
                          row-aligned with train_df.
        train_labels    : (N,) array of binary labels (los_gt7), row-aligned
                          with train_df.
        k               : number of peers to return per class.
        age_tol         : maximum absolute age difference (years) for a
                          candidate to be considered age-matched.

    Returns:
        positive_peer_indices : list of up to k row indices (into train_df)
                                whose label is 1 and who are nearest to the
                                target in embedding space.
        negative_peer_indices : list of up to k row indices (into train_df)
                                whose label is 0 and who are nearest to the
                                target in embedding space.
    """
    # Step 1: hard filter — same ICD chapter and ICU type
    candidates = _hard_filter(target_idx, train_df)

    # Step 2: soft filter — within age_tol years
    candidates = _age_filter(target_idx, train_df, candidates, age_tol)

    # Step 3: split by outcome label
    labels_candidates = train_labels[candidates]
    pos_candidates = candidates[labels_candidates == 1]  # prolonged stay
    neg_candidates = candidates[labels_candidates == 0]  # not prolonged

    # Step 4: rank each group by L2 distance in embedding space, take top k
    # Nearest neighbours are preferred over random — they are the most similar
    # patients within the filtered pool, which is what peer retrieval should use.
    target_emb = train_features[target_idx]

    def _nearest_k(idxs: np.ndarray) -> list[int]:
        if len(idxs) == 0:
            return []
        dists = np.linalg.norm(train_features[idxs] - target_emb, axis=1)
        top_k = idxs[np.argsort(dists)[:k]]
        return top_k.tolist()

    positive_peers = _nearest_k(pos_candidates)
    negative_peers = _nearest_k(neg_candidates)

    return positive_peers, negative_peers


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — build peer groups for all training patients
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__": # only run if directly started from this file
    from config import OUTPUT_DIR
    from prd_net.config_prd import EMBEDDING_CACHE_PATH, PEER_CACHE_PATH, K_PEERS, AGE_TOLERANCE

    # ── Load embeddings ───────────────────────────────────────────────────────
    print("Loading embeddings...")
    with open(EMBEDDING_CACHE_PATH, "rb") as f:
        emb_dict = pickle.load(f)

    # ── Load training data ────────────────────────────────────────────────────
    print("Loading training data...")
    # Use unscaled data as train_df — filtering only needs age, ICD, and ICU
    # columns which are binary (0/1); scaling would distort those values.
    train_df = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
    y_train  = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")

    stay_ids       = train_df["stay_id"].values
    train_features = np.stack([emb_dict[sid] for sid in stay_ids])
    train_labels   = y_train.set_index("stay_id").loc[stay_ids, "los_gt7"].values

    print(f"  Patients      : {len(train_df):,}")
    print(f"  Embedding dim : {train_features.shape[1]}")
    print(f"  K peers       : {K_PEERS}  |  Age tolerance : ±{AGE_TOLERANCE} years")

    # ── Build peer groups ─────────────────────────────────────────────────────
    print("\nBuilding peer groups...")
    peer_cache = {}          # {target_idx: (pos_peers, neg_peers)}
    n_short_pos = 0          # patients with fewer than K positive peers
    n_short_neg = 0          # patients with fewer than K negative peers
    n_empty     = 0          # patients with 0 peers on either side

    for i in tqdm(range(len(train_df)), desc="Peers", unit="patient"):
        pos, neg = get_peers(i, train_df, train_features, train_labels,
                             k=K_PEERS, age_tol=AGE_TOLERANCE)
        peer_cache[int(stay_ids[i])] = (pos, neg)

        if len(pos) < K_PEERS: n_short_pos += 1
        if len(neg) < K_PEERS: n_short_neg += 1
        if len(pos) == 0 or len(neg) == 0: n_empty += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total    = len(peer_cache)
    ge5_pos  = sum(1 for pos, _   in peer_cache.values() if len(pos) >= 5)
    ge5_neg  = sum(1 for _,   neg in peer_cache.values() if len(neg) >= 5)

    print(f"\nDone. Results for {total:,} patients:")

    # How many patients got fewer peers than requested (K=20)
    # This happens for rare diagnosis/ICU combinations with few matching patients
    print(f"  Patients with <{K_PEERS} positive peers : {n_short_pos:,}  "
          f"(got at least 1, just fewer than {K_PEERS})")
    print(f"  Patients with <{K_PEERS} negative peers : {n_short_neg:,}  "
          f"(got at least 1, just fewer than {K_PEERS})")
    print(f"  Patients with 0 peers on either side  : {n_empty:,}  "
          f"(no match survived hard+age filter — will be skipped in training)")

    # Quality check: what % of patients have at least 5 peers per class
    # 5 is the minimum to form a meaningful training signal — below that
    # the contrastive loss has too few examples to learn from
    print(f"\n  Coverage (>= 5 peers per class):")
    print(f"    Positive peers : {ge5_pos/total*100:.1f}%  "
          f"({ge5_pos:,} of {total:,} patients)")
    print(f"    Negative peers : {ge5_neg/total*100:.1f}%  "
          f"({ge5_neg:,} of {total:,} patients)")
    if ge5_pos / total < 0.9 or ge5_neg / total < 0.9:
        print("  WARNING: <90% coverage — consider loosening hard/age filters before continuing.")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(PEER_CACHE_PATH, "wb") as f:
        pickle.dump(peer_cache, f)
    print(f"\nSaved: {PEER_CACHE_PATH}")
