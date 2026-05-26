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

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    ICD chapter AND ICU type. The target itself is excluded.

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
    # Step 2: age filter — within age_tol years  [TODO: implement next]
    # Step 3: nearest neighbours per class       [TODO: implement next]
    candidates = _hard_filter(target_idx, train_df)
    raise NotImplementedError(f"{len(candidates)} candidates after hard filter — steps 2–3 not yet implemented")
