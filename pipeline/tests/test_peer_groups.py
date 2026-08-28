"""
prd_net/02_peer-groups.py — hard/age filter tests.

Regression coverage for the 2026-08-25 fix (see icd_hard_filter_fix memory):
the hard filter must match on the true primary_diag column, not on the
multi-label icd_* one-hot block.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import load_module

peer_groups = load_module("prd_net/02_peer-groups.py")


def make_train_df():
    return pd.DataFrame({
        "stay_id":        [1, 2, 3, 4, 5],
        "primary_diag":   ["circulatory", "circulatory", "endocrine", "circulatory", "unknown"],
        "age":            [60, 65, 62, 90, 61],
        "icu_micu":       [1, 1, 0, 1, 1],
        "icu_sicu":       [0, 0, 1, 0, 0],
        "icu_ccu":        [0, 0, 0, 0, 0],
        "icu_cvicu":      [0, 0, 0, 0, 0],
        "icu_micu_sicu":  [0, 0, 0, 0, 0],
        "icu_tsicu":      [0, 0, 0, 0, 0],
        "icu_neuro_sicu": [0, 0, 0, 0, 0],
        "adm_emergency":  [1, 1, 1, 1, 1],
        "adm_urgent":     [0, 0, 0, 0, 0],
        "adm_elective":   [0, 0, 0, 0, 0],
        "adm_observation": [0, 0, 0, 0, 0],
    })


def test_hard_filter_matches_primary_diag_not_first_icd_column():
    # Patient 0 is "circulatory"; only patients 1 and 3 share it (also same
    # ICU/admission type). Patient 2 is "endocrine" and must be excluded even
    # though it's otherwise identical, and patient 0 itself must be excluded.
    train_df = make_train_df()
    candidates = peer_groups._hard_filter(0, train_df)
    assert sorted(candidates.tolist()) == [1, 3]


def test_hard_filter_excludes_target_and_respects_icu_and_admission_type():
    train_df = make_train_df()
    # Patient 2 is "sicu", the only sicu patient -> no candidates survive.
    candidates = peer_groups._hard_filter(2, train_df)
    assert candidates.tolist() == []


def test_hard_filter_unknown_primary_diag_skips_diagnosis_match():
    # "unknown" primary_diag means the diagnosis criterion is not applied
    # (there's nothing to match on), so ICU/admission type alone decide.
    train_df = make_train_df()
    candidates = peer_groups._hard_filter(4, train_df)
    assert sorted(candidates.tolist()) == [0, 1, 3]


def test_age_filter_keeps_only_candidates_within_tolerance():
    train_df = make_train_df()
    # Target is patient 0 (age 60); candidates are patients 1 (65) and 3 (90).
    candidates = np.array([1, 3])
    within_5 = peer_groups._age_filter(0, train_df, candidates, age_tol=5)
    assert within_5.tolist() == [1]

    within_30 = peer_groups._age_filter(0, train_df, candidates, age_tol=30)
    assert sorted(within_30.tolist()) == [1, 3]
