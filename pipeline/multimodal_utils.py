"""
Shared utilities for multimodal explainability scripts.

Imported by: baseline/07_model_gru.py, baseline/08b_hyperparameter_search.py,
             baseline/09_shap.py, baseline/10_explainability.py, baseline/11_timeshap.py,
             explainability/shap_prdnet.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import OBS_WINDOW, CXR_FEATURES

# ── CXR feature names produced by preprocessing/01d_extract_radiology_features.py,
# merged into the static matrix by preprocessing/04_preprocessing.py when
# config.USE_CXR_FEATURES is True. Derived from config.CXR_FEATURES (minus the
# coverage flag, which get_cxr_feature_groups tracks separately) so this list
# can't drift from what's actually attached to the feature matrix.
CXR_STRUCT_FEATURES = [c for c in CXR_FEATURES if c != "has_cxr_report"]


def get_cxr_feature_groups(static_features: list[str]) -> dict:
    """
    Partition static_features into baseline / CXR-structured / BERT-PCA / has_cxr.

    Returns a dict with keys:
        bert_pca, cxr_struct, cxr_all, baseline
    """
    bert_pca = [c for c in static_features if c.startswith("bert_pca_")]
    cxr_struct = [c for c in CXR_STRUCT_FEATURES if c in static_features]
    has_cxr = ["has_cxr_report"] if "has_cxr_report" in static_features else []
    cxr_all = cxr_struct + bert_pca + has_cxr
    baseline = [c for c in static_features if c not in cxr_all]
    return {
        "bert_pca": bert_pca,
        "cxr_struct": cxr_struct,
        "has_cxr": has_cxr,
        "cxr_all": cxr_all,
        "baseline": baseline,
    }


# ── Correlated-feature grouping for SHAP display ─────────────────────────────
# Several static features are near-duplicates of each other (e.g. heart_rate_mean
# vs heart_rate_median, or dbp_mean vs map_mean, r > 0.9 on train). SHAP has no
# way to know two such columns carry almost the same information, so it can
# split credit between them arbitrarily -- observed empirically: dbp_mean and
# map_mean get opposite-signed SHAP values for 85% of test patients even
# though the two inputs move together. Grouping correlated features before
# display reports one combined, stable number instead of two that can
# contradict each other for the same patient.

def group_correlated_static_features(X_train_static, threshold: float = 0.85) -> dict:
    """Cluster columns of X_train_static (fit on train only) whose pairwise
    Pearson |correlation| exceeds `threshold`.

    Uses complete-linkage (clique) merging, not simple transitive union-find:
    a candidate merge is only accepted if EVERY pair in the resulting group
    exceeds `threshold`, not just a chain through one shared "hub" column.
    This matters because a hub like gcs_total_mean correlates > 0.85 with
    each of gcs_eye/verbal/motor_mean individually (it's their sum), but
    those three are only ~0.7-0.84 correlated with EACH OTHER -- clinically
    distinct sub-scores that a naive transitive merge would wrongly fold
    into one bucket. dbp/map's mean+median, by contrast, are all pairwise
    > 0.85 with each other and correctly form one group.

    Returns {group_id: [member_feature_names, ...]}, sorted; group_id is the
    shortest member name (deterministic, human-readable).
    """
    feats = list(X_train_static.columns)
    corr = X_train_static[feats].corr()
    corr_arr = corr.values
    n = len(feats)

    # groups[i] = the group (list of feature indices) that feature i belongs to
    groups = [[i] for i in range(n)]
    group_of = list(range(n))  # feature index -> position in `groups`

    edges = [
        (abs(corr_arr[i, j]), i, j)
        for i in range(n) for j in range(i + 1, n)
        if np.isfinite(corr_arr[i, j]) and abs(corr_arr[i, j]) > threshold
    ]
    edges.sort(reverse=True)  # merge strongest correlations first

    for _, i, j in edges:
        gi, gj = group_of[i], group_of[j]
        if gi == gj:
            continue
        candidate = groups[gi] + groups[gj]
        # Complete-linkage check: every pair in the merged group must clear threshold
        if all(
            abs(corr_arr[a, b]) > threshold
            for idx_a, a in enumerate(candidate) for b in candidate[idx_a + 1:]
        ):
            for m in candidate:
                group_of[m] = gi
            groups[gi] = candidate
            groups[gj] = []

    clusters = [g for g in groups if g]
    return {
        min((feats[i] for i in members), key=len): sorted(feats[i] for i in members)
        for members in clusters
    }


def grouped_shap(shap_values: np.ndarray, feature_names: list[str], groups: dict) -> tuple[np.ndarray, list[str]]:
    """Sum per-column SHAP values within each correlated group.

    shap_values: (N, F) array aligned with feature_names.
    groups: output of group_correlated_static_features().

    Returns (grouped_values (N, n_groups), display_labels) where a group of
    size 1 keeps its plain name and a group of size > 1 is labeled
    "name (+k correlated: ...)" so it's clear the number is a combined signal.
    """
    idx = {f: i for i, f in enumerate(feature_names)}
    grouped_cols, labels = [], []
    for group_id, members in groups.items():
        cols = [idx[m] for m in members if m in idx]
        if not cols:
            continue
        grouped_cols.append(shap_values[:, cols].sum(axis=1))
        if len(members) == 1:
            labels.append(group_id)
        else:
            others = ", ".join(m for m in members if m != group_id)
            labels.append(f"{group_id} (+{len(members) - 1} correlated: {others})")
    return np.stack(grouped_cols, axis=1), labels


# ── Dataset ────────────────────────────────────────────────────────────────

class ICUDataset(Dataset):
    def __init__(self, X_static, y, ts, ts_features):
        self.stay_ids = X_static["stay_id"].values
        self.static_arr = X_static.drop(columns=["stay_id"]).values.astype(np.float32)
        self.labels = y.set_index("stay_id").loc[self.stay_ids, "los_gt7"].values.astype(np.float32)
        self.ts_features = ts_features

        ts_pivot = (
            ts[ts["stay_id"].isin(self.stay_ids)]
            .sort_values(["stay_id", "hour"])
            .set_index(["stay_id", "hour"])[ts_features]
            .fillna(0.0)
        )
        self.ts_arr = np.zeros((len(self.stay_ids), OBS_WINDOW, len(ts_features)), dtype=np.float32)
        for i, sid in enumerate(self.stay_ids):
            if sid in ts_pivot.index.get_level_values("stay_id"):
                self.ts_arr[i] = ts_pivot.loc[sid].values

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ts_arr[idx]),
            torch.tensor(self.static_arr[idx]),
            torch.tensor(self.labels[idx]),
        )


# ── Model ──────────────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    def __init__(self, ts_input_size, static_input_size,
                 hidden_size=64, num_layers=2, static_dim=64, dropout=0.3,
                 use_gru: bool = True, use_text: bool = False, text_dim: int = 32):
        super().__init__()
        self.use_gru = use_gru
        self.use_text = use_text

        if use_gru:
            self.gru = nn.GRU(
                input_size=ts_input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        self.static_branch = nn.Sequential(
            nn.Linear(static_input_size, static_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        if use_text:
            self.text_branch = nn.Sequential(
                nn.Linear(1536, text_dim), nn.ReLU(), nn.Dropout(dropout),
            )
        fusion_input = (hidden_size if use_gru else 0) + static_dim + (text_dim if use_text else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fusion_input, 32),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, ts, static, text=None):
        if self.use_text and text is None:
            text = torch.zeros(ts.shape[0], self.text_branch[0].in_features, device=ts.device, dtype=ts.dtype)
        parts = []
        if self.use_gru:
            _, h_n = self.gru(ts)
            parts.append(h_n[-1])
        parts.append(self.static_branch(static))
        if self.use_text and text is not None:
            parts.append(self.text_branch(text))
        return self.classifier(torch.cat(parts, dim=1)).squeeze(1)

    @torch.no_grad()
    def gru_embedding(self, ts_tensor):
        """Returns the GRU final hidden state (batch, hidden_size)."""
        self.eval()
        _, h_n = self.gru(ts_tensor)
        return h_n[-1]


class SHAPWrapper(nn.Module):
    """Wraps GRUModel with sigmoid so output is a probability in [0, 1]."""
    def __init__(self, model: GRUModel):
        super().__init__()
        self.model = model

    def forward(self, ts, static):
        # NOTE: text branch receives an all-zeros tensor — SHAP attributions for text features are not meaningful.
        # Pass actual text embeddings for correct text-branch attribution.
        text = torch.zeros(ts.shape[0], 1536, device=ts.device) if self.model.use_text else None
        return torch.sigmoid(self.model(ts, static, text)).unsqueeze(1)


def load_multimodal_model(checkpoint_path, ts_input_size: int,
                          static_input_size: int, device) -> GRUModel:
    """Load a GRUModel checkpoint, inferring architecture from the state dict."""
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Infer num_layers from highest GRU layer index present
    gru_layer_indices = [
        int(k.split("_l")[1].split(".")[0])
        for k in sd if k.startswith("gru.weight_ih_l")
    ]
    num_layers = max(gru_layer_indices) + 1 if gru_layer_indices else 1

    hidden_size = sd["gru.weight_ih_l0"].shape[0] // 3
    static_dim = sd["static_branch.0.weight"].shape[0]
    use_text = "text_branch.0.weight" in sd
    text_dim = sd["text_branch.0.weight"].shape[0] if use_text else 32

    model = GRUModel(
        ts_input_size=ts_input_size,
        static_input_size=static_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        static_dim=static_dim,
        use_text=use_text,
        text_dim=text_dim,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model
