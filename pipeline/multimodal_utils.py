"""
Shared utilities for multimodal explainability scripts.

Imported by: 08b_shap_multimodal.py, 09b_explainability_multimodal.py,
             10b_timeshap_multimodal.py, 12b_lime_multimodal.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ── CXR feature names produced by cxr_03_extract_features.py ──────────────
CXR_STRUCT_FEATURES = [
    "pneumonia", "pleural_effusion", "pneumothorax", "edema", "atelectasis",
    "opacity", "cardiomegaly",
    "severity_bilateral", "severity_diffuse", "severity_multifocal",
    "severity_extensive", "severity_severe", "severity_marked",
    "severity_score",
    "worsening", "improved", "stable",
    "ventilator", "central_line", "chest_tube",
    "report_length", "sentence_count", "abnormality_count",
]


def get_cxr_feature_groups(static_features: list[str]) -> dict:
    """
    Partition static_features into baseline / CXR-structured / BERT-PCA / has_cxr.

    Returns a dict with keys:
        bert_pca, cxr_struct, cxr_all, baseline
    """
    bert_pca   = [c for c in static_features if c.startswith("bert_pca_")]
    cxr_struct = [c for c in CXR_STRUCT_FEATURES if c in static_features]
    has_cxr    = ["has_cxr"] if "has_cxr" in static_features else []
    cxr_all    = cxr_struct + bert_pca + has_cxr
    baseline   = [c for c in static_features if c not in cxr_all]
    return {
        "bert_pca":   bert_pca,
        "cxr_struct": cxr_struct,
        "has_cxr":    has_cxr,
        "cxr_all":    cxr_all,
        "baseline":   baseline,
    }


# ── Dataset ────────────────────────────────────────────────────────────────

class ICUDataset(Dataset):
    def __init__(self, X_static, y, ts, ts_features):
        self.stay_ids    = X_static["stay_id"].values
        self.static_arr  = X_static.drop(columns=["stay_id"]).values.astype(np.float32)
        self.labels      = y.set_index("stay_id").loc[self.stay_ids, "los_gt7"].values.astype(np.float32)
        self.ts_features = ts_features

        ts_pivot = (
            ts[ts["stay_id"].isin(self.stay_ids)]
            .sort_values(["stay_id", "hour"])
            .set_index(["stay_id", "hour"])[ts_features]
            .fillna(0.0)
        )
        self.ts_arr = np.zeros((len(self.stay_ids), 48, len(ts_features)), dtype=np.float32)
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
                 hidden_size=64, num_layers=2, static_dim=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=ts_input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_branch = nn.Sequential(
            nn.Linear(static_input_size, static_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size + static_dim, 32),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, ts, static):
        _, h_n     = self.gru(ts)
        gru_out    = h_n[-1]
        static_out = self.static_branch(static)
        return self.classifier(torch.cat([gru_out, static_out], dim=1)).squeeze(1)

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
        return torch.sigmoid(self.model(ts, static)).unsqueeze(1)


def load_multimodal_model(checkpoint_path, ts_input_size: int,
                          static_input_size: int, device) -> GRUModel:
    """Load a GRUModel checkpoint and set to eval mode."""
    model = GRUModel(
        ts_input_size=ts_input_size,
        static_input_size=static_input_size,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model
