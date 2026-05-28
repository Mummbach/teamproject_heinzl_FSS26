"""
PRD-Net — Step 3: Peer Retrieval Derivation Network
=====================================================
Takes a patient embedding (from the GRU) and two prototype vectors
(positive peer centroid, negative peer centroid) and learns to score
how similar the patient is to each group.

Architecture (to be implemented):
  encode(x)           — projects the raw embedding into a learned space
  forward(x, pos, neg) — scores similarity to positive vs. negative prototype
"""

import torch
import torch.nn as nn


class PRDNet(nn.Module):
    """
    Peer Retrieval Derivation Network.

    Args:
        input_dim  : dimensionality of the input patient embedding (128)
        hidden_dim : size of the internal projection layer (64, matches baseline)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # Encoder: single-layer GRU, same architecture as 07_model_gru.py
        # (baseline uses 2 layers; here 1 layer keeps the prototype branch lighter)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Head: takes the concatenation of two hidden_dim vectors (patient vs prototype)
        # and outputs a single logit — positive = closer to positive peers
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x):
        """
        Project a patient embedding into the learned representation space.

        Args:
            x : (batch, input_dim) — flat patient embeddings

        Returns:
            (batch, hidden_dim) — encoded representation
        """
        # GRU expects (batch, seq_len, input_size); unsqueeze adds seq_len=1
        _, h_n = self.encoder(x.unsqueeze(1))
        return h_n[-1]  # (batch, hidden_dim)

    def forward(self, x, pos_proto, neg_proto):
        """
        Score the patient against positive and negative peer prototypes.

        Encodes the patient, computes how far it sits from each peer group
        centroid, then passes the combined delta through the classifier head.

        Args:
            x         : (batch, input_dim) — target patient embeddings
            pos_proto : (batch, hidden_dim) — mean embedding of positive peers
            neg_proto : (batch, hidden_dim) — mean embedding of negative peers

        Returns:
            logit     : (batch,) — raw score; positive = closer to pos peers
            delta_pos : (batch, hidden_dim) — distance vector to positive proto
            delta_neg : (batch, hidden_dim) — distance vector to negative proto
        """
        h = self.encode(x)                        # (batch, hidden_dim)

        delta_pos = h - pos_proto                 # how far from positive peers
        delta_neg = h - neg_proto                 # how far from negative peers

        combined = torch.cat([delta_pos, delta_neg], dim=-1)  # (batch, 2*hidden_dim)
        logit = self.head(combined).squeeze(-1)   # (batch,)

        return logit, delta_pos, delta_neg
