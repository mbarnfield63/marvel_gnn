"""Shared GNN encoder + task heads.

One encoder, one lightweight head per task (marvel_gnn_plan.md): uncertainty
calibration (node -> log sigma) and outlier detection (transition -> logit).
Later heads (orphan linkage, label correction) read the same embeddings.
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from .data import EDGE_DIM, NODE_DIM


class Encoder(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            GATv2Conv(NODE_DIM if i == 0 else hidden, hidden, edge_dim=EDGE_DIM)
            for i in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))

    def forward(self, data):
        h = data.x
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(conv(h, data.edge_index, data.edge_attr)))
        return h


class MarvelGNN(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.encoder = Encoder(hidden, layers)
        self.unc_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.outlier_head = nn.Sequential(
            nn.Linear(2 * hidden + EDGE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def log_sigma(self, data):
        """Per-level log(sigma) in 1e-6 cm-1 units."""
        return self.unc_head(self.encoder(data)).squeeze(-1)

    def outlier_logits(self, data):
        """Per-transition badness logit. build_graph stores transition k as
        directed edges 2k (upper -> lower) and 2k+1, so even rows enumerate
        the transitions in input order."""
        h = self.encoder(data)
        up = data.edge_index[0, 0::2]
        low = data.edge_index[1, 0::2]
        z = torch.cat([h[up], h[low], data.edge_attr[0::2]], dim=1)
        return self.outlier_head(z).squeeze(-1)


def nll_loss(log_sigma, errors):
    """Heteroscedastic Gaussian NLL against a (n_levels, n_samples) error
    matrix (NaN = level absent from that masked refit)."""
    mask = torch.isfinite(errors)
    e = torch.where(mask, errors, torch.zeros_like(errors))
    ls = log_sigma.unsqueeze(1)
    per = ls + e**2 / (2.0 * torch.exp(2.0 * ls))
    return per[mask].mean()
