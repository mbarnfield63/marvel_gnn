"""Shared GNN encoder + uncertainty-calibration head.

The encoder is shared by design: later heads (outlier, orphan linkage, label
correction) read the same embeddings. Only the uncertainty head exists so far.
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


class UncertaintyModel(nn.Module):
    """Predicts per-level log(sigma) in 1e-6 cm-1 units."""

    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.encoder = Encoder(hidden, layers)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, data):
        return self.head(self.encoder(data)).squeeze(-1)


def nll_loss(log_sigma, errors):
    """Heteroscedastic Gaussian NLL against a (n_levels, n_samples) error
    matrix (NaN = level absent from that masked refit)."""
    mask = torch.isfinite(errors)
    e = torch.where(mask, errors, torch.zeros_like(errors))
    ls = log_sigma.unsqueeze(1)
    per = ls + e**2 / (2.0 * torch.exp(2.0 * ls))
    return per[mask].mean()
