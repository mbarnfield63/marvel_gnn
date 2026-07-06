"""Shared GNN encoder + task heads.

One encoder, one lightweight head per task (marvel_gnn_plan.md): uncertainty
calibration (node -> log sigma), outlier detection (transition -> logit),
orphan linkage (level pair -> plausible-transition logit), and label
correction (suspect transition x candidate upper -> logit).
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from .data import EDGE_DIM, MAX_QN, NODE_DIM
from .labelfix import FIX_EXTRA


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
        self.link_head = nn.Sequential(
            nn.Linear(2 * hidden + MAX_QN, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.fix_head = nn.Sequential(
            nn.Linear(2 * hidden + EDGE_DIM + FIX_EXTRA, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

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

    def link_logits(self, h_a, h_b, qn_a, qn_b):
        """Plausible-transition logit for level-embedding pairs. Symmetric in
        (a, b): absolute energy order across components is unknowable, so the
        head sees only the sum and the difference magnitude of the embeddings,
        plus the pair's per-token |ΔQN| (the MAX_QN-wide block from
        build_graph; padded slots differ by 0). The selection rule ΔJ = ±1 is
        the strongest prior on which two levels can share a real transition —
        too weak to recover from embeddings alone."""
        z = torch.cat([h_a + h_b, (h_a - h_b).abs(), (qn_a - qn_b).abs()], dim=-1)
        return self.link_head(z).squeeze(-1)

    def fix_logits(self, data, fix):
        """(n_suspects, max_candidates) logits over candidate upper-level
        relabelings of suspect transitions (see labelfix.fix_sample). Padded
        candidate slots are -inf, so softmax / cross-entropy ignore them."""
        h = self.encoder(data)
        c = fix["cand"].shape[1]
        z = torch.cat([h[fix["cand"]],
                       h[fix["lower"]].unsqueeze(1).expand(-1, c, -1),
                       data.edge_attr[fix["edge_row"]].unsqueeze(1).expand(-1, c, -1),
                       fix["extra"]], dim=-1)
        return self.fix_head(z).squeeze(-1).masked_fill(~fix["mask"], float("-inf"))


def nll_loss(log_sigma, errors):
    """Heteroscedastic Gaussian NLL against a (n_levels, n_samples) error
    matrix (NaN = level absent from that masked refit)."""
    mask = torch.isfinite(errors)
    e = torch.where(mask, errors, torch.zeros_like(errors))
    ls = log_sigma.unsqueeze(1)
    per = ls + e**2 / (2.0 * torch.exp(2.0 * ls))
    return per[mask].mean()
