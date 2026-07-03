"""Featurization of a solved spectroscopic component + masked-refit training
samples for the uncertainty-calibration head.

Diatomic (v, J) schema only for now — generalize only after this head is
validated on CO (see marvel_gnn_plan.md).

Training signal: mask a random fraction of transitions, re-solve the component
still containing the ground level, and record how far each surviving level
moved. A robustly determined level barely moves; a fragile one swings. The GNN
sees the *full* graph and predicts each level's marginal sensitivity, so a
prediction is interpretable as the level's uncertainty in the actual network.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from marvel_gnn.core.network import split_components
from marvel_gnn.core.solver import level_index, solve_energies

ERROR_SCALE = 1e6  # errors/sigmas are handled in 1e-6 cm-1 units

NODE_DIM = 7
EDGE_DIM = 3


def build_graph(transitions):
    """One connected component -> (torch_geometric Data, {assignment: node index}).

    Data extras: .assignments (list), .level_energies (float64 tensor, cm-1),
    .ground (int index of the zero-energy level).
    """
    energies = solve_energies(transitions)
    idx = level_index(transitions)
    n = len(idx)

    v = np.zeros(n)
    j = np.zeros(n)
    for a, i in idx.items():
        v_str, j_str = a.split()
        v[i], j[i] = float(v_str), float(j_str)

    e_arr = np.array([energies[a] for a in idx])
    incident = [[] for _ in range(n)]
    for t in transitions:
        incident[idx[t.upper]].append(t.unc)
        incident[idx[t.lower]].append(t.unc)
    incident = [np.array(u) for u in incident]

    x = np.column_stack([
        v / 10.0,
        j / 50.0,
        np.log1p([len(u) for u in incident]),
        np.array([np.log10(u.min()) for u in incident]) / 10.0,
        np.array([np.log10(np.median(u)) for u in incident]) / 10.0,
        np.array([np.log10((1.0 / u**2).sum()) for u in incident]) / 10.0,
        np.log1p(e_arr) / 10.0,
    ])

    src, dst, eattr = [], [], []
    for t in transitions:
        i, k = idx[t.upper], idx[t.lower]
        resid = abs(t.freq - (energies[t.upper] - energies[t.lower]))
        feat = [np.log10(t.unc) / 10.0, np.log1p(t.freq) / 10.0,
                min(resid / t.unc, 10.0) / 10.0]
        src += [i, k]
        dst += [k, i]
        eattr += [feat, feat]

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(eattr, dtype=torch.float32),
    )
    data.assignments = list(idx)
    data.level_energies = torch.tensor(e_arr, dtype=torch.float64)
    data.ground = int(np.argmin(e_arr))
    return data, idx


def refit_error_matrix(transitions, n_samples=200, mask_fraction=0.15, rng=None):
    """(n_levels, n_samples) matrix of masked-refit energy errors in 1e-6 cm-1.

    NaN where a level did not survive (disconnected from the ground level's
    component in that sample). Row order matches level_index(transitions).
    """
    rng = np.random.default_rng(rng)
    energies = solve_energies(transitions)
    idx = level_index(transitions)
    ground = min(energies, key=energies.get)

    errors = np.full((len(idx), n_samples), np.nan)
    n_mask = max(1, round(mask_fraction * len(transitions)))
    for s in range(n_samples):
        masked = set(rng.choice(len(transitions), size=n_mask, replace=False))
        kept = [t for i, t in enumerate(transitions) if i not in masked]
        comps, _ = split_components(kept, minsize=1)
        comp = next((c for c in comps
                     if any(ground in (t.upper, t.lower) for t in c)), None)
        if comp is None:
            continue
        refit = solve_energies(comp)
        for a, e in refit.items():
            errors[idx[a], s] = (e - energies[a]) * ERROR_SCALE
    return errors
