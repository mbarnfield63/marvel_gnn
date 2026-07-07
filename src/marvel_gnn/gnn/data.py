"""Featurization of a solved spectroscopic component + masked-refit training
samples for the uncertainty-calibration head.

Schema-oblivious QN handling: an assignment's space-separated tokens fill a
fixed MAX_QN-wide block (zero-padded, e/f parity letters -> 0/1), plus one
token-count column so mixed-schema training can tell a padded zero from a
real one. Token *order* differs per schema (CO: v J; CO2 CDSD: J v1 v2 l2 v3
r e/f) — only corrupt() and labelfix need to know which token is J, via j_pos.

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

MAX_QN = 7   # QN slots; longest in-scope schema is 7 (CO2 CDSD)
NODE_DIM = MAX_QN + 6
EDGE_DIM = 5

_TOKEN = {"e": 0.0, "f": 1.0}  # parity letters in CDSD-style assignments


def qn_array(assignments):
    """(n, MAX_QN) float array of QN tokens in assignment order, zero-padded.
    Unknown non-numeric tokens fail loudly (float raises)."""
    out = np.zeros((len(assignments), MAX_QN))
    for r, a in enumerate(assignments):
        toks = a.split()
        out[r, :len(toks)] = [float(_TOKEN.get(s, s)) for s in toks]
    return out


def edge_leverages(transitions, idx):
    """Leverage h of each transition in the weighted least-squares solve.

    h_e = w_e * x_e^T L+ x_e (x_e the incidence vector, w_e = 1/unc^2):
    h -> 1 means the solve absorbs the line into the level energies (a bridge
    is exactly 1 — any frequency shift there is undetectable from residuals);
    h -> 0 means the line is highly redundant. sum(h) = n_levels - 1.
    Computed on the reduced system (level 0 pinned, as in solve_energies);
    components here are small (<1k levels), so a dense inverse is fine.
    """
    n = len(idx)
    w = np.array([1.0 / (t.unc * t.unc) for t in transitions])
    lap = np.zeros((n, n))
    for t, wi in zip(transitions, w):
        i, j = idx[t.upper], idx[t.lower]
        lap[i, i] += wi
        lap[j, j] += wi
        lap[i, j] -= wi
        lap[j, i] -= wi
    g = np.zeros((n, n))
    g[1:, 1:] = np.linalg.inv(lap[1:, 1:])
    h = np.array([w[k] * (g[i, i] + g[j, j] - 2.0 * g[i, j])
                  for k, t in enumerate(transitions)
                  for i, j in [(idx[t.upper], idx[t.lower])]])
    return np.clip(h, 0.0, 1.0)


def build_graph(transitions):
    """One connected component -> (torch_geometric Data, {assignment: node index}).

    Data extras: .assignments (list), .level_energies (float64 tensor, cm-1),
    .ground (int index of the zero-energy level).
    """
    energies = solve_energies(transitions)
    idx = level_index(transitions)
    n = len(idx)

    qn = qn_array(list(idx)) / 50.0  # x cols 0..MAX_QN-1 (heads slice these)
    nqn = len(next(iter(idx)).split())

    e_arr = np.array([energies[a] for a in idx])
    incident = [[] for _ in range(n)]
    for t in transitions:
        incident[idx[t.upper]].append(t.unc)
        incident[idx[t.lower]].append(t.unc)
    incident = [np.array(u) for u in incident]

    x = np.column_stack([
        qn,
        np.full(n, nqn / MAX_QN),
        np.log1p([len(u) for u in incident]),
        np.array([np.log10(u.min()) for u in incident]) / 10.0,
        np.array([np.log10(np.median(u)) for u in incident]) / 10.0,
        np.array([np.log10((1.0 / u**2).sum()) for u in incident]) / 10.0,
        np.log1p(e_arr) / 10.0,
    ])

    lev = edge_leverages(transitions, idx)
    src, dst, eattr = [], [], []
    for t, h in zip(transitions, lev):
        i, k = idx[t.upper], idx[t.lower]
        resid = abs(t.freq - (energies[t.upper] - energies[t.lower]))
        # studentized leave-one-out residual: undoes the solve's absorption of
        # the line (resid ~ (1-h) * true error), so a shifted redundant line
        # scores its full amplitude while a bridge stays at 0 (undetectable)
        stud = resid / (t.unc * np.sqrt(max(1.0 - h, 1e-12)))
        feat = [np.log10(t.unc) / 10.0, np.log1p(t.freq) / 10.0,
                np.log10(1.0 + resid / t.unc) / 4.0,
                h,
                np.log10(1.0 + stud) / 4.0]
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
