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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from torch_geometric.data import Data

from marvel_gnn.core.network import split_components
from marvel_gnn.core.selinv import selected_inverse
from marvel_gnn.core.solver import level_index, solve_energies

ERROR_SCALE = 1e6  # errors/sigmas are handled in 1e-6 cm-1 units

MAX_QN = 15  # QN slots; longest in-scope schema is 15 (C2H4: v1..v12 J Ka Kc)
# Zero the QN block and hold the token-count column constant, so the heads see
# structure only: a CO-trained model has never seen another schema's labels.
QN_MASK = True
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
    Computed on the reduced system (level 0 pinned, as in solve_energies).
    Only the diagonal and edge entries of the inverse are needed, so it uses
    a sparse LDL^T plus selected inversion instead of a dense n x n inverse
    (12C2H4, 11.6k levels: 0.36 s vs 24 s / 5 GiB dense).
    """
    n = len(idx)
    ui = np.array([idx[t.upper] for t in transitions])
    li = np.array([idx[t.lower] for t in transitions])
    w = np.array([1.0 / (t.unc * t.unc) for t in transitions])
    lap = coo_matrix((np.r_[w, w, -w, -w], (np.r_[ui, li, ui, li], np.r_[ui, li, li, ui])),
                     shape=(n, n)).tocsc()
    # symmetric mode with no pivoting on an SPD matrix gives L U = L (D L^T)
    lu = splu(lap[1:, 1:], permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options={"SymmetricMode": True})
    assert (lu.perm_r == lu.perm_c).all()
    lower = lu.L.tocsc()
    lower.sort_indices()
    z = selected_inverse(lower, lu.U.diagonal())

    pos = np.r_[-1, lu.perm_c]  # level -> factor position; -1 = pinned (G row 0)
    p, q = pos[ui], pos[li]
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    zd = np.r_[z.diagonal(), 0.0]  # zd[-1] = 0 for the pinned level
    gij = np.zeros(len(transitions))
    ok = lo >= 0
    gij[ok] = np.asarray(z[hi[ok], lo[ok]]).ravel()
    h = w * (zd[p] + zd[q] - 2.0 * gij)
    assert np.isclose(h.sum(), n - 1, rtol=1e-6)  # catches a pattern miss
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
    if QN_MASK:
        qn[:] = 0.0
        nqn = MAX_QN

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
