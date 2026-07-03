"""Weighted least-squares energy solve (MARVEL() in MARVEL4.1.cpp, lines 976-1093).

The normal equations form a weighted graph Laplacian (weight = 1/unc^2),
singular with a constant nullspace per connected component. The C++ solves the
singular system with Eigen's LDLT and then shifts so the minimum energy is 0;
here we pin the first level to 0, solve the reduced SPD system, and apply the
same min-shift — identical energies up to float rounding.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def level_index(transitions):
    """{assignment: index} in first-appearance order (matches C++ insertion order)."""
    idx = {}
    for t in transitions:
        for a in (t.upper, t.lower):
            if a not in idx:
                idx[a] = len(idx)
    return idx


def solve_energies(transitions, unc=None):
    """Solve one connected component. Returns {assignment: energy}, min level = 0.

    `unc` optionally overrides each transition's uncertainty (same order as
    `transitions`) — used by the bootstrap to resample without mutating.
    """
    idx = level_index(transitions)
    n = len(idx)
    if unc is None:
        unc = [t.unc for t in transitions]

    rows, cols, vals = [], [], []
    y = np.zeros(n)
    for t, u in zip(transitions, unc):
        i, j = idx[t.upper], idx[t.lower]
        w = 1.0 / (u * u)
        rows += [i, j, i, j]
        cols += [i, j, j, i]
        vals += [w, w, -w, -w]
        y[i] += t.freq * w
        y[j] -= t.freq * w

    a = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # pin level 0 at energy 0: solve the reduced system for the rest
    x = np.zeros(n)
    if n > 1:
        x[1:] = spsolve(a[1:, 1:], y[1:])
    x -= x.min()

    return {assignment: x[i] for assignment, i in idx.items()}
