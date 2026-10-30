"""Takahashi selected inversion of a sparse LDL^T factor (torch-free, so
solver-only scripts can use it without importing the GNN stack)."""

import numpy as np
from scipy.sparse import csc_matrix


def selected_inverse(lower, d):
    """Entries of A^{-1} on the pattern of A's LDL^T factor (Takahashi
    recursion): `lower` unit-lower CSC with sorted indices, `d` the pivots.
    Returns a CSC matrix with `lower`'s pattern holding the lower triangle of
    A^{-1}. Exact; cost ~ sum over columns of nnz(col)^2, so it stays cheap
    while the factor has little fill (true of spectroscopic networks)."""
    n = len(d)
    ptr, rows, vals = lower.indptr, lower.indices, lower.data
    zv = np.empty_like(vals)
    for j in range(n - 1, -1, -1):
        a, b = ptr[j], ptr[j + 1]
        s, l = rows[a + 1:b], vals[a + 1:b]  # below-diagonal pattern of column j
        zs = np.empty((len(s), len(s)))      # A^{-1}[s, s], from later columns
        for c, k in enumerate(s):
            r = rows[ptr[k]:ptr[k + 1]]
            pos = ptr[k] + np.searchsorted(r, s[c:])
            zs[c:, c] = zv[pos]
            zs[c, c + 1:] = zs[c + 1:, c]
        col = -zs @ l
        zv[a + 1:b] = col
        zv[a] = 1.0 / d[j] - l @ col
    return csc_matrix((zv, rows, ptr), shape=lower.shape)
