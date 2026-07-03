import numpy as np
import pytest
import torch

from marvel_gnn.core.parse import Transition
from marvel_gnn.core.solver import level_index
from marvel_gnn.gnn.data import (EDGE_DIM, ERROR_SCALE, NODE_DIM, build_graph,
                                 edge_leverages, refit_error_matrix)
from marvel_gnn.gnn.model import MarvelGNN, nll_loss


def tr(upper, lower, freq, unc=1e-3):
    return Transition(freq=freq, unc=unc, orig_unc=unc, upper=upper, lower=lower, ref="00Test.1")


def ladder(n=10):
    """Chain of n levels 0..n-1, plus a redundant skip edge for cycles."""
    ts = [tr(f"0 {i+1}", f"0 {i}", 10.0 + i) for i in range(n - 1)]
    ts.append(tr("0 2", "0 0", 21.0))
    return ts


def test_edge_leverages():
    ts = ladder()  # chain + one skip edge: edges 0,1 and the skip form a cycle
    idx = level_index(ts)
    h = edge_leverages(ts, idx)
    assert h.shape == (len(ts),)
    assert np.isclose(h.sum(), len(idx) - 1)  # trace of the hat matrix = rank
    cycle = [0, 1, len(ts) - 1]
    bridges = [i for i in range(len(ts)) if i not in cycle]
    assert np.allclose(h[bridges], 1.0)  # bridge residual fully absorbed
    assert (h[cycle] < 1.0).all()
    # equal uncertainties: the 3-cycle splits leverage 2 across its edges
    assert np.allclose(h[cycle], 2.0 / 3.0)


def test_build_graph_shapes():
    ts = ladder()
    graph, idx = build_graph(ts)
    n = len(idx)
    assert graph.x.shape == (n, NODE_DIM)
    assert graph.edge_index.shape == (2, 2 * len(ts))  # both directions
    assert graph.edge_attr.shape == (2 * len(ts), EDGE_DIM)
    assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_attr).all()
    assert graph.ground == idx["0 0"]


def test_refit_error_matrix():
    errors = refit_error_matrix(ladder(), n_samples=20, mask_fraction=0.15, rng=0)
    assert errors.shape == (10, 20)
    finite = errors[np.isfinite(errors)]
    assert len(finite) > 0
    # reproducible
    again = refit_error_matrix(ladder(), n_samples=20, mask_fraction=0.15, rng=0)
    np.testing.assert_array_equal(np.isnan(errors), np.isnan(again))
    np.testing.assert_allclose(finite, again[np.isfinite(again)])


def test_refit_errors_scale_with_conflict():
    # a level attached by two conflicting lines must swing more than a clean one
    ts = ladder()
    ts.append(tr("0 9", "0 8", 18.0 + 0.01))  # conflicts with the exact 18.0 line
    errors = refit_error_matrix(ts, n_samples=50, mask_fraction=0.15, rng=0)
    swing = np.nanstd(errors, axis=1)
    assert swing[9] > 10 * max(swing[1], 1e-12)  # node "0 9" is the conflicted one


def test_model_forward_and_loss():
    graph, idx = build_graph(ladder())
    model = MarvelGNN(hidden=16, layers=2)
    log_sigma = model.log_sigma(graph)
    assert log_sigma.shape == (len(idx),)
    errors = torch.randn(len(idx), 5)
    errors[0] = float("nan")
    loss = nll_loss(log_sigma, errors)
    assert torch.isfinite(loss)


def test_training_reduces_loss():
    torch.manual_seed(0)
    graph, _ = build_graph(ladder(12))
    errors = torch.tensor(
        refit_error_matrix(ladder(12), n_samples=50, rng=1), dtype=torch.float32)
    errors[graph.ground] = float("nan")

    model = MarvelGNN(hidden=16, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = nll_loss(model.log_sigma(graph), errors)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first
