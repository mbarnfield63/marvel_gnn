import numpy as np
import torch

from marvel_gnn.core.parse import Transition
from marvel_gnn.gnn.corrupt import corrupt, largest_component
from marvel_gnn.gnn.data import build_graph
from marvel_gnn.gnn.model import MarvelGNN


def tr(upper, lower, freq, unc=1e-3):
    return Transition(freq=freq, unc=unc, orig_unc=unc, upper=upper, lower=lower, ref="00Test.1")


def ladder(n=30):
    ts = [tr(f"0 {i+1}", f"0 {i}", 10.0 + i) for i in range(n - 1)]
    ts.append(tr("0 2", "0 0", 21.0))
    return ts


def test_corrupt_counts_and_marks():
    ts = ladder(30)
    corrupted, kinds = corrupt(ts, fraction=0.1, rng=0)
    assert len(corrupted) == len(ts)
    assert (kinds > 0).sum() == 3  # round(0.1 * 30)
    changed = [i for i, (a, b) in enumerate(zip(ts, corrupted)) if a != b]
    assert changed == list(np.where(kinds > 0)[0])
    assert all(t.upper != t.lower for t in corrupted)
    assert all(int(t.upper.split()[1]) >= 0 for t in corrupted)


def test_corrupt_deterministic():
    a, ka = corrupt(ladder(), fraction=0.1, rng=7)
    b, kb = corrupt(ladder(), fraction=0.1, rng=7)
    assert a == b
    np.testing.assert_array_equal(ka, kb)


def test_largest_component_filters_labels():
    ts = ladder(10) + [tr("5 1", "5 0", 999.0)]  # disconnected 2-level island
    labels = np.zeros(len(ts), dtype=bool)
    labels[-1] = True
    kept, kept_labels = largest_component(ts, labels)
    assert len(kept) == len(ts) - 1
    assert not kept_labels.any()


def test_outlier_logits_shape():
    ts = ladder()
    graph, _ = build_graph(ts)
    logits = MarvelGNN(hidden=16, layers=2).outlier_logits(graph)
    assert logits.shape == (len(ts),)
    assert torch.isfinite(logits).all()


def test_outlier_training_reduces_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(4):
        bad, kinds = corrupt(ladder(30), fraction=0.1, rng=rng)
        bad, kinds = largest_component(bad, kinds)
        g, _ = build_graph(bad)
        graphs.append((g, torch.tensor(kinds > 0, dtype=torch.float32)))

    model = MarvelGNN(hidden=16, layers=2)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(9.0))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = sum(bce(model.outlier_logits(g), l) for g, l in graphs) / len(graphs)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first
