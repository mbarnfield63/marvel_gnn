import networkx as nx
import numpy as np
import torch

from marvel_gnn.core.parse import Transition
from marvel_gnn.gnn.data import MAX_QN
from marvel_gnn.gnn.linkage import detach_orphans, link_bce, link_metrics, prepare_linked
from marvel_gnn.gnn.model import MarvelGNN


def tr(upper, lower, freq, unc=1e-3):
    return Transition(freq=freq, unc=unc, orig_unc=unc, upper=upper, lower=lower, ref="00Test.1")


def network():
    """6-level cycle core (never detachable) with two pendant 3-level chains."""
    ts = [tr(f"0 {i+1}", f"0 {i}", 10.0 + i) for i in range(5)]
    ts.append(tr("0 5", "0 0", 60.0))  # closes the cycle
    for name, anchor, base in [("1", "0 2", 100.0), ("2", "0 4", 200.0)]:
        ts.append(tr(f"{name} 0", anchor, base))
        ts.append(tr(f"{name} 1", f"{name} 0", base + 1.0))
        ts.append(tr(f"{name} 2", f"{name} 1", base + 2.0))
    return ts


def test_detach_orphans():
    ts = network()
    main, orphans = detach_orphans(ts, n_detach=2, max_size=3, rng=0)
    assert 1 <= len(orphans) <= 2
    detached = set()
    for o_ts, o_root, m_anchor in orphans:
        nodes = {t.upper for t in o_ts} | {t.lower for t in o_ts}
        assert 2 <= len(nodes) <= 3
        assert o_root in nodes and m_anchor not in nodes
        detached |= nodes
    main_nodes = {t.upper for t in main} | {t.lower for t in main}
    assert not main_nodes & detached
    # every returned anchor must survive in the main graph (a nested detachment
    # can remove an earlier pick's anchor — those orphans are dropped)
    for _, _, m_anchor in orphans:
        assert m_anchor in main_nodes
    # the core cycle can never be detached, and main stays connected
    assert {"0 0", "0 5"} <= main_nodes
    g = nx.Graph((t.upper, t.lower) for t in main)
    assert nx.is_connected(g)
    # bridge transitions are masked: in neither main nor any orphan
    assert len(main) + sum(len(o) for o, _, _ in orphans) + len(orphans) == len(ts)


def test_link_head_symmetric():
    model = MarvelGNN(hidden=16, layers=2)
    h_a, h_b = torch.randn(4, 16), torch.randn(4, 16)
    vj_a, vj_b = torch.randn(4, MAX_QN), torch.randn(4, MAX_QN)
    assert torch.allclose(model.link_logits(h_a, h_b, vj_a, vj_b),
                          model.link_logits(h_b, h_a, vj_b, vj_a))


def test_link_training_reduces_loss():
    torch.manual_seed(0)
    samples = prepare_linked(network(), 4, n_detach=2, max_size=3, seed=0)
    assert samples
    model = MarvelGNN(hidden=16, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = sum(link_bce(model, s, n_neg=20) for s in samples) / len(samples)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first
    m = link_metrics(model, samples)
    assert set(m) == {"mrr", "hits1", "hits5", "median_rank", "n_pairs"}
    assert 0.0 < m["mrr"] <= 1.0
