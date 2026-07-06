"""Schema generalization: the 7-token CO2 CDSD schema (J v1 v2 l2 v3 r e/f)
through featurization, corruption, candidate windows, and the model heads.
j_pos marks the J token's position (CO "v J" default: 1; CDSD: 0)."""

from pathlib import Path

import pytest
import torch

from marvel_gnn.core.parse import Transition
from marvel_gnn.gnn.corrupt import QN_BUMP, corrupt
from marvel_gnn.gnn.data import MAX_QN, NODE_DIM, build_graph, qn_array
from marvel_gnn.gnn.labelfix import fix_sample
from marvel_gnn.gnn.model import MarvelGNN

CO2_DIR = Path(r"C:\Code\MARVEL\molecules\CO2")


def tr(upper, lower, freq, unc=1e-3):
    return Transition(freq=freq, unc=unc, orig_unc=unc, upper=upper, lower=lower, ref="00Test.1")


def cdsd_ladder(n=10):
    """Chain over J = 0..n-1 in one CDSD-style vibrational state, plus a skip
    edge for a cycle (the 7-token twin of test_gnn.ladder)."""
    a = lambda j: f"{j} 0 0 0 1 1 e"
    ts = [tr(a(j + 1), a(j), 10.0 + j) for j in range(n - 1)]
    ts.append(tr(a(2), a(0), 21.0))
    return ts


def test_qn_array_pads_and_maps_parity():
    q = qn_array(["3 0 1 1 0 2 f", "0 1"])
    assert q.shape == (2, MAX_QN)
    assert q[0].tolist() == [3, 0, 1, 1, 0, 2, 1, 0]
    assert q[1].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]


def test_build_graph_and_heads_on_7_token_schema():
    ts = cdsd_ladder()
    graph, idx = build_graph(ts)
    assert graph.x.shape == (len(idx), NODE_DIM)
    assert torch.isfinite(graph.x).all()
    # J (token 0) landed in QN column 0, scaled /50
    assert torch.allclose(graph.x[:, 0] * 50.0,
                          torch.tensor([float(a.split()[0]) for a in idx]))

    model = MarvelGNN(hidden=16, layers=2)
    h = model.encoder(graph)
    assert model.log_sigma(graph).shape == (len(idx),)
    assert model.outlier_logits(graph).shape == (len(ts),)
    qn = graph.x[:, :MAX_QN]
    assert model.link_logits(h[:3], h[3:6], qn[:3], qn[3:6]).shape == (3,)


def test_corrupt_bumps_j_at_j_pos_0():
    ts = cdsd_ladder(30)
    out, kinds, _ = corrupt(ts, fraction=0.5, rng=0, j_pos=0)
    bumped = [(o, t) for o, t, k in zip(out, ts, kinds) if k == QN_BUMP]
    assert bumped
    for o, t in bumped:
        ot, tt = o.upper.split(), t.upper.split()
        assert ot[1:] == tt[1:]  # only the J token changed
        assert abs(int(ot[0]) - int(tt[0])) == 1


def test_fix_sample_candidates_vary_j_token_0():
    ts = cdsd_ladder()
    graph, idx = build_graph(ts)
    fix = fix_sample(ts, [4], graph, idx, j_pos=0)
    levels = list(idx)
    ju = int(ts[4].upper.split()[0])
    cand = fix["cand"][0][fix["mask"][0]].tolist()
    assert cand
    for c in cand:
        toks = levels[c].split()
        assert toks[1:] == ts[4].upper.split()[1:]
        assert 1 <= abs(int(toks[0]) - ju) <= 2


@pytest.mark.skipif(not CO2_DIR.exists(), reason="CO2 data not present")
def test_real_co2_component_featurizes():
    from marvel_gnn.core.network import split_components
    from marvel_gnn.core.parse import infer_segments, parse_native, parse_native_levels

    levels = parse_native_levels(CO2_DIR / "EnergyLevels_737.txt")
    tr_path = CO2_DIR / "Transitions_737.txt"
    segments, _ = infer_segments(tr_path, levels)
    kept, _ = parse_native(tr_path, segments=segments)
    comp = split_components(kept, minsize=2)[0][0]

    graph, idx = build_graph(comp)
    assert graph.x.shape == (len(idx), NODE_DIM)
    assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_attr).all()
