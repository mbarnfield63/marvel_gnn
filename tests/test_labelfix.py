import numpy as np
import torch

from marvel_gnn.core.parse import Transition
from marvel_gnn.gnn.corrupt import QN_BUMP, corrupt, largest_component
from marvel_gnn.gnn.data import build_graph
from marvel_gnn.gnn.labelfix import correction_report, fix_metrics, fix_sample
from marvel_gnn.gnn.model import MarvelGNN


def tr(upper, lower, freq, unc=1e-3):
    return Transition(freq=freq, unc=unc, orig_unc=unc, upper=upper, lower=lower, ref="00Test.1")


def ladder(n=30):
    ts = [tr(f"0 {i+1}", f"0 {i}", 10.0 + i) for i in range(n - 1)]
    ts.append(tr("0 2", "0 0", 21.0))
    return ts


def corrupted_sample(rng):
    ts = ladder(30)
    bad, kinds, mags = corrupt(ts, fraction=0.2, rng=rng)
    orig = np.array([t.upper for t in ts], dtype=object)
    bad, kinds, mags, orig = largest_component(bad, kinds, mags, orig)
    g, idx = build_graph(bad)
    fix = fix_sample(bad, np.where(kinds == QN_BUMP)[0], g, idx, orig_upper=orig)
    return bad, g, idx, fix, kinds, orig


def test_fix_sample_targets_true_upper():
    bad, g, idx, fix, kinds, orig = corrupted_sample(rng=0)
    assert fix is not None
    levels = list(idx)
    for k, i in enumerate(fix["rows"]):
        assert kinds[i] == QN_BUMP
        names = [levels[x] for x in fix["cand"][k][fix["mask"][k]].tolist()]
        # window: same v, J within +-2, never the current upper or the lower
        assert bad[i].upper not in names and bad[i].lower not in names
        assert names[fix["target"][k]] == orig[i]
        v, ju = bad[i].upper.split()
        assert all(n.split()[0] == v and 1 <= abs(int(n.split()[1]) - int(ju)) <= 2
                   for n in names)


def test_fix_logits_shape_and_mask():
    _, g, _, fix, _, _ = corrupted_sample(rng=0)
    logits = MarvelGNN(hidden=16, layers=2).fix_logits(g, fix)
    assert logits.shape == fix["cand"].shape
    assert torch.isfinite(logits[fix["mask"]]).all()
    assert (logits[~fix["mask"]] == float("-inf")).all()


def test_fix_training_reduces_loss_and_reports():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    graphs = []
    while len(graphs) < 4:
        bad, g, idx, fix, kinds, _ = corrupted_sample(rng=rng)
        if fix is not None:
            graphs.append((g, None, kinds, None, fix))
    model = MarvelGNN(hidden=16, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = sum(torch.nn.functional.cross_entropy(model.fix_logits(g, fx), fx["target"])
                   for g, *_, fx in graphs) / len(graphs)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first
    m = fix_metrics(model, graphs)
    assert set(m) == {"acc", "mrr", "n", "n_cand"}
    assert m["n"] > 0 and 0.0 <= m["acc"] <= 1.0
    for r in correction_report(model, ladder(30)):
        assert set(r) == {"transition", "ref", "upper", "lower", "proposed_upper",
                          "outlier_p", "confidence"}
        assert r["outlier_p"] > 0.5
