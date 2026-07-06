"""Orphan-linkage head: data prep, ranking metrics, and the advisory report.

split_components drops components smaller than minsize from the solve — the
orphans. This head ranks (orphan level, main-network level) pairs by how
plausible a connecting transition is, from the shared-encoder embeddings.
No real labeled merges exist, so training and eval detach bridge subtrees
from solved networks and check that the true attachment pair is recovered
(marvel_gnn_plan.md). Output is advisory only — a ranked candidate report
for human literature review, never inserted into the solve.
"""

import networkx as nx
import numpy as np
import torch

from marvel_gnn.core.solver import solve_energies
from marvel_gnn.gnn.data import MAX_QN, build_graph


def detach_orphans(comp, n_detach=5, max_size=25, rng=None):
    """Detach up to n_detach bridge subtrees (2..max_size levels, never the
    ground side) from a connected component, simulating orphan networks with
    a known answer.

    Returns (main_transitions, orphans); each orphan is (transitions,
    orphan_level, main_level) where the removed bridge ran
    orphan_level -- main_level. Bridge transitions appear in neither list.
    """
    rng = np.random.default_rng(rng)
    g = nx.Graph((t.upper, t.lower) for t in comp)
    energies = solve_energies(comp)
    ground = min(energies, key=energies.get)
    # a bridge of the full graph stays a bridge in any subgraph containing it,
    # so one upfront bridge pass survives the sequential detachments below
    bridges = list(nx.bridges(g))
    rng.shuffle(bridges)

    picked = []
    for u, w in bridges:
        if len(picked) == n_detach:
            break
        if u not in g or w not in g:  # inside an already-detached subtree
            continue
        g.remove_edge(u, w)
        side_u = nx.node_connected_component(g, u)
        o_root, m_anchor = (w, u) if ground in side_u else (u, w)
        o_side = side_u if o_root == u else nx.node_connected_component(g, w)
        if not 2 <= len(o_side) <= max_size:
            g.add_edge(u, w)
            continue
        g.remove_nodes_from(o_side)
        picked.append((o_side, o_root, m_anchor))

    detached = set().union(*(nodes for nodes, _, _ in picked)) if picked else set()
    main = [t for t in comp if t.upper not in detached and t.lower not in detached]
    # an early pick's anchor can itself be detached by a later pick — drop those,
    # their anchor is no longer in the main graph
    orphans = [([t for t in comp if t.upper in nodes and t.lower in nodes],
                o_root, m_anchor)
               for nodes, o_root, m_anchor in picked if m_anchor not in detached]
    return main, orphans


def prepare_linked(comp, n_graphs, n_detach=5, max_size=25, seed=None, device="cpu"):
    """n_graphs independent detachment samples, featurized: each sample is
    (main_graph, [(orphan_graph, true orphan node idx, true main node idx), ...]).
    Samples where no eligible bridge existed are dropped."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_graphs):
        main, orphans = detach_orphans(comp, n_detach=n_detach, max_size=max_size, rng=rng)
        if not orphans:
            continue
        mg, midx = build_graph(main)
        outs = []
        for ts, o_root, m_anchor in orphans:
            og, oidx = build_graph(ts)
            outs.append((og.to(device), oidx[o_root], midx[m_anchor]))
        samples.append((mg.to(device), outs))
    return samples


def link_bce(model, sample, n_neg=100):
    """BCE on the true attachment pair vs n_neg random orphan x main pairs per
    orphan (pos_weight = n_neg balances the classes). Negatives may rarely
    collide with the true pair — harmless label noise at these graph sizes."""
    main_graph, orphans = sample
    h_main = model.encoder(main_graph)
    vj_main = main_graph.x[:, :MAX_QN]  # build_graph's QN block
    logits, labels = [], []
    for og, oi, mi in orphans:
        h_o = model.encoder(og)
        vj_o = og.x[:, :MAX_QN]
        oj = torch.cat([torch.tensor([oi], device=h_o.device),
                        torch.randint(len(h_o), (n_neg,), device=h_o.device)])
        mj = torch.cat([torch.tensor([mi], device=h_main.device),
                        torch.randint(len(h_main), (n_neg,), device=h_main.device)])
        logits.append(model.link_logits(h_o[oj], h_main[mj], vj_o[oj], vj_main[mj]))
        labels.append(torch.cat([torch.ones(1, device=h_o.device),
                                 torch.zeros(n_neg, device=h_o.device)]))
    logits, labels = torch.cat(logits), torch.cat(labels)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=torch.tensor(float(n_neg), device=logits.device))


def _all_pair_logits(model, h_o, vj_o, h_main, vj_main):
    """Logit for every orphan x main pair, row-major (orphan outer, main inner)."""
    a = h_o.repeat_interleave(len(h_main), 0)
    b = h_main.repeat(len(h_o), 1)
    va = vj_o.repeat_interleave(len(h_main), 0)
    vb = vj_main.repeat(len(h_o), 1)
    return model.link_logits(a, b, va, vb)


def link_metrics(model, samples):
    """Rank of the true attachment pair among all orphan x main pairs:
    MRR, hits@1, hits@5, median rank, mean candidate-pair count."""
    ranks, n_cands = [], []
    with torch.no_grad():
        for main_graph, orphans in samples:
            h_main = model.encoder(main_graph)
            vj_main = main_graph.x[:, :MAX_QN]
            for og, oi, mi in orphans:
                h_o = model.encoder(og)
                s = _all_pair_logits(model, h_o, og.x[:, :MAX_QN], h_main, vj_main)
                true = oi * len(h_main) + mi
                ranks.append(1 + (s > s[true]).sum().item())
                n_cands.append(len(s))
    ranks = np.array(ranks, dtype=float)
    return {"mrr": float((1.0 / ranks).mean()),
            "hits1": float((ranks <= 1).mean()),
            "hits5": float((ranks <= 5).mean()),
            "median_rank": float(np.median(ranks)),
            "n_pairs": float(np.mean(n_cands))}


def orphan_report(model, main_transitions, orphan_comps, top_k=5):
    """Advisory ranked candidate-merge report: for each orphan component, the
    top_k (orphan level, main level) pairs by link score. Scientific-integrity
    constraint (marvel_gnn_plan.md): for human literature review only — never
    auto-inserted into the solve; a GNN prediction is not a measurement."""
    device = next(model.parameters()).device
    mg, midx = build_graph(main_transitions)
    m_levels = list(midx)
    rows = []
    with torch.no_grad():
        mg = mg.to(device)
        h_main = model.encoder(mg)
        vj_main = mg.x[:, :MAX_QN]
        for ci, comp in enumerate(orphan_comps):
            og, oidx = build_graph(comp)
            o_levels = list(oidx)
            og = og.to(device)
            h_o = model.encoder(og)
            # ponytail: sigmoid score under 1:n_neg sampled training — the
            # rank is meaningful, the absolute value is not calibrated
            p = torch.sigmoid(_all_pair_logits(model, h_o, og.x[:, :MAX_QN], h_main, vj_main))
            for r in torch.argsort(p, descending=True)[:top_k].tolist():
                rows.append({"orphan_component": ci,
                             "orphan_level": o_levels[r // len(h_main)],
                             "main_level": m_levels[r % len(h_main)],
                             "score": float(p[r])})
    return rows
