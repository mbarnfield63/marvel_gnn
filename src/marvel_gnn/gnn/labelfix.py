"""Label-correction head: data prep, ranking metrics, and the advisory report.

Narrower head on the outlier subset whose likely cause is QN mislabeling
(marvel_gnn_plan.md): for a suspect transition it ranks candidate upper-level
reassignments and proposes the best one. Trained on corrupt()'s QN_BUMP lines,
where the pre-corruption upper is the known recovery target. Candidates are
existing levels with the same non-J tokens and J within +-2 of the current
upper (j_pos gives the J token's position, as in corrupt()) —
mirrors the corruption model (J off-by-one) with distractors; a true upper
whose level vanished from the graph is unrecoverable by this head (the
"recoverable subset" ceiling). Output is advisory only — a flagged suggestion
for human literature review, never applied to source data.
"""

import numpy as np
import torch

from marvel_gnn.gnn.data import build_graph

FIX_EXTRA = 3  # per-candidate features: |dJ vs current upper|, |dJ vs lower|, energy-freq mismatch


def fix_sample(transitions, suspects, graph, idx, orig_upper=None, device="cpu",
               j_pos=1):
    """Candidate-correction bundle for the suspect transitions (indices into
    transitions). Returns padded tensors: cand/mask/extra (n, C[, FIX_EXTRA]),
    lower/edge_row (n,), rows (list of kept suspect indices), and target (n,)
    when orig_upper (per-transition true upper assignment) is given. Suspects
    with no candidate — or, in training, an unrecoverable truth — are dropped;
    None if nothing survives.

    The mismatch extra is |(E_cand - E_lower) - freq| / unc, log-scaled like
    build_graph's residual feature — the direct "does this relabeling explain
    the measured frequency" signal.
    """
    energies = graph.level_energies.cpu().numpy()
    levels = list(idx)
    rows, cands, extras, targets = [], [], [], []
    for i in suspects:
        t = transitions[i]
        toks = t.upper.split()
        ju, jl = int(toks[j_pos]), int(t.lower.split()[j_pos])
        cand = [idx[a] for d in (-2, -1, 1, 2)
                for a in [" ".join(toks[:j_pos] + [str(ju + d)] + toks[j_pos + 1:])]
                if a in idx and a != t.lower]
        if not cand:
            continue
        if orig_upper is not None:
            true_node = idx.get(orig_upper[i])
            if true_node not in cand:
                continue
            targets.append(cand.index(true_node))
        ex = []
        for c in cand:
            jc = int(levels[c].split()[j_pos])
            m = abs((energies[c] - energies[idx[t.lower]]) - t.freq)
            ex.append([abs(jc - ju) / 50.0, abs(jc - jl) / 50.0,
                       np.log10(1.0 + m / t.unc) / 4.0])
        rows.append(int(i))
        cands.append(cand)
        extras.append(ex)
    if not rows:
        return None
    n, c_max = len(rows), max(len(c) for c in cands)
    cand_t = torch.zeros(n, c_max, dtype=torch.long)
    mask = torch.zeros(n, c_max, dtype=torch.bool)
    extra = torch.zeros(n, c_max, FIX_EXTRA)
    for k, (c, ex) in enumerate(zip(cands, extras)):
        cand_t[k, :len(c)] = torch.tensor(c)
        mask[k, :len(c)] = True
        extra[k, :len(c)] = torch.tensor(ex)
    fix = {"rows": rows,
           "cand": cand_t.to(device),
           "mask": mask.to(device),
           "extra": extra.to(device),
           "lower": torch.tensor([idx[transitions[i].lower] for i in rows], device=device),
           "edge_row": torch.tensor([2 * i for i in rows], device=device)}
    if orig_upper is not None:
        fix["target"] = torch.tensor(targets, device=device)
    return fix


def fix_metrics(model, graphs):
    """Top-1 accuracy and MRR of the true upper among the candidate window,
    pooled over corrupted copies (train_bad-style tuples with the fix bundle
    last). n = recoverable QN-bump lines; n_cand = mean window size (1/n_cand
    is the random baseline for acc)."""
    ranks, n_cand = [], []
    with torch.no_grad():
        for g, *_, fix in graphs:
            if fix is None:
                continue
            logits = model.fix_logits(g, fix)
            true = logits.gather(1, fix["target"].unsqueeze(1))
            ranks += (1 + (logits > true).sum(1)).tolist()
            n_cand += fix["mask"].sum(1).tolist()
    ranks = np.array(ranks, dtype=float)
    return {"acc": float((ranks == 1).mean()), "mrr": float((1.0 / ranks).mean()),
            "n": len(ranks), "n_cand": float(np.mean(n_cand))}


def correction_report(model, transitions, device="cpu", j_pos=1):
    """Advisory QN-correction report on a real network: every line the outlier
    head flags (p > 0.5) gets the top-ranked upper-level relabeling from the
    candidate window. Scientific-integrity constraint (marvel_gnn_plan.md):
    for human literature review only — never applied to source data.
    Confidence is a softmax over the window, conditional on the upper actually
    being wrong — not an absolute error probability."""
    graph, idx = build_graph(transitions)
    graph = graph.to(device)
    levels = list(idx)
    with torch.no_grad():
        out_p = torch.sigmoid(model.outlier_logits(graph))
        suspects = (out_p > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
        fix = fix_sample(transitions, suspects, graph, idx, device=device, j_pos=j_pos)
        if fix is None:
            return []
        p = torch.softmax(model.fix_logits(graph, fix), dim=1)
        best = p.argmax(1)
    return [{"transition": i, "ref": transitions[i].ref,
             "upper": transitions[i].upper, "lower": transitions[i].lower,
             "proposed_upper": levels[fix["cand"][k, best[k]].item()],
             "outlier_p": float(out_p[i]), "confidence": float(p[k, best[k]])}
            for k, i in enumerate(fix["rows"])]
