"""Train the shared encoder with the uncertainty and outlier heads on CO.

Run: uv run python -m marvel_gnn.gnn.train

Split is cross-isotopologue: train on the three large networks, validate on
the two small ones (never seen in training).

Uncertainty eval: NLL and 1-sigma coverage of held-out masked-refit errors,
with the legacy Dijkstra+bootstrap uncertainty and the published e_E as
controls on the same errors. Coverage against published energies is a sanity
check only.

Outlier eval: average precision and precision/recall on held-out corrupted
copies of the validation networks (5% injected corruptions; see corrupt.py).
Deliberately not benchmarked against the legacy ratio-threshold heuristic
(marvel_gnn_plan.md) — that heuristic is what this head replaces.
"""

from pathlib import Path

import numpy as np
import torch

from marvel_gnn.core.network import split_components
from marvel_gnn.core.parse import parse_mrt_levels, parse_mrt_transitions
from marvel_gnn.core.uncertainty import marvel_solve
from marvel_gnn.gnn.corrupt import corrupt, largest_component
from marvel_gnn.gnn.data import ERROR_SCALE, build_graph, refit_error_matrix
from marvel_gnn.gnn.model import MarvelGNN, nll_loss

CO_DIR = Path(r"C:\Code\MARVEL\molecules\CO")
MODEL_PATH = Path(__file__).parents[3] / "data" / "marvel_gnn.pt"

TRAIN_ISOS = ["13C16O", "12C18O", "13C18O"]
VAL_ISOS = ["12C17O", "13C17O"]

N_TRAIN_SAMPLES = 200   # masked refits per training network (uncertainty)
N_VAL_SAMPLES = 100
N_TRAIN_CORRUPT = 30    # corrupted copies per training network (outlier)
N_VAL_CORRUPT = 50      # val networks are tiny; more copies for stable stratified stats
CORRUPT_PER_EPOCH = 6   # corrupted graphs sampled into each epoch's loss
OUT_LOSS_WEIGHT = 10.0  # uncertainty-NLL gradients dominate the shared encoder otherwise
EPOCHS = 1000


def prepare(iso, transitions, n_samples, seed):
    kept, _ = transitions[iso]
    comp = split_components(kept, minsize=2)[0][0]  # main component only
    graph, idx = build_graph(comp)
    errors = torch.tensor(
        refit_error_matrix(comp, n_samples=n_samples, rng=seed), dtype=torch.float32)
    errors[graph.ground] = float("nan")  # shift reference: error 0 by construction
    return comp, graph, errors


def prepare_corrupted(comp, n_graphs, seed, device):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n_graphs):
        bad, kinds, mags = corrupt(comp, rng=rng)
        bad, kinds, mags = largest_component(bad, kinds, mags)
        g, _ = build_graph(bad)
        graphs.append((g.to(device),
                       torch.tensor(kinds > 0, dtype=torch.float32, device=device),
                       kinds, mags))
    return graphs


def fixed_sigma_metrics(sigma, errors):
    """NLL and 1-sigma coverage for a fixed per-level sigma vector (mu-cm-1)."""
    log_sigma = torch.log(torch.tensor(sigma, dtype=torch.float32))
    nll = nll_loss(log_sigma, errors).item()
    mask = torch.isfinite(errors)
    cover = (errors.abs() <= torch.tensor(sigma, dtype=torch.float32).unsqueeze(1))[mask]
    return nll, cover.float().mean().item()


def average_precision(labels, scores):
    order = np.argsort(-scores)
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


MAG_BINS = [5.0, 15.0, 50.0, 150.0, 500.0]  # freq-shift amplitude in units of unc


def outlier_metrics(model, graphs):
    """Pooled AP + precision/recall at p=0.5 + mean per-graph precision@k,
    per-corruption-kind AP (each kind's positives vs clean negatives), and
    freq-shift recall stratified by shift amplitude (detectability floor)."""
    all_scores, all_kinds, all_mags, p_at_k = [], [], [], []
    with torch.no_grad():
        for g, labels, kinds, mags in graphs:
            logits = model.outlier_logits(g).cpu().numpy()
            all_scores.append(logits)
            all_kinds.append(kinds)
            all_mags.append(mags)
            k = int((kinds > 0).sum())
            top = np.argsort(-logits)[:k]
            p_at_k.append((kinds[top] > 0).mean())
    scores = np.concatenate(all_scores)
    kinds = np.concatenate(all_kinds)
    mags = np.concatenate(all_mags)
    labels = kinds > 0
    flagged = scores > 0.0  # p = 0.5
    m = {"ap": average_precision(labels, scores),
         "precision": labels[flagged].mean() if flagged.any() else float("nan"),
         "recall": flagged[labels].mean(),
         "p_at_k": float(np.mean(p_at_k)),
         "base": labels.mean()}
    for kind, name in [(1, "freq"), (2, "qn")]:
        mask = (kinds == kind) | (kinds == 0)
        m[f"ap_{name}"] = average_precision(kinds[mask] == kind, scores[mask])
        m[f"recall_{name}"] = flagged[kinds == kind].mean()
    m["freq_by_mag"] = [
        (lo, hi, int(sel.sum()), flagged[sel].mean() if sel.any() else float("nan"))
        for lo, hi in zip(MAG_BINS, MAG_BINS[1:])
        for sel in [(kinds == 1) & (mags >= lo) & (mags < hi)]]
    return m


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    transitions = parse_mrt_transitions(CO_DIR / "CO_isotopologues_all_input.txt")
    published = parse_mrt_levels(CO_DIR / "CO_isotopologues_all_output.txt")

    train_set = [(c, g.to(device), e.to(device))
                 for c, g, e in (prepare(iso, transitions, N_TRAIN_SAMPLES, seed=1)
                                 for iso in TRAIN_ISOS)]
    val_set = [(c, g.to(device), e.to(device))
               for c, g, e in (prepare(iso, transitions, N_VAL_SAMPLES, seed=2)
                               for iso in VAL_ISOS)]

    train_bad = []
    for i, (comp, _, _) in enumerate(train_set):
        train_bad += prepare_corrupted(comp, N_TRAIN_CORRUPT, seed=100 + i, device=device)
    val_bad = {iso: prepare_corrupted(comp, N_VAL_CORRUPT, seed=200 + i, device=device)
               for i, (iso, (comp, _, _)) in enumerate(zip(VAL_ISOS, val_set))}

    n_pos = sum(labels.sum().item() for _, labels, _, _ in train_bad)
    n_all = sum(len(labels) for _, labels, _, _ in train_bad)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((n_all - n_pos) / n_pos, device=device))

    model = MarvelGNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    best = (float("inf"), None)  # early stopping: keep the best-val-NLL checkpoint
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        unc_loss = sum(nll_loss(model.log_sigma(g), e) for _, g, e in train_set) / len(train_set)
        batch = rng.choice(len(train_bad), size=CORRUPT_PER_EPOCH, replace=False)
        out_loss = sum(bce(model.outlier_logits(train_bad[k][0]), train_bad[k][1])
                       for k in batch) / CORRUPT_PER_EPOCH
        (unc_loss + OUT_LOSS_WEIGHT * out_loss).backward()
        opt.step()

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                v_unc = sum(nll_loss(model.log_sigma(g), e) for _, g, e in val_set) / len(val_set)
            v_ap = np.mean([outlier_metrics(model, graphs)["ap"] for graphs in val_bad.values()])
            print(f"epoch {epoch:4d}  train unc {unc_loss.item():9.4f}  out {out_loss.item():.4f}"
                  f"  | val unc NLL {v_unc.item():10.4f}  val outlier AP {v_ap:.3f}")
            if v_unc.item() < best[0]:
                best = (v_unc.item(), epoch,
                        {k: v.clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[2])
    print(f"\nbest checkpoint: epoch {best[1]} (val unc NLL {best[0]:.4f})")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"model saved to {MODEL_PATH}\n")

    model.eval()
    for iso, (comp, graph, errors) in zip(VAL_ISOS, val_set):
        with torch.no_grad():
            sigma_gnn = torch.exp(model.log_sigma(graph)).cpu().numpy().astype(np.float64)

        legacy = marvel_solve(comp, bootstrap_iterations=100, rng=42)
        sigma_legacy = np.array([legacy[a][1] for a in graph.assignments]) * ERROR_SCALE
        pub = published[iso]
        sigma_pub = np.array([pub[a].unc for a in graph.assignments]) * ERROR_SCALE

        keep = np.arange(len(graph.assignments)) != graph.ground  # legacy/pub sigma = 0 there
        err = errors[keep].cpu()
        print(f"== {iso} uncertainty (validation, {errors.shape[1]} held-out refits) ==")
        for name, sigma in [("GNN", sigma_gnn[keep]),
                            ("legacy baseline", sigma_legacy[keep]),
                            ("published e_E", sigma_pub[keep])]:
            nll, cover = fixed_sigma_metrics(sigma, err)
            print(f"  {name:16s} NLL {nll:10.4f}   1-sigma coverage {cover:.1%} (target ~68%)")

        gap = np.abs(graph.level_energies.numpy()
                     - np.array([pub[a].energy for a in graph.assignments])) * ERROR_SCALE
        for name, sigma in [("GNN", sigma_gnn), ("legacy baseline", sigma_legacy)]:
            print(f"  sanity: |E_ours - E_published| <= sigma_{name}: "
                  f"{np.mean(gap[keep] <= sigma[keep]):.1%}")

        m = outlier_metrics(model, val_bad[iso])
        print(f"== {iso} outlier detection ({N_VAL_CORRUPT} corrupted copies, "
              f"{m['base']:.1%} lines bad) ==")
        print(f"  average precision {m['ap']:.3f} (random baseline {m['base']:.3f})")
        print(f"  precision {m['precision']:.1%} / recall {m['recall']:.1%} at p=0.5")
        print(f"  precision@k (k = nb injected) {m['p_at_k']:.1%}")
        print(f"  by kind: freq-shift AP {m['ap_freq']:.3f} recall {m['recall_freq']:.1%}"
              f" | qn-bump AP {m['ap_qn']:.3f} recall {m['recall_qn']:.1%}")
        strat = "  ".join(f"{lo:.0f}-{hi:.0f}x: {r:.0%} (n={n})"
                          for lo, hi, n, r in m["freq_by_mag"])
        print(f"  freq-shift recall by amplitude: {strat}")
        print()


if __name__ == "__main__":
    main()
