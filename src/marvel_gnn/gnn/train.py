"""Train the uncertainty-calibration head on CO and evaluate it.

Run: uv run python -m marvel_gnn.gnn.train

Split is cross-isotopologue: train on the three large networks, validate on
the two small ones (never seen in training). Primary eval: NLL and 1-sigma
coverage of *held-out* masked-refit errors, with two controls scored on the
same errors — the legacy Dijkstra+bootstrap uncertainty and the published e_E.
Coverage against published energies is reported as a sanity check only (per
marvel_gnn_plan.md, not used for model selection).
"""

import math
from pathlib import Path

import numpy as np
import torch

from marvel_gnn.core.network import split_components
from marvel_gnn.core.parse import parse_mrt_levels, parse_mrt_transitions
from marvel_gnn.core.uncertainty import marvel_solve
from marvel_gnn.gnn.data import ERROR_SCALE, build_graph, refit_error_matrix
from marvel_gnn.gnn.model import UncertaintyModel, nll_loss

CO_DIR = Path(r"C:\Code\MARVEL\molecules\CO")
MODEL_PATH = Path(__file__).parents[3] / "data" / "uncertainty_model.pt"

TRAIN_ISOS = ["13C16O", "12C18O", "13C18O"]
VAL_ISOS = ["12C17O", "13C17O"]

N_TRAIN_SAMPLES = 200
N_VAL_SAMPLES = 100
EPOCHS = 1000


def prepare(iso, transitions, n_samples, seed):
    kept, _ = transitions[iso]
    comp = split_components(kept, minsize=2)[0][0]  # main component only
    graph, idx = build_graph(comp)
    errors = torch.tensor(
        refit_error_matrix(comp, n_samples=n_samples, rng=seed), dtype=torch.float32)
    errors[graph.ground] = float("nan")  # shift reference: error 0 by construction
    return comp, graph, errors


def fixed_sigma_metrics(sigma, errors):
    """NLL and 1-sigma coverage for a fixed per-level sigma vector (mu-cm-1)."""
    log_sigma = torch.log(torch.tensor(sigma, dtype=torch.float32))
    nll = nll_loss(log_sigma, errors).item()
    mask = torch.isfinite(errors)
    cover = (errors.abs() <= torch.tensor(sigma, dtype=torch.float32).unsqueeze(1))[mask]
    return nll, cover.float().mean().item()


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

    model = UncertaintyModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = sum(nll_loss(model(g), e) for _, g, e in train_set) / len(train_set)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                vloss = sum(nll_loss(model(g), e) for _, g, e in val_set) / len(val_set)
            print(f"epoch {epoch:4d}  train NLL {loss.item():8.4f}  val NLL {vloss.item():8.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nmodel saved to {MODEL_PATH}\n")

    model.eval()
    for iso, (comp, graph, errors) in zip(VAL_ISOS, val_set):
        with torch.no_grad():
            sigma_gnn = torch.exp(model(graph)).cpu().numpy().astype(np.float64)

        legacy = marvel_solve(comp, bootstrap_iterations=100, rng=42)
        sigma_legacy = np.array([legacy[a][1] for a in graph.assignments]) * ERROR_SCALE
        pub = published[iso]
        sigma_pub = np.array([pub[a].unc for a in graph.assignments]) * ERROR_SCALE

        keep = np.arange(len(graph.assignments)) != graph.ground  # legacy/pub sigma = 0 there
        err = errors[keep].cpu()
        print(f"== {iso} (validation, {errors.shape[1]} held-out refits) ==")
        for name, sigma in [("GNN", sigma_gnn[keep]),
                            ("legacy baseline", sigma_legacy[keep]),
                            ("published e_E", sigma_pub[keep])]:
            nll, cover = fixed_sigma_metrics(sigma, err)
            print(f"  {name:16s} NLL {nll:8.4f}   1-sigma coverage {cover:.1%} (target ~68%)")

        # sanity check: does sigma cover the gap to the published energies?
        gap = np.abs(graph.level_energies.numpy()
                     - np.array([pub[a].energy for a in graph.assignments])) * ERROR_SCALE
        for name, sigma in [("GNN", sigma_gnn), ("legacy baseline", sigma_legacy)]:
            print(f"  sanity: |E_ours - E_published| <= sigma_{name}: "
                  f"{np.mean(gap[keep] <= sigma[keep]):.1%}")
        print()


if __name__ == "__main__":
    main()
