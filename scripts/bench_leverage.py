"""Time build_graph / edge_leverages on the 12C2H4 v1 network (largest component).

Run: uv run python scripts/bench_leverage.py [--gpu] [--validate]

--validate checks the sparse edge_leverages against the dense reduced-Laplacian
inverse on every CO component and the largest CO2 component, plus bridges and
sum(h), and times it on C2H4 and CO2 628.

Two graphs: the v1 deliverable (edited, negated lines dropped) and the
unedited per-source inputs pooled. Timing only — no GNN heads are run.
C2H4 assignments have 15 QN tokens; data.MAX_QN is raised in-process only so
qn_array() can build the node block (no source or model change).
"""

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

import marvel_gnn.gnn.data as gdata
from marvel_gnn.core.network import split_components
from marvel_gnn.core.parse import Transition, parse_native
from marvel_gnn.core.solver import level_index

C2H4 = Path(__file__).parents[1] / "data" / "C2H4_v1"
gdata.MAX_QN = 15  # timing only; the trained model still expects 7


def load_unedited():
    """Pool the per-source TSVs: freq unc 15 upper QNs, 15 lower QNs, id."""
    kept = []
    for f in sorted((C2H4 / "input").glob("C2H4_12C2H4_*.txt")):
        for line in f.read_text().splitlines()[1:]:
            tok = line.split("\t")
            # blank unc (6 sources state none per line): 1e-3 cm-1, timing only
            freq, unc = float(tok[0]), float(tok[1] or 1e-3) or 1e-6
            if freq < 0.0:
                continue
            up, lo = " ".join(tok[2:17]), " ".join(tok[17:32])
            if up != lo:
                kept.append(Transition(freq, unc, unc, up, lo, tok[32]))
    return kept


def timed(fn, *args):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args)
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 2**30
    tracemalloc.stop()
    return out, dt, peak


def dense_laplacian(comp, idx):
    n = len(idx)
    lap = np.zeros((n, n))
    for t in comp:
        i, j, w = idx[t.upper], idx[t.lower], 1.0 / (t.unc * t.unc)
        lap[i, i] += w
        lap[j, j] += w
        lap[i, j] -= w
        lap[j, i] -= w
    return lap


def dense_leverages(comp, idx):
    """Reference: the original dense-inverse edge_leverages."""
    lap = dense_laplacian(comp, idx)
    g = np.zeros_like(lap)
    g[1:, 1:] = np.linalg.inv(lap[1:, 1:])
    return np.clip(np.array([(g[i, i] + g[j, j] - 2.0 * g[i, j]) / t.unc**2
                             for t in comp for i, j in [(idx[t.upper], idx[t.lower])]]),
                   0.0, 1.0)


def stable_leverages(comp, idx):
    """Cancellation-free reference h_e = w_e |C^-1 x_e|^2 (A = C C^T); dense, CO-size only."""
    from scipy.linalg import cholesky, solve_triangular
    lap = dense_laplacian(comp, idx)
    x = np.zeros((len(idx), len(comp)))
    for e, t in enumerate(comp):
        x[idx[t.upper], e], x[idx[t.lower], e] = 1.0, -1.0
    y = solve_triangular(cholesky(lap[1:, 1:], lower=True), x[1:], lower=True)
    return np.clip((y**2).sum(0) / np.array([t.unc**2 for t in comp]), 0.0, 1.0)


def gpu_leverages(comp, idx, dtype):
    """Same reduced-system dense inverse as edge_leverages, on the GPU."""
    import torch
    lap = torch.tensor(dense_laplacian(comp, idx)[1:, 1:], dtype=dtype, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g = torch.zeros((len(idx),) * 2, dtype=dtype, device="cuda")
    g[1:, 1:] = torch.linalg.inv(lap)
    del lap
    ui = torch.tensor([idx[t.upper] for t in comp], device="cuda")
    li = torch.tensor([idx[t.lower] for t in comp], device="cuda")
    w = torch.tensor([1.0 / t.unc**2 for t in comp], dtype=dtype, device="cuda")
    h = (w * (g[ui, ui] + g[li, li] - 2.0 * g[ui, li])).clamp(0.0, 1.0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return h.double().cpu().numpy(), dt, torch.cuda.max_memory_allocated() / 2**30


def validate():
    import networkx as nx
    from marvel_gnn.core.parse import infer_segments, parse_mrt_transitions, parse_native_levels
    from marvel_gnn.gnn.train import CO_DIR
    co2 = CO_DIR.parent / "CO2"
    sets = [(f"CO {iso}#{k}", c, iso == "13C18O" and k == 0)
            for iso, (kept, _) in parse_mrt_transitions(CO_DIR / "CO_isotopologues_all_input.txt").items()
            for k, c in enumerate(split_components(kept, minsize=2)[0])]
    seg, _ = infer_segments(co2 / "Transitions_628.txt", parse_native_levels(co2 / "EnergyLevels_628.txt"))
    sets.append(("CO2 628 main", split_components(
        parse_native(co2 / "Transitions_628.txt", segments=seg)[0], minsize=2)[0][0], False))
    worst = 0.0
    for name, comp, show in sets:
        idx = level_index(comp)
        hs = gdata.edge_leverages(comp, idx)
        g = nx.MultiGraph((t.upper, t.lower) for t in comp)
        bridges = {frozenset(e) for e in nx.bridges(nx.Graph(g))
                   if g.number_of_edges(*e) == 1}
        hb = np.array([hs[k] for k, t in enumerate(comp) if frozenset((t.upper, t.lower)) in bridges])
        assert np.isclose(hs.sum(), len(idx) - 1) and np.allclose(hb, 1.0)
        dmax = np.abs(hs - dense_leverages(comp, idx)).max()
        worst = max(worst, dmax) if name.startswith("CO ") else worst
        if len(idx) > 300 or show:
            extra = ""
            if len(idx) < 2000:
                st = stable_leverages(comp, idx)
                extra = (f"  vs cancellation-free ref: sparse {np.abs(hs - st).max():.1e}"
                         f" dense {np.abs(dense_leverages(comp, idx) - st).max():.1e}")
            print(f"{name}: {len(idx)} levels, {len(bridges)} bridges (h=1 ok), "
                  f"sum(h)=n-1 ok, max|sparse-dense| {dmax:.1e}{extra}")
    print(f"all {sum(n.startswith('CO ') for n, *_ in sets)} CO components: worst max|sparse-dense| {worst:.1e}")

    edited, _ = parse_native(C2H4 / "C2H4_12C2H4_trans.txt")
    for name, trans in [("C2H4 edited", edited), ("C2H4 unedited", load_unedited()),
                        ("CO2 628 main", sets[-1][1])]:
        comp = split_components(trans, minsize=2)[0][0]
        idx = level_index(comp)
        _, dt, peak = timed(gdata.edge_leverages, comp, idx)
        _, dtg, _ = timed(gdata.build_graph, comp)
        print(f"{name}: {len(idx)} levels, sparse edge_leverages {dt:.2f} s (peak {peak:.3f} GiB), "
              f"build_graph {dtg:.2f} s")


def main():
    if "--validate" in sys.argv:
        return validate()
    gpu = "--gpu" in sys.argv
    edited, _ = parse_native(C2H4 / "C2H4_12C2H4_trans.txt")
    for name, trans in [("edited", edited), ("unedited", load_unedited())]:
        comp = split_components(trans, minsize=2)[0][0]
        idx = level_index(comp)
        print(f"\n[{name}] {len(trans)} lines -> main component "
              f"{len(idx)} levels, {len(comp)} lines")
        h, dt, peak = timed(gdata.edge_leverages, comp, idx)
        print(f"  edge_leverages (numpy dense): {dt:7.1f} s  peak {peak:.2f} GiB  "
              f"sum(h)={h.sum():.2f} (n-1={len(idx) - 1})")
        _, dt, peak = timed(gdata.build_graph, comp)
        print(f"  build_graph (total):          {dt:7.1f} s  peak {peak:.2f} GiB")
        if gpu:
            import torch
            for dtype in (torch.float64, torch.float32):
                hg, dt, vram = gpu_leverages(comp, idx, dtype)
                err = np.abs(hg - h)
                print(f"  torch inv {str(dtype)[6:]:7s} GPU: {dt:7.1f} s  peak VRAM {vram:.2f} GiB  "
                      f"|dh| max {err.max():.2e} median {np.median(err):.2e}  "
                      f"sum(h)={hg.sum():.2f}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
