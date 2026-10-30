"""Blind C2H4 test, step C: one-pass predictions on the 12C2H4 v1 networks.

Run: uv run python scripts/c2h4_blind_predict.py

Pre-registered in reports/2026-10-29_c2h4_blind/PREREGISTRATION.md. Writes
per-line and per-level TSVs to data/C2H4_v1/predictions/ (plus their sha256).
Never reads the answer key (overrides, decisions.md).

Graphs (ground-state component only; other lines are written with in_main=0):
- unedited: pooled 0b56238 per-source inputs, source negations flipped back to
  positive, blank uncertainties filled per source policy.
- edited: v1 deliverable trans, negated lines dropped.
"""

import hashlib
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from marvel_gnn.core.parse import UNC_FLOOR, Transition, parse_native
from marvel_gnn.core.solver import level_index, solve_energies
from marvel_gnn.gnn.data import ERROR_SCALE, build_graph, edge_leverages
from marvel_gnn.gnn.model import MarvelGNN

ROOT = Path(__file__).parents[1]
C2H4 = ROOT / "data" / "C2H4_v1"
OUT = C2H4 / "predictions"
MODEL = ROOT / "data" / "marvel_gnn.pt"
GROUND = " ".join(["0"] * 15)

# Blank per-line uncertainties: stated_unc from MARVEL_network HEAD sources.tsv,
# else 1e-3 cm-1 (user policy, 2026-10-29).
BLANK_UNC = {"10LoBoLoRo": 0.0005, "24PeLeSuRu": 0.005, "26PeLeSuRu": 0.0013,
             "18AsRa": 1e-3, "23MrMaReAr": 1e-3, "24BeBeReMa": 1e-3}
N_ITER = 641  # iterative baseline: lines removed = 550 network + 91 source negations


def load_unedited():
    lines, flipped = [], set()
    for f in sorted((C2H4 / "input").glob("C2H4_12C2H4_*_qam.txt")):
        for row in f.read_text().splitlines()[1:]:
            t = row.split("\t")
            freq = float(t[0])
            if freq < 0.0:
                flipped.add(t[32])
            unc = float(t[1]) if t[1] else BLANK_UNC[t[32].split(".")[0]]
            up, lo = " ".join(t[2:17]), " ".join(t[17:32])
            assert up != lo, t[32]
            lines.append(Transition(abs(freq), unc or UNC_FLOOR, unc or UNC_FLOOR, up, lo, t[32]))
    return lines, flipped


def ground_component(lines):
    g = nx.Graph((t.upper, t.lower) for t in lines)
    main = nx.node_connected_component(g, GROUND)
    assert len(main) == max(len(c) for c in nx.connected_components(g))
    return [t for t in lines if t.upper in main]


def residuals(comp):
    e = solve_energies(comp)
    return np.array([abs(t.freq - (e[t.upper] - e[t.lower])) for t in comp])


def iterative_ranks(comp, n_iter):
    """MARVEL-style loop: drop the worst |resid|/unc line, re-solve the ground
    component, repeat. Returns rank (0 = removed first) per line; lines never
    removed rank after, by |resid|/unc of the final solve; lines cut off from
    the ground component by a removal rank last."""
    ids = [t.ref for t in comp]
    rank = {}
    cur = list(comp)
    for r in range(n_iter):
        z = residuals(cur) / np.array([t.unc for t in cur])
        rank[cur[int(np.argmax(z))].ref] = r
        cur = ground_component([t for t in cur if t.ref not in rank])
    z = residuals(cur) / np.array([t.unc for t in cur])
    for k in np.argsort(-z):
        rank[cur[k].ref] = len(rank)
    for i in ids:
        rank.setdefault(i, len(ids))
    return np.array([rank[i] for i in ids])


def predict(name, lines, model, rng, iterate=False):
    comp = ground_component(lines)
    idx = level_index(comp)
    graph, _ = build_graph(comp)
    with torch.no_grad():
        logit = model.outlier_logits(graph).numpy()
        sigma = np.exp(model.log_sigma(graph).numpy().astype(np.float64)) / ERROR_SCALE  # cm-1
    resid = residuals(comp)
    unc = np.array([t.unc for t in comp])
    h = edge_leverages(comp, idx)
    ui = np.array([idx[t.upper] for t in comp])
    li = np.array([idx[t.lower] for t in comp])
    line_sigma = np.hypot(sigma[ui], sigma[li])
    cols = {
        "resid_over_unc": resid / unc,
        "leverage": h,
        "studentized": resid / (unc * np.sqrt(np.maximum(1.0 - h, 1e-12))),
        "gnn_outlier_logit": logit,
        "gnn_line_sigma": line_sigma,
        "gnn_sigma_over_unc": line_sigma / unc,
        "random": rng.random(len(comp)),
    }
    if iterate:
        cols["iterative_rank"] = iterative_ranks(comp, N_ITER)

    main = {t.ref for t in comp}
    rows = ["\t".join(["id", "upper", "lower", "freq", "unc", "in_main", *cols])]
    pos = {t.ref: k for k, t in enumerate(comp)}
    for t in lines:
        k = pos.get(t.ref)
        vals = [f"{cols[c][k]:.10g}" if k is not None else "nan" for c in cols]
        rows.append("\t".join([t.ref, t.upper, t.lower, f"{t.freq:.6f}", f"{t.unc:.3e}",
                               str(int(t.ref in main)), *vals]))
    (OUT / f"{name}_lines.tsv").write_text("\n".join(rows) + "\n")

    e = graph.level_energies.numpy()
    lev = ["assignment\tenergy\tgnn_sigma"] + [
        f"{a}\t{e[i]:.6f}\t{sigma[i]:.6e}" for a, i in idx.items()]
    (OUT / f"{name}_levels.tsv").write_text("\n".join(lev) + "\n")
    print(f"{name}: {len(lines)} lines, ground component {len(idx)} levels / {len(comp)} lines")


def main():
    OUT.mkdir(exist_ok=True)
    model = MarvelGNN()
    model.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    model.eval()
    rng = np.random.default_rng(20261029)
    unedited, flipped = load_unedited()
    print(f"unedited: {len(flipped)} source negations flipped back in")
    predict("unedited", unedited, model, rng, iterate=True)
    predict("edited", parse_native(C2H4 / "C2H4_12C2H4_trans.txt")[0], model, rng)

    files = [MODEL] + sorted(OUT.glob("*.tsv"))
    sums = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in files]
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))


if __name__ == "__main__":
    main()
