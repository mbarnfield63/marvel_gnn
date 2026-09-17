"""Out-of-distribution eval: the CO-trained uncertainty-calibration head on SO.

Context (see SO_HANDOFF_from_marvel_scraping.md): a sibling repo's real
Marvel4.1.x bootstrap runs found SO's 34S16O network is ~60% wrong/very-bad,
almost entirely attributable to 22HeStLy's own blanket 0.05 cm-1 uncertainty
being miscalibrated against tighter sources; 36S16O's 22HeStLy data is, by
contrast, internally perfect (0% wrong/very-bad). tests/test_so_oracle.py
already validates this port's solve against that real run to ~1e-6 cm-1, so
the residual/uncertainty ratio computed here from our OWN solve is the same
signal Marvel4.1's BadLines.txt reports, without needing to parse its raw
output format.

This script asks one question: never having seen SO or any non-numeric QN
token during training (CO's schema is all-numeric v,J), does the CO-trained
model's predicted per-level sigma still track which levels sit on
inconsistent (high residual/unc) transitions? SO's QN schema (state, v, N, J,
F, ef) has three token kinds qn_array() cannot parse as-is -- an electronic
state name, an F1/F2/F3 spin label, and a "-" structural-disconnection dummy
(see data/SO_runs/*/oracle and .../fallback build scripts) -- so assignments
are re-encoded to plain integers below before featurization. This is an
eval-time adapter only; marvel_gnn/gnn/data.py is untouched.

Advisory/read-only: no training, no model changes, no data written back.
"""

from pathlib import Path

import numpy as np
import torch

from marvel_gnn.core.network import split_components
from marvel_gnn.core.parse import Transition, parse_native
from marvel_gnn.core.solver import level_index, solve_energies
from marvel_gnn.core.uncertainty import marvel_solve
from marvel_gnn.gnn.data import ERROR_SCALE, build_graph
from marvel_gnn.gnn.model import MarvelGNN

SO_DIR = Path(__file__).parent.parent / "data" / "SO_runs"
MODEL_PATH = Path(__file__).parent.parent / "data" / "marvel_gnn.pt"

# (isotopologue, [main-component indices to check], legacy bootstrap iterations)
CASES = [("34S16O", [0], 100), ("36S16O", [0, 1], 100)]

F_LABEL = {"F1": 1, "F2": 2, "F3": 3}


def _load_segments(path):
    segments = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            tag, unit = line.split()
            segments[tag] = unit
    return segments


def _encode_so(transitions):
    """Map SO's (state, v, N, J, F, ef) tokens to plain numeric strings:
    electronic state -> stable integer id, F1/F2/F3 -> 1/2/3, dummy "-" -> 0,
    e/f left alone (qn_array already handles parity letters). v/N/J are
    already numeric and pass through unchanged."""
    states = sorted({tok for t in transitions for a in (t.upper, t.lower)
                      for tok in [a.split()[0]]})
    state_id = {s: i for i, s in enumerate(states)}

    def encode(a):
        toks = a.split()
        toks[0] = str(state_id[toks[0]])
        toks[2] = "0" if toks[2] == "-" else toks[2]
        toks[4] = "0" if toks[4] == "-" else str(F_LABEL[toks[4]])
        return " ".join(toks)

    return [Transition(t.freq, t.unc, t.orig_unc, encode(t.upper), encode(t.lower), t.ref)
            for t in transitions], state_id


def per_level_badness(transitions):
    """Max |resid|/unc over each level's incident transitions, using our own
    (validated) solve -- the same quantity Marvel4.1's BadLines.txt reports."""
    energies = solve_energies(transitions)
    idx = level_index(transitions)
    badness = np.zeros(len(idx))
    source_22hestly = np.zeros(len(idx), dtype=bool)
    for t in transitions:
        ratio = abs(t.freq - (energies[t.upper] - energies[t.lower])) / t.unc
        for a in (t.upper, t.lower):
            i = idx[a]
            badness[i] = max(badness[i], ratio)
            if t.tag == "22HeStLy":
                source_22hestly[i] = True
    return badness, source_22hestly


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    model = MarvelGNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    for iso, comp_indices, boot_iter in CASES:
        oracle_dir = SO_DIR / iso / "oracle"
        if not oracle_dir.exists():
            print(f"{iso}: no oracle data at {oracle_dir}, skipping")
            continue

        raw = oracle_dir / "transitions_raw.txt"
        native = raw.with_name("_native_tmp.txt")
        lines = [" ".join([tok[0], tok[1], tok[1], *tok[2:]])
                 for tok in (l.split() for l in raw.read_text().splitlines() if l.strip())]
        native.write_text("\n".join(lines) + "\n")
        segments = _load_segments(oracle_dir / "segment.txt")
        kept, _ = parse_native(native, nqn=6, segments=segments)
        native.unlink()
        solvable, _ = split_components(kept, minsize=1)

        for ci in comp_indices:
            comp = solvable[ci]
            n_levels = len(level_index(comp))
            badness, is_22hestly = per_level_badness(comp)

            orig_assignments = list(level_index(comp))  # same order as the encoded graph
            encoded, _ = _encode_so(comp)
            graph, _ = build_graph(encoded)
            with torch.no_grad():
                sigma_gnn = torch.exp(model.log_sigma(graph)).numpy() / ERROR_SCALE

            legacy = marvel_solve(comp, bootstrap_iterations=boot_iter, rng=42)
            sigma_legacy = np.array([legacy[a][1] for a in orig_assignments])

            keep = np.arange(n_levels) != graph.ground  # ground level unc = 0 by construction
            print(f"== {iso} component {ci} ({n_levels} levels, {len(comp)} transitions, "
                  f"{is_22hestly.sum()} touched by 22HeStLy) ==")
            print(f"  median empirical badness (max resid/unc): "
                  f"22HeStLy levels {np.median(badness[is_22hestly]):.3f}  "
                  f"other levels {np.median(badness[~is_22hestly]) if (~is_22hestly).any() else float('nan'):.3f}")
            print(f"  Spearman(GNN sigma, empirical badness):    {spearman(sigma_gnn[keep], badness[keep]):.3f}")
            print(f"  Spearman(legacy sigma, empirical badness): {spearman(sigma_legacy[keep], badness[keep]):.3f}")
            print(f"  median sigma (cm-1): GNN {np.median(sigma_gnn[keep]):.4f}  "
                  f"legacy {np.median(sigma_legacy[keep]):.4f}")
            print()


if __name__ == "__main__":
    main()
