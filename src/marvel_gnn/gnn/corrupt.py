"""Synthetic corruption injection for outlier-detection training and eval.

Every in-scope dataset is post-review clean (marvel_gnn_plan.md): no real
labeled errors exist, so the outlier head trains and evaluates on known
injected corruptions.
"""

from dataclasses import replace

import networkx as nx
import numpy as np


FREQ_SHIFT = 1  # kind codes; 0 = clean
QN_BUMP = 2


def corrupt(transitions, fraction=0.05, rng=None, j_pos=1):
    """Corrupt a random subset of transitions. Returns (corrupted, kinds, mags):
    kinds an int8 array (0 clean, FREQ_SHIFT, or QN_BUMP), mags the shift
    magnitude in units of unc (0 for clean and QN_BUMP lines) — used for
    amplitude-stratified eval.

    j_pos is the J token's position in the assignment (CO "v J": 1, the
    default; CO2 CDSD "J v1 v2 l2 v3 r e/f": 0).

    Two failure modes, equal odds per corrupted line:
    - FREQ_SHIFT: frequency shifted by +-(5..500)x unc, log-uniform magnitude
      (measurement / transcription error)
    - QN_BUMP: J off-by-one on the upper level (QN misassignment; rewires the
      graph, possibly onto a level that does not otherwise exist)
    """
    rng = np.random.default_rng(rng)
    n_bad = max(1, round(fraction * len(transitions)))
    bad = set(rng.choice(len(transitions), size=n_bad, replace=False).tolist())
    kinds = np.zeros(len(transitions), dtype=np.int8)
    mags = np.zeros(len(transitions))

    out = []
    for i, t in enumerate(transitions):
        if i not in bad:
            out.append(t)
            continue

        new_upper = None
        if rng.random() < 0.5:
            toks = t.upper.split()
            j = int(toks[j_pos])
            delta = int(rng.choice([-1, 1]))
            for d in (delta, -delta):
                cand = " ".join(toks[:j_pos] + [str(j + d)] + toks[j_pos + 1:])
                if j + d >= 0 and cand != t.lower:
                    new_upper = cand
                    break
        if new_upper is not None:
            kinds[i] = QN_BUMP
            out.append(replace(t, upper=new_upper))
        else:  # frequency-shift mode (also the fallback when no valid J bump)
            kinds[i] = FREQ_SHIFT
            mags[i] = 10 ** rng.uniform(np.log10(5.0), np.log10(500.0))
            out.append(replace(t, freq=t.freq + float(rng.choice([-1.0, 1.0])) * t.unc * mags[i]))
    return out, kinds, mags


def largest_component(transitions, *labels):
    """Restrict (transitions, *label arrays) to the largest connected
    component — a QN rewire can split the network, and the solver needs it
    connected."""
    g = nx.Graph((t.upper, t.lower) for t in transitions)
    main = max(nx.connected_components(g), key=len)
    mask = np.array([t.upper in main for t in transitions])
    return ([t for t, m in zip(transitions, mask) if m],
            *(lab[mask] for lab in labels))
