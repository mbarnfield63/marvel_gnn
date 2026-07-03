"""Spectroscopic-network decomposition (DFS in MARVEL4.1.cpp, lines 505-576)."""

import networkx as nx


def split_components(transitions, minsize=1):
    """Split transitions into connected components (spectroscopic networks).

    Returns (solvable, dropped): each a list of transition lists, one per
    component, ordered largest-first. Components with fewer than `minsize`
    levels go to `dropped` — MARVEL never computes energies for them (the
    orphan problem the GNN work targets).
    """
    g = nx.Graph()
    for i, t in enumerate(transitions):
        g.add_edge(t.upper, t.lower, idx=i)

    solvable, dropped = [], []
    for nodes in sorted(nx.connected_components(g), key=len, reverse=True):
        comp = [t for t in transitions if t.upper in nodes]
        (solvable if len(nodes) >= minsize else dropped).append(comp)
    return solvable, dropped
