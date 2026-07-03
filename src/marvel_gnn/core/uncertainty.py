"""Legacy MARVEL uncertainty baseline (MARVEL4.1.cpp, lines 584-667).

Two independent estimates, combined via a second shortest-path pass:

1. Shortest-path: Dijkstra from the lowest-energy level with edge weight
   unc^2; a level's uncertainty is sqrt of its path distance.
2. Bootstrap: resample each transition's uncertainty as U{1..10} x orig_unc,
   re-solve, repeat; bootunc = sqrt(2) * max(|E - median|, stdev).
3. Combined: Dijkstra again, but the cost of an edge is the *target level's*
   max(shortest-path unc, bootstrap unc)^2; final unc = sqrt(distance).

This is the control the GNN uncertainty head is benchmarked against — ported
faithfully, not endorsed.
"""

import math

import networkx as nx
import numpy as np

from .solver import solve_energies


def _root(energies):
    return min(energies, key=energies.get)


def shortest_path_unc(transitions, energies):
    """{assignment: unc} via Dijkstra with edge weight unc^2 from the lowest level."""
    g = nx.MultiGraph()
    g.add_weighted_edges_from((t.upper, t.lower, t.unc * t.unc) for t in transitions)
    dist = nx.single_source_dijkstra_path_length(g, _root(energies))
    return {a: math.sqrt(d) for a, d in dist.items()}


def bootstrap_unc(transitions, energies, iterations=100, rng=None):
    """{assignment: bootunc} by re-solving with resampled uncertainties."""
    rng = np.random.default_rng(rng)
    orig = np.array([t.orig_unc for t in transitions])
    samples = {a: [] for a in energies}
    for _ in range(iterations):
        resampled = rng.integers(1, 11, len(transitions)) * orig
        for a, e in solve_energies(transitions, unc=resampled).items():
            samples[a].append(e)

    out = {}
    for a, s in samples.items():
        s = np.asarray(s)
        diff = abs(energies[a] - np.median(s))
        std = s.std(ddof=1)
        out[a] = math.sqrt(2.0) * max(diff, std)
    return out


def combined_unc(transitions, energies, sp_unc, boot_unc):
    """Final uncertainty: shortest path where entering a level costs its
    max(sp_unc, boot_unc)^2. Parallel transitions collapse (same node cost)."""
    cost = {a: max(sp_unc[a], boot_unc[a]) ** 2 for a in energies}
    g = nx.DiGraph()
    for t in transitions:
        g.add_edge(t.lower, t.upper, weight=cost[t.upper])
        g.add_edge(t.upper, t.lower, weight=cost[t.lower])
    dist = nx.single_source_dijkstra_path_length(g, _root(energies))
    return {a: math.sqrt(d) for a, d in dist.items()}


def marvel_solve(transitions, bootstrap_iterations=0, rng=None):
    """Full legacy treatment of one connected component.

    Returns {assignment: (energy, unc)}; unc is the shortest-path estimate,
    or the bootstrap-combined one when bootstrap_iterations > 0.
    """
    energies = solve_energies(transitions)
    sp = shortest_path_unc(transitions, energies)
    if bootstrap_iterations:
        boot = bootstrap_unc(transitions, energies, bootstrap_iterations, rng)
        unc = combined_unc(transitions, energies, sp, boot)
    else:
        unc = sp
    return {a: (energies[a], unc[a]) for a in energies}
