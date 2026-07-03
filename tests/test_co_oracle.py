"""Oracle validation: the ported solver vs the published CO isotopologue levels
(Grigorev et al., ApJS 283, 2026).

For every isotopologue the published level set is exactly the largest
connected component of the deposited transitions, with the ground state at 0.

Energy tolerances are per-isotopologue: the deposited MRT input carries a
single uncertainty column, while the published analysis solved with
*optimized* (reweighted) uncertainties for conflicting lines. Where the input
is conflict-free we match to ~1e-9 cm-1; where lines conflict (12C18O, and
13C18O's v=1 J=0 level, attached to the network only through three mutually
inconsistent measurements) the published energies are not reproducible from
the deposited data, and the tolerance documents that gap.
"""

from pathlib import Path

import pytest

from marvel_gnn.core.network import split_components
from marvel_gnn.core.parse import parse_mrt_levels, parse_mrt_transitions
from marvel_gnn.core.solver import solve_energies

CO_DIR = Path(r"C:\Code\MARVEL\molecules\CO")

TOLERANCE = {
    "12C17O": 5e-9,
    "13C16O": 2e-8,
    "13C17O": 5e-8,
    "12C18O": 1e-6,   # optimized-unc gap, systematic ~6.5e-7
    "13C18O": 5e-8,   # excluding "1 0", handled separately below
}

pytestmark = pytest.mark.skipif(not CO_DIR.exists(), reason="CO oracle data not present")


@pytest.fixture(scope="module")
def oracle():
    transitions = parse_mrt_transitions(CO_DIR / "CO_isotopologues_all_input.txt")
    published = parse_mrt_levels(CO_DIR / "CO_isotopologues_all_output.txt")
    return transitions, published


@pytest.mark.parametrize("iso", sorted(TOLERANCE))
def test_energies_match_published(oracle, iso):
    transitions, published = oracle
    kept, _ = transitions[iso]
    solvable, _ = split_components(kept, minsize=2)
    computed = solve_energies(solvable[0])  # published = largest component only
    pub = published[iso]

    assert set(computed) == set(pub)
    assert computed["0 0"] == 0.0 and pub["0 0"].energy == 0.0

    known_gap = {"13C18O": {"1 0"}}.get(iso, set())
    diffs = {a: abs(computed[a] - pub[a].energy) for a in pub if a not in known_gap}
    worst = max(diffs, key=diffs.get)
    assert diffs[worst] < TOLERANCE[iso], f"worst level {worst}: {diffs[worst]:.3e}"

    for a in known_gap:
        assert abs(computed[a] - pub[a].energy) < 5e-4
