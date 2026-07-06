"""Oracle validation: the ported solver vs the published CO2 isotopologue levels
(12 isotopologues, 7-token CDSD assignments, mixed-unit deposition).

For every isotopologue the published level set is exactly the largest
connected component of the deposited transitions, with the ground state at 0.
Unlike CO, the deposited uncertainties differ from the optimized ones used in
the published solve, so absolute energy agreement varies (1e-10 .. 2e-3 cm-1);
the invariant that holds everywhere is agreement within 3x the published
level uncertainty.
"""

from pathlib import Path

import pytest

from marvel_gnn.core.network import split_components
from marvel_gnn.core.parse import infer_segments, parse_native, parse_native_levels
from marvel_gnn.core.solver import solve_energies

CO2_DIR = Path(r"C:\Code\MARVEL\molecules\CO2")

ISOS = ["626", "627", "628", "636", "637", "638",
        "727", "728", "737", "738", "828", "838"]

pytestmark = pytest.mark.skipif(not CO2_DIR.exists(), reason="CO2 oracle data not present")


@pytest.mark.parametrize("iso", ISOS)
def test_energies_match_published(iso):
    published = parse_native_levels(CO2_DIR / f"EnergyLevels_{iso}.txt")
    tr_path = CO2_DIR / f"Transitions_{iso}.txt"
    segments, _ = infer_segments(tr_path, published)
    kept, _ = parse_native(tr_path, segments=segments)

    solvable, _ = split_components(kept, minsize=2)
    computed = solve_energies(solvable[0])  # published = largest component only

    assert set(computed) == set(published)
    assert min(lv.energy for lv in published.values()) == 0.0

    over = [a for a in published
            if abs(computed[a] - published[a].energy)
            > 3 * max(published[a].unc, 1e-6)]
    assert not over, f"{iso}: {len(over)} levels beyond 3 sigma, e.g. {over[:3]}"
