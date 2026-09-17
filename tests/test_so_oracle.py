"""Oracle validation: the ported solver vs a real Marvel4.1.x (C++) bootstrap run.

Unlike the CO/CO2 oracles (published, deposited data), this oracle comes from a
sibling repo (MARVEL_scraping) that ran the actual compiled Marvel4.1.x binary
on SO's 34S16O/36S16O "fallback" networks (X3Sigma- N/F-conflict sources kept
disconnected, per that repo's ADR-0005/0006) and recorded its real
EnergyLevels.txt output. Closes the gap noted in marvel_gnn_plan.md: SO has
never had a 3-way C++/Python-port/oracle check, only CO has.

`transitions_raw.txt` is MARVEL4.1's own single-uncertainty-column input
format (not this port's native two-column format), and most 34S16O sources
are natively in MHz per `segment.txt` -- both handled below the same way
`parse_native`'s `segments` parameter is designed for.
"""

from pathlib import Path

import pytest

from marvel_gnn.core.network import split_components
from marvel_gnn.core.parse import parse_native, parse_native_levels
from marvel_gnn.core.solver import solve_energies

SO_DIR = Path(__file__).parent.parent / "data" / "SO_runs"

# (isotopologue, [expected main-component indices to validate])
CASES = [
    ("34S16O", [0]),      # one main network; 03SaMcTh kept disconnected
    ("36S16O", [0, 1]),   # 22HeStLy's own F1-branch/F2-branch, both perfect
]

TOLERANCE = 1e-5  # cm-1; real solve agrees to ~1e-7-1e-6 once units are converted

pytestmark = pytest.mark.skipif(
    not SO_DIR.exists(), reason="SO oracle data not present (see data/SO_runs/*/oracle)"
)


def _load_segments(path):
    segments = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        tag, unit = line.split()
        segments[tag] = unit
    return segments


def _duplicate_unc_column(raw_path, dst_path):
    """MARVEL4.1's own input has one unc column; this port's native format has two."""
    lines = []
    for line in Path(raw_path).read_text().splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        lines.append(" ".join([tokens[0], tokens[1], tokens[1], *tokens[2:]]))
    Path(dst_path).write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("iso,components", CASES)
def test_energies_match_real_marvel4_run(tmp_path, iso, components):
    oracle_dir = SO_DIR / iso / "oracle"
    if not oracle_dir.exists():
        pytest.skip(f"no oracle data for {iso}")

    native_path = tmp_path / f"{iso}.txt"
    _duplicate_unc_column(oracle_dir / "transitions_raw.txt", native_path)
    segments = _load_segments(oracle_dir / "segment.txt")

    kept, _ = parse_native(native_path, nqn=6, segments=segments)
    solvable, _ = split_components(kept, minsize=1)
    published = parse_native_levels(oracle_dir / "EnergyLevels.txt")

    for comp_idx in components:
        computed = solve_energies(solvable[comp_idx])
        common = set(computed) & set(published)
        assert len(common) == len(computed), (
            f"{iso} component {comp_idx}: {len(computed) - len(common)} "
            "computed levels missing from oracle"
        )
        diffs = {a: abs(computed[a] - published[a].energy) for a in common}
        worst = max(diffs, key=diffs.get)
        assert diffs[worst] < TOLERANCE, (
            f"{iso} component {comp_idx}, worst level {worst!r}: {diffs[worst]:.3e} cm-1"
        )
