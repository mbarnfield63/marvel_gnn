"""Input parsers: native MARVEL transitions format and CDS machine-readable (MRT) tables.

Faithful port of the input handling in MARVEL4.1.cpp (lines 321-499): same
skip rules, same unc floor, same unit conversion, same negative-frequency
exclusion. Assignment keys are the quantum-number tokens joined by single
spaces, e.g. "4 17" for a diatomic (v, J) level.
"""

from dataclasses import dataclass
from pathlib import Path

C_CM_PER_S = 2.99792458e10  # speed of light in cm/s (value used by MARVEL4.1.cpp)

_UNIT_TO_HZ = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}

UNC_FLOOR = 1e-6  # cm-1, applied when the optimized uncertainty is exactly 0


@dataclass
class Transition:
    freq: float      # cm-1
    unc: float       # optimized uncertainty used in the solve (cm-1)
    orig_unc: float  # original measured uncertainty (cm-1), used by the bootstrap
    upper: str       # upper-level assignment key
    lower: str       # lower-level assignment key
    ref: str         # source tag, e.g. "15CaKaKa.1"

    @property
    def tag(self) -> str:
        """Segment/paper tag: the ref without its trailing .N counter."""
        return self.ref.split(".", 1)[0]


@dataclass
class Level:
    """One row of a published MARVEL energy-levels table (validation oracle)."""
    energy: float   # cm-1
    unc: float      # published e_E, cm-1
    n_trans: int    # number of incident transitions


def parse_native(path, nqn=None, segments=None):
    """Parse a native MARVEL transitions file.

    Line format: freq  orig_unc  optim_unc  <nqn upper QNs>  <nqn lower QNs>  ref

    nqn=None infers the QN count from the first data line (tokens = 2*nqn + 4);
    the method is otherwise QN-count-oblivious since assignments are opaque keys.

    Returns (kept, excluded): excluded holds negative-frequency transitions
    (freq kept negative, as in the C++), which MARVEL leaves out of the solve
    and later re-checks for revival.
    """
    kept, excluded = [], []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if "&" in line:
            continue
        tokens = line.split()
        if len(tokens) < 5 or tokens[0].startswith("#"):
            continue
        if nqn is None:
            nqn = (len(tokens) - 4) // 2
        freq = float(tokens[0])
        orig_unc = float(tokens[1])
        unc = float(tokens[2])
        if unc == 0.0:
            unc = UNC_FLOOR
        upper = " ".join(tokens[3:3 + nqn])
        lower = " ".join(tokens[3 + nqn:3 + 2 * nqn])
        ref = tokens[2 * nqn + 3]

        if segments is not None:
            tag = ref.split(".", 1)[0]
            if tag not in segments:
                raise KeyError(f"missing segment: {tag}")
            factor = _UNIT_TO_HZ.get(segments[tag])  # unknown units mean cm-1, as in the C++
            if factor is not None:
                freq *= factor / C_CM_PER_S
                unc *= factor / C_CM_PER_S
                orig_unc *= factor / C_CM_PER_S

        t = Transition(freq, unc, orig_unc, upper, lower, ref)
        if freq < 0.0:
            excluded.append(t)
            continue
        if upper == lower:
            raise ValueError(f"upper == lower assignment at line {lineno}: {upper!r}")
        kept.append(t)
    return kept, excluded


def parse_native_levels(path):
    """Parse a native MARVEL energy-levels file: <QN tokens>  E  unc  n_trans.

    QN-count-oblivious: everything before the last three columns is the
    assignment key. Returns {assignment: Level}.
    """
    levels = {}
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 4 or tokens[0].startswith("#"):
            continue
        levels[" ".join(tokens[:-3])] = Level(
            energy=float(tokens[-3]), unc=float(tokens[-2]), n_trans=int(tokens[-1]))
    return levels


_CANDIDATE_UNITS = ("cm-1",) + tuple(_UNIT_TO_HZ)  # cm-1 first: wins ties


def infer_segments(path, levels, nqn=None):
    """Reconstruct a missing segment file: infer each source tag's unit.

    For datasets deposited without their segment file (the unit tags live in a
    separate MARVEL input that is often not published), test every tag's
    positive-frequency lines against solved level differences under each
    candidate unit; a tag's unit is the one matching the most lines within
    3x the optimized uncertainty. Tags with no checkable line default to cm-1.

    Returns (segments, report): segments is {tag: unit} usable directly by
    parse_native; report is {tag: {unit: n_matching}} plus a "lines" count —
    inferred units are a provenance decision, so review the report before
    trusting a solve built on it.
    """
    kept, excluded = parse_native(path, nqn)  # raw, no conversion
    rows = {t.tag: [] for t in kept + excluded}
    for t in kept:
        eu, el = levels.get(t.upper), levels.get(t.lower)
        if eu is not None and el is not None:
            rows[t.tag].append((t.freq, t.unc, eu.energy - el.energy))
    segments, report = {}, {}
    for tag, checkable in rows.items():
        counts = {}
        for unit in _CANDIDATE_UNITS:
            f = _UNIT_TO_HZ.get(unit, C_CM_PER_S) / C_CM_PER_S
            counts[unit] = sum(1 for freq, unc, diff in checkable
                               if abs(freq * f - diff) <= 3 * max(unc * f, UNC_FLOOR))
        segments[tag] = max(_CANDIDATE_UNITS, key=lambda u: counts[u])
        report[tag] = {"lines": len(checkable), **counts}
    return segments, report


def _mrt_data_lines(path):
    lines = Path(path).read_text().splitlines()
    dividers = [i for i, l in enumerate(lines) if l.startswith("-" * 40)]
    return lines[dividers[-1] + 1:]


def parse_mrt_transitions(path):
    """Parse a CDS MRT transitions table (Iso Name v e_v v' J' v'' J'' Tag).

    Returns {isotopologue: (kept, excluded)}. The single e_v column serves as
    both the optimized and original uncertainty.
    """
    result = {}
    for line in _mrt_data_lines(path):
        t = line.split()
        if not t:
            continue
        freq = float(t[2])
        unc = float(t[3])
        if unc == 0.0:
            unc = UNC_FLOOR
        tr = Transition(freq, unc, unc, f"{t[4]} {t[5]}", f"{t[6]} {t[7]}", t[8])
        kept, excluded = result.setdefault(t[0], ([], []))
        (excluded if freq < 0.0 else kept).append(tr)
    return result


def parse_mrt_levels(path):
    """Parse a CDS MRT energy-levels table (Iso Name v J E e_E N).

    Returns {isotopologue: {assignment: Level}}.
    """
    result = {}
    for line in _mrt_data_lines(path):
        t = line.split()
        if not t:
            continue
        result.setdefault(t[0], {})[f"{t[2]} {t[3]}"] = Level(
            energy=float(t[4]), unc=float(t[5]), n_trans=int(t[6]))
    return result
