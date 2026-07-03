from pathlib import Path

import pytest

from marvel_gnn.core.parse import (
    C_CM_PER_S,
    load_segments,
    parse_mrt_levels,
    parse_mrt_transitions,
    parse_native,
)

CO_DIR = Path(r"C:\Code\MARVEL\molecules\CO")

NATIVE_SAMPLE = """\
2108.76597 1.1e-4 1.1e-4 1 1 0 2 79Guelachvili.2
& comment line to be skipped
too short line
2104.95112 1.1e-4 0.0 1 2 0 3 79Guelachvili.3
-2112.54742 1.1e-4 1.1e-4 1 0 0 1 79Guelachvili.1
"""


def test_parse_native(tmp_path):
    f = tmp_path / "tr.txt"
    f.write_text(NATIVE_SAMPLE)
    kept, excluded = parse_native(f, nqn=2)

    assert [t.ref for t in kept] == ["79Guelachvili.2", "79Guelachvili.3"]
    t = kept[0]
    assert (t.freq, t.orig_unc, t.unc) == (2108.76597, 1.1e-4, 1.1e-4)
    assert (t.upper, t.lower, t.tag) == ("1 1", "0 2", "79Guelachvili")
    assert kept[1].unc == 1e-6  # zero optimized unc floored

    assert len(excluded) == 1 and excluded[0].freq == -2112.54742


def test_parse_native_rejects_self_transition(tmp_path):
    f = tmp_path / "tr.txt"
    f.write_text("100.0 1e-4 1e-4 0 1 0 1 20AaBb.1\n")
    with pytest.raises(ValueError, match="upper == lower"):
        parse_native(f, nqn=2)


def test_segment_unit_conversion(tmp_path):
    tr = tmp_path / "tr.txt"
    tr.write_text("115271.2018 0.005 0.005 0 1 0 0 70RoDo.1\n")
    seg = tmp_path / "seg.txt"
    seg.write_text("70RoDo MHz\n")

    kept, _ = parse_native(tr, nqn=2, segments=load_segments(seg))
    assert kept[0].freq == pytest.approx(115271.2018e6 / C_CM_PER_S)
    assert kept[0].unc == pytest.approx(0.005e6 / C_CM_PER_S)


def test_segment_missing_tag_raises(tmp_path):
    tr = tmp_path / "tr.txt"
    tr.write_text("100.0 1e-4 1e-4 0 1 0 0 20AaBb.1\n")
    seg = tmp_path / "seg.txt"
    seg.write_text("99ZzYy cm-1\n")
    with pytest.raises(KeyError, match="missing segment"):
        parse_native(tr, nqn=2, segments=load_segments(seg))


@pytest.mark.skipif(not CO_DIR.exists(), reason="CO oracle data not present")
class TestCOOracleFiles:
    def test_transitions(self):
        result = parse_mrt_transitions(CO_DIR / "CO_isotopologues_all_input.txt")

        assert set(result) == {"12C17O", "12C18O", "13C16O", "13C17O", "13C18O"}
        assert sum(len(k) + len(e) for k, e in result.values()) == 6068

        kept, excluded = result["12C17O"]
        first = kept[0]
        assert (first.freq, first.unc) == (3.747901334, 1.670e-6)
        assert (first.upper, first.lower, first.ref) == ("0 1", "0 0", "03KlSuLeMu.1")
        # the one negative-frequency line in 12C17O
        assert [e.ref for e in excluded] == ["79Guelachvili.1"]

    def test_levels(self):
        levels = parse_mrt_levels(CO_DIR / "CO_isotopologues_all_output.txt")

        assert set(levels) == {"12C17O", "12C18O", "13C16O", "13C17O", "13C18O"}
        lv = levels["12C17O"]["0 1"]  # v=0, J=1
        assert lv.energy == 3.747902324
        assert lv.unc == 0.000002001
        assert lv.n_trans == 8
