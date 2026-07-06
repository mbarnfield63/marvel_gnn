from pathlib import Path

import pytest

from marvel_gnn.core.parse import (
    C_CM_PER_S,
    infer_segments,
    load_segments,
    parse_mrt_levels,
    parse_mrt_transitions,
    parse_native,
    parse_native_levels,
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


def test_parse_native_infers_nqn_and_skips_comments(tmp_path):
    f = tmp_path / "tr.txt"
    # CO2-style 7-token assignments, plus a #-commented line (as in 636)
    f.write_text(
        "671.3 0.1 0.1 5 0 1 1 0 1 e 4 0 0 0 0 1 e 32MaBa.1\n"
        "#-11156.4 0.8 0.8 2 0 0 0 2 1 e 3 1 0 0 1 1 e 80Siemen.1\n"
    )
    kept, excluded = parse_native(f)  # nqn inferred = 7
    assert len(kept) == 1 and not excluded
    assert kept[0].upper == "5 0 1 1 0 1 e"
    assert kept[0].lower == "4 0 0 0 0 1 e"


def test_parse_native_levels(tmp_path):
    f = tmp_path / "lv.txt"
    f.write_text(
        "0 0 0 0 0 1 e          0.0000000000  0.000e-00  132\n"
        "2 0 0 0 0 1 e          2.3413093517  3.749e-08  339\n"
    )
    levels = parse_native_levels(f)
    lv = levels["2 0 0 0 0 1 e"]
    assert (lv.energy, lv.unc, lv.n_trans) == (2.3413093517, 3.749e-08, 339)


def test_infer_segments(tmp_path):
    tr = tmp_path / "tr.txt"
    tr.write_text(
        "3.845033 1e-6 1e-6 0 1 0 0 20AaBb.1\n"       # cm-1
        "115271.2018 0.005 0.005 0 1 0 0 70RoDo.1\n"  # same line in MHz
        "-100.0 1.0 1.0 0 1 0 0 99ZzYy.1\n"           # deactivated only -> default
    )
    lv = tmp_path / "lv.txt"
    lv.write_text("0 0 0.0 0.0 2\n0 1 3.845033 1e-6 2\n")

    segments, report = infer_segments(tr, parse_native_levels(lv))
    assert segments == {"20AaBb": "cm-1", "70RoDo": "MHz", "99ZzYy": "cm-1"}
    assert report["70RoDo"]["MHz"] == 1 and report["70RoDo"]["cm-1"] == 0

    kept, _ = parse_native(tr, segments=segments)
    assert all(abs(t.freq - 3.845033) < 1e-4 for t in kept)


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


CO2_DIR = Path(r"C:\Code\MARVEL\molecules\CO2")

CO2_ISOS = ["626", "627", "628", "636", "637", "638",
            "727", "728", "737", "738", "828", "838"]


@pytest.mark.skipif(not CO2_DIR.exists(), reason="CO2 data not present")
@pytest.mark.parametrize("iso", CO2_ISOS)
def test_co2_unit_inference_closes_network(iso):
    """With inferred segments every active line must reproduce E' - E'' to 3 sigma.

    The CO2 deposition (12 isotopologues, 7-token CDSD assignments) ships
    without segment files and mixes cm-1, MHz and kHz sources — exactly the
    case infer_segments exists for.
    """
    levels = parse_native_levels(CO2_DIR / f"EnergyLevels_{iso}.txt")
    tr_path = CO2_DIR / f"Transitions_{iso}.txt"  # casing varies; NTFS is case-insensitive
    segments, _ = infer_segments(tr_path, levels)
    kept, _ = parse_native(tr_path, segments=segments)

    checked = misses = 0
    for t in kept:
        eu, el = levels.get(t.upper), levels.get(t.lower)
        if eu is None or el is None:  # floating-component lines: no published level
            continue
        checked += 1
        if abs(t.freq - (eu.energy - el.energy)) > 3 * max(t.unc, 1e-6):
            misses += 1
    assert checked > 0.9 * len(kept)
    assert misses == 0, f"{iso}: {misses}/{checked} lines fail closure"


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
