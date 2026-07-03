import math

import pytest

from marvel_gnn.core.parse import Transition
from marvel_gnn.core.solver import flag_bad_lines, solve_energies
from marvel_gnn.core.uncertainty import (
    bootstrap_unc,
    combined_unc,
    marvel_solve,
    shortest_path_unc,
)


def tr(upper, lower, freq, unc=1e-3):
    return Transition(freq=freq, unc=unc, orig_unc=unc, upper=upper, lower=lower, ref="00Test.1")


CHAIN = [tr("b", "a", 10.0, unc=1e-3), tr("c", "b", 5.0, unc=2e-3)]


def test_shortest_path_unc():
    energies = solve_energies(CHAIN)
    sp = shortest_path_unc(CHAIN, energies)
    assert sp["a"] == 0.0  # root = lowest level
    assert sp["b"] == pytest.approx(1e-3)
    assert sp["c"] == pytest.approx(math.sqrt(1e-6 + 4e-6))


def test_shortest_path_prefers_precise_parallel_edge():
    ts = [tr("b", "a", 10.0, unc=5e-3), tr("b", "a", 10.0, unc=1e-3)]
    sp = shortest_path_unc(ts, solve_energies(ts))
    assert sp["b"] == pytest.approx(1e-3)


def test_bootstrap_exact_network_has_zero_unc():
    # tree network: every resample still reproduces the same exact energies
    energies = solve_energies(CHAIN)
    boot = bootstrap_unc(CHAIN, energies, iterations=20, rng=0)
    assert all(u == pytest.approx(0.0, abs=1e-12) for u in boot.values())


def test_bootstrap_conflicting_lines_have_positive_unc():
    ts = [tr("b", "a", 10.0), tr("b", "a", 10.4)]
    energies = solve_energies(ts)
    boot = bootstrap_unc(ts, energies, iterations=50, rng=0)
    assert boot["b"] > 0.01  # resampled weights swing E(b) between ~10.0 and ~10.4


def test_combined_unc_sums_node_costs_along_path():
    energies = solve_energies(CHAIN)
    sp = {"a": 0.0, "b": 1e-3, "c": 2e-3}
    boot = {"a": 0.0, "b": 3e-3, "c": 1e-3}  # b: boot wins; c: sp wins
    final = combined_unc(CHAIN, energies, sp, boot)
    assert final["a"] == 0.0
    assert final["b"] == pytest.approx(3e-3)
    assert final["c"] == pytest.approx(math.sqrt(9e-6 + 4e-6))


def test_marvel_solve_end_to_end():
    result = marvel_solve(CHAIN, bootstrap_iterations=10, rng=0)
    assert result["a"] == (0.0, 0.0)
    assert result["c"][0] == pytest.approx(15.0)
    assert result["c"][1] >= 0.0


def test_flag_bad_lines():
    # consistent triangle plus one line off by 0.05 (50x its unc of 1e-3)
    ts = [tr("b", "a", 10.0), tr("c", "b", 5.0), tr("c", "a", 15.0), tr("c", "a", 15.05)]
    flagged = flag_bad_lines(ts, solve_energies(ts))
    by_cat = {cat for _, _, cat in flagged}
    assert "VERY BAD" in by_cat  # the injected line, ratio ~ 37-50
    # the consistent lines get dragged slightly, may flag as WRONG, but never VERY BAD_100+
    assert not {"VERY BAD_100", "VERY BAD_1000"} & by_cat


def test_flag_bad_lines_clean_network_is_quiet():
    ts = [tr("b", "a", 10.0), tr("c", "b", 5.0), tr("c", "a", 15.0)]
    assert flag_bad_lines(ts, solve_energies(ts)) == []
