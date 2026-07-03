import pytest

from marvel_gnn.core.network import split_components
from marvel_gnn.core.parse import Transition
from marvel_gnn.core.solver import solve_energies


def tr(upper, lower, freq, unc=1e-3):
    return Transition(freq=freq, unc=unc, orig_unc=unc, upper=upper, lower=lower, ref="00Test.1")


def test_split_components():
    transitions = [
        tr("a", "b", 1.0), tr("b", "c", 1.0),   # component of 3 levels
        tr("x", "y", 1.0),                       # component of 2 levels
    ]
    solvable, dropped = split_components(transitions, minsize=3)
    assert len(solvable) == 1 and len(dropped) == 1
    assert {t.upper for t in solvable[0]} == {"a", "b"}
    assert dropped[0][0].upper == "x"


def test_solve_chain():
    # b is 10 above a, c is 5 above b -> energies 0, 10, 15
    energies = solve_energies([tr("b", "a", 10.0), tr("c", "b", 5.0)])
    assert energies["a"] == pytest.approx(0.0, abs=1e-12)
    assert energies["b"] == pytest.approx(10.0)
    assert energies["c"] == pytest.approx(15.0)


def test_solve_weighted_average():
    # two conflicting measurements of the same gap: precise 10.0 (unc 1e-3),
    # sloppy 10.4 (unc 2e-3) -> weighted mean = (10.0/1 + 10.4/4)/(1 + 1/4)
    energies = solve_energies([tr("b", "a", 10.0, unc=1e-3), tr("b", "a", 10.4, unc=2e-3)])
    expected = (10.0 / 1e-6 + 10.4 / 4e-6) / (1 / 1e-6 + 1 / 4e-6)
    assert energies["b"] == pytest.approx(expected)


def test_solve_cycle_consistency():
    # over-determined triangle, exactly consistent -> exact recovery
    energies = solve_energies([
        tr("b", "a", 10.0), tr("c", "b", 5.0), tr("c", "a", 15.0),
    ])
    assert energies["b"] == pytest.approx(10.0)
    assert energies["c"] == pytest.approx(15.0)


def test_unc_override():
    # overriding uncertainties shifts the weighted average
    ts = [tr("b", "a", 10.0, unc=1e-3), tr("b", "a", 10.4, unc=1e-3)]
    assert solve_energies(ts)["b"] == pytest.approx(10.2)
    assert solve_energies(ts, unc=[1e-3, 2e-3])["b"] == pytest.approx(
        (10.0 / 1e-6 + 10.4 / 4e-6) / (1 / 1e-6 + 1 / 4e-6))
