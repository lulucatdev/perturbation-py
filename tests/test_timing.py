"""Tests for the lead-lag timing / incidence analysis.

The ``build_lead_lag_incidence`` function parses symbolic equation strings to
determine which endogenous variables appear with leads, lags, or at the
current period.  This information drives the partitioning of variables into
states (those that appear with a lag) and forward-looking (those with a lead).

The tests verify correct detection of timing structure for a simple two-
equation system and proper error handling for unknown symbols.
"""

import pytest


def test_build_lead_lag_incidence_for_mixed_system():
    """A two-equation system should correctly identify leads, lags, and current positions.

    Equation 1 contains x at the current period and with a lead, plus k with a
    lag.  Equation 2 contains k with a lag and a current-period shock.  The
    incidence table should reflect that k has a lag and x has a lead, but not
    vice versa.
    """
    from perturbation_py.timing import build_lead_lag_incidence

    equations = (
        "x = beta * x(+1) + k(-1)",
        "k = rho * k(-1) + eps",
    )
    incidence = build_lead_lag_incidence(equations, endogenous=("x", "k"))

    assert incidence.max_lag == 1
    assert incidence.max_lead == 1
    assert incidence.current_positions["x"] == 0
    assert incidence.current_positions["k"] == 1
    assert incidence.has_lag("k")
    assert incidence.has_lead("x")
    assert not incidence.has_lead("k")


def test_reject_unknown_symbol_in_equations():
    """An equation referencing a variable not in the endogenous list should raise ValueError."""
    from perturbation_py.timing import build_lead_lag_incidence

    equations = ("x = y(-1)",)
    with pytest.raises(ValueError, match="Unknown endogenous symbol"):
        build_lead_lag_incidence(equations, endogenous=("x",))
