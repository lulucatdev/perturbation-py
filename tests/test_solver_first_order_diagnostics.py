"""Tests for first-order solver diagnostics and error reporting.

When the Blanchard-Kahn conditions are not satisfied -- i.e., the number of
unstable eigenvalues does not equal the number of forward-looking variables --
the solver should raise a structured ``BlanchardKahnError`` that reports the
nature of the failure (indeterminacy vs. no stable equilibrium) and the count
of unstable roots.

The fixture model used here has an explosive transition (eigenvalue 1.1 > 1)
with no controls, which violates the rank conditions.
"""

import numpy as np
import pytest

from perturbation_py.model import DSGEModel
from perturbation_py.solver import BlanchardKahnError, solve_first_order


def make_indeterminate_fixture_model() -> DSGEModel:
    """Build a model that violates the Blanchard-Kahn conditions.

    The model has a single state with explosive dynamics (s' = 1.1 * s) and
    no controls or shocks.  Because the transition eigenvalue exceeds unity
    there is no stable saddle-path solution, and the solver should raise
    a ``BlanchardKahnError``.

    Returns
    -------
    DSGEModel
        A model designed to trigger a Blanchard-Kahn failure.
    """
    def transition(
        s: np.ndarray, x: np.ndarray, e: np.ndarray, params: dict[str, float]
    ) -> np.ndarray:
        return np.array([1.1 * s[0]], dtype=float)

    def arbitrage(
        s: np.ndarray,
        x: np.ndarray,
        s_next: np.ndarray,
        x_next: np.ndarray,
        params: dict[str, float],
    ) -> np.ndarray:
        return np.array([], dtype=float)

    return DSGEModel(
        state_names=("k",),
        control_names=(),
        shock_names=(),
        parameters={},
        steady_state_states=np.array([0.0]),
        steady_state_controls=np.array([]),
        steady_state_shocks=np.array([]),
        transition=transition,
        arbitrage=arbitrage,
    )


def test_solver_reports_indeterminacy_with_structured_error():
    """The solver should raise BlanchardKahnError with a meaningful reason and eigenvalue count.

    The error's ``reason`` field should be one of the recognised failure modes
    ('indeterminacy' or 'no_stable_equilibrium'), and ``unstable_count`` should
    be a non-negative integer reflecting the number of eigenvalues outside the
    unit circle.
    """
    model = make_indeterminate_fixture_model()

    with pytest.raises(BlanchardKahnError) as exc:
        solve_first_order(model)

    assert exc.value.reason in {"indeterminacy", "no_stable_equilibrium"}
    assert exc.value.unstable_count >= 0
