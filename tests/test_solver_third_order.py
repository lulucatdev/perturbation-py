"""Tests for the third-order perturbation solver.

The third-order solver extends the perturbation expansion with cubic
correction tensors (ghxxx, ghxxu, ghxuu, ghuuu).  These are rank-4 arrays
whose dimensions depend on n_controls, n_states, and n_shocks.

For the linear test model all third-order tensors should be zero, but this
test focuses on verifying that the solver produces tensors of the correct
rank (ndim == 4), which is a basic structural invariant.
"""

from helpers import make_scalar_model
from perturbation_py.solver_third_order import solve_third_order


def test_third_order_solver_emits_consistent_tensor_ranks():
    """All third-order policy tensors should be rank-4 (ndim == 4).

    This checks ghxxx, ghxxu, ghxuu, and ghuuu, each of which should have
    shape (n_controls, *, *, *) where the trailing dimensions correspond to
    combinations of states and shocks.
    """
    model = make_scalar_model()
    sol = solve_third_order(model)

    assert sol.ghxxx.ndim == 4
    assert sol.ghxxu.ndim == 4
    assert sol.ghxuu.ndim == 4
    assert sol.ghuuu.ndim == 4
