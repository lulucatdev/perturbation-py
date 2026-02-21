"""Tests for pruned simulation (naive and KKSS methods).

Higher-order perturbation solutions can generate explosive sample paths when
simulated directly, because the polynomial policy function is only a local
approximation and can diverge far from the steady state.  *Pruning* addresses
this by decomposing the state into first-order and higher-order components and
feeding only the stable first-order component back into the law of motion.

Two pruning methods are tested here:

- **naive**: a straightforward clipped simulation that evaluates the full
  polynomial but can still drift for very long horizons.
- **KKSS** (Kim, Kim, Schaumburg & Sims, 2008): a theoretically grounded
  decomposition that separates the state into first- and second-order parts,
  guaranteeing ergodicity and the correct unconditional mean.

Key properties verified:

1. Both methods produce finite simulation paths.
2. At order 1, KKSS and naive coincide because the second-order correction is
   identically zero.
3. Over long samples the KKSS ergodic mean converges to SS + 0.5 * ghs2.
"""

import numpy as np
from numpy.testing import assert_allclose

from helpers import make_scalar_model
from perturbation_py.policy import Policy
from perturbation_py.pruning import simulate_pruned
from perturbation_py.solver import solve_first_order
from perturbation_py.solver_second_order import solve_second_order


def test_pruned_simulation_stays_finite_for_second_order_policy():
    """Verify that the naive pruned simulation produces finite states and controls.

    An unpruned second-order simulation can explode; the pruned version should
    keep all values finite over a moderate horizon.
    """
    model = make_scalar_model()
    so = solve_second_order(model)
    policy = Policy.from_second_order(so)

    result = simulate_pruned(policy, horizon=40, shock_std=0.01, seed=42, method="naive")
    assert np.isfinite(result.states).all()
    assert np.isfinite(result.controls).all()


def test_kkss_pruned_simulation_stays_finite():
    """Verify that the KKSS pruned simulation produces finite states and controls.

    The KKSS decomposition is designed to be ergodic by construction; this test
    checks the minimal requirement that no NaN or Inf values appear.
    """
    model = make_scalar_model()
    so = solve_second_order(model)
    policy = Policy.from_second_order(so)

    result = simulate_pruned(policy, horizon=40, shock_std=0.01, seed=42, method="kkss")
    assert np.isfinite(result.states).all()
    assert np.isfinite(result.controls).all()


def test_kkss_equals_naive_for_first_order():
    """At order 1, KKSS should reduce to the same as naive (linear) simulation."""
    model = make_scalar_model()
    fo = solve_first_order(model)
    policy = Policy.from_first_order(fo)

    shocks = np.array([[0.01], [-0.005], [0.003], [0.0], [-0.002]])
    result_n = simulate_pruned(policy, horizon=5, shocks=shocks, method="naive")
    result_k = simulate_pruned(policy, horizon=5, shocks=shocks, method="kkss")

    assert_allclose(result_k.states, result_n.states, atol=1e-12)
    assert_allclose(result_k.controls, result_n.controls, atol=1e-12)


def test_kkss_ergodic_mean_converges_to_ghs2_correction():
    """The ergodic mean of the KKSS simulation converges to SS + 0.5 * ghs2.

    For the linear test model ghs2 = 0, so the long-run average should stay
    near the deterministic steady state.  A 10,000-period simulation (with a
    1,000-period burn-in) is used to keep sampling noise within the 0.01
    tolerance.
    """
    model = make_scalar_model()  # linear model: ghs2=0, mean should stay at SS
    so = solve_second_order(model)
    policy = Policy.from_second_order(so)

    result = simulate_pruned(policy, horizon=10000, shock_std=0.01, seed=42, method="kkss")
    # For a linear model, controls should have mean ≈ SS = 0
    mean_controls = result.controls[1000:].mean(axis=0)  # skip burn-in
    assert_allclose(mean_controls, policy.steady_state_controls, atol=0.01)
