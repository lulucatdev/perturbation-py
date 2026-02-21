"""Simulation and impulse-response tools for perturbation solutions.

This module provides routines for forward simulation of DSGE models using
first-order (linear) solutions, higher-order ``Policy`` objects, and
pruned second-order methods.  It also implements the Generalised Impulse
Response Function (GIRF) for nonlinear perturbation solutions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import Policy
from .pruning import (
    PrunedSimulationResult,
    impulse_response_pruned,
    simulate_pruned,
)
from .solver import FirstOrderSolution

Array = np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    """Result of a forward simulation.

    Attributes
    ----------
    states : ndarray, shape (horizon + 1, n_s)
        Simulated state path (deviations from steady state).  Index 0
        contains the initial condition.
    controls : ndarray, shape (horizon, n_x)
        Simulated controls.  For :func:`simulate_linear` these are
        deviations; for :func:`simulate_with_policy` they include the
        steady-state level.
    shocks : ndarray, shape (horizon, n_e)
        Shock sequence used in the simulation.
    """
    states: Array
    controls: Array
    shocks: Array


def simulate_linear(
    solution: FirstOrderSolution, initial_state: Array, shocks: Array
) -> SimulationResult:
    """Forward-simulate a first-order (linear) perturbation solution.

    Iterates the state-space system::

        x_t     = P  s_t
        s_{t+1} = T  s_t  +  R  e_t

    where ``P`` is the policy matrix, ``T`` the transition matrix, and
    ``R`` the shock-impact matrix from the :class:`FirstOrderSolution`.

    Parameters
    ----------
    solution : FirstOrderSolution
        First-order solution containing ``transition``, ``policy``, and
        ``shock_impact`` matrices.
    initial_state : ndarray, shape (n_s,)
        Initial state deviation from steady state.
    shocks : ndarray, shape (T, n_e)
        Exogenous shock realisations for each period.

    Returns
    -------
    SimulationResult
        States of shape ``(T+1, n_s)``, controls of shape ``(T, n_x)``,
        and the input shocks.

    Raises
    ------
    ValueError
        If array dimensions are inconsistent.
    """
    initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
    shocks = np.asarray(shocks, dtype=float)
    if shocks.ndim != 2:
        raise ValueError("shocks must have shape (T, n_shocks)")

    horizon = shocks.shape[0]
    n_states = solution.transition.shape[0]
    n_controls = solution.policy.shape[0]

    if initial_state.size != n_states:
        raise ValueError("initial_state size does not match transition matrix")
    if shocks.shape[1] != solution.shock_impact.shape[1]:
        raise ValueError("shock dimension does not match solution.shock_impact")

    states = np.zeros((horizon + 1, n_states), dtype=float)
    controls = np.zeros((horizon, n_controls), dtype=float)
    states[0] = initial_state

    for t in range(horizon):
        controls[t] = solution.policy @ states[t]
        states[t + 1] = (
            solution.transition @ states[t] + solution.shock_impact @ shocks[t]
        )

    return SimulationResult(states=states, controls=controls, shocks=shocks)


def impulse_response(
    solution: FirstOrderSolution,
    *,
    horizon: int,
    shock_index: int,
    shock_size: float = 1.0,
) -> SimulationResult:
    """Compute a linear impulse response function from a first-order solution.

    A unit (or scaled) shock is applied at ``t = 0`` to the exogenous
    variable at *shock_index*; all other shocks are zero.  The system
    is simulated from the steady state.

    Parameters
    ----------
    solution : FirstOrderSolution
        First-order solution.
    horizon : int
        Number of periods to simulate.
    shock_index : int
        Index of the shocked exogenous variable (0-based).
    shock_size : float, optional
        Magnitude of the initial shock (default 1.0).

    Returns
    -------
    SimulationResult
        Impulse-response paths for states and controls.

    Raises
    ------
    ValueError
        If *horizon* is non-positive or *shock_index* is out of bounds.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if shock_index < 0 or shock_index >= solution.shock_impact.shape[1]:
        raise ValueError("shock_index out of bounds")

    shocks = np.zeros((horizon, solution.shock_impact.shape[1]), dtype=float)
    shocks[0, shock_index] = shock_size
    initial_state = np.zeros(solution.transition.shape[0], dtype=float)
    return simulate_linear(solution, initial_state=initial_state, shocks=shocks)


def simulate_with_policy(
    policy: Policy,
    *,
    initial_state: Array,
    shocks: Array,
    include_higher_order: bool = True,
) -> SimulationResult:
    """Forward-simulate using a :class:`Policy` object (any perturbation order).

    States are propagated with the first-order transition::

        s_{t+1} = h_x  s_t  +  h_u  e_t

    while controls are evaluated through the full ``policy.controls()``
    method, which includes second- and third-order corrections when the
    policy order is >= 2 and *include_higher_order* is ``True``.

    Parameters
    ----------
    policy : Policy
        Perturbation policy (must include ``transition`` and
        ``shock_impact``).
    initial_state : ndarray, shape (n_s,)
        Initial state deviation from steady state.
    shocks : ndarray, shape (T, n_e)
        Exogenous shock realisations.
    include_higher_order : bool, optional
        Whether to include second- and third-order terms in the control
        evaluation (default ``True``).

    Returns
    -------
    SimulationResult
        States (deviations), controls (levels), and shocks.

    Raises
    ------
    ValueError
        If the policy lacks transition matrices or dimensions are
        inconsistent.
    """
    initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
    shocks = np.asarray(shocks, dtype=float)
    if shocks.ndim != 2:
        raise ValueError("shocks must have shape (T, n_shocks)")

    if policy.transition is None or policy.shock_impact is None:
        raise ValueError(
            "Policy must include transition and shock_impact for simulation"
        )

    horizon = shocks.shape[0]
    n_s = policy.n_states
    n_x = policy.n_controls
    n_e = policy.n_shocks

    if initial_state.size != n_s:
        raise ValueError(f"Expected initial_state size {n_s}, got {initial_state.size}")
    if shocks.shape[1] != n_e:
        raise ValueError(f"Expected shock size {n_e}, got {shocks.shape[1]}")

    states = np.zeros((horizon + 1, n_s), dtype=float)
    controls = np.zeros((horizon, n_x), dtype=float)
    states[0] = initial_state

    for t in range(horizon):
        controls[t] = policy.controls(
            state=states[t],
            shock=shocks[t],
            include_higher_order=include_higher_order,
        )
        states[t + 1] = policy.transition @ states[t] + policy.shock_impact @ shocks[t]

    return SimulationResult(states=states, controls=controls, shocks=shocks)


def impulse_response_with_policy(
    policy: Policy,
    *,
    horizon: int,
    shock_index: int,
    shock_size: float = 1.0,
    include_higher_order: bool = True,
) -> SimulationResult:
    """Compute an impulse response using a :class:`Policy` object.

    Convenience wrapper around :func:`simulate_with_policy` that
    constructs a shock matrix with a single impulse at ``t = 0``.

    Parameters
    ----------
    policy : Policy
        Perturbation policy.
    horizon : int
        Number of periods to simulate.
    shock_index : int
        Index of the shocked exogenous variable (0-based).
    shock_size : float, optional
        Magnitude of the initial shock (default 1.0).
    include_higher_order : bool, optional
        Include higher-order terms in the control evaluation
        (default ``True``).

    Returns
    -------
    SimulationResult
        Impulse-response paths.

    Raises
    ------
    ValueError
        If *horizon* is non-positive or *shock_index* is out of bounds.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if shock_index < 0 or shock_index >= policy.n_shocks:
        raise ValueError("shock_index out of bounds")

    shocks = np.zeros((horizon, policy.n_shocks), dtype=float)
    shocks[0, shock_index] = shock_size
    return simulate_with_policy(
        policy,
        initial_state=np.zeros(policy.n_states, dtype=float),
        shocks=shocks,
        include_higher_order=include_higher_order,
    )


@dataclass(frozen=True)
class GIRFResult:
    """Generalized impulse response function result.

    Attributes
    ----------
    states : array (horizon+1, n_s) — GIRF for states (shocked - baseline)
    controls : array (horizon, n_x) — GIRF for controls (shocked - baseline)
    baseline_states : array — baseline (no-shock) path
    baseline_controls : array — baseline (no-shock) path
    shocked_states : array — shocked path
    shocked_controls : array — shocked path
    """
    states: Array
    controls: Array
    baseline_states: Array
    baseline_controls: Array
    shocked_states: Array
    shocked_controls: Array


def generalized_irf(
    policy: Policy,
    *,
    horizon: int,
    shock_index: int,
    shock_size: float = 1.0,
    pruning_method: str = "kkss",
    initial_state: Array | None = None,
) -> GIRFResult:
    """Compute the Generalised Impulse Response Function (GIRF).

    Unlike the standard linear IRF, the GIRF accounts for nonlinearities
    by comparing two pruned simulations:

    1. **Baseline** -- simulate with ``e_t = 0`` for all ``t`` (the
       deterministic path from the initial state).
    2. **Shocked** -- simulate with ``e_0 = shock_size`` at
       *shock_index* and ``e_t = 0`` for ``t > 0``.
    3. **GIRF** = shocked path - baseline path.

    At first order (linear) the baseline is constant at the steady state
    and the GIRF collapses to the standard linear IRF.  At second order
    and above, the GIRF is *asymmetric*: a positive shock generally
    produces a different (in absolute value) response than a negative
    shock of the same magnitude.

    Parameters
    ----------
    policy : Policy
        Perturbation policy (order >= 1).
    horizon : int
        Number of periods to simulate after the initial shock.
    shock_index : int
        Index of the shocked exogenous variable (0-based).
    shock_size : float, optional
        Magnitude of the period-0 shock (default 1.0).
    pruning_method : str, optional
        Pruning strategy for the underlying simulations (``"kkss"`` or
        ``"naive"``; default ``"kkss"``).
    initial_state : ndarray, shape (n_s,), optional
        Initial state deviation.  Defaults to zero (steady state).

    Returns
    -------
    GIRFResult
        Contains the GIRF (shocked minus baseline) for states and
        controls, as well as the raw baseline and shocked paths.

    Raises
    ------
    ValueError
        If *horizon* is non-positive or *shock_index* is out of bounds.

    References
    ----------
    Koop, Pesaran, and Potter (1996), Journal of Econometrics 74(1),
    119-147 (original GIRF concept).
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    n_e = policy.n_shocks
    if shock_index < 0 or shock_index >= n_e:
        raise ValueError("shock_index out of bounds")

    n_s = policy.n_states

    if initial_state is None:
        s0 = np.zeros(n_s, dtype=float)
    else:
        s0 = np.asarray(initial_state, dtype=float).reshape(-1)

    # Baseline: no shocks
    shocks_baseline = np.zeros((horizon, n_e), dtype=float)
    baseline = simulate_pruned(
        policy,
        horizon=horizon,
        initial_state=s0,
        shocks=shocks_baseline,
        method=pruning_method,
    )

    # Shocked: impulse at t=0
    shocks_shocked = np.zeros((horizon, n_e), dtype=float)
    shocks_shocked[0, shock_index] = shock_size
    shocked = simulate_pruned(
        policy,
        horizon=horizon,
        initial_state=s0,
        shocks=shocks_shocked,
        method=pruning_method,
    )

    return GIRFResult(
        states=shocked.states - baseline.states,
        controls=shocked.controls - baseline.controls,
        baseline_states=baseline.states,
        baseline_controls=baseline.controls,
        shocked_states=shocked.states,
        shocked_controls=shocked.controls,
    )


__all__ = [
    "SimulationResult",
    "PrunedSimulationResult",
    "GIRFResult",
    "simulate_linear",
    "impulse_response",
    "simulate_with_policy",
    "impulse_response_with_policy",
    "simulate_pruned",
    "impulse_response_pruned",
    "generalized_irf",
]
