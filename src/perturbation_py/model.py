"""DSGE model specification following the dolo convention.

A rational-expectations DSGE model is specified as a pair of functions:

* **Transition** ``g(s, x, e, p) -> s'`` -- maps current states *s*,
  controls *x*, shocks *e*, and parameters *p* to next-period states.
* **Arbitrage** ``f(s, x, s', x', p) -> residual`` -- the set of
  expectational (Euler-type) equations that must equal zero in
  equilibrium.

This decomposition follows the convention of *dolo*
(EconForge/dolo) and is natural for perturbation methods because the
first-order conditions are already expressed in terms of today's and
tomorrow's endogenous variables.

References
----------
Villemot, S. (2011). "Solving rational expectations models at first
    order: what Dynare does." *Dynare Working Papers*, 2.
Winant, P. (2017). "dolo: a tool for perturbation analysis of
    rational expectations models." *EconForge/dolo*, GitHub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .timing import LeadLagIncidence, build_lead_lag_incidence

Array = np.ndarray
TransitionFunction = Callable[[Array, Array, Array, Mapping[str, float]], Array]
ArbitrageFunction = Callable[[Array, Array, Array, Array, Mapping[str, float]], Array]


def _to_1d(array: Array | Sequence[float], expected_size: int | None = None) -> Array:
    out = np.asarray(array, dtype=float).reshape(-1)
    if expected_size is not None and out.size != expected_size:
        raise ValueError(f"Expected vector of size {expected_size}, got {out.size}")
    return out


@dataclass
class DSGEModel:
    """Specification of a DSGE model for perturbation solution.

    The model is specified as a pair of functions (transition *g*,
    arbitrage *f*) following the convention of *dolo*
    (EconForge/dolo).  The transition maps current states, controls,
    and shocks to next-period states; the arbitrage conditions are the
    Euler-type equations that hold in rational-expectations equilibrium.

    Parameters
    ----------
    state_names : Sequence[str]
        Names of the state (predetermined) variables.
    control_names : Sequence[str]
        Names of the control (forward-looking) variables.
    shock_names : Sequence[str]
        Names of the exogenous shock processes.
    parameters : Mapping[str, float]
        Model parameters passed through to *transition* and *arbitrage*.
    steady_state_states : Array
        Deterministic steady-state values of the state variables.
    steady_state_controls : Array
        Deterministic steady-state values of the control variables.
    transition : TransitionFunction
        Callable ``g(s, x, e, p) -> s'`` returning next-period states.
    arbitrage : ArbitrageFunction
        Callable ``f(s, x, s', x', p) -> residual`` returning the
        Euler-equation residuals (should be zero in equilibrium).
    steady_state_shocks : Array or None
        Steady-state shock values (defaults to zero vector).
    shock_covariance : Array or None
        Variance-covariance matrix of the shocks, shape ``(n_e, n_e)``.
    equations : Sequence[str] or None
        Optional symbolic equation strings for lead-lag analysis.
    lead_lag_incidence : LeadLagIncidence or None
        Pre-computed lead-lag incidence; built automatically from
        *equations* if not supplied.

    Raises
    ------
    ValueError
        If dimension mismatches are detected between names and
        steady-state vectors, or if the transition / arbitrage functions
        return arrays of unexpected size.
    """

    state_names: Sequence[str]
    control_names: Sequence[str]
    shock_names: Sequence[str]
    parameters: Mapping[str, float]
    steady_state_states: Array
    steady_state_controls: Array
    transition: TransitionFunction
    arbitrage: ArbitrageFunction
    steady_state_shocks: Array | None = None
    shock_covariance: Array | None = None
    equations: Sequence[str] | None = None
    lead_lag_incidence: LeadLagIncidence | None = None

    def __post_init__(self) -> None:
        self.steady_state_states = _to_1d(self.steady_state_states)
        self.steady_state_controls = _to_1d(self.steady_state_controls)

        if self.steady_state_shocks is None:
            self.steady_state_shocks = np.zeros(len(self.shock_names), dtype=float)
        else:
            self.steady_state_shocks = _to_1d(self.steady_state_shocks)

        if len(self.state_names) != self.steady_state_states.size:
            raise ValueError("state_names length does not match steady_state_states")
        if len(self.control_names) != self.steady_state_controls.size:
            raise ValueError(
                "control_names length does not match steady_state_controls"
            )
        if len(self.shock_names) != self.steady_state_shocks.size:
            raise ValueError("shock_names length does not match steady_state_shocks")

        if self.lead_lag_incidence is None and self.equations is not None:
            self.lead_lag_incidence = build_lead_lag_incidence(
                self.equations,
                endogenous=self.endogenous_names,
            )

        t = self.transition(
            self.steady_state_states,
            self.steady_state_controls,
            self.steady_state_shocks,
            self.parameters,
        )
        a = self.arbitrage(
            self.steady_state_states,
            self.steady_state_controls,
            self.steady_state_states,
            self.steady_state_controls,
            self.parameters,
        )

        t = _to_1d(t)
        a = _to_1d(a)

        if t.size != self.n_states:
            raise ValueError("transition must return n_states values")
        if a.size != self.n_controls:
            raise ValueError("arbitrage must return n_controls values")

    @property
    def n_states(self) -> int:
        """Number of state (predetermined) variables."""
        return len(self.state_names)

    @property
    def n_controls(self) -> int:
        """Number of control (forward-looking) variables."""
        return len(self.control_names)

    @property
    def n_shocks(self) -> int:
        """Number of exogenous shock processes."""
        return len(self.shock_names)

    @property
    def endogenous_names(self) -> tuple[str, ...]:
        """All endogenous variable names (states followed by controls)."""
        return tuple(self.state_names) + tuple(self.control_names)

    def split_state_control_vector(
        self, vector: Array | Sequence[float]
    ) -> tuple[Array, Array]:
        """Split a stacked endogenous vector into state and control parts.

        Parameters
        ----------
        vector : array_like
            Concatenated vector ``[states, controls]`` of length
            ``n_states + n_controls``.

        Returns
        -------
        states : Array
            The first ``n_states`` elements.
        controls : Array
            The remaining ``n_controls`` elements.
        """
        vec = _to_1d(vector, expected_size=self.n_states + self.n_controls)
        return vec[: self.n_states], vec[self.n_states :]

    def static_residual(
        self,
        states: Array | Sequence[float],
        controls: Array | Sequence[float],
        shocks: Array | Sequence[float] | None = None,
    ) -> Array:
        """Evaluate the static (steady-state) residual.

        Sets ``s' = s`` and ``x' = x`` to compute the fixed-point
        residual ``[g(s, x, e) - s, f(s, x, s, x)]``.  A zero residual
        indicates a valid deterministic steady state.

        Parameters
        ----------
        states : array_like
            State variable values, shape ``(n_states,)``.
        controls : array_like
            Control variable values, shape ``(n_controls,)``.
        shocks : array_like or None
            Shock values; defaults to the steady-state shocks (zeros).

        Returns
        -------
        Array
            Residual vector of length ``n_states + n_controls``.
        """
        s = _to_1d(states, expected_size=self.n_states)
        x = _to_1d(controls, expected_size=self.n_controls)
        if shocks is None:
            e = np.asarray(self.steady_state_shocks, dtype=float)
        else:
            e = _to_1d(shocks, expected_size=self.n_shocks)

        transition_values = _to_1d(
            self.transition(s, x, e, self.parameters), expected_size=self.n_states
        )
        arbitrage_values = _to_1d(
            self.arbitrage(s, x, s, x, self.parameters), expected_size=self.n_controls
        )

        transition_residual = transition_values - s
        return np.concatenate([transition_residual, arbitrage_values])

    def static_residual_from_vector(
        self,
        vector: Array | Sequence[float],
        shocks: Array | Sequence[float] | None = None,
    ) -> Array:
        """Evaluate the static residual from a stacked endogenous vector.

        Convenience wrapper around :meth:`static_residual` that accepts
        a single concatenated ``[states, controls]`` vector.

        Parameters
        ----------
        vector : array_like
            Stacked vector of length ``n_states + n_controls``.
        shocks : array_like or None
            Shock values; defaults to steady-state shocks.

        Returns
        -------
        Array
            Residual vector of length ``n_states + n_controls``.
        """
        s, x = self.split_state_control_vector(vector)
        return self.static_residual(s, x, shocks=shocks)
