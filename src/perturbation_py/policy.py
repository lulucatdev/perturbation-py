"""Perturbation policy function evaluation.

Provides a unified :class:`Policy` object that encapsulates the Taylor
expansion coefficients of the policy function up to third order and
evaluates the approximation at arbitrary (state, shock) deviations from
the steady state.

The policy function is a Taylor expansion of the control vector:

.. math::

    x(s, \\epsilon) = \\bar{x}
        + g_x \\hat{s} + g_u \\epsilon
        + \\tfrac{1}{2}\\bigl(g_{xx}(\\hat{s}\\otimes\\hat{s})
          + 2 g_{xu}(\\hat{s}\\otimes\\epsilon)
          + g_{uu}(\\epsilon\\otimes\\epsilon)
          + g_{\\sigma\\sigma}\\bigr)
        + \\tfrac{1}{6}\\bigl(g_{xxx}(\\hat{s}^{\\otimes 3})
          + 3 g_{xxu}(\\hat{s}^{\\otimes 2}\\otimes\\epsilon) + \\cdots\\bigr)

where :math:`\\hat{s} = s - \\bar{s}`.  The factorial scaling
(1/2 for second order, 1/6 for third order) is applied automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

Array = np.ndarray


def _as_1d(array: Array | list[float] | tuple[float, ...], size: int) -> Array:
    out = np.asarray(array, dtype=float).reshape(-1)
    if out.size != size:
        raise ValueError(f"Expected vector of size {size}, got {out.size}")
    return out


def _quadratic_term(tensor: Array, a: Array, b: Array) -> Array:
    if tensor is None or tensor.size == 0:
        return np.zeros(0, dtype=float)
    return np.einsum("ijk,j,k->i", tensor, a, b)


def _cubic_term(tensor: Array, a: Array, b: Array, c: Array) -> Array:
    if tensor is None or tensor.size == 0:
        return np.zeros(0, dtype=float)
    return np.einsum("ijkl,j,k,l->i", tensor, a, b, c)


@dataclass(frozen=True)
class Policy:
    """Perturbation policy function for evaluating controls at any point.

    Encapsulates the Taylor expansion coefficients of the policy
    function up to the specified order and provides a unified
    :meth:`controls` method that evaluates the approximation at
    arbitrary (state, shock) deviations from the steady state.

    The expansion is:

    * **Order 1:** :math:`x = \\bar{x} + g_x \\hat{s} + g_u \\epsilon`
    * **Order 2:** adds
      :math:`\\tfrac{1}{2}(g_{xx} \\hat{s}^2 + 2 g_{xu} \\hat{s} \\epsilon
      + g_{uu} \\epsilon^2 + g_{\\sigma\\sigma})`
    * **Order 3:** adds
      :math:`\\tfrac{1}{6} g_{xxx} \\hat{s}^3 + \\tfrac{1}{2} g_{xxu}
      \\hat{s}^2 \\epsilon + \\tfrac{1}{2} g_{xuu} \\hat{s} \\epsilon^2
      + \\tfrac{1}{6} g_{uuu} \\epsilon^3`

    where :math:`\\hat{s} = s - \\bar{s}` and Kronecker-product notation
    is implied.

    Attributes
    ----------
    order : int
        Approximation order (1, 2, or 3).
    steady_state_controls : Array, shape (n_x,)
        Steady-state control values :math:`\\bar{x}`.
    ghx : Array, shape (n_x, n_s)
        First-order state coefficient.
    ghu : Array, shape (n_x, n_e)
        First-order shock coefficient.
    ghxx, ghxu, ghuu : Array or None
        Second-order coefficients (populated when ``order >= 2``).
    ghs2 : Array or None, shape (n_x,)
        Uncertainty correction (populated when ``order >= 2``).
    ghxxx, ghxxu, ghxuu, ghuuu : Array or None
        Third-order coefficients (populated when ``order == 3``).
    transition : Array or None, shape (n_s, n_s)
        First-order state transition matrix.
    shock_impact : Array or None, shape (n_s, n_e)
        First-order shock impact on states.
    """

    order: int
    steady_state_controls: Array
    ghx: Array
    ghu: Array
    ghxx: Array | None = None
    ghxu: Array | None = None
    ghuu: Array | None = None
    ghs2: Array | None = None
    ghxxx: Array | None = None
    ghxxu: Array | None = None
    ghxuu: Array | None = None
    ghuuu: Array | None = None
    transition: Array | None = None
    shock_impact: Array | None = None

    @property
    def n_controls(self) -> int:
        """Number of control variables."""
        return int(self.ghx.shape[0])

    @property
    def n_states(self) -> int:
        """Number of state variables."""
        return int(self.ghx.shape[1])

    @property
    def n_shocks(self) -> int:
        """Number of exogenous shocks."""
        return int(self.ghu.shape[1])

    @classmethod
    def from_first_order(
        cls,
        solution: "FirstOrderSolution",
        *,
        steady_state_controls: Array | None = None,
    ) -> "Policy":
        """Construct a first-order Policy from a FirstOrderSolution.

        Parameters
        ----------
        solution : FirstOrderSolution
            The solved first-order perturbation result.
        steady_state_controls : Array or None
            Steady-state control values; defaults to a zero vector.

        Returns
        -------
        Policy
            A Policy object with ``order=1``.
        """
        if steady_state_controls is None:
            steady_state_controls = np.zeros(solution.policy.shape[0], dtype=float)
        return cls(
            order=1,
            steady_state_controls=np.asarray(steady_state_controls, dtype=float),
            ghx=np.asarray(solution.policy, dtype=float),
            ghu=np.asarray(solution.control_shock_impact, dtype=float),
            transition=np.asarray(solution.transition, dtype=float),
            shock_impact=np.asarray(solution.shock_impact, dtype=float),
        )

    @classmethod
    def from_second_order(
        cls,
        solution: "SecondOrderSolution",
        *,
        steady_state_controls: Array | None = None,
    ) -> "Policy":
        """Construct a second-order Policy from a SecondOrderSolution.

        Parameters
        ----------
        solution : SecondOrderSolution
            The solved second-order perturbation result.
        steady_state_controls : Array or None
            Steady-state control values; defaults to a zero vector.

        Returns
        -------
        Policy
            A Policy object with ``order=2``.
        """
        if steady_state_controls is None:
            steady_state_controls = np.zeros(solution.ghx.shape[0], dtype=float)
        return cls(
            order=2,
            steady_state_controls=np.asarray(steady_state_controls, dtype=float),
            ghx=np.asarray(solution.ghx, dtype=float),
            ghu=np.asarray(solution.ghu, dtype=float),
            ghxx=np.asarray(solution.ghxx, dtype=float),
            ghxu=np.asarray(solution.ghxu, dtype=float),
            ghuu=np.asarray(solution.ghuu, dtype=float),
            ghs2=np.asarray(solution.ghs2, dtype=float),
            transition=np.asarray(solution.first_order.transition, dtype=float),
            shock_impact=np.asarray(solution.first_order.shock_impact, dtype=float),
        )

    @classmethod
    def from_third_order(
        cls,
        solution: "ThirdOrderSolution",
        *,
        steady_state_controls: Array | None = None,
    ) -> "Policy":
        """Construct a third-order Policy from a ThirdOrderSolution.

        Parameters
        ----------
        solution : ThirdOrderSolution
            The solved third-order perturbation result.
        steady_state_controls : Array or None
            Steady-state control values; defaults to a zero vector.

        Returns
        -------
        Policy
            A Policy object with ``order=3``.
        """
        if steady_state_controls is None:
            steady_state_controls = np.zeros(solution.ghx.shape[0], dtype=float)
        return cls(
            order=3,
            steady_state_controls=np.asarray(steady_state_controls, dtype=float),
            ghx=np.asarray(solution.ghx, dtype=float),
            ghu=np.asarray(solution.ghu, dtype=float),
            ghxx=np.asarray(solution.ghxx, dtype=float),
            ghxu=np.asarray(solution.ghxu, dtype=float),
            ghuu=np.asarray(solution.ghuu, dtype=float),
            ghs2=np.asarray(solution.ghs2, dtype=float),
            ghxxx=np.asarray(solution.ghxxx, dtype=float),
            ghxxu=np.asarray(solution.ghxxu, dtype=float),
            ghxuu=np.asarray(solution.ghxuu, dtype=float),
            ghuuu=np.asarray(solution.ghuuu, dtype=float),
            transition=np.asarray(
                solution.second_order.first_order.transition, dtype=float
            ),
            shock_impact=np.asarray(
                solution.second_order.first_order.shock_impact, dtype=float
            ),
        )

    def controls(
        self,
        *,
        state: Array | list[float] | tuple[float, ...],
        shock: Array | list[float] | tuple[float, ...] | None = None,
        include_higher_order: bool = True,
    ) -> Array:
        """Evaluate the policy function at given state and shock deviations.

        Computes the control vector using the Taylor expansion up to
        the stored order.  The factorial scaling (1/2 for quadratic
        terms, 1/6 for cubic terms) is applied automatically.

        Parameters
        ----------
        state : array_like
            State deviation from steady state :math:`\\hat{s} = s - \\bar{s}`,
            shape ``(n_states,)``.
        shock : array_like or None
            Shock realisation :math:`\\epsilon`, shape ``(n_shocks,)``.
            Defaults to zero.
        include_higher_order : bool
            If False, only the first-order terms are included regardless
            of the stored order.

        Returns
        -------
        Array, shape (n_controls,)
            The evaluated control vector.
        """
        s = _as_1d(state, self.n_states)
        if shock is None:
            e = np.zeros(self.n_shocks, dtype=float)
        else:
            e = _as_1d(shock, self.n_shocks)

        out = np.asarray(self.steady_state_controls, dtype=float).copy()
        out = out + self.ghx @ s + self.ghu @ e

        if not include_higher_order or self.order < 2:
            return out

        if self.ghxx is not None and self.ghxx.size:
            out = out + 0.5 * _quadratic_term(self.ghxx, s, s)
        if self.ghxu is not None and self.ghxu.size:
            out = out + _quadratic_term(self.ghxu, s, e)
        if self.ghuu is not None and self.ghuu.size:
            out = out + 0.5 * _quadratic_term(self.ghuu, e, e)
        if self.ghs2 is not None and self.ghs2.size:
            out = out + 0.5 * np.asarray(self.ghs2, dtype=float).reshape(-1)

        if self.order < 3:
            return out

        if self.ghxxx is not None and self.ghxxx.size:
            out = out + (1.0 / 6.0) * _cubic_term(self.ghxxx, s, s, s)
        if self.ghxxu is not None and self.ghxxu.size:
            out = out + 0.5 * _cubic_term(self.ghxxu, s, s, e)
        if self.ghxuu is not None and self.ghxuu.size:
            out = out + 0.5 * _cubic_term(self.ghxuu, s, e, e)
        if self.ghuuu is not None and self.ghuuu.size:
            out = out + (1.0 / 6.0) * _cubic_term(self.ghuuu, e, e, e)

        return out


if TYPE_CHECKING:
    from .solver import FirstOrderSolution
    from .solver_second_order import SecondOrderSolution
    from .solver_third_order import ThirdOrderSolution
