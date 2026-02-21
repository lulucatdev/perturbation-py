"""Low-level numerical differentiation backends.

Provides two strategies for computing Jacobian matrices of multivariate
vector-valued functions:

* **Finite differences** -- central-difference approximation with
  O(epsilon^2) truncation error.
* **Complex step** -- the Squire-Trapp (1998) method, which achieves
  machine-precision accuracy (no subtractive cancellation) at the cost
  of requiring the target function to support complex arithmetic.

References
----------
Squire, W. and Trapp, G. (1998). "Using Complex Variables to Estimate
    Derivatives of Real Functions." *SIAM Review*, 40(1), 110-112.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray


def finite_difference_jacobian(
    func: Callable[..., Array],
    args: tuple[Array, ...],
    arg_index: int,
    epsilon: float,
) -> Array:
    """Compute the Jacobian via central finite differences.

    Uses the two-point central difference formula:

    .. math::

        \\frac{\\partial f}{\\partial x_i}
        \\approx \\frac{f(x + h e_i) - f(x - h e_i)}{2h}

    where :math:`h = \\epsilon \\cdot \\max(1, |x_i|)`.  The truncation
    error is :math:`O(\\epsilon^2)`.

    Parameters
    ----------
    func : callable
        Multivariate vector-valued function ``(*args) -> Array``.
    args : tuple of Array
        Argument tuple at which to evaluate the Jacobian.
    arg_index : int
        Index into *args* specifying which argument to differentiate
        with respect to.
    epsilon : float
        Base step size.

    Returns
    -------
    Array, shape (n_out, n_in)
        The Jacobian matrix :math:`\\partial f / \\partial x_{\\text{arg\\_index}}`.
    """
    baseline = np.asarray(func(*args), dtype=float).reshape(-1)
    x = np.asarray(args[arg_index], dtype=float).reshape(-1)

    jac = np.zeros((baseline.size, x.size), dtype=float)

    for i in range(x.size):
        step = epsilon * max(1.0, abs(x[i]))
        delta = np.zeros_like(x)
        delta[i] = step

        args_plus = list(args)
        args_minus = list(args)
        args_plus[arg_index] = x + delta
        args_minus[arg_index] = x - delta

        f_plus = np.asarray(func(*tuple(args_plus)), dtype=float).reshape(-1)
        f_minus = np.asarray(func(*tuple(args_minus)), dtype=float).reshape(-1)
        jac[:, i] = (f_plus - f_minus) / (2.0 * step)

    return jac


def complex_step_jacobian(
    func: Callable[..., Array],
    args: tuple[Array, ...],
    arg_index: int,
    epsilon: float,
) -> Array:
    """Compute the Jacobian via the complex-step method.

    Uses the Squire-Trapp (1998) formula:

    .. math::

        \\frac{\\partial f}{\\partial x_i}
        \\approx \\frac{\\mathrm{Im}\\, f(x + i h e_i)}{h}

    This avoids subtractive cancellation entirely, yielding accuracy at
    or near machine epsilon regardless of step size (as long as the
    target function supports complex arithmetic).

    Parameters
    ----------
    func : callable
        Multivariate vector-valued function that accepts complex arrays.
    args : tuple of Array
        Argument tuple at which to evaluate the Jacobian.
    arg_index : int
        Index into *args* specifying which argument to differentiate
        with respect to.
    epsilon : float
        Step size for the imaginary perturbation.

    Returns
    -------
    Array, shape (n_out, n_in)
        The Jacobian matrix.

    References
    ----------
    Squire, W. and Trapp, G. (1998). *SIAM Review*, 40(1), 110-112.
    """
    baseline = np.asarray(func(*args), dtype=complex).reshape(-1)
    x = np.asarray(args[arg_index], dtype=float).reshape(-1)

    jac = np.zeros((baseline.size, x.size), dtype=float)

    args_complex = [np.asarray(a, dtype=complex).reshape(-1) for a in args]
    for i in range(x.size):
        step = epsilon * max(1.0, abs(x[i]))
        perturbed = np.asarray(args_complex[arg_index], dtype=complex)
        perturbed[i] += 1j * step

        args_cs = list(args_complex)
        args_cs[arg_index] = perturbed

        f_cs = np.asarray(func(*tuple(args_cs)), dtype=complex).reshape(-1)
        jac[:, i] = np.imag(f_cs) / step

    return jac
