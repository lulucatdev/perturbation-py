"""Tensor operations for higher-order perturbation methods.

Ported from dolo's ``numeric/tensor.py`` and ``numeric/matrix_equations.py``.
"""

from __future__ import annotations

import numpy as np
from numpy import einsum

Array = np.ndarray


def sdot(U: Array, V: Array) -> Array:
    """Tensor product contracting the last axis of *U* with the first axis of *V*.

    Computes the generalised inner product:

        ``result[i1,...,iN, j2,...,jM] = sum_k U[i1,...,iN, k] * V[k, j2,...,jM]``

    For 2-D arrays this is equivalent to the matrix product ``U @ V``.

    Parameters
    ----------
    U : ndarray
        Left operand of arbitrary dimension (>= 1).
    V : ndarray
        Right operand of arbitrary dimension (>= 1).  ``V.shape[0]`` must
        equal ``U.shape[-1]``.

    Returns
    -------
    ndarray
        Array of shape ``U.shape[:-1] + V.shape[1:]``.

    References
    ----------
    Ported from dolo ``numeric/tensor.py``.
    """
    return np.tensordot(U, V, axes=(U.ndim - 1, 0))


def mdot(M: Array, *C: Array) -> Array:
    """Multi-index tensor contraction via dynamic ``einsum``.

    Given a tensor ``M`` of shape ``(a, b, c, ...)`` and matrices
    ``C1, C2, ...``, the function contracts the *last* ``len(C)`` indices
    of ``M`` with the *first* index of each ``Ci``, producing a result
    whose trailing indices come from the remaining dimensions of the ``Ci``.

    In Einstein notation (example with two matrices)::

        result[a, i, j] = sum_{b,c} M[a, b, c] * C1[b, i] * C2[c, j]

    Parameters
    ----------
    M : ndarray
        Core tensor whose trailing axes are contracted.
    *C : ndarray
        One or more 2-D (or higher) arrays.  ``len(C)`` must not exceed
        ``M.ndim``, and the first axis of each ``Ci`` must match the
        corresponding trailing axis of ``M``.

    Returns
    -------
    ndarray
        Contracted tensor whose shape is the concatenation of the
        uncontracted leading dimensions of ``M`` with the trailing
        dimensions of each ``Ci``.

    References
    ----------
    Ported from dolo ``numeric/tensor.py``.
    """
    sig = _mdot_signature(M.shape, *(c.shape for c in C))
    return einsum(sig, M, *C)


def solve_generalized_sylvester(A: Array, B: Array, C: Array, D: Array) -> Array:
    r"""Solve the generalised Sylvester equation for higher-order perturbation.

    Finds ``X`` satisfying::

        A @ X + B @ X @ kron(C, ..., C) + D = 0

    where the Kronecker power ``kron(C, ..., C)`` contains ``D.ndim - 1``
    copies of ``C``.

    The equation is vectorised as::

        (A \otimes I  +  B \otimes C^{\top \otimes p}) \, \text{vec}(X) = -\text{vec}(D)

    where ``p = D.ndim - 1`` is the Kronecker order, and solved via a
    direct linear solve.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Left coefficient matrix (typically the identity minus a Schur factor).
    B : ndarray, shape (n, n)
        Right coefficient matrix (typically a Schur factor).
    C : ndarray, shape (m, m)
        Base matrix for the Kronecker product (typically a first-order
        transition or policy matrix).
    D : ndarray, shape (n, m, ..., m)
        Right-hand side tensor.  Must have ``ndim >= 2``.  The number of
        trailing axes determines the Kronecker power.

    Returns
    -------
    ndarray
        Solution tensor ``X`` with the same shape as ``D``.

    Notes
    -----
    This solver appears in the second- and third-order perturbation steps
    of Schmitt-Grohe and Uribe (2004) to solve for ``g_xx``, ``g_xxx``,
    etc.

    References
    ----------
    Ported from dolo ``numeric/matrix_equations.py``.
    Schmitt-Grohe and Uribe (2004), JEDC 28, 755-775.
    """
    n_d = D.ndim - 1
    n_v = C.shape[1]
    n_c = D.size // n_v**n_d

    DD = D.reshape(n_c, n_v**n_d)

    # Build kron(C, ..., C)  (n_d copies)
    CC = C
    for _ in range(n_d - 1):
        CC = np.kron(CC, C)

    # Vectorised Sylvester:  (A ⊗ I + B ⊗ CC^T) vec(X) = -vec(D)
    I = np.eye(CC.shape[0])
    L = np.kron(A, I) + np.kron(B, CC.T)
    XX = np.linalg.solve(L, -DD.ravel()).reshape(n_c, n_v**n_d)

    return XX.reshape((n_c,) + (n_v,) * n_d)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mdot_signature(M_shape: tuple[int, ...], *C_shapes: tuple[int, ...]) -> str:
    """Build an einsum signature for :func:`mdot`."""
    M_syms = [chr(97 + e) for e in range(len(M_shape))]
    fC_syms = M_syms[-len(C_shapes):]
    ic = 97 + len(M_syms)
    C_syms: list[list[str]] = []
    for i in range(len(C_shapes)):
        c_sym = [fC_syms[i]]
        for _ in range(len(C_shapes[i]) - 1):
            c_sym.append(chr(ic))
            ic += 1
        C_syms.append(c_sym)
    C_sig = [M_syms] + C_syms
    out_sig = [M_syms[: -len(C_shapes)]] + [cc[1:] for cc in C_syms]
    args = ",".join("".join(g) for g in C_sig)
    out = "".join("".join(g) for g in out_sig)
    return args + "->" + out
