"""Tests for tensor operations (sdot, mdot, solve_generalized_sylvester).

These tests verify the core linear-algebra primitives used throughout the
perturbation solver:

- ``sdot``: a shape-aware dot product that contracts the last axis of the
  first operand with the first axis of the second (generalising ``@`` to
  higher-rank tensors).
- ``mdot``: multi-mode contraction ``M x_1 A_1 x_2 A_2 ...`` that applies
  separate matrices along each trailing axis of a tensor.
- ``solve_generalized_sylvester``: solves A X + B X C^{otimes d} + D = 0
  for X, which arises at every order >= 2 of the perturbation expansion.
"""

import numpy as np
from numpy.testing import assert_allclose

from perturbation_py.tensor_ops import mdot, sdot, solve_generalized_sylvester


def test_sdot_matrices_equals_matmul():
    """Verify that sdot on two 2-D arrays reproduces standard matrix multiplication."""
    A = np.random.default_rng(0).standard_normal((3, 4))
    B = np.random.default_rng(1).standard_normal((4, 5))
    assert_allclose(sdot(A, B), A @ B, atol=1e-14)


def test_sdot_tensor_vector():
    """Verify that sdot contracts a rank-3 tensor with a vector along the last axis."""
    T = np.random.default_rng(2).standard_normal((2, 3, 4))
    v = np.random.default_rng(3).standard_normal(4)
    expected = np.tensordot(T, v, axes=(2, 0))
    assert_allclose(sdot(T, v), expected, atol=1e-14)


def test_mdot_matrix_matrix():
    """Verify mdot(M, A, B) equals the einsum contraction M_{ijk} A_{jm} B_{kn}."""
    M = np.random.default_rng(4).standard_normal((2, 3, 4))
    A = np.random.default_rng(5).standard_normal((3, 5))
    B = np.random.default_rng(6).standard_normal((4, 6))
    result = mdot(M, A, B)
    expected = np.einsum("ijk,jm,kn->imn", M, A, B)
    assert_allclose(result, expected, atol=1e-13)


def test_mdot_single_matrix():
    """Verify that mdot with a single matrix reduces to ordinary matrix multiplication."""
    M = np.random.default_rng(7).standard_normal((2, 3))
    A = np.random.default_rng(8).standard_normal((3, 4))
    result = mdot(M, A)
    assert_allclose(result, M @ A, atol=1e-14)


def test_solve_generalized_sylvester_2d_known():
    """A X + B X C + D = 0 with known solution X = I."""
    n = 3
    rng = np.random.default_rng(42)
    X_true = rng.standard_normal((n, n))
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    C = rng.standard_normal((n, n))
    D = -(A @ X_true + B @ X_true @ C)
    X = solve_generalized_sylvester(A, B, C, D)
    assert_allclose(X, X_true, atol=1e-10)


def test_solve_generalized_sylvester_3d():
    """A X + B X [C,C] + D = 0 with n_d=2 (3D tensor)."""
    n = 2
    m = 3
    rng = np.random.default_rng(99)
    X_true = rng.standard_normal((n, m, m))
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    C = rng.standard_normal((m, m))
    CC = np.kron(C, C)
    # D = -(A X + B X [C,C]) where the matrix equation is A @ X_flat + B @ X_flat @ CC
    D_flat = -(A @ X_true.reshape(n, m * m) + B @ X_true.reshape(n, m * m) @ CC)
    D = D_flat.reshape(n, m, m)
    X = solve_generalized_sylvester(A, B, C, D)
    assert_allclose(X, X_true, atol=1e-10)


def test_solve_generalized_sylvester_residual():
    """Check that A X + B X C + D ≈ 0 for random system."""
    n = 4
    rng = np.random.default_rng(123)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    C = rng.standard_normal((n, n))
    D = rng.standard_normal((n, n))
    X = solve_generalized_sylvester(A, B, C, D)
    residual = A @ X + B @ X @ C + D
    assert_allclose(residual, 0.0, atol=1e-10)
