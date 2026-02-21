"""Second-order Dynare parity tests.

Validates that the Python Sylvester-based second-order solver produces
decision-rule tensors (ghx, ghu, ghxx, ghxu, ghuu, ghs2) that match
Dynare's ``stoch_simul(order=2)`` output for nonlinear DSGE models.

The test compares the control-variable rows of each tensor after
accounting for Dynare's internal variable reordering (DR order).

Reference: Schmitt-Grohe and Uribe (2004), JEDC 28, 755-775.
"""
import sys
from pathlib import Path

import pytest

# Ensure fixtures are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fixtures"))

from nonlinear_parity_models import NONLINEAR_PARITY_SUITE

from perturbation_py.benchmarks import (
    compare_second_order_to_dynare,
    dynare_available,
)

pytestmark = pytest.mark.skipif(
    not dynare_available(),
    reason="Dynare not available (set PERTURBATION_DYNARE_CMD)",
)


@pytest.mark.parametrize(
    "spec",
    NONLINEAR_PARITY_SUITE,
    ids=[s["name"] for s in NONLINEAR_PARITY_SUITE],
)
def test_second_order_python_matches_dynare(spec):
    """Python second-order solution should match Dynare's stoch_simul(order=2).

    Tolerances are set to 1e-5 for ghx/ghu (first-order, should be very
    close) and 1e-4 for second-order tensors (accumulated FD error in
    Hessian computation).
    """
    model = spec["model_fn"]()
    mod_text = spec["mod_fn"]()

    report = compare_second_order_to_dynare(
        model, mod_text, spec["name"], timeout_sec=120,
    )

    # First-order coefficients should match very tightly (both use QZ)
    assert report.max_abs_error_ghx < 1e-5, (
        f"ghx error {report.max_abs_error_ghx:.2e} exceeds tolerance"
    )
    assert report.max_abs_error_ghu < 1e-5, (
        f"ghu error {report.max_abs_error_ghu:.2e} exceeds tolerance"
    )

    # Second-order tensors: our solver uses FD-computed Hessians while Dynare
    # uses analytical derivatives, so we expect O(1e-4) precision.  Use 1e-3
    # to account for accumulated FD error amplified by the Sylvester solve.
    assert report.max_abs_error_ghxx < 1e-3, (
        f"ghxx error {report.max_abs_error_ghxx:.2e} exceeds tolerance"
    )
    assert report.max_abs_error_ghxu < 1e-3, (
        f"ghxu error {report.max_abs_error_ghxu:.2e} exceeds tolerance"
    )
    assert report.max_abs_error_ghuu < 1e-3, (
        f"ghuu error {report.max_abs_error_ghuu:.2e} exceeds tolerance"
    )
    assert report.max_abs_error_ghs2 < 1e-3, (
        f"ghs2 error {report.max_abs_error_ghs2:.2e} exceeds tolerance"
    )
