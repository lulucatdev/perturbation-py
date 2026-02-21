import pytest

from fixtures.dynare_parity_suite import PARITY_SUITE
from perturbation_py.benchmarks import compare_first_order_to_dynare, dynare_available


@pytest.mark.integration
@pytest.mark.parametrize("spec", PARITY_SUITE, ids=[spec.name for spec in PARITY_SUITE])
def test_dynare_and_python_first_order_match_for_suite(spec):
    if not dynare_available():
        pytest.skip("Dynare binary not available; set PERTURBATION_DYNARE_CMD")

    report = compare_first_order_to_dynare(spec)

    assert report.max_abs_error_transition < 1e-8
    assert report.max_abs_error_policy < 1e-8
    assert report.max_abs_error_state_shock < 1e-8
    assert report.max_abs_error_control_shock < 1e-8
