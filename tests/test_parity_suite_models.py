import pytest

from fixtures.dynare_parity_suite import PARITY_SUITE
from perturbation_py.solver import solve_first_order


@pytest.mark.parametrize("spec", PARITY_SUITE, ids=[spec.name for spec in PARITY_SUITE])
def test_parity_suite_models_have_stable_first_order_solutions(spec):
    solution = solve_first_order(spec.to_model())
    assert solution.blanchard_kahn_satisfied
    assert solution.stable


def test_parity_suite_is_comprehensive():
    assert len(PARITY_SUITE) >= 6
    references = {ref for spec in PARITY_SUITE for ref in spec.reference_models}
    assert any("RBC_baseline" in ref for ref in references)
    assert any("Gali_2015" in ref for ref in references)
    assert any("Smets_Wouters_2007" in ref for ref in references)
