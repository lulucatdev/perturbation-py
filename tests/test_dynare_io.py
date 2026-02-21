from perturbation_py.io import load_dynare_reference, parse_dynare_mod_file


def test_parse_minimal_dynare_mod_and_read_reference_json():
    spec = parse_dynare_mod_file("tests/fixtures/dynare/minimal_rbc.mod")
    ref = load_dynare_reference("tests/fixtures/dynare/minimal_rbc_reference.json")

    assert "k" in spec.endogenous
    assert "c" in spec.endogenous
    assert "eps" in spec.exogenous
    assert ref.order == 1
    assert ref.ghx is not None
