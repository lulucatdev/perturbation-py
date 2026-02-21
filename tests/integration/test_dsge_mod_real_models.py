from pathlib import Path

import pytest

from perturbation_py.benchmarks import dynare_available, run_dynare_mod_file
from perturbation_py.io import (
    compare_reconstructed_irfs_to_dynare,
    load_dynare_results,
    results_mat_path_for_mod,
)


REAL_DSGE_MOD_MODELS = (
    Path("references/DSGE_mod/Gali_2008/Gali_2008_chapter_3.mod"),
    Path("references/DSGE_mod/Gali_2015/Gali_2015_chapter_3.mod"),
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "mod_file", REAL_DSGE_MOD_MODELS, ids=[p.stem for p in REAL_DSGE_MOD_MODELS]
)
def test_real_dsge_mod_irf_reconstruction_matches_dynare(mod_file: Path):
    if not dynare_available():
        pytest.skip("Dynare runtime not available")

    run = run_dynare_mod_file(mod_file, timeout_sec=900)
    assert run.return_code == 0

    mat_path = results_mat_path_for_mod(mod_file)
    results = load_dynare_results(mat_path)
    comparison = compare_reconstructed_irfs_to_dynare(results)

    assert len(comparison.per_series_max_abs_error) > 0
    assert comparison.max_abs_error < 1e-8
