# Dynare Parity Suite

This project includes a Dynare-vs-Python first-order parity suite built with model families inspired by `references/DSGE_mod`.

## Coverage

The current suite is defined in `tests/fixtures/dynare_parity_suite.py` and includes six linearized model cases:

- `rbc_baseline_like` (RBC baseline style)
- `rbc_news_like` (news-shock style)
- `sgu_small_open_like` (small open economy style)
- `nk_chapter3_like` (baseline New Keynesian style)
- `smets_wouters_like` (medium-scale NK style)
- `small_open_economy_like` (SOE style, multi-shock)

Each case stores `reference_models` paths pointing back to related `DSGE_mod` folders.

In addition, real `DSGE_mod` integration parity tests are implemented in
`tests/integration/test_dsge_mod_real_models.py` and currently cover:

- `Gali_2008_chapter_3.mod`
- `Gali_2015_chapter_3.mod`

These tests run Dynare on the original `.mod` files, load Dynare result matrices,
reconstruct IRFs in Python from `oo_.dr`, and assert near-zero max error.

## How parity is checked

1. Python solves each model with `solve_first_order`.
2. A Dynare `.mod` file is auto-generated from the exact same linear coefficients.
3. Dynare computes `oo_.dr` at order 1.
4. The test compares:
   - transition matrix
   - policy matrix
   - state shock impact matrix
   - control shock impact matrix

Tolerance is currently `1e-8` on max absolute error.

## Running the suite

You need a Dynare CLI command available.

If Dynare is on PATH:

```bash
pytest tests/integration/test_dynare_parity_suite.py -v
```

If Dynare requires a custom command, set:

```bash
export PERTURBATION_DYNARE_CMD="/path/to/dynare"
pytest tests/integration/test_dynare_parity_suite.py -v
```

For Docker-based execution, the recommended wrapper is:

```bash
export PERTURBATION_DYNARE_CMD="$PWD/scripts/dynare-octave-docker"
pytest tests/integration/test_dynare_parity_suite.py -v
```

`scripts/dynare-docker` (official `dynare/dynare`) may require a MATLAB license.

If Dynare is not installed, integration parity tests are skipped by design.

## Full parity run

```bash
export PERTURBATION_DYNARE_CMD="$PWD/scripts/dynare-octave-docker"
pytest tests/integration/test_dynare_parity_suite.py tests/integration/test_dsge_mod_real_models.py -v
```
