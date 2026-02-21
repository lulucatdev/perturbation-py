# Validation

## Unit-level validation

- Jacobian consistency checks
- BK diagnostics and error classification
- Steady-state residual checks
- Simulation shape and sanity checks

## Dynare parity validation

`tests/integration/test_dynare_parity_suite.py` compares Dynare and Python outputs on a curated model suite inspired by `references/DSGE_mod`.

Compared objects:

- Transition matrix
- Policy matrix
- State-shock impact matrix
- Control-shock impact matrix

## Runtime prerequisites

- Dynare binary available on `PATH`, or
- `PERTURBATION_DYNARE_CMD` pointing to Dynare executable

If Dynare is missing, integration tests are skipped.
