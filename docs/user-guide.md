# User Guide

## 1. Model contract

`perturbation_py` expects a `DSGEModel` with:

- `transition(s, x, e, params) -> s_next`
- `arbitrage(s, x, s_next, x_next, params) -> residual`

where:

- `s`: state vector
- `x`: control vector
- `e`: shock vector

Steady-state vectors are provided in `steady_state_states` and `steady_state_controls`.

## 2. Solving policies

- `solve_first_order(model)` for `order=1`
- `solve_second_order(model)` for `order=2`
- `solve_third_order(model)` for `order=3`

Convert to unified policy object:

- `Policy.from_first_order(...)`
- `Policy.from_second_order(...)`
- `Policy.from_third_order(...)`

## 3. Simulation

- Linear: `simulate_with_policy(policy, ...)`
- Pruned: `simulate_pruned(policy, ...)`
- IRF: `impulse_response_with_policy(...)` and `impulse_response_pruned(...)`

## 4. CLI

- `perturbation-py solve`
- `perturbation-py steady-state`
- `perturbation-py simulate`
- `perturbation-py irf`
- `perturbation-py parse-mod`

## 5. Dynare tools

- Parse `.mod`: `parse_dynare_mod_file(path)`
- Run parity benchmark: `compare_first_order_to_dynare(spec)`
- Run Dynare fixture directly: `run_dynare_mod_file(path)`
- Load Dynare `.mat` output: `load_dynare_results(path)`
- Reconstruct and compare IRFs: `compare_reconstructed_irfs_to_dynare(results)`
