# Math Notes

## First order

Around deterministic steady state, linearized system is represented as:

- Transition block
  - `s_{t+1} = g_s s_t + g_x x_t + g_e e_t`
- Arbitrage block
  - `0 = f_s s_t + f_x x_t + f_S s_{t+1} + f_X x_{t+1}`

The first-order policy uses generalized Schur decomposition and BK checks.

## Second order

Controls are approximated by:

`x_t = ghx s_t + ghu e_t + 1/2 ghxx[s,s] + ghxu[s,e] + 1/2 ghuu[e,e] + 1/2 ghs2`

Current implementation computes these tensors using local implicit maps and numerical derivatives around the steady state.

## Third order

Adds:

- `1/6 ghxxx[s,s,s]`
- `1/2 ghxxu[s,s,e]`
- `1/2 ghxuu[s,e,e]`
- `1/6 ghuuu[e,e,e]`

Third-order tensors are computed numerically from local derivatives of the implicit control map.

## Pruning

Pruning simulation is provided to stabilize higher-order simulation paths by evaluating nonlinear policy terms while keeping a stable first-order state propagation backbone.
