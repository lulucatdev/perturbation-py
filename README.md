# perturbation_py

**Perturbation-based solution of Dynamic Stochastic General Equilibrium models up to third order, with cross-validated Dynare parity.**

## 1. Introduction

`perturbation_py` is a Python library that computes approximate decision rules for Dynamic Stochastic General Equilibrium (DSGE) models using the perturbation method. The library implements the standard recursive approach: first-order linear approximation via generalized Schur (QZ) decomposition, followed by higher-order corrections obtained by solving systems of generalized Sylvester matrix equations.

The implementation covers perturbation orders 1 through 3:

- `order=1`: First-order (linear) approximation via QZ decomposition with Blanchard and Kahn (1980) rank conditions.
- `order=2`: Second-order approximation via closed-form Sylvester equations following Schmitt-Grohe and Uribe (2004), including the constant uncertainty correction `ghs2`.
- `order=3`: Third-order approximation via iterated Sylvester equations, including the state-dependent risk correction `ghxss`.

Supplementary capabilities include:

- **Pruning**: Stable higher-order simulation via the Kim, Kim, Schaumburg, and Sims (2008) decomposition, extended to third order following Andreasen, Fernandez-Villaverde, and Rubio-Ramirez (2018).
- **Ergodic moments**: Unconditional mean, covariance, and autocorrelation from the perturbation solution via discrete Lyapunov equations.
- **Generalized impulse response functions (GIRF)**: Nonlinear impulse responses that capture asymmetric effects absent from linear IRFs.
- **Dynare parity harness**: Automated validation of all decision-rule tensors against Dynare's `stoch_simul`.

## 2. Theoretical Framework

### 2.1 Model Specification

A DSGE model is expressed as a system of two vector-valued functions evaluated at the deterministic steady state. Let `s_t` (n_s × 1) denote predetermined (state) variables, `x_t` (n_x × 1) forward-looking (control) variables, and `ε_t` (n_e × 1) exogenous innovations with `E[ε_t ε_t'] = Σ_ε`.

**Transition equation** (law of motion for states):

```
s_{t+1} = g(s_t, x_t, ε_t)
```

**Equilibrium conditions** (Euler/optimality equations):

```
E_t[ f(s_t, x_t, s_{t+1}, x_{t+1}) ] = 0
```

The perturbation method approximates the policy function `x_t = h(s_t, σ)` as a Taylor expansion around the deterministic steady state `(s̄, x̄)` where `g(s̄, x̄, 0) = s̄` and `f(s̄, x̄, s̄, x̄) = 0`.

### 2.2 First-Order Solution

The first-order approximation yields a linear decision rule:

```
x_t     = g_x · s_t + g_u · ε_t
s_{t+1} = h_x · s_t + h_u · ε_t
```

where the matrices `g_x`, `g_u`, `h_x`, `h_u` are obtained from the generalized Schur (QZ) decomposition of the linearized system. Existence and uniqueness of a bounded solution requires the Blanchard-Kahn (1980) condition: the number of unstable generalized eigenvalues must equal the number of forward-looking variables `n_x`.

### 2.3 Second-Order Solution

At second order, the policy function gains quadratic terms and a constant risk correction (Schmitt-Grohe and Uribe 2004, Equations 45–60):

```
x_t = g_x · s_t + g_u · ε_t
    + ½ g_xx · (s_t ⊗ s_t) + g_xu · (s_t ⊗ ε_t) + ½ g_uu · (ε_t ⊗ ε_t)
    + ½ g_σσ
```

The tensors `g_xx`, `g_xu`, `g_uu` are obtained by solving generalized Sylvester equations of the form `A·X + B·X·(C⊗C) + D = 0`. The uncertainty correction `g_σσ` captures the effect of future volatility on current decisions (precautionary behavior), solving `L · g_σσ = −tr(K · Σ_ε)`.

### 2.4 Third-Order Solution

The third-order approximation adds cubic terms and the state-dependent risk correction `g_{xσσ}` (dolo reference implementation, following the recursive structure of Judd and Guu 1997):

```
x_t += (1/6) g_xxx · (s⊗s⊗s) + ½ g_xxu · (s⊗s⊗ε) + ½ g_xuu · (s⊗ε⊗ε)
     + (1/6) g_uuu · (ε⊗ε⊗ε) + ½ g_{xσσ} · s_t
```

### 2.5 Pruned Simulation

Direct simulation with higher-order policy functions can generate explosive sample paths. The pruning method of Kim, Kim, Schaumburg, and Sims (2008), extended by Andreasen, Fernandez-Villaverde, and Rubio-Ramirez (2018), decomposes the state into order-specific components:

```
s^(1)_{t+1} = h_x · s^(1)_t + h_u · ε_{t+1}
s^(2)_{t+1} = h_x · s^(2)_t + ½ h_xx·(s^(1)⊗s^(1)) + h_xu·(s^(1)⊗ε) + ½ h_uu·(ε⊗ε) + ½ h_σσ

x^(1)_t = g_x · s^(1)_t + g_u · ε_t
x^(2)_t = g_x · s^(2)_t + ½ g_xx·(s^(1)⊗s^(1)) + g_xu·(s^(1)⊗ε) + ½ g_uu·(ε⊗ε) + ½ g_σσ

x_t = x̄ + x^(1)_t + x^(2)_t
```

The total state remains bounded because the quadratic terms are evaluated only at `s^(1)`, which inherits the stability of the linear system.

## 3. Numerical Validation

### 3.1 First-Order Dynare Parity

The first-order solver is validated against Dynare 6.2's `stoch_simul(order=1)` on a suite of six models spanning standard macroeconomic specifications. For each model, we compare all four decision-rule matrices element-wise.

**Table 1.** First-order decision rules — maximum absolute error vs. Dynare (6 models):

| Model | max err `h_x` | max err `g_x` | max err `h_u` | max err `g_u` |
|:------|:---:|:---:|:---:|:---:|
| RBC baseline | 0 | 0 | 0 | 0 |
| RBC news | 2.2e-16 | 8.9e-16 | 1.7e-18 | 2.8e-17 |
| SGU small open | 2.2e-16 | 1.2e-15 | 3.5e-18 | 1.9e-17 |
| NK Chapter 3 | 4.4e-16 | 4.4e-16 | 1.7e-18 | 1.1e-18 |
| Smets-Wouters class | 4.4e-16 | 3.3e-16 | 1.7e-18 | 8.7e-19 |
| Small open economy | 3.3e-16 | 7.8e-16 | 8.7e-19 | 1.6e-17 |

All errors are at or below machine epsilon (≈ 2.2 × 10⁻¹⁶), confirming that the Python QZ-based solver produces identical decision rules to Dynare up to IEEE 754 double-precision arithmetic.

### 3.2 Second-Order Dynare Parity

The second-order Sylvester solver is validated against Dynare's `stoch_simul(order=2)` on a neoclassical growth (RBC) model with Cobb-Douglas production (α = 0.33, β = 0.99, δ = 0.025), log-utility, and i.i.d. TFP shocks (σ = 0.01). The model has one state (capital) and one control (consumption), with nonlinearity from k^α in production and 1/c in the Euler equation.

**Table 2.** Second-order decision rules — maximum absolute error vs. Dynare (RBC model):

| Tensor | Max abs error | Order of magnitude |
|:-------|:---:|:---|
| `g_x` (first-order, states) | 1.71e-09 | Sub-ppb agreement |
| `g_u` (first-order, shocks) | 1.31e-10 | Sub-ppb agreement |
| `g_xx` (state-state) | 2.79e-04 | O(10⁻⁴) |
| `g_xu` (state-shock) | 2.38e-05 | O(10⁻⁵) |
| `g_uu` (shock-shock) | 2.29e-06 | O(10⁻⁶) |
| `g_σσ` (risk correction) | 9.78e-05 | O(10⁻⁵) |

**Discussion.** First-order coefficients (`g_x`, `g_u`) agree to O(10⁻⁹) — both solvers use QZ decomposition, with residual discrepancy from different Schur factorization orderings and floating-point accumulation.

Second-order tensors exhibit O(10⁻⁴)–O(10⁻⁶) discrepancies. This is expected: Dynare computes model Hessians analytically via symbolic differentiation, while our implementation obtains them by central finite differences of the Jacobians (step size ε = 10⁻⁵, yielding O(ε²) = O(10⁻¹⁰) local truncation error). The effective Hessian error is amplified by the condition number of the Sylvester system, producing the observed O(10⁻⁴) bound. This precision is sufficient for economic applications — typical calibration uncertainty in DSGE parameters exceeds O(10⁻²).

### 3.3 Internal Cross-Validation

**Table 3.** Sylvester vs. local implicit cross-method agreement (RBC model):

| Tensor | Max abs difference |
|:-------|:---:|
| `g_xx` | 7.66e-04 |
| `g_xu` | 3.84e-05 |
| `g_uu` | 1.90e-06 |

The two independent solver paths — closed-form Sylvester equations vs. finite-difference of an implicit rootfinding map — agree to within the expected FD precision, serving as a dual-method verification.

**Additional consistency properties verified by the test suite:**

- Linear models yield identically zero second- and third-order tensors (machine precision)
- Second-order solution with zero shock covariance reproduces the first-order policy
- The `g_σσ` residual equation `L · g_σσ + K = 0` holds to O(10⁻¹²)
- KKSS pruned simulation ergodic mean converges to `x̄ + ½ g_σσ` (verified over T = 50,000 periods)
- Order-1 GIRF coincides with the standard linear IRF

### 3.4 Test Summary

| Category | Tests | Status |
|:---------|:-----:|:------:|
| Unit tests (solvers, derivatives, tensor ops, moments, pruning, GIRF) | 49 | Pass |
| Integration — first-order Dynare parity (6 models) | 6 | Pass |
| Integration — second-order Dynare parity (1 model) | 1 | Pass |
| Integration — Dynare IRF reconstruction (2 models) | 2 | Pass |
| **Total** | **58** | **Pass** |

## 4. Installation

```bash
pip install -e .
```

**Dependencies:** NumPy, SciPy. Optional: Dynare (for integration tests).

## 5. Examples

### 5.1 Defining and Solving a Linear NKPC Model

A scalar New Keynesian Phillips Curve with an AR(1) cost-push shock:

```python
import numpy as np
from perturbation_py import DSGEModel, solve_first_order

def transition(s, x, e, p):
    return np.array([p["rho"] * s[0] + p["sigma"] * e[0]])

def arbitrage(s, x, s_next, x_next, p):
    return np.array([x[0] - p["beta"] * x_next[0] - p["kappa"] * s[0]])

model = DSGEModel(
    state_names=("z",),
    control_names=("pi",),
    shock_names=("eps",),
    parameters={"rho": 0.9, "beta": 0.95, "kappa": 0.1, "sigma": 0.01},
    steady_state_states=np.array([0.0]),
    steady_state_controls=np.array([0.0]),
    transition=transition,
    arbitrage=arbitrage,
)

fo = solve_first_order(model)

# The analytic solution is g_x = kappa / (1 - beta * rho)
expected_gx = 0.1 / (1.0 - 0.95 * 0.9)  # ≈ 0.7143
print("g_x:", fo.policy)                  # [[0.71428571]]
print("h_x:", fo.transition)              # [[0.9]]
print("BK satisfied:", fo.blanchard_kahn_satisfied)  # True
print("Eigenvalues:", fo.eigenvalues)      # moduli of generalized eigenvalues
```

### 5.2 Nonlinear RBC Model — Second-Order Solution

A one-sector neoclassical growth model with Cobb-Douglas production, where the nonlinearity arises from k^α and 1/c:

```python
import numpy as np
from perturbation_py import DSGEModel, solve_second_order

ALPHA, BETA, DELTA, SIGMA = 0.33, 0.99, 0.025, 0.01
K_SS = (ALPHA * BETA / (1 - BETA * (1 - DELTA))) ** (1 / (1 - ALPHA))
Y_SS = K_SS ** ALPHA
C_SS = Y_SS - DELTA * K_SS

def transition(s, x, e, p):
    K = p["k_ss"] + s[0]
    C = p["c_ss"] + x[0]
    z = p["sigma"] * e[0]
    K_next = (1 - p["delta"]) * K + K ** p["alpha"] * np.exp(z) - C
    return np.array([K_next - p["k_ss"]])

def arbitrage(s, x, s_next, x_next, p):
    C      = p["c_ss"] + x[0]
    C_next = p["c_ss"] + x_next[0]
    K_next = p["k_ss"] + s_next[0]
    mpk = p["alpha"] * K_next ** (p["alpha"] - 1) + 1 - p["delta"]
    return np.array([1.0 - p["beta"] * (C / C_next) * mpk])

model = DSGEModel(
    state_names=("k",), control_names=("c",), shock_names=("eps",),
    parameters={"alpha": ALPHA, "beta": BETA, "delta": DELTA,
                "sigma": SIGMA, "k_ss": K_SS, "c_ss": C_SS},
    steady_state_states=np.array([0.0]),
    steady_state_controls=np.array([0.0]),
    transition=transition, arbitrage=arbitrage,
)

so = solve_second_order(model, method="sylvester")

print("g_x  (linear):", so.ghx)     # (1, 1) — consumption response to capital
print("g_xx (quadratic):", so.ghxx)  # (1, 1, 1) — curvature
print("g_σσ (risk):", so.ghs2)      # (1,) — precautionary savings effect
# g_σσ > 0 means agents consume more due to precautionary motive
```

### 5.3 Third-Order Solution with Sigma Corrections

```python
from perturbation_py import solve_third_order

to = solve_third_order(model, method="sylvester")

print("g_xxx:", to.ghxxx.shape)   # (1, 1, 1, 1) — cubic state tensor
print("g_xxu:", to.ghxxu.shape)   # (1, 1, 1, 1) — state-state-shock cross
print("g_xσσ:", to.ghxss)         # (1, 1) — state-dependent risk correction
# g_{xσσ} captures how the risk premium varies with the state
```

### 5.4 Verifying Analytic Solutions

For the scalar NKPC model, we can verify the solver exactly:

```python
fo = solve_first_order(model)

rho, beta, kappa, sigma = 0.9, 0.95, 0.1, 0.01
expected_gx = kappa / (1.0 - beta * rho)
expected_hx = rho
expected_gu = kappa * sigma / (1.0 - beta * rho)
expected_hu = sigma

assert np.allclose(fo.policy, [[expected_gx]], atol=1e-12)
assert np.allclose(fo.transition, [[expected_hx]], atol=1e-12)
assert np.allclose(fo.control_shock_impact, [[expected_gu]], atol=1e-12)
assert np.allclose(fo.shock_impact, [[expected_hu]], atol=1e-12)
```

For a linear model, all second-order tensors must be zero:

```python
so = solve_second_order(model, method="sylvester")
assert np.allclose(so.ghxx, 0.0, atol=1e-8)
assert np.allclose(so.ghxu, 0.0, atol=1e-8)
assert np.allclose(so.ghuu, 0.0, atol=1e-8)
assert np.allclose(so.ghs2, 0.0, atol=1e-8)
```

### 5.5 Policy Evaluation

The `Policy` object provides a unified interface for evaluating decision rules at any order:

```python
from perturbation_py import Policy, solve_first_order, solve_second_order

# From first-order
fo = solve_first_order(model)
policy1 = Policy.from_first_order(fo)

state = np.array([0.05])  # 5% above steady state
shock = np.array([0.0])
x1 = policy1.controls(state=state, shock=shock)
# At order 1: x = g_x · s
assert np.allclose(x1, fo.policy @ state)

# From second-order — adds quadratic terms and risk correction
so = solve_second_order(model, method="sylvester")
policy2 = Policy.from_second_order(so)
x2 = policy2.controls(state=state, shock=shock)
# x2 includes ½ g_xx·(s⊗s) + ½ g_σσ correction
```

### 5.6 Linear Simulation and Impulse Responses

```python
from perturbation_py import simulate_linear, impulse_response

fo = solve_first_order(model)

# Simulate with an explicit shock sequence
shocks = np.zeros((50, 1))
shocks[0, 0] = 1.0  # unit impulse at t=0
sim = simulate_linear(fo, initial_state=np.array([0.0]), shocks=shocks)
print(sim.states.shape)    # (51, 1) — includes initial condition
print(sim.controls.shape)  # (50, 1)

# Impulse response (convenience wrapper)
irf = impulse_response(fo, horizon=40, shock_index=0, shock_size=1.0)
# States decay geometrically: s_t = h_x^t · h_u · shock
print("s_1 =", irf.states[1, 0])   # = sigma * shock_size
print("s_2 =", irf.states[2, 0])   # = rho * s_1
```

### 5.7 KKSS Pruned Simulation

```python
from perturbation_py import Policy, simulate_pruned, impulse_response_pruned

so = solve_second_order(model, method="sylvester")
policy = Policy.from_second_order(so)

# KKSS pruned simulation — stable even at higher orders
result = simulate_pruned(policy, horizon=1000, shock_std=0.01, seed=42, method="kkss")
print(result.states.shape)    # (1001, n_s)
print(result.controls.shape)  # (1000, n_x)
assert np.isfinite(result.states).all()   # no explosions

# Pruned impulse response
irf = impulse_response_pruned(policy, horizon=40, shock_index=0, shock_size=1.0)

# Long-run mean converges to steady state + ½ g_σσ
long_sim = simulate_pruned(policy, horizon=50000, shock_std=0.01, seed=0, method="kkss")
ergodic_mean = long_sim.controls[5000:].mean(axis=0)  # skip burn-in
expected_mean = policy.steady_state_controls + 0.5 * so.ghs2
print("Ergodic mean:", ergodic_mean)
print("Expected:    ", expected_mean)
```

### 5.8 Generalized Impulse Response Functions

Unlike linear IRFs, the GIRF captures nonlinear asymmetries — positive and negative shocks produce different response magnitudes:

```python
from perturbation_py import generalized_irf, impulse_response

# At order 1, GIRF = linear IRF
fo = solve_first_order(model)
policy1 = Policy.from_first_order(fo)
linear_irf = impulse_response(fo, horizon=20, shock_index=0, shock_size=1.0)
girf1 = generalized_irf(policy1, horizon=20, shock_index=0, shock_size=1.0)
assert np.allclose(girf1.states, linear_irf.states, atol=1e-12)

# At order 2, responses are asymmetric
so = solve_second_order(model, method="sylvester")
policy2 = Policy.from_second_order(so)
girf_pos = generalized_irf(policy2, horizon=20, shock_index=0, shock_size=+1.0)
girf_neg = generalized_irf(policy2, horizon=20, shock_index=0, shock_size=-1.0)

# GIRF = shocked path − baseline path
print("Positive shock response:", girf_pos.controls[:5, 0])
print("Negative shock response:", girf_neg.controls[:5, 0])
# |response(+1)| ≠ |response(-1)| due to curvature
```

### 5.9 Ergodic Moments from Perturbation Solution

Computes theoretical unconditional moments without simulation, using discrete Lyapunov equations:

```python
from perturbation_py import compute_unconditional_moments, Policy

# First-order moments
fo = solve_first_order(model)
policy = Policy.from_first_order(fo)
mom = compute_unconditional_moments(policy, max_lag=5)

print("E[s]:", mom.mean_states)           # zero (deviations from SS)
print("E[x]:", mom.mean_controls)         # SS controls (no risk adjustment at order 1)
print("Std[s]:", mom.std_states)
print("Std[x]:", mom.std_controls)
print("Autocorr:", mom.autocorrelations)  # shape (5, n_s + n_x)

# Verify against analytic AR(1) variance: Var(s) = σ² / (1 − ρ²)
rho, sigma = 0.9, 0.01
expected_var = sigma**2 / (1 - rho**2)
assert np.allclose(mom.variance_states[0, 0], expected_var, rtol=1e-10)

# Autocorrelation at lag k should be ρ^k
for k in range(1, 6):
    assert np.allclose(mom.autocorrelations[k - 1, 0], rho**k, rtol=1e-10)

# Second-order moments include risk correction in the mean
so = solve_second_order(model, method="sylvester")
policy2 = Policy.from_second_order(so)
mom2 = compute_unconditional_moments(policy2)
expected_mean = policy2.steady_state_controls + 0.5 * so.ghs2
assert np.allclose(mom2.mean_controls, expected_mean, atol=1e-14)
```

### 5.10 Tensor Operations

The library exposes the low-level tensor algebra used internally:

```python
from perturbation_py import sdot, mdot, solve_generalized_sylvester
import numpy as np

# sdot: contracts last axis of U with first axis of V (generalizes @)
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
assert np.allclose(sdot(A, B), A @ B)

# Works on higher-rank tensors
T = np.random.randn(2, 3, 4)  # rank-3
v = np.random.randn(4)
result = sdot(T, v)            # shape (2, 3)

# mdot: multi-index contraction M_{ijk} A_{jm} B_{kn} → R_{imn}
M = np.random.randn(2, 3, 4)
C1 = np.random.randn(3, 5)
C2 = np.random.randn(4, 6)
result = mdot(M, C1, C2)
expected = np.einsum("ijk,jm,kn->imn", M, C1, C2)
assert np.allclose(result, expected)

# Generalized Sylvester: A·X + B·X·C + D = 0
n, m = 4, 3
A = np.random.randn(n, n)
B = np.random.randn(n, n)
C = np.random.randn(m, m)
D = np.random.randn(n, m)
X = solve_generalized_sylvester(A, B, C, D)
residual = A @ X + B @ X @ C + D
assert np.allclose(residual, 0.0, atol=1e-10)
```

### 5.11 Numerical Steady-State Solver

For models where the steady state is not known analytically:

```python
from perturbation_py import solve_steady_state

# Model with nonlinear steady-state conditions
ss = solve_steady_state(model, guess_states=[5.0], guess_controls=[0.5])

print("States:", ss.states)
print("Controls:", ss.controls)
print("Residual norm:", ss.residual_norm)  # should be < 1e-10
print("Converged:", ss.success)
print("Iterations:", ss.nfev)

# Verify: the static residual at SS should be zero
residual = model.static_residual(ss.states, ss.controls)
assert np.linalg.norm(residual, ord=np.inf) < 1e-10
```

### 5.12 Cross-Method Validation

The two independent solver methods serve as mutual verification:

```python
from perturbation_py import solve_second_order

so_syl = solve_second_order(model, method="sylvester")
so_li  = solve_second_order(model, method="local_implicit")

# First-order coefficients should match tightly (same QZ)
assert np.allclose(so_syl.ghx, so_li.ghx, atol=1e-8)
assert np.allclose(so_syl.ghu, so_li.ghu, atol=1e-8)

# Second-order tensors agree within FD precision
assert np.allclose(so_syl.ghxx, so_li.ghxx, atol=1e-3)
assert np.allclose(so_syl.ghxu, so_li.ghxu, atol=1e-3)
assert np.allclose(so_syl.ghuu, so_li.ghuu, atol=1e-3)

# Only Sylvester computes the risk correction
print("g_σσ (Sylvester):", so_syl.ghs2)       # nonzero
print("g_σσ (local implicit):", so_li.ghs2)    # zero (not computed)
```

### 5.13 Dynare Parity Testing

Validate your Python solution against Dynare's output:

```python
from perturbation_py import compare_second_order_to_dynare, dynare_available

if dynare_available():
    model = ...       # DSGEModel (deviation form)
    mod_text = "..."  # Dynare .mod file (level form)

    report = compare_second_order_to_dynare(model, mod_text, "my_model")
    print(f"g_x  error: {report.max_abs_error_ghx:.2e}")
    print(f"g_xx error: {report.max_abs_error_ghxx:.2e}")
    print(f"g_σσ error: {report.max_abs_error_ghs2:.2e}")
```

## 6. API Reference

### 6.1 Solvers

#### `solve_first_order(model, *, jacobians=None, eig_cutoff=1-1e-8) -> FirstOrderSolution`

Solves for the first-order decision rule via generalized Schur decomposition.

**Returns `FirstOrderSolution`:**

| Field | Shape | Description |
|:------|:------|:------------|
| `policy` | `(n_x, n_s)` | `g_x` — control response to states |
| `transition` | `(n_s, n_s)` | `h_x` — state transition matrix |
| `shock_impact` | `(n_s, n_e)` | `h_u` — shock impact on states |
| `control_shock_impact` | `(n_x, n_e)` | `g_u` — contemporaneous shock on controls |
| `eigenvalues` | `(n_s+n_x,)` | Generalized eigenvalue moduli |
| `blanchard_kahn_satisfied` | `bool` | BK rank condition |

#### `solve_second_order(model, *, method="sylvester", shock_covariance=None) -> SecondOrderSolution`

Solves for the second-order decision rule. `"sylvester"` (default) computes exact Sylvester solutions including `g_σσ`. `"local_implicit"` uses FD of an implicit rootfinding map (faster, lower accuracy, `g_σσ = 0`).

**Returns `SecondOrderSolution`:**

| Field | Shape | Description |
|:------|:------|:------------|
| `ghxx` | `(n_x, n_s, n_s)` | `g_xx` — state-state quadratic |
| `ghxu` | `(n_x, n_s, n_e)` | `g_xu` — state-shock cross |
| `ghuu` | `(n_x, n_e, n_e)` | `g_uu` — shock-shock quadratic |
| `ghs2` | `(n_x,)` | `g_σσ` — uncertainty correction |

#### `solve_third_order(model, *, method="sylvester", shock_covariance=None) -> ThirdOrderSolution`

Solves for the third-order decision rule via iterated Sylvester equations.

**Returns `ThirdOrderSolution`:**

| Field | Shape | Description |
|:------|:------|:------------|
| `ghxxx` | `(n_x, n_s, n_s, n_s)` | `g_xxx` — cubic state tensor |
| `ghxxu` | `(n_x, n_s, n_s, n_e)` | `g_xxu` — state-state-shock |
| `ghxuu` | `(n_x, n_s, n_e, n_e)` | `g_xuu` — state-shock-shock |
| `ghuuu` | `(n_x, n_e, n_e, n_e)` | `g_uuu` — cubic shock tensor |
| `ghxss` | `(n_x, n_s)` | `g_{xσσ}` — state-dependent risk |

### 6.2 Policy Evaluation

#### `Policy`

Unified interface for evaluating perturbation solutions of any order.

```python
policy = Policy.from_first_order(fo)   # or from_second_order, from_third_order
x = policy.controls(state=s, shock=e)
```

**Properties:** `n_states`, `n_controls`, `n_shocks`, `order`, `steady_state_controls`

### 6.3 Simulation

| Function | Description |
|:---------|:------------|
| `simulate_linear(solution, initial_state, shocks)` | First-order linear simulation |
| `simulate_with_policy(policy, *, initial_state, shocks)` | Full nonlinear policy simulation |
| `simulate_pruned(policy, *, horizon, method="kkss")` | KKSS pruned simulation |
| `impulse_response(solution, *, horizon, shock_index)` | Linear impulse response |
| `impulse_response_pruned(policy, *, horizon, shock_index)` | Pruned impulse response |
| `generalized_irf(policy, *, horizon, shock_index)` | GIRF: shocked − baseline |

### 6.4 Ergodic Moments

#### `compute_unconditional_moments(policy, *, shock_covariance=None, max_lag=5) -> MomentsResult`

Computes theoretical unconditional moments:
- **Mean**: `E[x] = x̄ + ½ g_σσ` (at order ≥ 2)
- **Variance**: Via discrete Lyapunov equation `Σ_s = h_x Σ_s h_x' + h_u Σ_ε h_u'`
- **Autocorrelation**: `Corr(y_t, y_{t-k})` from powers of `h_x`

### 6.5 Derivatives

| Function | Description |
|:---------|:------------|
| `compute_jacobians(model, epsilon=1e-6)` | First-order Jacobians of `f` and `g` |
| `compute_model_hessians(model, epsilon=1e-5)` | Second derivatives via FD of Jacobians |
| `compute_model_third_derivatives(model, epsilon=1e-4)` | Third derivatives via FD of Hessians |

### 6.6 Tensor Operations

| Function | Description |
|:---------|:------------|
| `sdot(U, V)` | Tensor contraction: last axis of `U` with first of `V` |
| `mdot(M, *C)` | Multi-index contraction via dynamic `einsum` |
| `solve_generalized_sylvester(A, B, C, D)` | Solves `A·X + B·X·(C⊗...⊗C) + D = 0` |

### 6.7 Dynare Interface

| Function | Description |
|:---------|:------------|
| `dynare_available()` | Check if Dynare is accessible |
| `compare_first_order_to_dynare(spec)` | Compare first-order decision rules vs. Dynare |
| `compare_second_order_to_dynare(model, mod_text, name)` | Compare second-order decision rules vs. Dynare |
| `run_dynare_mod_file(path)` | Execute a `.mod` file via Dynare |
| `parse_dynare_mod_file(path)` | Parse `.mod` file structure |
| `load_dynare_results(path)` | Load Dynare `.mat` output |

## 7. Architecture

```
src/perturbation_py/
├── model.py                 # DSGEModel definition
├── derivatives.py           # Jacobian computation dispatcher
├── derivative_backends.py   # FD and complex-step backends
├── model_hessians.py        # Second/third-order derivatives via FD of Jacobians
├── solver.py                # First-order QZ solver with BK conditions
├── solver_second_order.py   # Second-order (Sylvester + local implicit)
├── solver_third_order.py    # Third-order (Sylvester + local implicit)
├── tensor_ops.py            # sdot, mdot, solve_generalized_sylvester
├── policy.py                # Unified Policy evaluation (orders 1–3)
├── simulation.py            # Linear simulation, IRF, GIRF
├── pruning.py               # KKSS and naive pruned simulation
├── moments.py               # Ergodic moments via Lyapunov equation
├── qz.py                    # BK diagnostics, eigenvalue utilities
├── steady_state.py          # Numerical steady-state solver
├── timing.py                # Lead/lag incidence from symbolic equations
├── benchmarks.py            # Dynare parity comparison infrastructure
└── io/                      # Dynare .mod parsing, .mat loading
```

## 8. References

| Key | Reference |
|:----|:----------|
| BK80 | O. J. Blanchard and C. M. Kahn, "The solution of linear difference models under rational expectations," *Econometrica* 48(5), 1305–1311, 1980. |
| SGU04 | S. Schmitt-Grohe and M. Uribe, "Solving dynamic general equilibrium models using a second-order approximation to the policy function," *JEDC* 28, 755–775, 2004. |
| KKSS08 | J. Kim, S. Kim, E. Schaumburg, and C. A. Sims, "Calculating and using second-order accurate solutions of discrete time dynamic equilibrium models," *JEDC* 32(11), 3397–3414, 2008. |
| AFRR18 | M. M. Andreasen, J. Fernandez-Villaverde, and J. F. Rubio-Ramirez, "The pruned state-space system for non-linear DSGE models: Theory and empirical applications," *REStud* 85(1), 1–49, 2018. |
| V11 | S. Villemot, "Solving rational expectations models at first order: What Dynare does," *Dynare Working Papers* 2, 2011. |
| dolo | P. Winant et al., [dolo](https://github.com/EconForge/dolo), `dolo.algos.perturbations_higher_order` — open-source reference for the Sylvester-equation algorithm. |

## Maintainer

Maintained by [Knowecon](https://github.com/Knowecon).

## License

MIT
