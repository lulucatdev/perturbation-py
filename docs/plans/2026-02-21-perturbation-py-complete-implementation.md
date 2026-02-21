# perturbation_py Complete Perturbation Method Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a complete perturbation-method package for DSGE models in Python, including robust first-order parity with Dynare conventions, second/third-order tensors, pruning, interoperability, and validation benchmarks.

**Architecture:** The implementation is layered: model specification and timing metadata -> steady-state and derivative engines -> order-specific solvers -> policy evaluation and simulation (including pruning) -> I/O adapters and benchmark parity harness. Numerical kernels rely on NumPy/SciPy linear algebra, while higher-order objects are represented as explicit tensors and evaluated with deterministic contraction order. Validation is test-first and benchmark-driven at every layer.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pytest, Typer, pydantic (optional for schemas), matplotlib (optional for plotting), GitHub Actions

---

### Task 1: Timing system and canonical model representation

**Files:**
- Create: `src/perturbation_py/timing.py`
- Modify: `src/perturbation_py/model.py`
- Modify: `src/perturbation_py/__init__.py`
- Test: `tests/test_timing.py`

**Step 1: Write the failing test**

```python
def test_build_lead_lag_incidence_for_mixed_system():
    equations = (
        "x = beta * x(+1) + k(-1)",
        "k = rho * k(-1) + eps",
    )
    incidence = build_lead_lag_incidence(equations, endogenous=("x", "k"))
    assert incidence.max_lag == 1
    assert incidence.max_lead == 1
    assert incidence.current_positions["x"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_timing.py::test_build_lead_lag_incidence_for_mixed_system -v`
Expected: FAIL with `NameError`/`ImportError` because timing parser is missing.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class LeadLagIncidence:
    max_lag: int
    max_lead: int
    current_positions: dict[str, int]

def build_lead_lag_incidence(equations, endogenous):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_timing.py::test_build_lead_lag_incidence_for_mixed_system -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/timing.py src/perturbation_py/model.py src/perturbation_py/__init__.py tests/test_timing.py
git commit -m "feat: add Dynare-style timing incidence representation"
```

### Task 2: Steady-state solving and validation contracts

**Files:**
- Create: `src/perturbation_py/steady_state.py`
- Modify: `src/perturbation_py/model.py`
- Modify: `src/perturbation_py/cli.py`
- Test: `tests/test_steady_state.py`

**Step 1: Write the failing test**

```python
def test_newton_solver_finds_consistent_steady_state():
    model = make_nonlinear_fixture_model()
    ss = solve_steady_state(model, guess_states=[0.3], guess_controls=[0.3])
    residual = model.static_residual(ss.states, ss.controls)
    assert np.linalg.norm(residual, ord=np.inf) < 1e-10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_steady_state.py::test_newton_solver_finds_consistent_steady_state -v`
Expected: FAIL because `solve_steady_state` and `static_residual` do not exist.

**Step 3: Write minimal implementation**

```python
def solve_steady_state(model, guess_states, guess_controls):
    z0 = np.concatenate([guess_states, guess_controls])
    root = scipy.optimize.root(lambda z: model.static_residual_from_vector(z), z0, method="hybr")
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_steady_state.py::test_newton_solver_finds_consistent_steady_state -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/steady_state.py src/perturbation_py/model.py src/perturbation_py/cli.py tests/test_steady_state.py
git commit -m "feat: add steady-state solver and residual validation"
```

### Task 3: Derivative backend abstraction (finite-difference + complex-step)

**Files:**
- Create: `src/perturbation_py/derivative_backends.py`
- Modify: `src/perturbation_py/derivatives.py`
- Modify: `src/perturbation_py/model.py`
- Test: `tests/test_derivative_backends.py`

**Step 1: Write the failing test**

```python
def test_complex_step_matches_finite_difference_on_linear_fixture():
    model = make_scalar_model()
    jac_fd = compute_jacobians(model, backend="finite_difference")
    jac_cs = compute_jacobians(model, backend="complex_step")
    assert np.allclose(jac_fd.f_x, jac_cs.f_x, atol=1e-10)
    assert np.allclose(jac_fd.g_s, jac_cs.g_s, atol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_derivative_backends.py::test_complex_step_matches_finite_difference_on_linear_fixture -v`
Expected: FAIL because backend selection is unsupported.

**Step 3: Write minimal implementation**

```python
def compute_jacobians(model, epsilon=1e-6, backend="finite_difference"):
    if backend == "finite_difference":
        return _compute_with_finite_difference(...)
    if backend == "complex_step":
        return _compute_with_complex_step(...)
    raise ValueError("Unknown backend")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_derivative_backends.py::test_complex_step_matches_finite_difference_on_linear_fixture -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/derivative_backends.py src/perturbation_py/derivatives.py src/perturbation_py/model.py tests/test_derivative_backends.py
git commit -m "feat: add pluggable derivative backends"
```

### Task 4: First-order solver diagnostics and Dynare-compatible classification

**Files:**
- Create: `src/perturbation_py/qz.py`
- Modify: `src/perturbation_py/solver.py`
- Modify: `src/perturbation_py/__init__.py`
- Test: `tests/test_solver_first_order_diagnostics.py`

**Step 1: Write the failing test**

```python
def test_solver_reports_indeterminacy_with_structured_error():
    model = make_indeterminate_fixture_model()
    with pytest.raises(BlanchardKahnError) as exc:
        solve_first_order(model)
    assert exc.value.reason in {"indeterminacy", "no_stable_equilibrium"}
    assert exc.value.unstable_count >= 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver_first_order_diagnostics.py::test_solver_reports_indeterminacy_with_structured_error -v`
Expected: FAIL because error payload does not expose diagnostic fields.

**Step 3: Write minimal implementation**

```python
@dataclass
class BKDiagnostics:
    unstable_count: int
    expected_unstable: int
    reason: str

class BlanchardKahnError(RuntimeError):
    def __init__(self, diagnostics: BKDiagnostics):
        ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver_first_order_diagnostics.py::test_solver_reports_indeterminacy_with_structured_error -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/qz.py src/perturbation_py/solver.py src/perturbation_py/__init__.py tests/test_solver_first_order_diagnostics.py
git commit -m "feat: add BK diagnostics and robust QZ classification"
```

### Task 5: Second-order perturbation tensors (`ghxx`, `ghxu`, `ghuu`, `ghs2`)

**Files:**
- Create: `src/perturbation_py/solver_second_order.py`
- Modify: `src/perturbation_py/derivatives.py`
- Modify: `src/perturbation_py/__init__.py`
- Test: `tests/test_solver_second_order.py`

**Step 1: Write the failing test**

```python
def test_second_order_solver_returns_expected_tensor_shapes():
    model = make_quadratic_fixture_model()
    sol = solve_second_order(model)
    assert sol.ghxx.shape == (model.n_controls, model.n_states, model.n_states)
    assert sol.ghxu.shape == (model.n_controls, model.n_states, model.n_shocks)
    assert sol.ghuu.shape == (model.n_controls, model.n_shocks, model.n_shocks)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver_second_order.py::test_second_order_solver_returns_expected_tensor_shapes -v`
Expected: FAIL because second-order solver is missing.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class SecondOrderSolution:
    ghx: np.ndarray
    ghu: np.ndarray
    ghxx: np.ndarray
    ghxu: np.ndarray
    ghuu: np.ndarray
    ghs2: np.ndarray

def solve_second_order(model, first_order=None):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver_second_order.py::test_second_order_solver_returns_expected_tensor_shapes -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/solver_second_order.py src/perturbation_py/derivatives.py src/perturbation_py/__init__.py tests/test_solver_second_order.py
git commit -m "feat: implement second-order perturbation tensors"
```

### Task 6: Third-order perturbation tensors (`ghxxx`, `ghxxu`, `ghxuu`, `ghuuu`)

**Files:**
- Create: `src/perturbation_py/solver_third_order.py`
- Modify: `src/perturbation_py/solver_second_order.py`
- Modify: `src/perturbation_py/__init__.py`
- Test: `tests/test_solver_third_order.py`

**Step 1: Write the failing test**

```python
def test_third_order_solver_emits_consistent_tensor_ranks():
    model = make_cubic_fixture_model()
    sol = solve_third_order(model)
    assert sol.ghxxx.ndim == 4
    assert sol.ghxxu.ndim == 4
    assert sol.ghxuu.ndim == 4
    assert sol.ghuuu.ndim == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver_third_order.py::test_third_order_solver_emits_consistent_tensor_ranks -v`
Expected: FAIL because third-order solver is missing.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class ThirdOrderSolution:
    second_order: SecondOrderSolution
    ghxxx: np.ndarray
    ghxxu: np.ndarray
    ghxuu: np.ndarray
    ghuuu: np.ndarray

def solve_third_order(model, second_order=None):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver_third_order.py::test_third_order_solver_emits_consistent_tensor_ranks -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/solver_third_order.py src/perturbation_py/solver_second_order.py src/perturbation_py/__init__.py tests/test_solver_third_order.py
git commit -m "feat: implement third-order perturbation tensors"
```

### Task 7: Unified policy object and tensor evaluation engine

**Files:**
- Create: `src/perturbation_py/policy.py`
- Modify: `src/perturbation_py/solver.py`
- Modify: `src/perturbation_py/solver_second_order.py`
- Modify: `src/perturbation_py/solver_third_order.py`
- Test: `tests/test_policy_evaluation.py`

**Step 1: Write the failing test**

```python
def test_policy_evaluator_matches_first_order_when_higher_terms_zero():
    sol1 = solve_first_order(make_scalar_model())
    policy = Policy.from_first_order(sol1)
    x = policy.controls(state=np.array([0.2]), shock=np.array([0.0]))
    assert np.allclose(x, sol1.policy @ np.array([0.2]))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_policy_evaluation.py::test_policy_evaluator_matches_first_order_when_higher_terms_zero -v`
Expected: FAIL because `Policy` abstraction is missing.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class Policy:
    order: int
    ghx: np.ndarray
    ghu: np.ndarray
    ghxx: np.ndarray | None = None
    ...

    def controls(self, state, shock):
        ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_policy_evaluation.py::test_policy_evaluator_matches_first_order_when_higher_terms_zero -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/policy.py src/perturbation_py/solver.py src/perturbation_py/solver_second_order.py src/perturbation_py/solver_third_order.py tests/test_policy_evaluation.py
git commit -m "feat: add unified perturbation policy evaluator"
```

### Task 8: Pruning simulation for higher-order solutions

**Files:**
- Create: `src/perturbation_py/pruning.py`
- Modify: `src/perturbation_py/simulation.py`
- Modify: `src/perturbation_py/policy.py`
- Test: `tests/test_pruning.py`

**Step 1: Write the failing test**

```python
def test_pruned_simulation_stays_finite_for_second_order_policy():
    policy = make_explosive_second_order_fixture_policy()
    result = simulate_pruned(policy, horizon=50, shock_std=0.01, seed=42)
    assert np.isfinite(result.controls).all()
    assert np.isfinite(result.states).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pruning.py::test_pruned_simulation_stays_finite_for_second_order_policy -v`
Expected: FAIL because pruning simulator is missing.

**Step 3: Write minimal implementation**

```python
def simulate_pruned(policy, horizon, shock_std, seed=None):
    rng = np.random.default_rng(seed)
    # Kim-Kim-Schaumburg-Sims style recursive decomposition
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pruning.py::test_pruned_simulation_stays_finite_for_second_order_policy -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/pruning.py src/perturbation_py/simulation.py src/perturbation_py/policy.py tests/test_pruning.py
git commit -m "feat: implement higher-order pruning simulation"
```

### Task 9: Dynare interoperability layer (`.mod` subset + reference IO)

**Files:**
- Create: `src/perturbation_py/io/__init__.py`
- Create: `src/perturbation_py/io/dynare_mod.py`
- Create: `src/perturbation_py/io/dynare_reference.py`
- Modify: `src/perturbation_py/cli.py`
- Test: `tests/test_dynare_io.py`
- Create: `tests/fixtures/dynare/minimal_rbc.mod`
- Create: `tests/fixtures/dynare/minimal_rbc_reference.json`

**Step 1: Write the failing test**

```python
def test_parse_minimal_dynare_mod_and_read_reference_json():
    model_spec = parse_dynare_mod_file("tests/fixtures/dynare/minimal_rbc.mod")
    ref = load_dynare_reference("tests/fixtures/dynare/minimal_rbc_reference.json")
    assert "k" in model_spec.endogenous
    assert ref.order == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynare_io.py::test_parse_minimal_dynare_mod_and_read_reference_json -v`
Expected: FAIL because Dynare IO adapters are missing.

**Step 3: Write minimal implementation**

```python
def parse_dynare_mod_file(path):
    text = Path(path).read_text(encoding="utf-8")
    ...

def load_dynare_reference(path):
    return DynareReference(**json.loads(Path(path).read_text(encoding="utf-8")))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dynare_io.py::test_parse_minimal_dynare_mod_and_read_reference_json -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/io/__init__.py src/perturbation_py/io/dynare_mod.py src/perturbation_py/io/dynare_reference.py src/perturbation_py/cli.py tests/test_dynare_io.py tests/fixtures/dynare/minimal_rbc.mod tests/fixtures/dynare/minimal_rbc_reference.json
git commit -m "feat: add Dynare mod subset parser and reference IO"
```

### Task 10: Parity benchmark harness against Dynare outputs

**Files:**
- Create: `src/perturbation_py/benchmarks.py`
- Create: `tests/integration/test_dynare_parity_order1.py`
- Create: `tests/integration/test_dynare_parity_order2.py`
- Create: `tests/fixtures/dynare/rbc_order1_reference.json`
- Create: `tests/fixtures/dynare/rbc_order2_reference.json`

**Step 1: Write the failing test**

```python
def test_order1_policy_matches_dynare_reference_within_tolerance():
    report = compare_with_dynare_fixture("rbc", order=1)
    assert report.max_abs_error_policy < 1e-6
    assert report.max_abs_error_transition < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_dynare_parity_order1.py::test_order1_policy_matches_dynare_reference_within_tolerance -v`
Expected: FAIL because benchmark comparison utility is missing.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class ParityReport:
    max_abs_error_policy: float
    max_abs_error_transition: float

def compare_with_dynare_fixture(model_name, order):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_dynare_parity_order1.py::test_order1_policy_matches_dynare_reference_within_tolerance -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/benchmarks.py tests/integration/test_dynare_parity_order1.py tests/integration/test_dynare_parity_order2.py tests/fixtures/dynare/rbc_order1_reference.json tests/fixtures/dynare/rbc_order2_reference.json
git commit -m "test: add Dynare parity benchmark harness"
```

### Task 11: Full CLI workflow (`steady-state`, `solve`, `simulate`, `irf`)

**Files:**
- Modify: `src/perturbation_py/cli.py`
- Create: `tests/test_cli_workflow.py`
- Create: `tests/fixtures/models/scalar_model.py`

**Step 1: Write the failing test**

```python
def test_cli_solve_and_irf_commands_produce_output_files(tmp_path):
    result = runner.invoke(app, [
        "solve",
        "--model", "tests/fixtures/models/scalar_model.py",
        "--order", "2",
        "--output", str(tmp_path / "sol.json"),
    ])
    assert result.exit_code == 0
    assert (tmp_path / "sol.json").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_workflow.py::test_cli_solve_and_irf_commands_produce_output_files -v`
Expected: FAIL because command set is incomplete.

**Step 3: Write minimal implementation**

```python
@app.command("solve")
def solve_cmd(model: str, order: int, output: str):
    ...

@app.command("irf")
def irf_cmd(solution: str, horizon: int, output: str):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_workflow.py::test_cli_solve_and_irf_commands_produce_output_files -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/cli.py tests/test_cli_workflow.py tests/fixtures/models/scalar_model.py
git commit -m "feat: add end-to-end CLI commands for perturbation workflow"
```

### Task 12: Documentation, examples, and CI quality gates

**Files:**
- Modify: `README.md`
- Create: `docs/user-guide.md`
- Create: `docs/math-notes.md`
- Create: `docs/validation.md`
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_docs_claims.py`

**Step 1: Write the failing test**

```python
def test_readme_mentions_orders_1_2_3_and_pruning():
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "order=1" in text
    assert "order=2" in text
    assert "order=3" in text
    assert "pruning" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_claims.py::test_readme_mentions_orders_1_2_3_and_pruning -v`
Expected: FAIL because docs do not yet reflect full implementation scope.

**Step 3: Write minimal implementation**

```markdown
## Features
- order=1, order=2, order=3 perturbation
- pruning simulation for higher-order policies
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_docs_claims.py::test_readme_mentions_orders_1_2_3_and_pruning -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add README.md docs/user-guide.md docs/math-notes.md docs/validation.md .github/workflows/ci.yml tests/test_docs_claims.py
git commit -m "docs: publish full perturbation documentation and CI gates"
```

### Task 13: Final integration gate and release candidate packaging

**Files:**
- Modify: `pyproject.toml`
- Create: `CHANGELOG.md`
- Create: `src/perturbation_py/version.py`
- Test: `tests/test_release_smoke.py`

**Step 1: Write the failing test**

```python
def test_release_smoke_import_and_cli_help():
    import perturbation_py
    assert perturbation_py.__version__.count(".") == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_release_smoke.py::test_release_smoke_import_and_cli_help -v`
Expected: FAIL because release metadata and changelog are incomplete.

**Step 3: Write minimal implementation**

```python
__version__ = "0.2.0"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_release_smoke.py::test_release_smoke_import_and_cli_help -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml CHANGELOG.md src/perturbation_py/version.py tests/test_release_smoke.py
git commit -m "chore: prepare release candidate metadata"
```

## Completion criteria (must all be true)

1. `pytest -q` passes for unit + integration suites.
2. `pytest tests/integration -q` passes with Dynare parity fixtures.
3. CLI supports solving and simulation for orders 1/2/3 plus pruning.
4. Documentation includes math notes, API guide, and parity validation report.
5. The package can be installed with `pip install -e .` and run with `perturbation-py --help`.

## Execution discipline

- Required implementation flow skill: `@superpowers/executing-plans`
- Required coding workflow skill for each task: `@superpowers/test-driven-development`
- Required pre-completion skill before final claim: `@superpowers/verification-before-completion`
