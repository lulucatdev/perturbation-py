# perturbation_py Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an open-source Python package that ports Dynare-style first-order perturbation workflows for DSGE models, with docs, tests, and runnable examples.

**Architecture:** The package uses a state-control model interface (`transition` + `arbitrage`), finite-difference derivatives around steady state, and a generalized Schur (QZ) solver to compute first-order policy/transition matrices. A simulation layer then produces paths and impulse responses from solved linear policy rules.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pytest, Typer (CLI), PEP 621 (`pyproject.toml`)

---

### Task 1: Scaffold package and metadata

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `LICENSE`
- Create: `src/perturbation_py/__init__.py`
- Create: `src/perturbation_py/cli.py`
- Test: `tests/test_package_smoke.py`

**Step 1: Write the failing test**

```python
def test_package_import_and_version():
    import perturbation_py
    assert hasattr(perturbation_py, "__version__")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_smoke.py::test_package_import_and_version -v`
Expected: FAIL with `ModuleNotFoundError`.

**Step 3: Write minimal implementation**

```python
__version__ = "0.1.0"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_smoke.py::test_package_import_and_version -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml README.md LICENSE src/perturbation_py/__init__.py src/perturbation_py/cli.py tests/test_package_smoke.py
git commit -m "feat: initialize perturbation_py package skeleton"
```

### Task 2: Add model contracts and derivative engine

**Files:**
- Create: `src/perturbation_py/model.py`
- Create: `src/perturbation_py/derivatives.py`
- Test: `tests/test_derivatives.py`

**Step 1: Write the failing test**

```python
def test_finite_difference_shapes_for_rbc_fixture():
    jac = compute_jacobians(rbc_fixture())
    assert jac.g_s.shape == (2, 2)
    assert jac.f_X.shape == (1, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_derivatives.py::test_finite_difference_shapes_for_rbc_fixture -v`
Expected: FAIL because derivative functions are missing.

**Step 3: Write minimal implementation**

```python
def central_difference_jacobian(func, x):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_derivatives.py::test_finite_difference_shapes_for_rbc_fixture -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/model.py src/perturbation_py/derivatives.py tests/test_derivatives.py
git commit -m "feat: add model interface and numerical derivatives"
```

### Task 3: Implement Dynare-style first-order perturbation solver

**Files:**
- Create: `src/perturbation_py/solver.py`
- Modify: `src/perturbation_py/__init__.py`
- Test: `tests/test_solver_first_order.py`

**Step 1: Write the failing test**

```python
def test_first_order_solver_returns_stable_solution_for_rbc():
    sol = solve_first_order(rbc_fixture())
    assert sol.policy.shape == (1, 2)
    assert sol.transition.shape == (2, 2)
    assert sol.stable
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver_first_order.py::test_first_order_solver_returns_stable_solution_for_rbc -v`
Expected: FAIL because solver is missing.

**Step 3: Write minimal implementation**

```python
def solve_first_order(model):
    # build pencil and solve with scipy.linalg.ordqz
    return solution
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver_first_order.py::test_first_order_solver_returns_stable_solution_for_rbc -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/solver.py src/perturbation_py/__init__.py tests/test_solver_first_order.py
git commit -m "feat: implement first-order perturbation solver with BK checks"
```

### Task 4: Add simulation and IRF utilities

**Files:**
- Create: `src/perturbation_py/simulation.py`
- Test: `tests/test_simulation.py`

**Step 1: Write the failing test**

```python
def test_irf_decays_under_stable_transition():
    irf = impulse_response(solution, horizon=40, shock_index=0, shock_size=1.0)
    assert abs(irf.states[-1, 1]) < abs(irf.states[1, 1])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation.py::test_irf_decays_under_stable_transition -v`
Expected: FAIL because simulation functions are missing.

**Step 3: Write minimal implementation**

```python
def simulate_linear(solution, initial_state, shocks):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation.py::test_irf_decays_under_stable_transition -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/perturbation_py/simulation.py tests/test_simulation.py
git commit -m "feat: add simulation and impulse response utilities"
```

### Task 5: Documentation and reference catalog

**Files:**
- Create: `references/README.md`
- Create: `docs/feasibility.md`
- Create: `docs/contribution-map.md`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
def test_readme_mentions_first_order_scope():
    text = Path("README.md").read_text()
    assert "first-order" in text.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_smoke.py::test_readme_mentions_first_order_scope -v`
Expected: FAIL because docs are missing.

**Step 3: Write minimal implementation**

```markdown
# perturbation_py
First-order perturbation package inspired by Dynare.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_smoke.py::test_readme_mentions_first_order_scope -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add references/README.md docs/feasibility.md docs/contribution-map.md README.md
git commit -m "docs: add feasibility analysis and reference catalog"
```
