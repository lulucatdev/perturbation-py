from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence

import numpy as np
from scipy.io import loadmat

from .io.dynare_reference import DynareReference
from .model import DSGEModel
from .policy import Policy
from .solver import FirstOrderSolution, solve_first_order
from .solver_second_order import SecondOrderSolution, solve_second_order

Array = np.ndarray


def dynare_command_from_env(default: str = "dynare") -> str:
    return os.environ.get("PERTURBATION_DYNARE_CMD", default)


_DYNARE_RUNTIME_CACHE: dict[str, bool] = {}


def dynare_available(command: str | None = None, *, check_runtime: bool = True) -> bool:
    cmd = command if command is not None else dynare_command_from_env()
    parts = shlex.split(cmd)
    if not parts:
        return False

    executable = parts[0]
    if os.path.isabs(executable):
        if not (Path(executable).exists() and os.access(executable, os.X_OK)):
            return False
    elif shutil.which(executable) is None:
        return False

    if not check_runtime:
        return True

    cached = _DYNARE_RUNTIME_CACHE.get(cmd)
    if cached is not None:
        return cached

    try:
        probe = subprocess.run(
            parts + ["--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        ok = probe.returncode == 0
    except Exception:
        ok = False

    _DYNARE_RUNTIME_CACHE[cmd] = ok
    return ok


def _as_float_matrix(array: Array | list[list[float]]) -> Array:
    out = np.asarray(array, dtype=float)
    if out.ndim != 2:
        raise ValueError("Expected 2D matrix")
    return out


@dataclass(frozen=True)
class LinearRESpec:
    name: str
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    shock_names: tuple[str, ...]
    g_s: Array
    g_x: Array
    g_e: Array
    f_s: Array
    f_x: Array
    f_S: Array
    f_X: Array
    reference_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        g_s = _as_float_matrix(self.g_s)
        g_x = _as_float_matrix(self.g_x)
        g_e = _as_float_matrix(self.g_e)
        f_s = _as_float_matrix(self.f_s)
        f_x = _as_float_matrix(self.f_x)
        f_S = _as_float_matrix(self.f_S)
        f_X = _as_float_matrix(self.f_X)

        n_s = len(self.state_names)
        n_x = len(self.control_names)
        n_e = len(self.shock_names)

        expected = {
            "g_s": (n_s, n_s),
            "g_x": (n_s, n_x),
            "g_e": (n_s, n_e),
            "f_s": (n_x, n_s),
            "f_x": (n_x, n_x),
            "f_S": (n_x, n_s),
            "f_X": (n_x, n_x),
        }
        actual = {
            "g_s": g_s.shape,
            "g_x": g_x.shape,
            "g_e": g_e.shape,
            "f_s": f_s.shape,
            "f_x": f_x.shape,
            "f_S": f_S.shape,
            "f_X": f_X.shape,
        }
        for key, shape in expected.items():
            if actual[key] != shape:
                raise ValueError(
                    f"{key} shape mismatch: expected {shape}, got {actual[key]}"
                )

        object.__setattr__(self, "g_s", g_s)
        object.__setattr__(self, "g_x", g_x)
        object.__setattr__(self, "g_e", g_e)
        object.__setattr__(self, "f_s", f_s)
        object.__setattr__(self, "f_x", f_x)
        object.__setattr__(self, "f_S", f_S)
        object.__setattr__(self, "f_X", f_X)

    @property
    def endogenous_names(self) -> tuple[str, ...]:
        return self.state_names + self.control_names

    def to_model(self) -> DSGEModel:
        def transition(s: Array, x: Array, e: Array, _: Mapping[str, float]) -> Array:
            return self.g_s @ s + self.g_x @ x + self.g_e @ e

        def arbitrage(
            s: Array,
            x: Array,
            s_next: Array,
            x_next: Array,
            _: Mapping[str, float],
        ) -> Array:
            return self.f_s @ s + self.f_x @ x + self.f_S @ s_next + self.f_X @ x_next

        return DSGEModel(
            state_names=self.state_names,
            control_names=self.control_names,
            shock_names=self.shock_names,
            parameters={},
            steady_state_states=np.zeros(len(self.state_names), dtype=float),
            steady_state_controls=np.zeros(len(self.control_names), dtype=float),
            transition=transition,
            arbitrage=arbitrage,
        )

    def to_dynare_mod(self) -> str:
        lines: list[str] = []
        lines.append(f"var {' '.join(self.endogenous_names)};")
        lines.append(f"varexo {' '.join(self.shock_names)};")
        if self.state_names:
            lines.append(f"predetermined_variables {' '.join(self.state_names)};")
        lines.append("")
        lines.append("model(linear);")

        for i, state in enumerate(self.state_names):
            rhs_terms: list[str] = []
            rhs_terms.extend(_terms_from_row(self.g_s[i], list(self.state_names)))
            rhs_terms.extend(_terms_from_row(self.g_x[i], list(self.control_names)))
            rhs_terms.extend(_terms_from_row(self.g_e[i], list(self.shock_names)))
            rhs = " + ".join(rhs_terms) if rhs_terms else "0"
            lines.append(f"  {state}(+1) = {rhs};")

        for i, _control in enumerate(self.control_names):
            expr_terms: list[str] = []
            expr_terms.extend(_terms_from_row(self.f_s[i], list(self.state_names)))
            expr_terms.extend(_terms_from_row(self.f_x[i], list(self.control_names)))
            expr_terms.extend(
                _terms_from_row(self.f_S[i], [f"{n}(+1)" for n in self.state_names])
            )
            expr_terms.extend(
                _terms_from_row(self.f_X[i], [f"{n}(+1)" for n in self.control_names])
            )
            expr = " + ".join(expr_terms) if expr_terms else "0"
            lines.append(f"  {expr} = 0;")

        lines.append("end;")
        lines.append("")
        lines.append("initval;")
        for name in self.endogenous_names:
            lines.append(f"  {name} = 0;")
        for name in self.shock_names:
            lines.append(f"  {name} = 0;")
        lines.append("end;")
        lines.append("")
        lines.append("steady;")
        lines.append("check;")
        lines.append("")
        lines.append("shocks;")
        for name in self.shock_names:
            lines.append(f"  var {name}; stderr 1;")
        lines.append("end;")
        lines.append("")
        lines.append(
            "stoch_simul(order=1, irf=0, noprint, nograph, nocorr, nomoments);"
        )
        lines.append("")
        return "\n".join(lines)


def _terms_from_row(coeffs: Array, symbols: list[str], tol: float = 1e-14) -> list[str]:
    terms: list[str] = []
    for coeff, symbol in zip(coeffs.tolist(), symbols):
        if abs(coeff) <= tol:
            continue
        terms.append(f"{coeff:.16g}*{symbol}")
    return terms


@dataclass(frozen=True)
class DynareLinearSolution:
    transition: Array
    policy: Array
    state_shock_impact: Array
    control_shock_impact: Array


@dataclass(frozen=True)
class ParityReport:
    model_name: str
    max_abs_error_transition: float
    max_abs_error_policy: float
    max_abs_error_state_shock: float
    max_abs_error_control_shock: float
    dynare_workdir: Path


@dataclass(frozen=True)
class DynareFixtureReport:
    mod_file: Path
    return_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ReferenceComparison:
    max_abs_error_ghx: float | None
    max_abs_error_ghu: float | None
    max_abs_error_ghxx: float | None
    max_abs_error_ghxu: float | None
    max_abs_error_ghuu: float | None


def compare_first_order_to_dynare(
    spec: LinearRESpec,
    *,
    dynare_command: str | None = None,
    timeout_sec: int = 300,
) -> ParityReport:
    if not dynare_available(dynare_command):
        raise RuntimeError(
            "Dynare command is not available. Set PERTURBATION_DYNARE_CMD or install Dynare."
        )

    py_solution = solve_first_order(spec.to_model())

    with tempfile.TemporaryDirectory(prefix=f"{spec.name}_dynare_") as tmp:
        workdir = Path(tmp)
        mod_path = workdir / f"{spec.name}.mod"
        mod_path.write_text(spec.to_dynare_mod(), encoding="utf-8")

        _run_dynare_file(
            mod_path=mod_path, command=dynare_command, timeout_sec=timeout_sec
        )
        dyn_solution = _read_dynare_solution_from_mat(spec, workdir)

        return ParityReport(
            model_name=spec.name,
            max_abs_error_transition=float(
                np.max(np.abs(dyn_solution.transition - py_solution.transition))
            ),
            max_abs_error_policy=float(
                np.max(np.abs(dyn_solution.policy - py_solution.policy))
            ),
            max_abs_error_state_shock=float(
                np.max(
                    np.abs(dyn_solution.state_shock_impact - py_solution.shock_impact)
                )
            ),
            max_abs_error_control_shock=float(
                np.max(
                    np.abs(
                        dyn_solution.control_shock_impact
                        - py_solution.control_shock_impact
                    )
                )
            ),
            dynare_workdir=workdir,
        )


def compare_with_dynare_fixture(
    model_name: str,
    *,
    order: int = 1,
    suite: Sequence[LinearRESpec] | None = None,
    dynare_command: str | None = None,
    timeout_sec: int = 300,
) -> ParityReport:
    if order != 1:
        raise NotImplementedError(
            "compare_with_dynare_fixture currently supports order=1"
        )

    specs = list(suite) if suite is not None else _default_fixture_suite()
    spec_map = {spec.name: spec for spec in specs}
    if model_name not in spec_map:
        raise KeyError(f"Unknown fixture model: {model_name}")
    return compare_first_order_to_dynare(
        spec_map[model_name],
        dynare_command=dynare_command,
        timeout_sec=timeout_sec,
    )


def _run_dynare_file(
    *, mod_path: Path, command: str | None = None, timeout_sec: int = 300
) -> None:
    dyn_cmd = command if command is not None else dynare_command_from_env()
    cmd = shlex.split(dyn_cmd) + [mod_path.name, "nolog", "noclearall"]
    proc = subprocess.run(
        cmd,
        cwd=mod_path.parent,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        message = (
            f"Dynare run failed for {mod_path.name}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
        raise RuntimeError(message)


def run_dynare_mod_file(
    mod_file: str | Path,
    *,
    dynare_command: str | None = None,
    timeout_sec: int = 600,
) -> DynareFixtureReport:
    mod_path = Path(mod_file)
    if not mod_path.exists():
        raise FileNotFoundError(f"Dynare mod file not found: {mod_path}")
    if not dynare_available(dynare_command):
        raise RuntimeError(
            "Dynare command is not available. Set PERTURBATION_DYNARE_CMD or install Dynare."
        )

    dyn_cmd = (
        dynare_command if dynare_command is not None else dynare_command_from_env()
    )
    cmd = shlex.split(dyn_cmd) + [mod_path.name, "nolog", "noclearall"]
    proc = subprocess.run(
        cmd,
        cwd=mod_path.parent,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    return DynareFixtureReport(
        mod_file=mod_path,
        return_code=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def compare_policy_to_dynare_reference(
    policy: Policy, reference: DynareReference
) -> ReferenceComparison:
    return ReferenceComparison(
        max_abs_error_ghx=_max_abs_error(policy.ghx, reference.ghx),
        max_abs_error_ghu=_max_abs_error(policy.ghu, reference.ghu),
        max_abs_error_ghxx=_max_abs_error(policy.ghxx, reference.ghxx),
        max_abs_error_ghxu=_max_abs_error(policy.ghxu, reference.ghxu),
        max_abs_error_ghuu=_max_abs_error(policy.ghuu, reference.ghuu),
    )


@dataclass(frozen=True)
class SecondOrderParityReport:
    """Comparison report for second-order perturbation solutions.

    All ``max_abs_error_*`` fields report the element-wise maximum absolute
    difference between Python and Dynare solutions for the control-variable
    rows of each decision-rule tensor.
    """
    model_name: str
    max_abs_error_ghx: float
    max_abs_error_ghu: float
    max_abs_error_ghxx: float
    max_abs_error_ghxu: float
    max_abs_error_ghuu: float
    max_abs_error_ghs2: float


def compare_second_order_to_dynare(
    model: DSGEModel,
    mod_text: str,
    model_name: str,
    *,
    dynare_command: str | None = None,
    timeout_sec: int = 300,
    method: str = "sylvester",
) -> SecondOrderParityReport:
    """Compare Python second-order solution against Dynare ``stoch_simul(order=2)``.

    Parameters
    ----------
    model : DSGEModel
        Python model (deviation form, with steady state at zero).
    mod_text : str
        Complete Dynare ``.mod`` file text (level form, with ``stoch_simul(order=2, ...)``).
    model_name : str
        Model identifier (used for temp file naming and .mat lookup).
    dynare_command : str, optional
        Dynare invocation command.
    method : str
        Python solver method (``"sylvester"`` or ``"local_implicit"``).

    Returns
    -------
    SecondOrderParityReport
    """
    if not dynare_available(dynare_command):
        raise RuntimeError(
            "Dynare command is not available. Set PERTURBATION_DYNARE_CMD or install Dynare."
        )

    py_sol = solve_second_order(model, method=method)

    with tempfile.TemporaryDirectory(prefix=f"{model_name}_dynare_o2_") as tmp:
        workdir = Path(tmp)
        mod_path = workdir / f"{model_name}.mod"
        mod_path.write_text(mod_text, encoding="utf-8")

        _run_dynare_file(
            mod_path=mod_path, command=dynare_command, timeout_sec=timeout_sec
        )
        dyn = _read_dynare_second_order_from_mat(
            workdir,
            model_name,
            state_names=list(model.state_names),
            control_names=list(model.control_names),
            shock_names=list(model.shock_names),
            endogenous_names=list(model.endogenous_names),
        )

        return SecondOrderParityReport(
            model_name=model_name,
            max_abs_error_ghx=float(np.max(np.abs(dyn["ghx"] - py_sol.ghx))),
            max_abs_error_ghu=float(np.max(np.abs(dyn["ghu"] - py_sol.ghu))),
            max_abs_error_ghxx=float(np.max(np.abs(dyn["ghxx"] - py_sol.ghxx))),
            max_abs_error_ghxu=float(np.max(np.abs(dyn["ghxu"] - py_sol.ghxu))),
            max_abs_error_ghuu=float(np.max(np.abs(dyn["ghuu"] - py_sol.ghuu))),
            max_abs_error_ghs2=float(np.max(np.abs(dyn["ghs2"] - py_sol.ghs2))),
        )


def _read_dynare_second_order_from_mat(
    workdir: Path,
    model_name: str,
    state_names: list[str],
    control_names: list[str],
    shock_names: list[str],
    endogenous_names: list[str],
) -> dict[str, Array]:
    """Extract second-order decision-rule tensors from Dynare's ``.mat`` output.

    Dynare stores ``dr.ghxx`` as ``(n_endo, nspred^2)``, ``dr.ghxu`` as
    ``(n_endo, nspred*nexo)``, etc.  This function reshapes them to 3-D
    tensors and reindexes from Dynare's internal DR ordering to the user's
    declaration ordering, returning only the control-variable rows.

    Returns
    -------
    dict with keys ``"ghx"``, ``"ghu"``, ``"ghxx"``, ``"ghxu"``, ``"ghuu"``, ``"ghs2"``
    """
    n_endo = len(endogenous_names)
    n_state = len(state_names)
    n_control = len(control_names)
    n_shock = len(shock_names)

    mat_file = _locate_results_mat(workdir, model_name)
    data = loadmat(mat_file, squeeze_me=False, struct_as_record=False)
    oo = data["oo_"][0, 0]
    dr = oo.dr[0, 0]

    # Variable ordering maps
    order_var = np.asarray(dr.order_var, dtype=int).reshape(-1) - 1
    state_var = np.asarray(dr.state_var, dtype=int).reshape(-1) - 1
    nspred = state_var.size

    row_for_decl = {decl_idx: row_idx for row_idx, decl_idx in enumerate(order_var)}
    col_for_state_decl = {decl_idx: col_idx for col_idx, decl_idx in enumerate(state_var)}

    # Load raw matrices
    raw_ghx = np.asarray(dr.ghx, dtype=float)  # (n_endo, nspred)
    raw_ghu = np.asarray(dr.ghu, dtype=float)  # (n_endo, n_shock)
    raw_ghxx = np.asarray(dr.ghxx, dtype=float)  # (n_endo, nspred^2)
    raw_ghxu = np.asarray(dr.ghxu, dtype=float)  # (n_endo, nspred*n_shock)
    raw_ghuu = np.asarray(dr.ghuu, dtype=float)  # (n_endo, n_shock^2)
    raw_ghs2 = np.asarray(dr.ghs2, dtype=float).reshape(n_endo)  # (n_endo,)

    # Reshape 2D → 3D tensors (Dynare uses kron ordering, C-order reshape works)
    raw_ghxx_3d = raw_ghxx.reshape(n_endo, nspred, nspred)
    raw_ghxu_3d = raw_ghxu.reshape(n_endo, nspred, n_shock)
    raw_ghuu_3d = raw_ghuu.reshape(n_endo, n_shock, n_shock)

    # Extract control rows, reindexing state columns to match user ordering
    ghx = np.zeros((n_control, n_state), dtype=float)
    ghu = np.zeros((n_control, n_shock), dtype=float)
    ghxx = np.zeros((n_control, n_state, n_state), dtype=float)
    ghxu = np.zeros((n_control, n_state, n_shock), dtype=float)
    ghuu = np.zeros((n_control, n_shock, n_shock), dtype=float)
    ghs2 = np.zeros(n_control, dtype=float)

    for i, ctrl_name in enumerate(control_names):
        decl_idx = endogenous_names.index(ctrl_name)
        dr_row = row_for_decl[decl_idx]

        ghs2[i] = raw_ghs2[dr_row]

        for k in range(n_shock):
            ghu[i, k] = raw_ghu[dr_row, k]
            for l in range(n_shock):
                ghuu[i, k, l] = raw_ghuu_3d[dr_row, k, l]

        for j, state_name in enumerate(state_names):
            state_decl = endogenous_names.index(state_name)
            sc = col_for_state_decl[state_decl]

            ghx[i, j] = raw_ghx[dr_row, sc]

            for k in range(n_shock):
                ghxu[i, j, k] = raw_ghxu_3d[dr_row, sc, k]

            for j2, state_name2 in enumerate(state_names):
                state_decl2 = endogenous_names.index(state_name2)
                sc2 = col_for_state_decl[state_decl2]
                ghxx[i, j, j2] = raw_ghxx_3d[dr_row, sc, sc2]

    return {
        "ghx": ghx,
        "ghu": ghu,
        "ghxx": ghxx,
        "ghxu": ghxu,
        "ghuu": ghuu,
        "ghs2": ghs2,
    }


def _read_dynare_solution_from_mat(
    spec: LinearRESpec, workdir: Path
) -> DynareLinearSolution:
    n_endo = len(spec.endogenous_names)
    n_state = len(spec.state_names)
    n_control = len(spec.control_names)
    n_shock = len(spec.shock_names)

    mat_file = _locate_results_mat(workdir, spec.name)

    data = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    oo = data["oo_"]
    dr = oo.dr

    ghx = _as_2d(np.asarray(dr.ghx, dtype=float), rows=n_endo, cols=n_state)
    ghu = _as_2d(np.asarray(dr.ghu, dtype=float), rows=n_endo, cols=n_shock)
    order_var = np.asarray(dr.order_var, dtype=int).reshape(-1) - 1
    state_var = np.asarray(dr.state_var, dtype=int).reshape(-1) - 1

    row_for_decl_idx = {decl_idx: row_idx for row_idx, decl_idx in enumerate(order_var)}
    col_for_state_decl_idx = {
        decl_idx: col_idx for col_idx, decl_idx in enumerate(state_var)
    }

    endo_decl = list(spec.endogenous_names)
    exo_decl = list(spec.shock_names)

    transition = np.zeros((n_state, n_state), dtype=float)
    policy = np.zeros((n_control, n_state), dtype=float)
    state_shock = np.zeros((n_state, n_shock), dtype=float)
    control_shock = np.zeros((n_control, n_shock), dtype=float)

    for i, state_name in enumerate(spec.state_names):
        decl_idx = endo_decl.index(state_name)
        row = row_for_decl_idx[decl_idx]

        for j, state_name_col in enumerate(spec.state_names):
            decl_col = endo_decl.index(state_name_col)
            col = col_for_state_decl_idx[decl_col]
            transition[i, j] = ghx[row, col]

        for k, shock_name in enumerate(spec.shock_names):
            shock_col = exo_decl.index(shock_name)
            state_shock[i, k] = ghu[row, shock_col]

    for i, control_name in enumerate(spec.control_names):
        decl_idx = endo_decl.index(control_name)
        row = row_for_decl_idx[decl_idx]

        for j, state_name_col in enumerate(spec.state_names):
            decl_col = endo_decl.index(state_name_col)
            col = col_for_state_decl_idx[decl_col]
            policy[i, j] = ghx[row, col]

        for k, shock_name in enumerate(spec.shock_names):
            shock_col = exo_decl.index(shock_name)
            control_shock[i, k] = ghu[row, shock_col]

    return DynareLinearSolution(
        transition=transition,
        policy=policy,
        state_shock_impact=state_shock,
        control_shock_impact=control_shock,
    )


def _locate_results_mat(workdir: Path, model_name: str) -> Path:
    candidates = [
        workdir / f"{model_name}_results.mat",
        workdir / model_name / "Output" / f"{model_name}_results.mat",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = list(workdir.rglob(f"{model_name}_results.mat"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Dynare output missing: {model_name}_results.mat under {workdir}"
    )


def _as_2d(array: Array, *, rows: int, cols: int) -> Array:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if cols == 1:
            arr = arr.reshape(rows, 1)
        elif rows == 1:
            arr = arr.reshape(1, cols)
        else:
            arr = arr.reshape(rows, cols)
    if arr.shape != (rows, cols):
        raise ValueError(f"Expected shape {(rows, cols)}, got {arr.shape}")
    return arr


def _read_ascii_vector(path: Path) -> Array:
    if not path.exists():
        raise FileNotFoundError(f"Dynare output missing: {path}")
    data = np.loadtxt(path, dtype=float)
    return np.asarray(data, dtype=float).reshape(-1)


def _default_fixture_suite() -> list[LinearRESpec]:
    try:
        from fixtures.dynare_parity_suite import PARITY_SUITE  # type: ignore[import-not-found]

        return list(PARITY_SUITE)
    except Exception as exc:
        raise RuntimeError(
            "No default fixture suite available. Provide suite=... explicitly."
        ) from exc


def _max_abs_error(lhs: Array | None, rhs: Array | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    lhs_arr = np.asarray(lhs, dtype=float)
    rhs_arr = np.asarray(rhs, dtype=float)
    if lhs_arr.shape != rhs_arr.shape:
        raise ValueError(
            f"Shape mismatch in reference comparison: {lhs_arr.shape} vs {rhs_arr.shape}"
        )
    return float(np.max(np.abs(lhs_arr - rhs_arr)))


def _read_ascii_matrix(path: Path, rows: int, cols: int) -> Array:
    if not path.exists():
        raise FileNotFoundError(f"Dynare output missing: {path}")
    data = np.loadtxt(path, dtype=float)
    arr = np.asarray(data, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if cols == 1:
            arr = arr.reshape(rows, 1)
        elif rows == 1:
            arr = arr.reshape(1, cols)
        else:
            arr = arr.reshape(rows, cols)

    if arr.shape != (rows, cols):
        raise ValueError(
            f"Matrix shape mismatch for {path.name}: expected {(rows, cols)}, got {arr.shape}"
        )
    return arr
