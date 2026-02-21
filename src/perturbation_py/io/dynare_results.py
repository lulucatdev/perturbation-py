from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

Array = np.ndarray


@dataclass(frozen=True)
class DynareResults:
    mat_path: Path
    model_name: str
    endogenous_names: tuple[str, ...]
    exogenous_names: tuple[str, ...]
    order_var: Array
    state_var: Array
    ghx: Array
    ghu: Array
    irfs: dict[str, Array]

    @property
    def n_endogenous(self) -> int:
        return len(self.endogenous_names)

    @property
    def n_exogenous(self) -> int:
        return len(self.exogenous_names)

    @property
    def n_states(self) -> int:
        return int(self.state_var.size)


@dataclass(frozen=True)
class IRFComparisonReport:
    model_name: str
    max_abs_error: float
    per_series_max_abs_error: dict[str, float]
    shock_scales: dict[str, float]


def load_dynare_results(path: str | Path) -> DynareResults:
    mat_path = Path(path)
    payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    M = payload["M_"]
    oo = payload["oo_"]
    dr = oo.dr

    endogenous_names = tuple(_to_name_tuple(M.endo_names))
    exogenous_names = tuple(_to_name_tuple(M.exo_names))

    ghx = np.asarray(dr.ghx, dtype=float)
    ghu = np.asarray(dr.ghu, dtype=float)
    order_var = np.asarray(dr.order_var, dtype=int).reshape(-1) - 1
    state_var = np.asarray(dr.state_var, dtype=int).reshape(-1) - 1

    irfs = _extract_irf_dict(oo.irfs)

    return DynareResults(
        mat_path=mat_path,
        model_name=mat_path.stem.replace("_results", ""),
        endogenous_names=endogenous_names,
        exogenous_names=exogenous_names,
        order_var=order_var,
        state_var=state_var,
        ghx=ghx,
        ghu=ghu,
        irfs=irfs,
    )


def compare_reconstructed_irfs_to_dynare(results: DynareResults) -> IRFComparisonReport:
    endo_to_decl = {name: idx for idx, name in enumerate(results.endogenous_names)}
    row_for_decl = {
        decl_idx: row_idx for row_idx, decl_idx in enumerate(results.order_var.tolist())
    }

    per_series: dict[str, float] = {}
    shock_scales: dict[str, float] = {}

    for shock_idx, shock_name in enumerate(results.exogenous_names):
        keys = [
            key
            for key in results.irfs.keys()
            if key.endswith(f"_{shock_name}")
            and key[: -len(shock_name) - 1] in endo_to_decl
        ]
        if not keys:
            continue

        scale = _infer_shock_scale(
            results=results,
            shock_idx=shock_idx,
            shock_name=shock_name,
            keys=keys,
            endo_to_decl=endo_to_decl,
            row_for_decl=row_for_decl,
        )
        shock_scales[shock_name] = scale

        horizon = max(int(np.asarray(results.irfs[k]).reshape(-1).size) for k in keys)
        simulated = _simulate_irf_by_shock(
            results=results,
            shock_idx=shock_idx,
            horizon=horizon,
            shock_size=scale,
        )

        for key in keys:
            var_name = key[: -len(shock_name) - 1]
            dynare_series = np.asarray(results.irfs[key], dtype=float).reshape(-1)
            py_series = simulated[var_name][: dynare_series.size]
            per_series[key] = float(np.max(np.abs(py_series - dynare_series)))

    max_abs_error = max(per_series.values()) if per_series else 0.0
    return IRFComparisonReport(
        model_name=results.model_name,
        max_abs_error=max_abs_error,
        per_series_max_abs_error=per_series,
        shock_scales=shock_scales,
    )


def results_mat_path_for_mod(mod_file: str | Path) -> Path:
    mod_path = Path(mod_file)
    candidates = [
        mod_path.parent / f"{mod_path.stem}_results.mat",
        mod_path.parent / mod_path.stem / "Output" / f"{mod_path.stem}_results.mat",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = list(mod_path.parent.rglob(f"{mod_path.stem}_results.mat"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate Dynare results MAT for {mod_path}")


def _simulate_irf_by_shock(
    *,
    results: DynareResults,
    shock_idx: int,
    horizon: int,
    shock_size: float,
) -> dict[str, Array]:
    n_endo = results.n_endogenous
    n_shocks = results.n_exogenous

    state = np.zeros(results.n_states, dtype=float)
    out = {name: np.zeros(horizon, dtype=float) for name in results.endogenous_names}

    for t in range(horizon):
        innovation = np.zeros(n_shocks, dtype=float)
        if t == 0:
            innovation[shock_idx] = shock_size

        y_order = results.ghx @ state + results.ghu @ innovation
        y_decl = np.zeros(n_endo, dtype=float)
        y_decl[results.order_var] = y_order

        for i, name in enumerate(results.endogenous_names):
            out[name][t] = y_decl[i]

        state = y_decl[results.state_var]

    return out


def _infer_shock_scale(
    *,
    results: DynareResults,
    shock_idx: int,
    shock_name: str,
    keys: list[str],
    endo_to_decl: dict[str, int],
    row_for_decl: dict[int, int],
) -> float:
    scales: list[float] = []
    for key in keys:
        var_name = key[: -len(shock_name) - 1]
        decl_idx = endo_to_decl[var_name]
        row_idx = row_for_decl[decl_idx]
        unit_response = float(results.ghu[row_idx, shock_idx])
        if abs(unit_response) <= 1e-14:
            continue
        dyn_first = float(np.asarray(results.irfs[key], dtype=float).reshape(-1)[0])
        scales.append(dyn_first / unit_response)

    if not scales:
        return 1.0
    return float(np.median(np.asarray(scales, dtype=float)))


def _to_name_tuple(names_obj: Any) -> tuple[str, ...]:
    arr = np.asarray(names_obj).reshape(-1)
    return tuple(str(x).strip() for x in arr)


def _extract_irf_dict(irfs_obj: Any) -> dict[str, Array]:
    out: dict[str, Array] = {}
    for attr in dir(irfs_obj):
        if attr.startswith("_"):
            continue
        value = getattr(irfs_obj, attr)
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            continue
        if arr.size == 0:
            continue
        out[attr] = arr.reshape(-1)
    return out
