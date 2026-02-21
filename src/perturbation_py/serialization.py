from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .policy import Policy


class _SimulationLike(Protocol):
    states: Array
    controls: Array
    shocks: Array


Array = np.ndarray


def save_policy(policy: Policy, path: str | Path) -> None:
    target = Path(path)
    payload = {
        "order": int(policy.order),
        "steady_state_controls": _to_list(policy.steady_state_controls),
        "ghx": _to_list(policy.ghx),
        "ghu": _to_list(policy.ghu),
        "ghxx": _to_list(policy.ghxx),
        "ghxu": _to_list(policy.ghxu),
        "ghuu": _to_list(policy.ghuu),
        "ghs2": _to_list(policy.ghs2),
        "ghxxx": _to_list(policy.ghxxx),
        "ghxxu": _to_list(policy.ghxxu),
        "ghxuu": _to_list(policy.ghxuu),
        "ghuuu": _to_list(policy.ghuuu),
        "transition": _to_list(policy.transition),
        "shock_impact": _to_list(policy.shock_impact),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_policy(path: str | Path) -> Policy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Policy(
        order=int(payload["order"]),
        steady_state_controls=np.asarray(payload["steady_state_controls"], dtype=float),
        ghx=np.asarray(payload["ghx"], dtype=float),
        ghu=np.asarray(payload["ghu"], dtype=float),
        ghxx=_to_array(payload.get("ghxx")),
        ghxu=_to_array(payload.get("ghxu")),
        ghuu=_to_array(payload.get("ghuu")),
        ghs2=_to_array(payload.get("ghs2")),
        ghxxx=_to_array(payload.get("ghxxx")),
        ghxxu=_to_array(payload.get("ghxxu")),
        ghxuu=_to_array(payload.get("ghxuu")),
        ghuuu=_to_array(payload.get("ghuuu")),
        transition=_to_array(payload.get("transition")),
        shock_impact=_to_array(payload.get("shock_impact")),
    )


def save_simulation(result: _SimulationLike, path: str | Path) -> None:
    target = Path(path)
    payload = {
        "states": _to_list(result.states),
        "controls": _to_list(result.controls),
        "shocks": _to_list(result.shocks),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _to_list(array: Array | None) -> Any:
    if array is None:
        return None
    return np.asarray(array, dtype=float).tolist()


def _to_array(value: Any) -> Array | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)
