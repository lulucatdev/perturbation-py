from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class DynareReference:
    order: int
    ghx: Array | None = None
    ghu: Array | None = None
    ghxx: Array | None = None
    ghxu: Array | None = None
    ghuu: Array | None = None
    ghs2: Array | None = None
    ghxxx: Array | None = None
    ghxxu: Array | None = None
    ghxuu: Array | None = None
    ghuuu: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_dynare_reference(path: str | Path) -> DynareReference:
    ref_path = Path(path)
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    return DynareReference(
        order=int(payload["order"]),
        ghx=_to_array(payload.get("ghx")),
        ghu=_to_array(payload.get("ghu")),
        ghxx=_to_array(payload.get("ghxx")),
        ghxu=_to_array(payload.get("ghxu")),
        ghuu=_to_array(payload.get("ghuu")),
        ghs2=_to_array(payload.get("ghs2")),
        ghxxx=_to_array(payload.get("ghxxx")),
        ghxxu=_to_array(payload.get("ghxxu")),
        ghxuu=_to_array(payload.get("ghxuu")),
        ghuuu=_to_array(payload.get("ghuuu")),
        metadata=dict(payload.get("metadata", {})),
    )


def save_dynare_reference(reference: DynareReference, path: str | Path) -> None:
    ref_path = Path(path)
    payload = {
        "order": int(reference.order),
        "ghx": _to_list(reference.ghx),
        "ghu": _to_list(reference.ghu),
        "ghxx": _to_list(reference.ghxx),
        "ghxu": _to_list(reference.ghxu),
        "ghuu": _to_list(reference.ghuu),
        "ghs2": _to_list(reference.ghs2),
        "ghxxx": _to_list(reference.ghxxx),
        "ghxxu": _to_list(reference.ghxxu),
        "ghxuu": _to_list(reference.ghxuu),
        "ghuuu": _to_list(reference.ghuuu),
        "metadata": dict(reference.metadata),
    }
    ref_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _to_array(value: Any) -> Array | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _to_list(value: Array | None) -> Any:
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()
