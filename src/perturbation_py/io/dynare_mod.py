from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class DynareModSpec:
    path: Path
    endogenous: tuple[str, ...]
    exogenous: tuple[str, ...]
    parameters: tuple[str, ...]
    model_equations: tuple[str, ...]
    initval: dict[str, float]


_COMMENT_PATTERNS = [
    re.compile(r"%.*$", flags=re.MULTILINE),
    re.compile(r"//.*$", flags=re.MULTILINE),
]
_DECL_PATTERN = re.compile(
    r"\b(var|varexo|parameters)\b\s+([^;]+);", flags=re.IGNORECASE
)
_MODEL_BLOCK = re.compile(
    r"(?mis)^\s*model(?:\s*\([^;]*\))?\s*;(?P<body>.*?)^\s*end\s*;"
)
_INITVAL_BLOCK = re.compile(r"(?mis)^\s*initval\s*;(?P<body>.*?)^\s*end\s*;")
_ASSIGN = re.compile(r"\b([A-Za-z_]\w*)\b\s*=\s*([^;]+);")


def parse_dynare_mod_file(path: str | Path) -> DynareModSpec:
    mod_path = Path(path)
    text = mod_path.read_text(encoding="utf-8")
    clean = _strip_comments(text)

    model_match = _MODEL_BLOCK.search(clean)
    declaration_region = clean if model_match is None else clean[: model_match.start()]

    endogenous: list[str] = []
    exogenous: list[str] = []
    parameters: list[str] = []

    for kind, symbols in _DECL_PATTERN.findall(declaration_region):
        names = _extract_symbol_list(symbols)
        if kind.lower() == "var":
            endogenous.extend(names)
        elif kind.lower() == "varexo":
            exogenous.extend(names)
        elif kind.lower() == "parameters":
            parameters.extend(names)

    equations: list[str] = []
    if model_match is not None:
        model_body = model_match.group("body")
        equations = [eq.strip() for eq in model_body.split(";") if eq.strip()]

    initval: dict[str, float] = {}
    initval_match = _INITVAL_BLOCK.search(clean)
    if initval_match is not None:
        body = initval_match.group("body")
        for name, value in _ASSIGN.findall(body):
            try:
                initval[name] = float(_safe_eval_numeric(value))
            except Exception:
                continue

    return DynareModSpec(
        path=mod_path,
        endogenous=tuple(dict.fromkeys(endogenous)),
        exogenous=tuple(dict.fromkeys(exogenous)),
        parameters=tuple(dict.fromkeys(parameters)),
        model_equations=tuple(equations),
        initval=initval,
    )


def _strip_comments(text: str) -> str:
    out = text
    for pattern in _COMMENT_PATTERNS:
        out = pattern.sub("", out)
    return out


def _safe_eval_numeric(expr: str) -> float:
    allowed_chars = set("0123456789.+-*/()eE ")
    if any(ch not in allowed_chars for ch in expr):
        raise ValueError("Expression contains unsupported characters")
    return float(eval(expr, {"__builtins__": {}}, {}))


def _extract_symbol_list(raw: str) -> list[str]:
    text = re.sub(r"\$[^$]*\$", " ", raw)
    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = re.findall(r"\b[A-Za-z_]\w*\b", text)
    blacklist = {"long_name", "tex_name"}
    return [tok for tok in tokens if tok not in blacklist]
