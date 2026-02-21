from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Annotated, Any, Callable, Mapping, cast

import numpy as np
import typer

from .io import parse_dynare_mod_file
from .model import DSGEModel
from .policy import Policy
from .pruning import impulse_response_pruned, simulate_pruned
from .serialization import load_policy, save_policy, save_simulation
from .simulation import impulse_response_with_policy, simulate_with_policy
from .solver import solve_first_order
from .solver_second_order import solve_second_order
from .solver_third_order import solve_third_order
from .steady_state import solve_steady_state

app = typer.Typer(help="Dynare-style perturbation tools in Python")


def _demo_model(rho: float, beta: float, kappa: float, sigma: float) -> DSGEModel:
    def transition(
        s: np.ndarray, x: np.ndarray, e: np.ndarray, params: Mapping[str, float]
    ) -> np.ndarray:
        return np.array([params["rho"] * s[0] + params["sigma"] * e[0]], dtype=float)

    def arbitrage(
        s: np.ndarray,
        x: np.ndarray,
        s_next: np.ndarray,
        x_next: np.ndarray,
        params: Mapping[str, float],
    ) -> np.ndarray:
        return np.array(
            [x[0] - params["beta"] * x_next[0] - params["kappa"] * s[0]], dtype=float
        )

    return DSGEModel(
        state_names=("k",),
        control_names=("c",),
        shock_names=("eps",),
        parameters={"rho": rho, "beta": beta, "kappa": kappa, "sigma": sigma},
        steady_state_states=np.array([0.0]),
        steady_state_controls=np.array([0.0]),
        transition=transition,
        arbitrage=arbitrage,
    )


def _load_model_builder(module_path: Path, symbol: str) -> Callable[[], DSGEModel]:
    spec = spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    builder_obj = getattr(module, symbol, None)
    builder = cast(Callable[[], DSGEModel] | None, builder_obj)
    if builder is None or not callable(builder):
        raise RuntimeError(
            f"Model module {module_path} must define callable {symbol}() -> DSGEModel"
        )
    return builder


def _load_model(module_file: str, builder: str) -> DSGEModel:
    module_path = Path(module_file)
    if not module_path.exists():
        raise FileNotFoundError(f"Model module not found: {module_path}")
    model_builder = _load_model_builder(module_path, builder)
    model: Any = model_builder()
    if not isinstance(model, DSGEModel):
        raise TypeError("Model builder must return DSGEModel")
    return model


def _policy_from_order(model: DSGEModel, order: int) -> Policy:
    if order == 1:
        return Policy.from_first_order(
            solve_first_order(model),
            steady_state_controls=np.asarray(model.steady_state_controls, dtype=float),
        )
    if order == 2:
        return Policy.from_second_order(
            solve_second_order(model),
            steady_state_controls=np.asarray(model.steady_state_controls, dtype=float),
        )
    if order == 3:
        return Policy.from_third_order(
            solve_third_order(model),
            steady_state_controls=np.asarray(model.steady_state_controls, dtype=float),
        )
    raise ValueError("order must be one of {1,2,3}")


@app.command("demo")
def demo(
    rho: Annotated[float, typer.Option(help="State persistence")] = 0.9,
    beta: Annotated[float, typer.Option(help="Discount factor")] = 0.95,
    kappa: Annotated[float, typer.Option(help="Slope in arbitrage equation")] = 0.1,
    sigma: Annotated[float, typer.Option(help="Shock scale in state equation")] = 0.01,
    horizon: Annotated[int, typer.Option(help="IRF horizon")] = 20,
) -> None:
    model = _demo_model(rho=rho, beta=beta, kappa=kappa, sigma=sigma)
    policy = _policy_from_order(model, order=3)
    irf = impulse_response_with_policy(
        policy,
        horizon=horizon,
        shock_index=0,
        shock_size=1.0,
        include_higher_order=True,
    )

    typer.echo("policy matrix ghx:")
    typer.echo(str(policy.ghx))
    typer.echo("transition matrix A:")
    typer.echo(str(policy.transition))
    typer.echo("first five state responses:")
    typer.echo(str(irf.states[:5]))


@app.command("steady-state")
def steady_state_cmd(
    model: Annotated[str, typer.Option(help="Python model module path")],
    output: Annotated[str, typer.Option(help="Output JSON path")],
    builder: Annotated[str, typer.Option(help="Model builder symbol")] = "build_model",
    tol: Annotated[float, typer.Option(help="Solver tolerance")] = 1e-10,
) -> None:
    m = _load_model(model, builder)
    result = solve_steady_state(m, tol=tol)
    Path(output).write_text(
        (
            "{\n"
            f'  "states": {result.states.tolist()},\n'
            f'  "controls": {result.controls.tolist()},\n'
            f'  "residual_norm": {result.residual_norm:.16g},\n'
            f'  "success": {str(result.success).lower()}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    typer.echo(f"Steady state written to {output}")


@app.command("solve")
def solve_cmd(
    model: Annotated[str, typer.Option(help="Python model module path")],
    output: Annotated[str, typer.Option(help="Output solution JSON path")],
    order: Annotated[int, typer.Option(help="Perturbation order {1,2,3}")] = 1,
    builder: Annotated[str, typer.Option(help="Model builder symbol")] = "build_model",
) -> None:
    m = _load_model(model, builder)
    policy = _policy_from_order(m, order=order)
    save_policy(policy, output)
    typer.echo(f"Order-{order} solution written to {output}")


@app.command("simulate")
def simulate_cmd(
    solution: Annotated[str, typer.Option(help="Path to saved solution JSON")],
    output: Annotated[str, typer.Option(help="Output simulation JSON path")],
    horizon: Annotated[int, typer.Option(help="Simulation horizon")] = 100,
    shock_std: Annotated[
        float, typer.Option(help="Std dev for generated shocks")
    ] = 1.0,
    seed: Annotated[int, typer.Option(help="RNG seed")] = 0,
    pruned: Annotated[bool, typer.Option(help="Use pruning simulation")] = False,
) -> None:
    policy = load_policy(solution)
    if pruned:
        result = simulate_pruned(
            policy,
            horizon=horizon,
            shock_std=shock_std,
            seed=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        shocks = rng.normal(0.0, shock_std, size=(horizon, policy.n_shocks))
        result = simulate_with_policy(
            policy,
            initial_state=np.zeros(policy.n_states, dtype=float),
            shocks=shocks,
            include_higher_order=True,
        )
    save_simulation(result, output)
    typer.echo(f"Simulation written to {output}")


@app.command("irf")
def irf_cmd(
    solution: Annotated[str, typer.Option(help="Path to saved solution JSON")],
    output: Annotated[str, typer.Option(help="Output IRF JSON path")],
    horizon: Annotated[int, typer.Option(help="IRF horizon")] = 40,
    shock_index: Annotated[int, typer.Option(help="Shock index")] = 0,
    shock_size: Annotated[float, typer.Option(help="Shock size")] = 1.0,
    pruned: Annotated[bool, typer.Option(help="Use pruning IRF")] = False,
) -> None:
    policy = load_policy(solution)
    if pruned:
        result = impulse_response_pruned(
            policy,
            horizon=horizon,
            shock_index=shock_index,
            shock_size=shock_size,
        )
    else:
        result = impulse_response_with_policy(
            policy,
            horizon=horizon,
            shock_index=shock_index,
            shock_size=shock_size,
            include_higher_order=True,
        )
    save_simulation(result, output)
    typer.echo(f"IRF written to {output}")


@app.command("parse-mod")
def parse_mod_cmd(
    mod_file: Annotated[str, typer.Option(help="Dynare .mod file path")],
) -> None:
    spec = parse_dynare_mod_file(mod_file)
    typer.echo(f"Model: {spec.path}")
    typer.echo(f"  endogenous: {', '.join(spec.endogenous)}")
    typer.echo(f"  exogenous: {', '.join(spec.exogenous)}")
    typer.echo(f"  parameters: {', '.join(spec.parameters)}")
    typer.echo(f"  equations: {len(spec.model_equations)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
