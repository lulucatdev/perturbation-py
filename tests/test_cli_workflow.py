from pathlib import Path

from typer.testing import CliRunner

from perturbation_py.cli import app


def test_cli_solve_and_irf_commands_produce_output_files(tmp_path: Path):
    runner = CliRunner()
    solution_path = tmp_path / "solution.json"
    irf_path = tmp_path / "irf.json"

    solve_result = runner.invoke(
        app,
        [
            "solve",
            "--model",
            "tests/fixtures/models/scalar_model.py",
            "--order",
            "2",
            "--output",
            str(solution_path),
        ],
    )
    assert solve_result.exit_code == 0
    assert solution_path.exists()

    irf_result = runner.invoke(
        app,
        [
            "irf",
            "--solution",
            str(solution_path),
            "--output",
            str(irf_path),
            "--horizon",
            "12",
            "--shock-index",
            "0",
        ],
    )
    assert irf_result.exit_code == 0
    assert irf_path.exists()
