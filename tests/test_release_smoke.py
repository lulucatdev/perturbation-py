from typer.testing import CliRunner

import perturbation_py
from perturbation_py.cli import app


def test_release_smoke_import_and_cli_help():
    assert perturbation_py.__version__.count(".") == 2

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Dynare-style perturbation tools" in result.stdout
