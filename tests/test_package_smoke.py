from pathlib import Path


def test_package_import_and_version():
    import perturbation_py

    assert isinstance(perturbation_py.__version__, str)
    assert perturbation_py.__version__


def test_readme_mentions_first_order_scope():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "first-order" in text.lower()
