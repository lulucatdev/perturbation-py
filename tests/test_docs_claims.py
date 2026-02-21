from pathlib import Path


def test_readme_mentions_orders_1_2_3_and_pruning():
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "order=1" in text
    assert "order=2" in text
    assert "order=3" in text
    assert "pruning" in text
