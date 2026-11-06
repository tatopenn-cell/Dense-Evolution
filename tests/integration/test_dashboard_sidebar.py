"""The sidebar of the Build page starts with Run, and the controls follow in order."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[2] / "tools" / "dashboard"


def _labels(block):
    for node in block.children.values():
        if hasattr(node, "children") and not hasattr(node, "label"):
            yield from _labels(node)
        elif getattr(node, "label", None):
            yield node.label


def test_run_is_first_and_the_circuit_controls_follow(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    for name in ("dashboard_core", "sota_views", "engine_explorer", "circuit_editor"):
        sys.modules.pop(name, None)
    app = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=180).run()
    assert list(_labels(app.sidebar))[:5] == [
        "▶ Run",
        "Preset",
        "OpenQASM 2.0",
        "Shots",
        "Seed",
    ]
