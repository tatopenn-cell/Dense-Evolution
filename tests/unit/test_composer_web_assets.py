"""The files of the browser mode of the Composer stay in step with the package.

``docs/assets/composer/py`` holds copies of the two Python files that Pyodide
runs, and ``web_kernel.js`` repeats the gate palette of the graphical builder.
A copy that drifts would make the page run a different simulator than the one
that the package tests, so these tests compare them.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSER = REPO_ROOT / "docs" / "assets" / "composer"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize("name", ["mps_numpy.py", "qasm_subset.py"])
def test_the_python_files_of_the_page_are_the_ones_of_the_package(name):
    source = REPO_ROOT / "dense_evolution" / "backends" / name
    assert _text(COMPOSER / "py" / name) == _text(source)


def test_the_palette_of_the_page_is_the_one_of_the_graphical_builder():
    from dashboard_core.graphical_builder import GATE_PALETTE

    js = _text(COMPOSER / "web_kernel.js")
    block = js[js.index("const PALETTE = [") : js.index("];", js.index("const PALETTE = ["))]
    entries = re.findall(
        r'\{ id: "(\w+)", label: "([^"]+)", kind: "(\w+)", gate: (null|"\w+") \}', block
    )
    from_js = [
        (i, label, kind, None if gate == "null" else gate.strip('"'))
        for i, label, kind, gate in entries
    ]
    from_python = [(g["id"], g["label"], g["kind"], g["gate"]) for g in GATE_PALETTE]
    assert from_js == from_python


def test_the_page_loads_the_web_kernel_before_the_composer():
    page = _text(REPO_ROOT / "docs" / "composer.md")
    assert page.index("web_kernel.js") < page.index("composer/app.js")


def test_the_default_preset_of_the_composer_exists_in_web_mode():
    app = _text(COMPOSER / "app.js")
    web = _text(COMPOSER / "web_kernel.js")
    key = re.search(r'presets\["(Bell state[^"]+)"\]', app).group(1)
    assert f'"{key}"' in web
