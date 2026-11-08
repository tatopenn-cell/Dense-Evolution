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


def test_the_noise_of_the_page_is_the_numpy_path_of_the_package():
    import importlib.util

    import numpy as np

    from dense_evolution import NoiseModel

    spec = importlib.util.spec_from_file_location("noise_numpy", COMPOSER / "py" / "noise_numpy.py")
    nz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nz)
    assert nz.MODELS == NoiseModel.MODELS
    js = _text(COMPOSER / "web_kernel.js")
    listed = re.search(r"const NOISE_MODELS = \[([^\]]*)\]", js).group(1)
    assert re.findall(r'"(\w+)"', listed) == NoiseModel.MODELS
    n = 4
    rng0 = np.random.default_rng(3)
    sv = rng0.normal(size=2**n) + 1j * rng0.normal(size=2**n)
    sv = (sv / np.linalg.norm(sv)).astype(np.complex128)
    for model in NoiseModel.MODELS:
        for p in (0.05, 0.4):
            for seed in range(5):
                want = NoiseModel.apply_to_sv(sv.copy(), n, model, p, rng=np.random.default_rng(seed))
                got = nz.apply_to_sv(sv, n, model, p, np.random.default_rng(seed))
                np.testing.assert_allclose(got, np.asarray(want), atol=1e-12, err_msg=f"{model} p={p} seed={seed}")


@pytest.mark.parametrize("name", ["mps_numpy.py", "qasm_subset.py"])
def test_the_python_files_of_the_page_are_the_ones_of_the_package(name):
    source = REPO_ROOT / "dense_evolution" / "backends" / name
    assert _text(COMPOSER / "py" / name) == _text(source)


@pytest.mark.parametrize("name", ["state_visuals.py", "circuit_diagram.py", "_gate_tables.py"])
def test_the_figure_files_of_the_page_are_the_ones_of_the_dashboard(name):
    source = REPO_ROOT / "tools" / "dashboard" / "core" / name
    assert _text(COMPOSER / "py" / "dashboard_core" / name) == _text(source)


def test_quirk_of_the_page_draws_more_than_16_wires_and_simulates_only_16():
    built = _text(COMPOSER / "quirk" / "quirk.html")
    assert "SIMULATED_WIRE_COUNT" in built
    assert re.search(r"MAX_WIRE_COUNT\s*=\s*1500", built)
    source = _text(REPO_ROOT / "third_party" / "quirk" / "src" / "Config.js")
    assert "Config.MAX_WIRE_COUNT = 1500;" in source
    assert "Config.SIMULATED_WIRE_COUNT = 16;" in source


def test_the_presets_of_the_page_are_the_library_of_the_dashboard():
    import json

    from dashboard_core.qasm_library import QASM_LIBRARY

    path = COMPOSER / "qasm_library.json"
    assert json.loads(path.read_text(encoding="utf-8")) == QASM_LIBRARY


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


def test_every_element_the_composer_reads_is_in_the_page():
    app = _text(COMPOSER / "app.js")
    page = _text(REPO_ROOT / "docs" / "composer.md")
    wanted = set(re.findall(r'\$\("([\w-]+)"\)', app))
    present = set(re.findall(r'id="([\w-]+)"', page))
    guarded = {"palette", "grid", "clear-btn"}
    assert wanted - present - guarded == set()


def test_the_page_has_the_settings_on_the_left_and_the_results_on_the_right():
    page = _text(REPO_ROOT / "docs" / "composer.md")
    side, main = page.index('<aside class="de-side">'), page.index('<div class="de-main">')
    assert side < main
    assert page.index('id="run-btn"') < main
    assert page.index('id="qasm"') < main
    assert page.index('id="statevector-table"') > main
    assert page.index('id="histogram-img"') > main


def test_every_view_and_tab_button_has_its_panel():
    page = _text(REPO_ROOT / "docs" / "composer.md")
    for view in re.findall(r'data-view="(\w+)"', page):
        assert f'id="de-view-{view}"' in page
    tabs = re.findall(r'class="de-tab(?: active)?" data-tab="(\w+)"', page)
    panels = re.findall(r'class="de-tabpanel" data-tab="(\w+)"', page)
    assert tabs and tabs == panels


def test_the_quirk_editor_and_its_license_are_shipped_and_in_the_page():
    quirk = COMPOSER / "quirk"
    assert (quirk / "quirk.html").stat().st_size > 100_000
    assert "Apache License" in _text(quirk / "LICENSE.txt")
    assert "../assets/composer/quirk/quirk.html" in _text(REPO_ROOT / "docs" / "composer.md")


def test_every_gate_id_the_bridge_reads_exists_in_the_source_of_quirk():
    source = "".join(
        _text(f) for f in (REPO_ROOT / "third_party" / "quirk" / "src" / "gates").glob("*.js")
    )
    ids = ["H", "X", "Y", "Z", "Swap", "•", "◦", "Rxft", "Ryft", "Rzft"]
    ids += [a + p for a in "XYZ" for p in ("^½", "^-½", "^¼", "^-¼", "^⅓", "^-⅓")]
    ids += ["Bloch", "Amps", "Chance", "Density", "Sample", "…"]
    missing = [i for i in ids if f'"{i}"' not in source and f"'{i}'" not in source]
    assert missing == []


@pytest.mark.parametrize(
    "name",
    [
        "install-composer.bat",
        "install-composer.sh",
        "uninstall-composer.bat",
        "uninstall-composer.sh",
    ],
)
def test_the_installers_of_the_site_are_the_ones_of_tools(name):
    site = REPO_ROOT / "docs" / "assets" / "installer" / name
    assert _text(site) == _text(REPO_ROOT / "tools" / "installer" / name)


def test_the_download_links_of_the_kernel_banner_work_from_the_page_of_the_composer():
    app = _text(COMPOSER / "app.js")
    assert 'href="assets/installer/' not in app
    assert 'href="../assets/installer/install-composer.bat"' in app
    assert 'href="../assets/installer/install-composer.sh"' in app
