"""The browser mode of the Composer, in a real Chromium.

``docs/assets/composer/web_kernel.js`` starts a Web Worker that loads Pyodide
and the NumPy MPS simulator. The files are served over HTTP from the folder of
the Composer and driven from a page of that origin: a circuit goes in as
OpenQASM and the answer is the one the page draws.

Pyodide comes from jsDelivr. When it cannot be reached the tests skip and the
reason names it; any other error fails the test.
"""

import functools
import http.server
import threading
from pathlib import Path

import pytest

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

pytestmark = pytest.mark.slow

COMPOSER = Path(__file__).resolve().parents[2] / "docs" / "assets" / "composer"
_NETWORK_ERRORS = (
    "Failed to fetch",
    "NetworkError",
    "net::ERR",
    "Load failed",
    "importScripts",
    "failed to load",
)

_RUN = """
async ([qasm]) => {
    const t0 = performance.now();
    try {
        const result = await window.DEWebKernel.run(qasm, 32);
        return { ok: true, result, seconds: (performance.now() - t0) / 1000 };
    } catch (err) {
        return { ok: false, error: String(err.message || err) };
    }
}
"""


@pytest.fixture(scope="module")
def page():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(COMPOSER))
    handler.log_message = lambda *args, **kwargs: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                tab = browser.new_page()
                tab.goto(f"http://127.0.0.1:{server.server_port}/")
                tab.add_script_tag(url="/web_kernel.js")
                tab.wait_for_function("typeof window.DEWebKernel === 'object'")
                started = tab.evaluate(
                    "() => window.DEWebKernel.ready().then(() => 'ready', (e) => String(e.message || e))"
                )
                if started != "ready":
                    if any(signature in started for signature in _NETWORK_ERRORS):
                        pytest.skip(f"Pyodide CDN unreachable from the worker: {started}")
                    pytest.fail(f"the web kernel did not start: {started}")
                yield tab
            finally:
                browser.close()
    finally:
        server.shutdown()


def _run(tab, qasm):
    return tab.evaluate(_RUN, [qasm])


def test_1_bell_state_built_from_the_grid_ops(page):
    qasm = page.evaluate(
        "() => window.DEWebKernel.opsToQasm(2, [{gate: 'h', qubits: [0]},"
        " {gate: 'cx', qubits: [0, 1]}])"
    )
    out = _run(page, qasm)
    assert out["ok"], out
    res = out["result"]
    assert res["large_scale"] is True
    assert res["engine"] == "numpy-web"
    assert res["n_qubits"] == 2
    states = {row["state"]: row["probability"] for row in res["top_k_states"]}
    assert set(states) >= {"00", "11"}
    assert states["00"] == pytest.approx(0.5, abs=1e-5)
    assert states["11"] == pytest.approx(0.5, abs=1e-5)
    assert res["mps_max_bond_used"] == 2


def test_2_every_preset_of_the_page_runs(page):
    names = page.evaluate("() => Object.keys(window.DEWebKernel.presets())")
    assert "Bell state (2 qubit)" in names
    for name in names:
        if "1500" in name:
            continue
        qasm = page.evaluate("(name) => window.DEWebKernel.presets()[name]", name)
        out = _run(page, qasm)
        assert out["ok"], (name, out)
        total = sum(row["probability"] for row in out["result"]["top_k_states"])
        assert 0.0 < total <= 1.0 + 1e-4, name


def test_3_ghz_with_1500_qubits(page):
    qasm = page.evaluate("() => window.DEWebKernel.presets()['GHZ state (1500 qubit)']")
    out = _run(page, qasm)
    assert out["ok"], out
    res = out["result"]
    print(f"BROWSER GHZ-1500: seconds={out['seconds']:.2f} bond={res['mps_max_bond_used']}")
    assert res["n_qubits"] == 1500
    states = {row["state"]: row["probability"] for row in res["top_k_states"]}
    assert states["0" * 1500] == pytest.approx(0.5, abs=2e-4)
    assert states["1" * 1500] == pytest.approx(0.5, abs=2e-4)
    assert res["mps_max_bond_used"] == 2


def test_4_an_error_of_the_circuit_names_its_line(page):
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nfoo q[0];\n'
    out = _run(page, qasm)
    assert out["ok"] is False
    assert "line 4" in out["error"]
    assert "unknown gate" in out["error"]


def test_5_a_gate_the_builder_cannot_place_is_refused(page):
    message = page.evaluate(
        "() => { try { window.DEWebKernel.opsToQasm(1, [{gate: 'foo', qubits: [0]}]); }"
        " catch (e) { return String(e.message); } return ''; }"
    )
    assert "foo" in message
