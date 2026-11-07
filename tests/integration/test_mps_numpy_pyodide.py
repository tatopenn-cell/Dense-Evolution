"""Run MPSSimulatorNumpy inside Pyodide, in a real Chromium browser.

This is a portability check: the same code in CPython and in the
WebAssembly Python of the browser must give the same numbers. The
correctness of the algorithm is covered by ``tests/unit/test_mps_numpy.py``
(items a to i of Task MPS-01, item C); this file checks only that the
engine runs on Pyodide and matches CPython.

The CDN of Pyodide is fetched from jsdelivr; when it cannot be reached
the tests skip, and the reason names the CDN.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js"

_JS_RUNNER = """
async ([mn_src, qs_src, code]) => {
    const t0 = performance.now();
    const py = await loadPyodide();
    await py.loadPackage(['numpy']);
    py.FS.mkdir('/work');
    py.FS.writeFile('/work/mps_numpy.py', mn_src);
    py.FS.writeFile('/work/qasm_subset.py', qs_src);
    const load_s = (performance.now() - t0) / 1000;

    let jax_status = 'not_attempted';
    try {
        await py.loadPackage('micropip');
        const mp = py.pyimport('micropip');
        await mp.install('jax');
        jax_status = 'installed';
    } catch (e) {
        jax_status = 'FAILED: ' + String(e).slice(0, 160);
    }

    const out = await py.runPythonAsync(code);
    const heap_mb = performance.memory
        ? Math.round(performance.memory.usedJSHeapSize / 1e6)
        : null;
    return JSON.stringify({
        load_seconds: load_s,
        jax_in_pyodide: jax_status,
        result: JSON.parse(String(out)),
        js_heap_mb: heap_mb,
    });
}
"""


def _ensure_qasm_library():
    """Build a fake dashboard_core package for gate_tuples_to_qasm."""
    fake = Path("/tmp/_mpsnumpy_pyodide_qasm") / "dashboard_core"
    if not fake.exists():
        fake.mkdir(parents=True)
        (fake / "__init__.py").write_text("")
        shutil.copy(
            REPO_ROOT / "tools/dashboard/core/qasm_library.py",
            fake / "qasm_library.py",
        )
        shutil.copy(
            REPO_ROOT / "tools/dashboard/core/_gate_tables.py",
            fake / "_gate_tables.py",
        )
    if str(fake.parent) not in sys.path:
        sys.path.insert(0, str(fake.parent))


def _qasm_of(ops, n: int) -> str:
    _ensure_qasm_library()
    from dashboard_core.qasm_library import gate_tuples_to_qasm

    return gate_tuples_to_qasm(ops, n_qubits=n)


def _run_in_pyodide_once(code: str) -> dict:
    mn_src = (REPO_ROOT / "dense_evolution/backends/mps_numpy.py").read_text()
    qs_src = (REPO_ROOT / "dense_evolution/backends/qasm_subset.py").read_text()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto("about:blank")
            page.add_script_tag(url=PYODIDE_CDN)
            page.wait_for_function("typeof loadPyodide === 'function'", timeout=120_000)
            raw = page.evaluate(_JS_RUNNER, [mn_src, qs_src, code])
        finally:
            browser.close()
    return json.loads(raw)


def _run_in_pyodide(code: str, *, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        try:
            return _run_in_pyodide_once(code)
        except Exception as exc:  # noqa: BLE001 -- CDN failures are opaque
            if attempt == attempts - 1:
                pytest.skip(
                    f"Pyodide CDN {PYODIDE_CDN} unreachable: " f"{type(exc).__name__}: {exc}"
                )
            time.sleep(3)


def test_1_ghz_200_inside_pyodide():
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '/work')\n"
        "import mps_numpy\n"
        "n = 200\n"
        "ops = [('h', 0)] + [('cx', q, q + 1) for q in range(n - 1)]\n"
        "sim = mps_numpy.MPSSimulatorNumpy(n)\n"
        "t0 = time.time()\n"
        "sim.run_circuit(ops)\n"
        "idx, p = sim.get_top_k_probable_states(2)\n"
        "dt = time.time() - t0\n"
        "json.dumps({'seconds': dt, 'bond': sim.max_bond_used(),"
        " 'idx': sorted(int(i) for i in idx),"
        " 'p': [float(x) for x in p]})\n"
    )
    out = _run_in_pyodide(code)
    res = out["result"]
    print(
        f"PYODIDE GHZ-200: seconds={res['seconds']:.2f}  bond={res['bond']}  "
        f"load_seconds={out['load_seconds']:.1f}  js_heap_mb={out['js_heap_mb']}"
    )
    print(f"jax_in_pyodide: {out['jax_in_pyodide']}")
    assert res["bond"] == 2
    assert res["idx"] == [0, 2**200 - 1]
    assert abs(res["p"][0] - 0.5) < 1e-9
    assert abs(res["p"][1] - 0.5) < 1e-9


def test_2_qft_20_matches_cpython():
    import dense_evolution as de
    from dense_evolution.backends.mps_numpy import MPSSimulatorNumpy
    from dense_evolution.backends.qasm_subset import parse_qasm_subset

    ops = de.qft(20)
    qasm = _qasm_of(ops, 20)
    max_bond = 128
    k = 8

    n_cp, ops_cp = parse_qasm_subset(qasm)
    sim_cp = MPSSimulatorNumpy(n_cp, max_bond=max_bond)
    sim_cp.run_circuit(ops_cp)
    idx_cp, p_cp = sim_cp.get_top_k_probable_states(k)
    cp_dict = {int(i): float(x) for i, x in zip(idx_cp, p_cp)}

    qasm_for_js = json.dumps(qasm)
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '/work')\n"
        "import mps_numpy, qasm_subset\n"
        "qasm = " + qasm_for_js + "\n"
        "n, ops = qasm_subset.parse_qasm_subset(qasm)\n"
        "sim = mps_numpy.MPSSimulatorNumpy(n, max_bond=" + str(max_bond) + ")\n"
        "t0 = time.time()\n"
        "sim.run_circuit(ops)\n"
        "idx, p = sim.get_top_k_probable_states(" + str(k) + ")\n"
        "dt = time.time() - t0\n"
        "json.dumps({'seconds': dt,"
        " 'idx': [int(i) for i in idx],"
        " 'p': [float(x) for x in p]})\n"
    )
    out = _run_in_pyodide(code)
    res = out["result"]
    pyodide_dict = {int(i): float(x) for i, x in zip(res["idx"], res["p"])}

    print(
        f"PYODIDE QFT-20: seconds={res['seconds']:.2f}  "
        f"load_seconds={out['load_seconds']:.1f}  js_heap_mb={out['js_heap_mb']}"
    )
    print(f"jax_in_pyodide: {out['jax_in_pyodide']}")

    shared = set(cp_dict) & set(pyodide_dict)
    assert len(shared) >= 4, (
        f"CPython top-{k} = {sorted(cp_dict)}; " f"Pyodide top-{k} = {sorted(pyodide_dict)}"
    )
    for i in shared:
        diff = abs(cp_dict[i] - pyodide_dict[i])
        assert diff < 1e-9, (i, cp_dict[i], pyodide_dict[i], diff)
