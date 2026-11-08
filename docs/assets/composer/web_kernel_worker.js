// Web worker of the browser mode of the Composer.
//
// It loads Pyodide (Python in WebAssembly) and NumPy, writes the two Python
// files of the NumPy MPS simulator (py/mps_numpy.py and py/qasm_subset.py,
// copies of dense_evolution/backends) into the Pyodide file system, and
// answers two messages: "init" and "run". JAX is not installed: nothing here
// imports it. It runs in a worker so that a long circuit does not freeze the
// page.

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
const PY_FILES = ["mps_numpy.py", "qasm_subset.py", "noise_numpy.py"];
// The figure modules of the dashboard (copies of tools/dashboard/core), which need matplotlib.
const FIGURE_FILES = ["state_visuals.py", "circuit_diagram.py", "_gate_tables.py"];
// Up to this many qubits the MPS is contracted to 2**n amplitudes, to sample the shots and draw
// the figures; above, the most probable states. The same constant as in web_kernel.js.
const CONTRACT_LIMIT = 20;

const GLUE = `
import base64, io, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mps_numpy as mn
import qasm_subset as qs
import noise_numpy as nz
from dashboard_core import circuit_diagram as cd
from dashboard_core import state_visuals as sv

CONTRACT_LIMIT = ${CONTRACT_LIMIT}
STATEVECTOR_ROWS = 4096


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _qiskit_order(values, n):
    # the engine puts qubit 0 in the most significant bit, the Composer shows it in the least
    return np.asarray(values).reshape((2,) * n).transpose(list(range(n))[::-1]).reshape(-1)


def _top_states(n, ops, k):
    sim = mn.MPSSimulatorNumpy(n)
    sim.run_circuit(ops)
    idx, probs = sim.get_top_k_probable_states(int(k))
    width = "0%db" % n
    return {
        "large_scale": True,
        "engine": "numpy-web",
        "backend": "mps",
        "n_qubits": n,
        "n_gates": len(ops),
        "k_requested": int(k),
        "top_k_states": [
            {"state": format(int(i), width)[::-1], "probability": float(p)}
            for i, p in zip(idx, probs)
        ],
        "mps_max_bond_used": int(sim.max_bond_used()),
        "mps_memory_mb": float(sim.memory_mb()),
        "mps_avg_jsd": float(sim.avg_jsd()),
    }


def run(qasm, shots, seed, k, noise_model="ideal", noise_p=0.0):
    n, ops = qs.parse_qasm_subset(qasm)
    if n > CONTRACT_LIMIT:
        return json.dumps(_top_states(n, ops, k))
    sim = mn.MPSSimulatorNumpy(n, max_bond=2 ** (n // 2))
    sim.run_circuit(ops)
    native = np.asarray(sim.contract_to_statevector(), dtype=np.complex128)
    rng = np.random.default_rng(int(seed))
    fidelity = None
    if noise_model != "ideal" and float(noise_p) > 0:
        ideal = native.copy()
        native = nz.apply_to_sv(native, n, noise_model, float(noise_p), rng)
        fidelity = float(abs(np.vdot(ideal, native)) ** 2)
    probs = np.abs(native) ** 2
    probs = probs / probs.sum()
    # Generator.choice(p=...) written out: the same cdf and the same uniform draws, so the same
    # counts, but without its cast to int64, which WebAssembly's 32-bit NumPy refuses
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]
    outcomes = np.searchsorted(cdf, rng.random(int(shots)), side="right")
    values, freqs = np.unique(outcomes, return_counts=True)
    width = "0%db" % n
    counts = {format(int(v), width)[::-1]: int(f) for v, f in zip(values, freqs)}
    state = _qiskit_order(native, n)
    idx = np.flatnonzero(np.abs(state) > 1e-10)
    truncated = len(idx) > STATEVECTOR_ROWS
    if truncated:
        idx = np.sort(idx[np.argsort(-np.abs(state[idx]))[:STATEVECTOR_ROWS]])
    rows = [
        {"state": format(int(i), width), "re": float(state[i].real), "im": float(state[i].imag),
         "abs": float(abs(state[i])), "phase": float(np.angle(state[i]))}
        for i in idx
    ]
    with plt.style.context("default"):
        result = {
            "large_scale": False,
            "engine": "numpy-web",
            "backend": "mps",
            "n_qubits": n,
            "n_gates": len(ops),
            "counts": counts,
            "statevector": rows,
            "statevector_truncated": truncated,
            "fidelity_vs_ideal": fidelity,
            "noise_model": noise_model,
            "noise_p": float(noise_p),
            "circuit_png": _png(cd.draw_native_circuit_diagram(ops, n)),
            "histogram_png": _png(sv.native_histogram_figure(counts)),
            "qsphere_png": _png(sv.native_qsphere_figure(state)),
            "bloch_png": _png(sv.native_bloch_multivector_figure(state)),
            "mps_max_bond_used": int(sim.max_bond_used()),
            "mps_memory_mb": float(sim.memory_mb()),
            "mps_avg_jsd": float(sim.avg_jsd()),
        }
    return json.dumps(result)
`;

let py = null;

function lastLine(message) {
  const lines = String(message).trim().split("\n").filter((l) => l.trim());
  return lines.length ? lines[lines.length - 1] : String(message);
}

async function init(base) {
  importScripts(PYODIDE_URL);
  py = await loadPyodide();
  await py.loadPackage(["numpy", "matplotlib"]);
  py.FS.mkdir("/work");
  py.FS.mkdir("/work/dashboard_core");
  const write = async (path, url) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Impossibile scaricare ${url} (${res.status})`);
    py.FS.writeFile(path, await res.text());
  };
  for (const name of PY_FILES) await write("/work/" + name, base + name);
  for (const name of FIGURE_FILES) await write("/work/dashboard_core/" + name, base + "dashboard_core/" + name);
  py.FS.writeFile("/work/dashboard_core/__init__.py", "");
  py.runPython("import sys; sys.path.insert(0, '/work')");
  py.runPython(GLUE);
}

self.onmessage = async (event) => {
  const { id, type } = event.data;
  try {
    if (type === "init") {
      await init(event.data.base);
      self.postMessage({ id, ok: true });
    } else if (type === "run") {
      const t0 = performance.now();
      const fn = py.globals.get("run");
      let text;
      try {
        text = fn(event.data.qasm, event.data.shots, event.data.seed, event.data.k, event.data.noiseModel || "ideal", event.data.noiseP || 0);
      } finally {
        fn.destroy();
      }
      const result = JSON.parse(text);
      result.elapsed_s = (performance.now() - t0) / 1000;
      self.postMessage({ id, ok: true, result });
    } else {
      throw new Error("messaggio sconosciuto: " + type);
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: lastLine(err && err.message ? err.message : err) });
  }
};
