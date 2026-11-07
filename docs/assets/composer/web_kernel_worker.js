// Web worker of the browser mode of the Composer.
//
// It loads Pyodide (Python in WebAssembly) and NumPy, writes the two Python
// files of the NumPy MPS simulator (py/mps_numpy.py and py/qasm_subset.py,
// copies of dense_evolution/backends) into the Pyodide file system, and
// answers two messages: "init" and "run". JAX is not installed: nothing here
// imports it. It runs in a worker so that a long circuit does not freeze the
// page.

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
const PY_FILES = ["mps_numpy.py", "qasm_subset.py"];

const GLUE = `
import json
import mps_numpy as mn
import qasm_subset as qs


def run(qasm, k):
    n, ops = qs.parse_qasm_subset(qasm)
    sim = mn.MPSSimulatorNumpy(n)
    sim.run_circuit(ops)
    idx, probs = sim.get_top_k_probable_states(int(k))
    width = "0%db" % n
    return json.dumps({
        "large_scale": True,
        "engine": "numpy-web",
        "backend": "mps",
        "n_qubits": n,
        "n_gates": len(ops),
        "k_requested": int(k),
        "top_k_states": [
            {"state": format(int(i), width), "probability": float(p)}
            for i, p in zip(idx, probs)
        ],
        "mps_max_bond_used": int(sim.max_bond_used()),
        "mps_memory_mb": float(sim.memory_mb()),
        "mps_avg_jsd": float(sim.avg_jsd()),
    })
`;

let py = null;

function lastLine(message) {
  const lines = String(message).trim().split("\n").filter((l) => l.trim());
  return lines.length ? lines[lines.length - 1] : String(message);
}

async function init(base) {
  importScripts(PYODIDE_URL);
  py = await loadPyodide();
  await py.loadPackage("numpy");
  py.FS.mkdir("/work");
  for (const name of PY_FILES) {
    const res = await fetch(base + name);
    if (!res.ok) throw new Error(`Impossibile scaricare ${name} (${res.status})`);
    py.FS.writeFile("/work/" + name, await res.text());
  }
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
        text = fn(event.data.qasm, event.data.k);
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
