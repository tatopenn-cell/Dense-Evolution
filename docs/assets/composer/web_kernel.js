// Browser mode of the Composer: the NumPy MPS simulator of
// dense_evolution/backends/mps_numpy.py, run by Pyodide inside the page.
//
// No server and nothing to install: the circuit never leaves the browser.
// The simulator is the NumPy one (no JAX), so only what it supports is here:
// the circuit run with the top most probable states, for up to MAX_QUBITS
// qubits. VQE, Hamiltonians, noise models and mitigation need the local kernel.
//
// The work happens in web_kernel_worker.js, so a long circuit does not freeze
// the page. This file is the small API the Composer (app.js) calls.
(function () {

const SCRIPT_SRC =
  (document.currentScript && document.currentScript.src) ||
  new URL("../assets/composer/web_kernel.js", window.location.href).href;
const BASE = new URL(".", SCRIPT_SRC).href;

// Largest qubit count tried with this engine (GHZ state, in Pyodide).
const MAX_QUBITS = 1500;
// Up to this many qubits the state is contracted to 2^n amplitudes, to sample the shots and draw
// the figures (the same constant as in web_kernel_worker.js).
const CONTRACT_LIMIT = 20;

// Same gates as the palette of tools/dashboard/core/graphical_builder.py.
const PALETTE = [
  { id: "h", label: "H", kind: "single", gate: "h" },
  { id: "x", label: "X", kind: "single", gate: "x" },
  { id: "y", label: "Y", kind: "single", gate: "y" },
  { id: "z", label: "Z", kind: "single", gate: "z" },
  { id: "s", label: "S", kind: "single", gate: "s" },
  { id: "sdg", label: "S†", kind: "single", gate: "sdg" },
  { id: "t", label: "T", kind: "single", gate: "t" },
  { id: "tdg", label: "T†", kind: "single", gate: "tdg" },
  { id: "sx", label: "√X", kind: "single", gate: "sx" },
  { id: "rx", label: "Rx(π/2)", kind: "single", gate: "rx" },
  { id: "ry", label: "Ry(π/2)", kind: "single", gate: "ry" },
  { id: "rz", label: "Rz(π/2)", kind: "single", gate: "rz" },
  { id: "ctrl", label: "●", kind: "control", gate: null },
  { id: "tgt_x", label: "⊕", kind: "target", gate: "x" },
  { id: "tgt_z", label: "● Z", kind: "target", gate: "z" },
  { id: "tgt_y", label: "⊗ Y", kind: "target", gate: "y" },
  { id: "swap", label: "×", kind: "swap", gate: null },
];

const ONE_QUBIT = new Set(["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"]);
const ROTATION = new Set(["rx", "ry", "rz"]);
const TWO_QUBIT = new Set(["cx", "cy", "cz", "swap"]);

function header(n) {
  return ['OPENQASM 2.0;', 'include "qelib1.inc";', `qreg q[${n}];`, `creg c[${n}];`];
}

// The builder's op list -> OpenQASM text (the same job as /api/build_from_ops).
// A placed Rx/Ry/Rz is a pi/2 rotation, as in the local kernel.
function opsToQasm(n, ops) {
  const lines = header(n);
  for (const op of ops) {
    const gate = op.gate;
    const q = op.qubits;
    if (ONE_QUBIT.has(gate)) lines.push(`${gate} q[${q[0]}];`);
    else if (ROTATION.has(gate)) lines.push(`${gate}(pi/2) q[${q[0]}];`);
    else if (TWO_QUBIT.has(gate)) lines.push(`${gate} q[${q[0]}],q[${q[1]}];`);
    else throw new Error(`Gate non supportato in modalita browser: ${gate}`);
  }
  lines.push("measure q -> c;");
  return lines.join("\n") + "\n";
}

function ghz(n) {
  const lines = header(n);
  lines.push("h q[0];");
  for (let i = 0; i + 1 < n; i++) lines.push(`cx q[${i}],q[${i + 1}];`);
  lines.push("measure q -> c;");
  return lines.join("\n") + "\n";
}

function qft(n) {
  const lines = header(n);
  for (let j = 0; j < n; j++) {
    lines.push(`h q[${j}];`);
    for (let k = j + 1; k < n; k++) lines.push(`cp(pi/${2 ** (k - j)}) q[${k}],q[${j}];`);
  }
  for (let i = 0; i < Math.floor(n / 2); i++) lines.push(`swap q[${i}],q[${n - 1 - i}];`);
  lines.push("measure q -> c;");
  return lines.join("\n") + "\n";
}

function generatedPresets() {
  const bell = header(2).concat(["h q[0];", "cx q[0],q[1];", "measure q -> c;"]).join("\n") + "\n";
  const plus = header(1).concat(["h q[0];", "measure q -> c;"]).join("\n") + "\n";
  return {
    "Bell state (2 qubit)": bell,
    "GHZ state (3 qubit)": ghz(3),
    "GHZ state (10 qubit)": ghz(10),
    "GHZ state (100 qubit)": ghz(100),
    "GHZ state (500 qubit)": ghz(500),
    "GHZ state (1500 qubit)": ghz(1500),
    "Superposition (1 qubit)": plus,
    "Quantum Fourier Transform (5 qubit)": qft(5),
    "Quantum Fourier Transform (20 qubit)": qft(20),
  };
}

// The circuits of the library of the dashboard (qasm_library.json is written from
// tools/dashboard/core/qasm_library.py and a test keeps the two equal), then the larger ones that
// only this page generates and that the library does not have.
async function presets() {
  const out = {};
  try {
    const res = await fetch(new URL("qasm_library.json", BASE).href);
    if (res.ok) Object.assign(out, await res.json());
  } catch (err) {
    // the generated circuits below are still offered
  }
  for (const [name, text] of Object.entries(generatedPresets())) {
    if (!(name in out)) out[name] = text;
  }
  return out;
}

let worker = null;
let readyPromise = null;
let nextId = 0;
const pending = new Map();

function call(message) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...message, id });
  });
}

function failAll(error) {
  for (const { reject } of pending.values()) reject(error);
  pending.clear();
}

async function start() {
  if (typeof Worker === "undefined") {
    throw new Error("Questo browser non supporta i Web Worker.");
  }
  worker = new Worker(BASE + "web_kernel_worker.js");
  worker.onmessage = (event) => {
    const { id, ok, result, error } = event.data;
    const entry = pending.get(id);
    if (!entry) return;
    pending.delete(id);
    if (ok) entry.resolve(result);
    else entry.reject(new Error(error));
  };
  worker.onerror = (event) => {
    failAll(new Error(event.message || "errore nel worker del motore NumPy"));
  };
  await call({ type: "init", base: BASE + "py/" });
}

// Loads Pyodide, NumPy and the simulator the first time; the same promise after.
function ready() {
  if (!readyPromise) {
    readyPromise = start().catch((err) => {
      readyPromise = null;
      if (worker) worker.terminate();
      worker = null;
      throw err;
    });
  }
  return readyPromise;
}

// OpenQASM text -> the result /api/run of the local kernel gives (up to CONTRACT_LIMIT qubits),
// or the "large_scale" one with the most probable states.
async function run(qasm, options) {
  const o = options || {};
  await ready();
  return call({
    type: "run",
    qasm,
    shots: o.shots || 1000,
    seed: o.seed === undefined ? 42 : o.seed,
    k: o.k || 32,
    noiseModel: o.noiseModel || "ideal",
    noiseP: o.noiseP || 0,
  });
}

// The models of NoiseModel.MODELS; a test compares the list with the package.
const NOISE_MODELS = ["ideal", "depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined"];

window.DEWebKernel = { MAX_QUBITS, CONTRACT_LIMIT, PALETTE, NOISE_MODELS, opsToQasm, presets, ready, run };

})();
