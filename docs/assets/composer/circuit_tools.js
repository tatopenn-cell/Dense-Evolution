// The entangling topologies, the input values of the qubits and the tables of a run, for the page.
//
// Each function is the translation of one of tools/dashboard/circuit_tools.py (topology_qasm,
// with_input_values, qubits_in_qasm, outcome_rows) and of dense_evolution/circuits/topology.py
// (entangling_layer); tests/unit/test_composer_circuit_tools.py runs this file with Node and compares
// every result with those functions, so a change of one that is not copied here fails.
(function () {

const PATTERNS = ["linear", "circular", "full", "star", "brick"];
const GATES = ["cx", "cz", "cy"];

// The pairs (control, target) of dense_evolution.circuits.topology.entangling_layer.
function topologyEdges(n, pattern, reverse, hub) {
  if (!(n >= 2)) throw new Error(`n_qubits must be >= 2, got ${n}`);
  if (!PATTERNS.includes(pattern)) throw new Error(`pattern must be one of ${PATTERNS.join(", ")}, got ${pattern}`);
  const h = hub || 0;
  let edges = [];
  if (pattern === "linear" || pattern === "circular") {
    for (let q = 0; q < n - 1; q++) edges.push([q, q + 1]);
    if (pattern === "circular" && n > 2) edges.push([n - 1, 0]);
  } else if (pattern === "full") {
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) edges.push([i, j]);
  } else if (pattern === "star") {
    if (!(h >= 0 && h < n)) throw new Error(`hub must satisfy 0 <= hub < ${n}, got ${h}`);
    for (let q = 0; q < n; q++) if (q !== h) edges.push([h, q]);
  } else {
    for (let q = 0; q < n - 1; q += 2) edges.push([q, q + 1]);
    for (let q = 1; q < n - 1; q += 2) edges.push([q, q + 1]);
  }
  if (reverse) edges = edges.map(([a, b]) => [b, a]);
  return edges;
}

// topology_qasm: an optional layer of h, then the layer, written as OpenQASM 2.0.
function topologyQasm(n, pattern, gate, reverse, hub, hadamards) {
  if (!GATES.includes(gate)) throw new Error(`gate must be one of ${GATES.join(", ")}, got ${gate}`);
  const edges = topologyEdges(n, pattern, reverse, hub);
  const lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', `qreg q[${n}];`, `creg c[${n}];`];
  if (hadamards) for (let q = 0; q < n; q++) lines.push(`h q[${q}];`);
  for (const [a, b] of edges) lines.push(`${gate} q[${a}],q[${b}];`);
  lines.push("measure q -> c;");
  return lines.join("\n") + "\n";
}

const QREG = /^\s*qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;/m;
const HEAD = /^\s*(OPENQASM|include|qreg|creg)\b/;

// qubits_in_qasm: the size of the first qreg, 0 if there is none.
function qubitsInQasm(text) {
  const m = QREG.exec(text);
  return m ? parseInt(m[2], 10) : 0;
}

// with_input_values: an x line for each chosen qubit, right after the declarations at the head.
function withInputValues(text, ones) {
  const indices = [...new Set((ones || []).map((i) => Math.trunc(Number(i))))].sort((a, b) => a - b);
  for (const i of indices) if (i < 0) throw new Error(`qubit index must be >= 0, got ${i}`);
  if (!indices.length) return text;
  const m = QREG.exec(text);
  if (!m) throw new Error("the circuit has no qreg declaration");
  const size = parseInt(m[2], 10);
  const kept = indices.filter((i) => i < size);
  if (!kept.length) return text;
  const lines = text.split(/(?<=\n)/);
  let last = text.slice(0, m.index).split("\n").length - 1;
  for (let k = 0; k < lines.length; k++) {
    if (!lines[k].trim()) continue;
    if (!HEAD.test(lines[k])) break;
    last = Math.max(last, k);
  }
  if (!lines[last].endsWith("\n")) lines[last] += "\n";
  lines.splice(last + 1, 0, ...kept.map((i) => `x ${m[1]}[${i}];\n`));
  return lines.join("");
}

// outcome_rows: the outcomes of a run, most frequent first, equal counts by outcome.
function outcomeRows(counts) {
  const total = Object.values(counts || {}).reduce((s, v) => s + v, 0);
  if (!(total > 0)) return [];
  const rows = Object.entries(counts).map(([outcome, shots]) => ({ outcome, shots, probability: shots / total }));
  rows.sort((a, b) => b.shots - a.shots || (a.outcome < b.outcome ? -1 : a.outcome > b.outcome ? 1 : 0));
  return rows;
}

// P(1) of each qubit from the statevector rows of a run ({state: bitstring of Qiskit order, abs}), and
// the same for a state that is cut to its rows: the weights are |amplitude|^2 and the qubit q is the
// character q from the right, as in dense_evolution's own tables.
function qubitP1(rows, n) {
  const out = Array(n).fill(0);
  let norm = 0;
  for (const r of rows) {
    const w = r.abs * r.abs;
    norm += w;
    for (let q = 0; q < n; q++) if (r.state[n - 1 - q] === "1") out[q] += w;
  }
  return norm > 0 ? out.map((v) => v / norm) : out;
}

window.DECircuitTools = { PATTERNS, GATES, topologyEdges, topologyQasm, qubitsInQasm, withInputValues, outcomeRows, qubitP1 };

})();
