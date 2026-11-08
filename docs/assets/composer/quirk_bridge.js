// Quirk circuit (the JSON of its address) -> OpenQASM 2.0 for the engines of Dense-Evolution.
//
// Quirk (third_party/quirk, Apache-2.0) is the editor; this file reads the circuit it keeps
// in its address (#circuit={"cols":[...],"init":[...]}) and writes the OpenQASM that
// /api/run and the NumPy engine read. Only identities are used: a power of a Pauli gate is
// a phase gate between two basis changes (X^t = H Z^t H, Y^t = S X^t S-dagger,
// Z^t = diag(1, e^(i pi t))), so a controlled power is a controlled phase between the same
// changes, exactly and with no global phase. Whatever has no such translation is returned
// in `skipped`, with the reason, and is not run.
(function () {

const POWERS = {
  "": 1, "^½": 0.5, "^-½": -0.5, "^¼": 0.25, "^-¼": -0.25, "^⅓": 1 / 3, "^-⅓": -1 / 3,
};
// Quirk's ids of the gates that show something and change nothing.
const DISPLAYS = new Set(["Bloch", "Amps", "Chance", "Density", "Sample", "…", "Detector"]);
const INIT = { 0: [], 1: ["x"], "+": ["h"], "-": ["x", "h"], "i": ["h", "s"], "-i": ["h", "sdg"] };

function angleText(exponent) {
  return `pi*${exponent}`;
}

// A plain number or an expression of numbers and pi, with no time variable t.
function constantAngle(arg) {
  const text = String(arg).replace(/pi/g, "3.141592653589793");
  if (!/^[0-9.+\-*/() ]+$/.test(text)) return null;
  try {
    const value = Function(`"use strict"; return (${text});`)();
    return Number.isFinite(value) ? value : null;
  } catch (err) {
    return null;
  }
}

function parseId(id) {
  const m = /^([XYZ])(\^.*)?$/.exec(id);
  if (!m) return null;
  const suffix = m[2] || "";
  return suffix in POWERS ? { axis: m[1], exponent: POWERS[suffix] } : null;
}

// The lines of one gate on `target`, with the control qubits `controls` (every one must be 1).
function targetLines(id, arg, target, controls, why) {
  const t = `q[${target}]`;
  const cs = controls.map((c) => `q[${c}]`);
  const named = parseId(id);
  if (named) {
    const { axis, exponent } = named;
    const phase = angleText(exponent);
    const pre = { X: ["h"], Y: ["sdg", "h"], Z: [] }[axis];
    const post = { X: ["h"], Y: ["h", "s"], Z: [] }[axis];
    const wrap = (inner) => [...pre.map((g) => `${g} ${t};`), ...inner, ...post.map((g) => `${g} ${t};`)];
    if (cs.length === 0) {
      if (exponent === 1) return [`${axis.toLowerCase()} ${t};`];
      return wrap([`p(${phase}) ${t};`]);
    }
    if (cs.length === 1) {
      if (exponent === 1 && axis !== "Z") return [`c${axis.toLowerCase()} ${cs[0]},${t};`];
      return wrap([`cp(${phase}) ${cs[0]},${t};`]);
    }
    if (cs.length === 2 && axis === "X" && exponent === 1) return [`ccx ${cs[0]},${cs[1]},${t};`];
    why.push(`${id} with ${cs.length} controls`);
    return null;
  }
  if (id === "H" && cs.length === 0) return [`h ${t};`];
  if (["Rxft", "Ryft", "Rzft"].includes(id) && cs.length === 0) {
    const angle = constantAngle(arg);
    if (angle !== null) return [`${id.slice(0, 2).toLowerCase()}(${angle}) ${t};`];
    why.push(`${id} with a formula of time`);
    return null;
  }
  why.push(cs.length ? `${id} with controls` : `gate ${id}`);
  return null;
}

// Returns {n, qasm, skipped}: n qubits, the OpenQASM text, and one text for each gate not run.
function quirkToQasm(circuit) {
  const cols = circuit.cols || [];
  const init = circuit.init || [];
  let n = init.length;
  for (const col of cols) n = Math.max(n, col.length);
  n = Math.max(n, 1);
  const lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', `qreg q[${n}];`, `creg c[${n}];`];
  const skipped = [];
  init.forEach((state, wire) => {
    const prep = INIT[state];
    if (prep === undefined) skipped.push(`initial state ${state} of wire ${wire}`);
    else prep.forEach((g) => lines.push(`${g} q[${wire}];`));
  });
  cols.forEach((col, index) => {
    const controls = [];
    const antis = [];
    const swaps = [];
    const targets = [];
    col.forEach((cell, wire) => {
      const id = cell && typeof cell === "object" ? cell.id : cell;
      if (id === 1 || id === undefined || id === null) return;
      if (id === "•") controls.push(wire);
      else if (id === "◦") antis.push(wire);
      else if (id === "Swap") swaps.push(wire);
      else if (DISPLAYS.has(id)) return;
      else targets.push({ id, arg: cell && typeof cell === "object" ? cell.arg : undefined, wire });
    });
    const here = `column ${index + 1}`;
    const flips = antis.map((w) => `x q[${w}];`);
    const all = controls.concat(antis);
    const body = [];
    const why = [];
    if (swaps.length) {
      if (swaps.length === 2 && all.length === 0 && targets.length === 0) body.push(`swap q[${swaps[0]}],q[${swaps[1]}];`);
      else why.push("Swap with controls or other gates");
    }
    for (const { id, arg, wire } of targets) {
      const made = targetLines(id, arg, wire, all, why);
      if (made) body.push(...made);
    }
    if (why.length) skipped.push(...why.map((w) => `${w} (${here})`));
    if (body.length) lines.push(...flips, ...body, ...flips);
  });
  lines.push("measure q -> c;");
  return { n, qasm: lines.join("\n") + "\n", skipped };
}

// The circuit of a Quirk address, or null when there is none.
function circuitFromHash(hash) {
  const m = /^#?circuit=(.*)$/.exec(hash || "");
  if (!m) return null;
  try {
    return JSON.parse(decodeURIComponent(m[1]));
  } catch (err) {
    return null;
  }
}

// ---- the other way: OpenQASM -> the circuit of Quirk ----------------------------------------

const QUIRK_MAX_WIRES = 16;
const SINGLE = {
  h: "H", x: "X", y: "Y", z: "Z", s: "Z^½", sdg: "Z^-½", t: "Z^¼", tdg: "Z^-¼", sx: "X^½",
};
const CONTROLLED_TARGET = { cx: "X", cy: "Y", cz: "Z" };
// Quirk's ids of Z^t for the exponents t that it has, found from t = angle / pi.
const Z_POWERS = [[1, "Z"], [0.5, "Z^½"], [-0.5, "Z^-½"], [0.25, "Z^¼"], [-0.25, "Z^-¼"], [1 / 3, "Z^⅓"], [-1 / 3, "Z^-⅓"]];

function zPowerId(angle) {
  const t = angle / Math.PI;
  const hit = Z_POWERS.find(([e]) => Math.abs(e - t) < 1e-9);
  return hit ? hit[1] : null;
}

// "name(angle) q[0],q[1]" -> {name, angle (the text inside the parentheses, or null), qubits}
function splitStatement(statement) {
  const m = /^([A-Za-z_][A-Za-z0-9_]*)\s*/.exec(statement);
  if (!m) return null;
  let rest = statement.slice(m[0].length);
  let angle = null;
  if (rest.startsWith("(")) {
    let depth = 0;
    let end = -1;
    for (let i = 0; i < rest.length; i++) {
      if (rest[i] === "(") depth++;
      else if (rest[i] === ")" && --depth === 0) {
        end = i;
        break;
      }
    }
    if (end < 0) return null;
    angle = rest.slice(1, end);
    rest = rest.slice(end + 1);
  }
  const qubits = [];
  for (const part of rest.split(",")) {
    const q = /\[\s*(\d+)\s*\]/.exec(part);
    if (!q) return null;
    qubits.push(parseInt(q[1], 10));
  }
  return { name: m[1].toLowerCase(), angle, qubits };
}

// Returns {n, circuit, skipped}: the number of qubits, the circuit of Quirk ({cols: [...]}, one
// gate in each column, in the order of the text) and one text for each gate with no Quirk gate.
// With maxWires, the gates on a qubit from maxWires up are not drawn (Quirk has no such wire): they
// are named in `skipped` and stay in the text, and the circuit has min(n, maxWires) wires.
function qasmToQuirk(text, maxWires) {
  const limit = maxWires || Infinity;
  const skipped = [];
  const gates = [];
  let n = 0;
  const statements = String(text).replace(/\/\/[^\n]*/g, "").split(";").map((s) => s.trim()).filter(Boolean);
  statements.forEach((statement, index) => {
    const head = statement.split(/[\s(]/)[0].toLowerCase();
    if (["openqasm", "include", "creg", "barrier", "measure", "id", "i"].includes(head)) return;
    if (head === "qreg") {
      const q = /\[\s*(\d+)\s*\]/.exec(statement);
      if (q) n = parseInt(q[1], 10);
      return;
    }
    const g = splitStatement(statement);
    if (!g) {
      skipped.push(`statement ${index + 1} not understood`);
      return;
    }
    gates.push({ ...g, index });
  });
  for (const g of gates) n = Math.max(n, ...g.qubits.map((q) => q + 1));
  const wires = Math.min(n, limit);
  const cols = [];
  const place = (cells) => {
    const col = Array(wires).fill(1);
    Object.entries(cells).forEach(([wire, id]) => { col[wire] = id; });
    cols.push(col);
  };
  for (const { name, angle, qubits, index } of gates) {
    const q = qubits;
    if (q.some((w) => w >= limit)) {
      skipped.push(`${name} (statement ${index + 1}) is on a qubit beyond the ${limit} wires of Quirk`);
      continue;
    }
    const theta = angle === null ? null : constantAngle(angle);
    if (name in SINGLE && q.length === 1) place({ [q[0]]: SINGLE[name] });
    else if (["rx", "ry", "rz"].includes(name) && q.length === 1 && theta !== null) {
      place({ [q[0]]: { id: `R${name[1]}ft`, arg: String(theta) } });
    } else if (["p", "u1", "phase"].includes(name) && q.length === 1 && theta !== null) {
      const id = zPowerId(theta);
      place({ [q[0]]: id || { id: "Rzft", arg: String(theta) } });
    } else if (name in CONTROLLED_TARGET && q.length === 2) place({ [q[0]]: "•", [q[1]]: CONTROLLED_TARGET[name] });
    else if (name === "swap" && q.length === 2) place({ [q[0]]: "Swap", [q[1]]: "Swap" });
    else if (name === "ccx" && q.length === 3) place({ [q[0]]: "•", [q[1]]: "•", [q[2]]: "X" });
    else if (name === "cp" && q.length === 2 && theta !== null && zPowerId(theta)) {
      place({ [q[0]]: "•", [q[1]]: zPowerId(theta) });
    } else skipped.push(`${name} (statement ${index + 1}) has no gate in Quirk`);
  }
  return { n: Math.max(n, 1), circuit: { cols }, skipped };
}

// The gate lines of an OpenQASM text that touch a qubit from `from` up (what Quirk cannot show),
// to keep them when the lines on the lower qubits are written again from the circuit of Quirk.
function linesFromQubit(text, from) {
  const out = [];
  String(text).replace(/\/\/[^\n]*/g, "").split(";").map((s) => s.trim()).filter(Boolean).forEach((st) => {
    const head = st.split(/[\s(]/)[0].toLowerCase();
    if (["openqasm", "include", "qreg", "creg", "barrier", "measure"].includes(head)) return;
    const g = splitStatement(st);
    if (g && g.qubits.some((q) => q >= from)) out.push(st + ";");
  });
  return out;
}

window.DEQuirkBridge = { quirkToQasm, circuitFromHash, qasmToQuirk, linesFromQubit, QUIRK_MAX_WIRES };

})();
