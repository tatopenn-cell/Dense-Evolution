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

window.DEQuirkBridge = { quirkToQasm, circuitFromHash };

})();
