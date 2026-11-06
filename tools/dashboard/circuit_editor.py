"""
Drag-and-drop circuit editor for the Dashboard: the circuit is kept in the
page URL (`circuit` query parameter, a JSON list of [row, column, gate]
cells), with undo/redo (a history list and an index) and the gate matrix
shown on hover.
"""
import json
import math

import streamlit as st

from dashboard_core.graphical_builder import GATE_PALETTE

MATRICES = {
    "h": "H = 1/√2 · [[1, 1], [1, −1]]",
    "x": "X = [[0, 1], [1, 0]]",
    "y": "Y = [[0, −i], [i, 0]]",
    "z": "Z = [[1, 0], [0, −1]]",
    "s": "S = [[1, 0], [0, i]]",
    "sdg": "S† = [[1, 0], [0, −i]]",
    "t": "T = [[1, 0], [0, e^(iπ/4)]]",
    "tdg": "T† = [[1, 0], [0, e^(−iπ/4)]]",
    "sx": "√X = ½ · [[1+i, 1−i], [1−i, 1+i]]",
    "rx": "Rx(θ) = [[cos θ/2, −i sin θ/2], [−i sin θ/2, cos θ/2]]",
    "ry": "Ry(θ) = [[cos θ/2, −sin θ/2], [sin θ/2, cos θ/2]]",
    "rz": "Rz(θ) = [[e^(−iθ/2), 0], [0, e^(iθ/2)]]",
    "p": "P(θ) = [[1, 0], [0, e^(iθ)]]",
    "ctrl": "Control ●: drag it into the column of a gate and the gate acts when this qubit is |1⟩ "
    "(two ● on an X make a Toffoli)",
    "tgt_x": "Target of CX: X = [[0, 1], [1, 0]] when the control is |1⟩",
    "tgt_y": "Target of CY: Y = [[0, −i], [i, 0]] when the control is |1⟩",
    "tgt_z": "Target of CZ: Z = [[1, 0], [0, −1]] when the control is |1⟩",
    "tgt_p": "Target of CP: P(θ) = [[1, 0], [0, e^(iθ)]] when the control is |1⟩",
    "tgt_rz": "Target of CRz: Rz(θ) = [[e^(−iθ/2), 0], [0, e^(iθ/2)]] when the control is |1⟩",
    "swap": "SWAP = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]] (two × in one column)",
}

_CSS = """
.ce-root { display: flex; flex-direction: column; gap: 10px; font-family: monospace; }
.ce-palette { display: flex; flex-wrap: wrap; gap: 6px; padding: 6px; border: 1px solid var(--st-border-color, #444); border-radius: 6px; }
.ce-chip {
  padding: 4px 10px; border-radius: 4px; cursor: grab; user-select: none; font-size: 14px;
  background: var(--st-secondary-background-color, #262730); color: var(--st-text-color, #fafafa);
  border: 1px solid var(--st-border-color, #444);
}
.ce-chip[data-kind="control"] { background: #3a5; color: #fff; }
.ce-chip[data-kind="target"] { background: #35a; color: #fff; }
.ce-chip[data-kind="swap"] { background: #a53; color: #fff; }
.ce-grid { display: grid; gap: 2px; align-items: center; }
.ce-qlabel { font-size: 12px; opacity: 0.75; text-align: right; padding-right: 6px; }
.ce-cell {
  width: 40px; height: 32px; border: 1px solid var(--st-border-color, #444); border-radius: 4px;
  display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 13px; cursor: pointer;
  background: var(--st-background-color, #0e1117); color: var(--st-text-color, #fafafa);
}
.ce-cell.ce-filled { background: var(--st-secondary-background-color, #262730); font-weight: bold; }
.ce-cell.ce-over { outline: 2px dashed #4af; }
.ce-hint { font-size: 12px; opacity: 0.75; }
.ce-cell { position: relative; z-index: 0; }
.ce-cell.ce-wire-up::before, .ce-cell.ce-wire-down::after {
  content: ''; position: absolute; left: 50%; width: 2px; margin-left: -1px; background: #4af; z-index: -1;
}
.ce-cell.ce-wire-up::before { top: -2px; height: calc(50% + 2px); }
.ce-cell.ce-wire-down::after { bottom: -2px; height: calc(50% + 2px); }
.ce-cell small { display: block; font-size: 9px; opacity: 0.85; line-height: 1; }
.ce-angle { font-size: 13px; display: flex; align-items: center; gap: 6px; margin-left: 8px; }
.ce-angle input { width: 90px; font-family: monospace; }
.ce-angle input.ce-bad { outline: 2px solid #e55; }
"""

_JS = r"""
export default function(component) {
  const { data, setStateValue, setTriggerValue, parentElement } = component;
  const nQubits = Math.max(1, data.n_qubits || 1);
  const nCols = Math.max(1, data.n_columns || 12);
  const palette = data.palette || [];
  const byId = Object.fromEntries(palette.map((g) => [g.id, g]));
  const grid = Array.from({ length: nQubits }, (_, r) =>
    Array.from({ length: nCols }, (_, c) => ((data.initial || [])[r] || [])[c] || null));
  const cellEls = Array.from({ length: nQubits }, () => Array(nCols).fill(null));

  parentElement.querySelectorAll('.ce-root').forEach((el) => el.remove());
  const root = document.createElement('div');
  root.className = 'ce-root';
  const paletteEl = document.createElement('div');
  paletteEl.className = 'ce-palette';
  palette.forEach((g) => {
    const chip = document.createElement('div');
    chip.className = 'ce-chip';
    chip.draggable = true;
    chip.textContent = g.label;
    chip.dataset.kind = g.kind;
    chip.title = g.matrix;
    chip.addEventListener('dragstart', (e) => {
      let id = g.id;
      if (g.param) {
        const a = parseAngle(angleInput.value);
        id = `${g.id}:${(a === null ? Math.PI / 2 : a).toPrecision(12)}`;
      }
      e.dataTransfer.setData('text/plain', id);
      e.dataTransfer.effectAllowed = 'copy';
    });
    paletteEl.appendChild(chip);
  });
  const angleWrap = document.createElement('label');
  angleWrap.className = 'ce-angle';
  angleWrap.textContent = 'θ for Rx Ry Rz P ● P ● Rz:';
  const angleInput = document.createElement('input');
  angleInput.type = 'text';
  angleInput.value = 'pi/2';
  angleInput.title = 'Radians: a number or an expression such as pi/4, 3*pi/8, 0.5';
  angleInput.addEventListener('input', () => {
    angleInput.classList.toggle('ce-bad', parseAngle(angleInput.value) === null);
  });
  angleWrap.appendChild(angleInput);
  paletteEl.appendChild(angleWrap);
  root.appendChild(paletteEl);

  const gridEl = document.createElement('div');
  gridEl.className = 'ce-grid';
  gridEl.style.gridTemplateColumns = `36px repeat(${nCols}, 40px)`;
  for (let r = 0; r < nQubits; r++) {
    const label = document.createElement('div');
    label.className = 'ce-qlabel';
    label.textContent = `q${r}`;
    gridEl.appendChild(label);
    for (let c = 0; c < nCols; c++) {
      const cell = document.createElement('div');
      cell.className = 'ce-cell';
      cell.addEventListener('dragover', (e) => { e.preventDefault(); cell.classList.add('ce-over'); });
      cell.addEventListener('dragleave', () => cell.classList.remove('ce-over'));
      cell.addEventListener('drop', (e) => {
        e.preventDefault();
        cell.classList.remove('ce-over');
        const id = e.dataTransfer.getData('text/plain');
        if (!byId[id.split(':')[0]]) return;
        grid[r][c] = id;
        renderCell(r, c);
        emit();
      });
      cell.addEventListener('click', () => {
        if (grid[r][c]) { grid[r][c] = null; renderCell(r, c); emit(); }
      });
      gridEl.appendChild(cell);
      cellEls[r][c] = cell;
    }
  }
  root.appendChild(gridEl);
  const hint = document.createElement('div');
  hint.className = 'ce-hint';
  hint.textContent = 'Drag a ● into the column of a gate to control it · hover a gate for its matrix · click a cell to remove · Ctrl+Z undo · Ctrl+Y redo';
  root.appendChild(hint);
  parentElement.appendChild(root);

  function renderCell(r, c) {
    const cell = cellEls[r][c];
    const raw = grid[r][c];
    const parts = raw ? raw.split(':') : [null];
    const g = byId[parts[0]];
    cell.classList.toggle('ce-filled', !!g);
    cell.textContent = g ? g.label : '';
    if (g && parts.length > 1) {
      const small = document.createElement('small');
      small.textContent = fmtAngle(Number(parts[1]));
      cell.appendChild(small);
    }
    cell.title = g ? g.matrix + (parts.length > 1 ? ` · θ = ${Number(parts[1])} rad` : '') : '';
    renderWires(c);
  }

  function renderWires(c) {
    const base = (r) => (grid[r][c] ? grid[r][c].split(':')[0] : null);
    const filled = [];
    let linked = false;
    for (let r = 0; r < nQubits; r++) {
      if (!grid[r][c]) continue;
      filled.push(r);
      if (base(r) === 'ctrl' || base(r) === 'swap') linked = true;
    }
    const lo = filled[0];
    const hi = filled[filled.length - 1];
    for (let r = 0; r < nQubits; r++) {
      const on = linked && filled.length > 1 && r >= lo && r <= hi;
      cellEls[r][c].classList.toggle('ce-wire-up', on && r > lo);
      cellEls[r][c].classList.toggle('ce-wire-down', on && r < hi);
    }
  }

  function fmtAngle(x) {
    for (let d = 1; d <= 8; d++) {
      const k = Math.round((x * d) / Math.PI);
      if (Math.abs(x - (k * Math.PI) / d) < 1e-6) {
        if (k === 0) return '0';
        const top = Math.abs(k) === 1 ? (k < 0 ? '-π' : 'π') : `${k}π`;
        return d === 1 ? top : `${top}/${d}`;
      }
    }
    return x.toFixed(2);
  }

  function parseAngle(text) {
    const src = String(text).toLowerCase().replace(/π/g, 'pi').replace(/\s+/g, '');
    let pos = 0;
    function number() {
      const m = /^(\d+\.?\d*|\.\d+)/.exec(src.slice(pos));
      if (m) { pos += m[0].length; return parseFloat(m[0]); }
      if (src.startsWith('pi', pos)) { pos += 2; return Math.PI; }
      if (src[pos] === '(') {
        pos++;
        const v = sum();
        if (src[pos] !== ')') return NaN;
        pos++;
        return v;
      }
      return NaN;
    }
    function unary() {
      if (src[pos] === '-') { pos++; return -unary(); }
      if (src[pos] === '+') { pos++; return unary(); }
      return number();
    }
    function product() {
      let v = unary();
      while (src[pos] === '*' || src[pos] === '/') {
        const op = src[pos++];
        const w = unary();
        v = op === '*' ? v * w : v / w;
      }
      return v;
    }
    function sum() {
      let v = product();
      while (src[pos] === '+' || src[pos] === '-') {
        const op = src[pos++];
        const w = product();
        v = op === '+' ? v + w : v - w;
      }
      return v;
    }
    if (src === '') return null;
    const v = sum();
    return pos === src.length && Number.isFinite(v) ? v : null;
  }

  function emit() { setStateValue('grid', grid.map((row) => row.slice())); }

  root.tabIndex = 0;
  root.addEventListener('keydown', (e) => {
    if (!e.ctrlKey || e.altKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    if (k === 'z' && !e.shiftKey) { setTriggerValue('history', 'undo'); e.preventDefault(); }
    if ((k === 'z' && e.shiftKey) || k === 'y') { setTriggerValue('history', 'redo'); e.preventDefault(); }
  });

  for (let r = 0; r < nQubits; r++) for (let c = 0; c < nCols; c++) renderCell(r, c);
}
"""


def encode_grid(grid):
    cells = [[r, c, g] for r, row in enumerate(grid) for c, g in enumerate(row) if g]
    return json.dumps({"n": len(grid), "cells": cells}, separators=(",", ":"))


def _split(cell):
    base, _, angle = cell.partition(":")
    return base, (float(angle) if angle else math.pi / 2)


def decode_grid(text, n_qubits, n_columns=12):
    grid = [[None] * n_columns for _ in range(n_qubits)]
    for r, c, g in json.loads(text).get("cells", []):
        if _split(g)[0] not in MATRICES:
            raise ValueError(f"unknown gate {g!r}")
        if r < n_qubits and c < n_columns:
            grid[r][c] = g
    return grid


_PARAM_SINGLES = {"rx", "ry", "rz", "p"}
_CONTROLLED = {"y": "cy", "z": "cz", "p": "cp", "rz": "crz"}


def _unify(base):
    """Cells written with a separate target (``tgt_x``, saved links) are the plain gate."""
    return base[4:] if base.startswith("tgt_") else base


def grid_to_ops(grid):
    """Op list of the grid. A column with ● control dots and one other gate is that gate
    controlled by them (one dot: cx cy cz cp crz; two dots on X: ccx)."""
    ops = []
    for c in range(len(grid[0]) if grid else 0):
        cells = [
            (r, _unify(g), a)
            for r in range(len(grid))
            if grid[r][c]
            for g, a in [_split(grid[r][c])]
        ]
        ctrls = [r for r, g, _ in cells if g == "ctrl"]
        swaps = [r for r, g, _ in cells if g == "swap"]
        gates = [(r, g, a) for r, g, a in cells if g not in ("ctrl", "swap")]
        if not ctrls and not swaps:
            for r, g, a in gates:
                ops.append(
                    {"gate": g, "qubits": [r], **({"angle": a} if g in _PARAM_SINGLES else {})}
                )
        elif ctrls and len(gates) == 1 and len(cells) == len(ctrls) + 1:
            r, g, a = gates[0]
            if g == "x" and len(ctrls) == 1:
                ops.append({"gate": "cx", "qubits": [ctrls[0], r]})
            elif g == "x" and len(ctrls) == 2:
                ops.append({"gate": "ccx", "qubits": [*ctrls, r]})
            elif g in _CONTROLLED and len(ctrls) == 1:
                op = {"gate": _CONTROLLED[g], "qubits": [ctrls[0], r]}
                if g in _PARAM_SINGLES:
                    op["angle"] = a
                ops.append(op)
        elif len(swaps) == 2 and len(cells) == 2:
            ops.append({"gate": "swap", "qubits": swaps})
    return ops


def ops_to_tuples(ops):
    """The gate tuples of dense_evolution for the op list of ``grid_to_ops``."""
    return [(op["gate"], *op["qubits"], *([op["angle"]] if "angle" in op else [])) for op in ops]


_SINGLE_GATES = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"}
_ROTATIONS = {"rx": "rx", "ry": "ry", "rz": "rz", "p": "p", "u1": "p", "phase": "p"}
_TARGETS = {"cx": "x", "cy": "y", "cz": "z"}
_ANGLED_TARGETS = {"cp": "p", "crz": "rz"}


def grid_from_ops(ops, n_qubits):
    """Draw a gate-tuple circuit on the editor grid.

    Returns ``(grid, skipped)``: ``grid`` has ``n_qubits`` rows and at least twelve columns,
    ``skipped`` counts the gates the grid cannot show (``id``, ``u3``, ...). Single-qubit gates
    on different qubits share a column; a gate on two or three qubits has a column of its own,
    as the editor requires. Angles are kept to 12 significant digits.
    """
    columns = []
    singles_open = False
    skipped = 0
    for op in ops:
        name, args = op[0], op[1:]
        single = False
        if name in _SINGLE_GATES:
            cells, single = {args[0]: name}, True
        elif name in _ROTATIONS:
            cells, single = {args[0]: f"{_ROTATIONS[name]}:{format(args[1], '.12g')}"}, True
        elif name in _TARGETS:
            cells = {args[0]: "ctrl", args[1]: _TARGETS[name]}
        elif name in _ANGLED_TARGETS:
            cells = {args[0]: "ctrl", args[1]: f"{_ANGLED_TARGETS[name]}:{format(args[2], '.12g')}"}
        elif name == "ccx":
            cells = {args[0]: "ctrl", args[1]: "ctrl", args[2]: "x"}
        elif name == "swap":
            cells = {args[0]: "swap", args[1]: "swap"}
        else:
            skipped += 1
            continue
        if single and singles_open and not set(cells) & set(columns[-1]):
            columns[-1].update(cells)
        else:
            columns.append(dict(cells))
            singles_open = single
    n_columns = max(len(columns), 12)
    grid = [[None] * n_columns for _ in range(n_qubits)]
    for c, cells in enumerate(columns):
        for r, g in cells.items():
            grid[r][c] = g
    return grid, skipped


_ANGLE_CHIPS = [
    {"id": "rx", "label": "Rx", "kind": "single", "param": True},
    {"id": "ry", "label": "Ry", "kind": "single", "param": True},
    {"id": "rz", "label": "Rz", "kind": "single", "param": True},
    {"id": "p", "label": "P", "kind": "single", "param": True},
]


def _palette():
    skip = ("rx", "ry", "rz", "tgt_x", "tgt_y", "tgt_z")
    chips = [g for g in GATE_PALETTE if g["id"] not in skip] + _ANGLE_CHIPS
    return [dict(g, matrix=MATRICES[g["id"]]) for g in chips]


def _push(grid):
    h, i = st.session_state["ce_hist"], st.session_state["ce_idx"]
    if grid != h[i]:
        del h[i + 1 :]
        h.append(grid)
        st.session_state["ce_idx"] = len(h) - 1


def _move(step):
    i = st.session_state["ce_idx"] + step
    if 0 <= i < len(st.session_state["ce_hist"]):
        st.session_state["ce_idx"] = i
        st.session_state["ce_version"] += 1


def circuit_editor(n_qubits, n_columns=12):
    """Mount the editor; returns the op list of the current grid.

    A change of ``n_columns`` keeps the gates that still fit in the new number of
    steps and starts a new history.
    """
    if st.session_state.get("ce_n") != n_qubits:
        start = [[None] * n_columns for _ in range(n_qubits)]
        link = st.query_params.get("circuit")
        if link and "ce_hist" not in st.session_state:
            try:
                start = decode_grid(link, n_qubits, n_columns)
            except (ValueError, KeyError, TypeError):
                pass
        st.session_state.update(
            ce_n=n_qubits, ce_hist=[start], ce_idx=0, ce_version=0, ce_start_version=None
        )
    elif len(st.session_state["ce_hist"][st.session_state["ce_idx"]][0]) != n_columns:
        grid = st.session_state["ce_hist"][st.session_state["ce_idx"]]
        resized = [list(row[:n_columns]) + [None] * (n_columns - len(row)) for row in grid]
        st.session_state.update(
            ce_hist=[resized],
            ce_idx=0,
            ce_version=st.session_state["ce_version"] + 1,
            ce_start_version=None,
        )

    current = st.session_state["ce_hist"][st.session_state["ce_idx"]]
    if st.session_state.get("ce_start_version") != st.session_state["ce_version"]:
        st.session_state.update(ce_start=current, ce_start_version=st.session_state["ce_version"])
    start = st.session_state["ce_start"]
    comp = st.components.v2.component("dense_evolution_circuit_editor", css=_CSS, js=_JS)
    res = comp(
        key=f"ce_{n_qubits}_{st.session_state['ce_version']}",
        data={
            "n_qubits": n_qubits,
            "n_columns": len(start[0]),
            "palette": _palette(),
            "initial": start,
        },
        default={"grid": start},
        on_grid_change=lambda: None,
        on_history_change=lambda: None,
    )
    if res.history in ("undo", "redo"):
        _move(-1 if res.history == "undo" else 1)
        st.rerun()
    if res.grid is not None and len(res.grid) == n_qubits and len(res.grid[0]) == len(start[0]):
        _push(res.grid)

    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("↶ Undo", disabled=st.session_state["ce_idx"] == 0, key="ce_undo"):
        _move(-1)
        st.rerun()
    if b2.button(
        "↷ Redo",
        disabled=st.session_state["ce_idx"] >= len(st.session_state["ce_hist"]) - 1,
        key="ce_redo",
    ):
        _move(1)
        st.rerun()

    grid = st.session_state["ce_hist"][st.session_state["ce_idx"]]
    st.query_params["circuit"] = encode_grid(grid)
    return grid_to_ops(grid)
