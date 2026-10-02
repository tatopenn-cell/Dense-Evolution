"""
Drag-and-drop circuit editor for the Dashboard: the circuit is kept in the
page URL (`circuit` query parameter, a JSON list of [row, column, gate]
cells), with undo/redo (a history list and an index) and the gate matrix
shown on hover.
"""
import json

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
    "rx": "Rx(π/2) = 1/√2 · [[1, −i], [−i, 1]]",
    "ry": "Ry(π/2) = 1/√2 · [[1, −1], [1, 1]]",
    "rz": "Rz(π/2) = [[e^(−iπ/4), 0], [0, e^(iπ/4)]]",
    "ctrl": "Control ●: the column's gate acts when this qubit is |1⟩",
    "tgt_x": "Target of CX: X = [[0, 1], [1, 0]] when the control is |1⟩",
    "tgt_y": "Target of CY: Y = [[0, −i], [i, 0]] when the control is |1⟩",
    "tgt_z": "Target of CZ: Z = [[1, 0], [0, −1]] when the control is |1⟩",
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
  display: flex; align-items: center; justify-content: center; font-size: 13px; cursor: pointer;
  background: var(--st-background-color, #0e1117); color: var(--st-text-color, #fafafa);
}
.ce-cell.ce-filled { background: var(--st-secondary-background-color, #262730); font-weight: bold; }
.ce-cell.ce-over { outline: 2px dashed #4af; }
.ce-hint { font-size: 12px; opacity: 0.75; }
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
      e.dataTransfer.setData('text/plain', g.id);
      e.dataTransfer.effectAllowed = 'copy';
    });
    paletteEl.appendChild(chip);
  });
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
        if (!byId[id]) return;
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
  hint.textContent = 'Hover a gate for its matrix · click a cell to remove · Ctrl+Z undo · Ctrl+Y redo';
  root.appendChild(hint);
  parentElement.appendChild(root);

  function renderCell(r, c) {
    const cell = cellEls[r][c];
    const g = byId[grid[r][c]];
    cell.classList.toggle('ce-filled', !!g);
    cell.textContent = g ? g.label : '';
    cell.title = g ? g.matrix : '';
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


def decode_grid(text, n_qubits, n_columns=12):
    grid = [[None] * n_columns for _ in range(n_qubits)]
    for r, c, g in json.loads(text).get("cells", []):
        if g not in MATRICES:
            raise ValueError(f"unknown gate {g!r}")
        if r < n_qubits and c < n_columns:
            grid[r][c] = g
    return grid


def grid_to_ops(grid):
    ops = []
    for c in range(len(grid[0]) if grid else 0):
        cells = [(r, grid[r][c]) for r in range(len(grid)) if grid[r][c]]
        singles = [(r, g) for r, g in cells if g in MATRICES and not g.startswith(("tgt_", "ctrl", "swap"))]
        ctrls = [r for r, g in cells if g == "ctrl"]
        tgts = [(r, g[4:]) for r, g in cells if g.startswith("tgt_")]
        swaps = [r for r, g in cells if g == "swap"]
        if len(singles) == len(cells):
            ops += [{"gate": g, "qubits": [r]} for r, g in singles]
        elif len(ctrls) == 1 and len(tgts) == 1 and len(cells) == 2:
            ops.append({"gate": "c" + tgts[0][1], "qubits": [ctrls[0], tgts[0][0]]})
        elif len(swaps) == 2 and len(cells) == 2:
            ops.append({"gate": "swap", "qubits": swaps})
    return ops


def _palette():
    return [dict(g, matrix=MATRICES[g["id"]]) for g in GATE_PALETTE]


def _push(grid):
    h, i = st.session_state["ce_hist"], st.session_state["ce_idx"]
    if grid != h[i]:
        del h[i + 1:]
        h.append(grid)
        st.session_state["ce_idx"] = len(h) - 1


def _move(step):
    i = st.session_state["ce_idx"] + step
    if 0 <= i < len(st.session_state["ce_hist"]):
        st.session_state["ce_idx"] = i
        st.session_state["ce_version"] += 1


def circuit_editor(n_qubits, n_columns=12):
    """Mount the editor; returns the op list of the current grid."""
    if st.session_state.get("ce_n") != n_qubits:
        start = [[None] * n_columns for _ in range(n_qubits)]
        link = st.query_params.get("circuit")
        if link and "ce_hist" not in st.session_state:
            try:
                start = decode_grid(link, n_qubits, n_columns)
            except (ValueError, KeyError, TypeError):
                pass
        st.session_state.update(ce_n=n_qubits, ce_hist=[start], ce_idx=0, ce_version=0, ce_start_version=None)

    current = st.session_state["ce_hist"][st.session_state["ce_idx"]]
    if st.session_state.get("ce_start_version") != st.session_state["ce_version"]:
        st.session_state.update(ce_start=current, ce_start_version=st.session_state["ce_version"])
    start = st.session_state["ce_start"]
    comp = st.components.v2.component("dense_evolution_circuit_editor", css=_CSS, js=_JS)
    res = comp(
        key=f"ce_{n_qubits}_{st.session_state['ce_version']}",
        data={"n_qubits": n_qubits, "n_columns": len(start[0]), "palette": _palette(), "initial": start},
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
    if b2.button("↷ Redo", disabled=st.session_state["ce_idx"] >= len(st.session_state["ce_hist"]) - 1, key="ce_redo"):
        _move(1)
        st.rerun()

    grid = st.session_state["ce_hist"][st.session_state["ce_idx"]]
    st.query_params["circuit"] = encode_grid(grid)
    return grid_to_ops(grid)
