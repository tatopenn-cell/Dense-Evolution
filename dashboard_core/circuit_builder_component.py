"""
Graphical (drag-and-drop) circuit builder -- the fourth pillar of IBM
Quantum Composer's layout (graphical editor + code editor + Statevector +
Probabilities), the one missing piece the rest of dashboard_core didn't
have yet.

Built on Streamlit's own native bidirectional component API
(st.components.v2.component -- real HTML/CSS/JS running in the browser,
not a hand-waved "coming soon" panel). Gates are dragged from a palette
onto a qubit x time-step grid using plain HTML5 drag-and-drop; every drop
re-serializes the grid into an ops list and pushes it back to Python via
setStateValue, so Streamlit always has the real current state of what's
on the canvas.

The ops list this emits is consumed by dashboard_core.graphical_builder.
ops_to_native_tuples, which builds dense_evolution's own gate tuples
(no Qiskit) executed by the exact same dense_evolution engine as typed
OpenQASM -- no separate/fake execution path for graphically-built
circuits.
"""

import streamlit as st

from .graphical_builder import GATE_PALETTE

__all__ = ['mount_circuit_builder']

_CSS = """
.cb-root { display: flex; flex-direction: column; gap: 10px; font-family: monospace; }
.cb-palette { display: flex; flex-wrap: wrap; gap: 6px; padding: 6px; border: 1px solid var(--st-border-color, #444); border-radius: 6px; }
.cb-chip {
  padding: 4px 10px; border-radius: 4px; cursor: grab; user-select: none;
  background: var(--st-secondary-background-color, #262730);
  color: var(--st-text-color, #fafafa);
  border: 1px solid var(--st-border-color, #444);
  font-size: 14px;
}
.cb-chip:active { cursor: grabbing; }
.cb-chip[data-kind="control"] { background: #3a5; color: #fff; }
.cb-chip[data-kind="target"] { background: #35a; color: #fff; }
.cb-chip[data-kind="swap"] { background: #a53; color: #fff; }
.cb-toolbar { display: flex; gap: 8px; align-items: center; }
.cb-toolbar button {
  padding: 4px 10px; border-radius: 4px; cursor: pointer;
  background: var(--st-secondary-background-color, #262730);
  color: var(--st-text-color, #fafafa);
  border: 1px solid var(--st-border-color, #444);
}
.cb-grid { display: grid; gap: 2px; align-items: center; position: relative; }
.cb-qlabel { font-size: 12px; opacity: 0.75; text-align: right; padding-right: 6px; }
.cb-cell {
  width: 40px; height: 32px; border: 1px solid var(--st-border-color, #444);
  border-radius: 4px; display: flex; align-items: center; justify-content: center;
  font-size: 13px; cursor: pointer;
  background: var(--st-background-color, #0e1117);
  color: var(--st-text-color, #fafafa);
}
.cb-cell.cb-filled { background: var(--st-secondary-background-color, #262730); font-weight: bold; }
.cb-cell.cb-over { outline: 2px dashed #4af; }
.cb-count { font-size: 12px; opacity: 0.75; }
"""

_HTML = """
<div class="cb-toolbar">
  <span class="cb-count" data-role="count">0 porte piazzate</span>
  <button data-role="clear">Pulisci griglia</button>
</div>
"""

_JS = r"""
export default function(component) {
  const { data, setStateValue, parentElement } = component;
  const nQubits = Math.max(1, data.n_qubits || 1);
  const nCols = Math.max(1, data.n_columns || 12);
  const palette = data.palette || [];

  const grid = Array.from({ length: nQubits }, () => Array(nCols).fill(null));
  const cellEls = Array.from({ length: nQubits }, () => Array(nCols).fill(null));

  const root = document.createElement('div');
  root.className = 'cb-root';

  const paletteEl = document.createElement('div');
  paletteEl.className = 'cb-palette';
  palette.forEach((g) => {
    const chip = document.createElement('div');
    chip.className = 'cb-chip';
    chip.draggable = true;
    chip.textContent = g.label;
    chip.dataset.kind = g.kind;
    chip.title = g.kind === 'single' ? `Porta a 1 qubit: ${g.label}`
      : g.kind === 'control' ? 'Controllo -- trascina un target nella stessa colonna'
      : g.kind === 'target' ? `Target -- forma C${g.gate.toUpperCase()} con un controllo nella stessa colonna`
      : 'SWAP -- trascina due marcatori nella stessa colonna';
    chip.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', g.id);
      e.dataTransfer.effectAllowed = 'copy';
    });
    paletteEl.appendChild(chip);
  });
  root.appendChild(paletteEl);

  const gridEl = document.createElement('div');
  gridEl.className = 'cb-grid';
  gridEl.style.gridTemplateColumns = `36px repeat(${nCols}, 40px)`;

  for (let r = 0; r < nQubits; r++) {
    const label = document.createElement('div');
    label.className = 'cb-qlabel';
    label.textContent = `q${r}`;
    gridEl.appendChild(label);
    for (let c = 0; c < nCols; c++) {
      const cell = document.createElement('div');
      cell.className = 'cb-cell';
      cell.addEventListener('dragover', (e) => { e.preventDefault(); cell.classList.add('cb-over'); });
      cell.addEventListener('dragleave', () => cell.classList.remove('cb-over'));
      cell.addEventListener('drop', (e) => {
        e.preventDefault();
        cell.classList.remove('cb-over');
        const gateId = e.dataTransfer.getData('text/plain');
        const g = palette.find((x) => x.id === gateId);
        if (!g) return;
        grid[r][c] = { kind: g.kind, gate: g.gate };
        renderCell(r, c);
        emit();
      });
      cell.addEventListener('click', () => {
        if (grid[r][c]) {
          grid[r][c] = null;
          renderCell(r, c);
          emit();
        }
      });
      gridEl.appendChild(cell);
      cellEls[r][c] = cell;
    }
  }
  root.appendChild(gridEl);
  parentElement.appendChild(root);

  const countEl = root.querySelector('[data-role="count"]');
  root.querySelector('[data-role="clear"]').addEventListener('click', () => {
    for (let r = 0; r < nQubits; r++) {
      for (let c = 0; c < nCols; c++) {
        grid[r][c] = null;
        renderCell(r, c);
      }
    }
    emit();
  });

  const SYMS = { control: '●', swap: '×' };

  function renderCell(r, c) {
    const cell = cellEls[r][c];
    const v = grid[r][c];
    if (!v) {
      cell.textContent = '';
      cell.classList.remove('cb-filled');
      return;
    }
    cell.classList.add('cb-filled');
    if (v.kind === 'single') cell.textContent = v.gate.toUpperCase();
    else if (v.kind === 'target') cell.textContent = v.gate === 'x' ? '⊕' : v.gate.toUpperCase();
    else cell.textContent = SYMS[v.kind] || '?';
  }

  function buildOps() {
    const ops = [];
    for (let c = 0; c < nCols; c++) {
      let single = null;
      const controls = [];
      const targets = [];
      const swaps = [];
      for (let r = 0; r < nQubits; r++) {
        const v = grid[r][c];
        if (!v) continue;
        if (v.kind === 'single') single = { row: r, gate: v.gate };
        else if (v.kind === 'control') controls.push(r);
        else if (v.kind === 'target') targets.push({ row: r, gate: v.gate });
        else if (v.kind === 'swap') swaps.push(r);
      }
      const isEmpty = !single && controls.length === 0 && targets.length === 0 && swaps.length === 0;
      if (isEmpty) continue;
      if (single && controls.length === 0 && targets.length === 0 && swaps.length === 0) {
        ops.push({ gate: single.gate, qubits: [single.row] });
      } else if (!single && controls.length === 1 && targets.length === 1 && swaps.length === 0) {
        ops.push({ gate: 'c' + targets[0].gate, qubits: [controls[0], targets[0].row] });
      } else if (!single && controls.length === 0 && targets.length === 0 && swaps.length === 2) {
        ops.push({ gate: 'swap', qubits: swaps });
      }
      // Any other combination in a column (e.g. a lone control with no
      // target yet) is an incomplete gate -- shown on the grid but not
      // emitted as an op until it's completed.
    }
    return ops;
  }

  function emit() {
    const ops = buildOps();
    countEl.textContent = `${ops.length} porte piazzate`;
    setStateValue('circuit', ops);
  }

  emit();
}
"""

_COMPONENT_NAME = "dense_evolution_circuit_builder"


def mount_circuit_builder(n_qubits: int, n_columns: int = 12, key: str = "circuit_builder"):
    """Mount the drag-and-drop circuit builder and return its current ops
    list (list[dict] with 'gate'/'qubits'), or [] if nothing is placed yet.

    `key` should change whenever `n_qubits` changes (e.g. include it in the
    caller's key) so Streamlit mounts a fresh grid instead of reusing a
    stale one sized for a different qubit count.
    """
    builder = st.components.v2.component(_COMPONENT_NAME, html=_HTML, css=_CSS, js=_JS)
    result = builder(
        key=key,
        data={"n_qubits": n_qubits, "n_columns": n_columns, "palette": GATE_PALETTE},
        default={"circuit": []},
        on_circuit_change=lambda: None,
    )
    return result.circuit if result.circuit is not None else []
