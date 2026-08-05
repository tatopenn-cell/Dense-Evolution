// Dense-Evolution Composer -- plain vanilla JS, no framework, no Streamlit.
// Talks to the local FastAPI backend (server.py), which runs every circuit
// on the real dense_evolution.DenseSVSimulator and renders every figure
// with Qiskit's own real visualization functions. Nothing here is mocked:
// every number/image on this page comes back from a real /api/run call.
//
// Wrapped in an IIFE: the docs site's Material theme uses
// navigation.instant (client-side page routing), which can re-evaluate
// this script without a full page reload -- top-level `let`/`const`
// would then throw "Identifier has already been declared" on the second
// run. A function scope makes every re-run independent.
(function () {

const N_COLS = 12;
let nQubits = 2;
let grid = [];       // grid[row][col] = {kind, gate} | null
let palette = [];
let presets = {};
let maxQubits = 20;  // replaced with the real per-machine limit at init(), see loadSystemLimits()
let maxQubitsDense = 20;  // the dense-backend (RAM-based) figure loadSystemLimits() fetches
// MPS above this many qubits never materializes a dense (2**n,) array (server.py's
// large-scale path, dashboard_core.run_large_circuit_mps) -- so it isn't bounded by
// maxQubitsDense at all. 100 isn't a hard ceiling of the engine itself (MPSSimulator's
// memory scales with bond dimension, not qubit count) -- it's the largest figure that
// function's own docstring reports actually measuring (30/50/100 qubits, "stays under 2s
// even at 100 qubits"), so it's what's honestly claimable as tested rather than a guess.
const MAX_QUBITS_MPS_LARGE_SCALE = 100;

const $ = (id) => document.getElementById(id);

const COLOR_BY_GATE = {
  h: "var(--de-red)",
  x: "var(--de-navy)", y: "var(--de-navy)", z: "var(--de-navy)",
  s: "var(--de-blue)", sdg: "var(--de-blue)", t: "var(--de-blue)", tdg: "var(--de-blue)",
  sx: "var(--de-blue)",
  rx: "var(--de-purple)", ry: "var(--de-purple)", rz: "var(--de-purple)",
};
const CELL_LABEL = { sdg: "S†", tdg: "T†", sx: "√X" };
function chipColor(g) {
  if (g.kind === "control" || g.kind === "target" || g.kind === "swap") return "var(--de-navy)";
  return COLOR_BY_GATE[g.gate] || "var(--de-gray)";
}

// Real periodic table data (symbol, Italian name), Z=1..54 plus Au/Pb as
// recognizable heavy examples -- clicking a tile just appends the real
// symbol to the custom-molecule "symbols" field, the same field typing
// would fill. Any combination is accepted; whether this simulator can
// actually handle the resulting molecule (exact diagonalization / VQE
// stay in the ~12-qubit range) is decided honestly by the real qubit
// count PennyLane reports back, not by this list -- picking a heavy
// element like Pb is expected to hit that limit with a real error, not
// a fake result.
const PERIODIC_ELEMENTS = [
  [1, "H", "Idrogeno"], [2, "He", "Elio"],
  [3, "Li", "Litio"], [4, "Be", "Berillio"], [5, "B", "Boro"], [6, "C", "Carbonio"],
  [7, "N", "Azoto"], [8, "O", "Ossigeno"], [9, "F", "Fluoro"], [10, "Ne", "Neon"],
  [11, "Na", "Sodio"], [12, "Mg", "Magnesio"], [13, "Al", "Alluminio"], [14, "Si", "Silicio"],
  [15, "P", "Fosforo"], [16, "S", "Zolfo"], [17, "Cl", "Cloro"], [18, "Ar", "Argon"],
  [19, "K", "Potassio"], [20, "Ca", "Calcio"], [21, "Sc", "Scandio"], [22, "Ti", "Titanio"],
  [23, "V", "Vanadio"], [24, "Cr", "Cromo"], [25, "Mn", "Manganese"], [26, "Fe", "Ferro"],
  [27, "Co", "Cobalto"], [28, "Ni", "Nichel"], [29, "Cu", "Rame"], [30, "Zn", "Zinco"],
  [31, "Ga", "Gallio"], [32, "Ge", "Germanio"], [33, "As", "Arsenico"], [34, "Se", "Selenio"],
  [35, "Br", "Bromo"], [36, "Kr", "Kripton"],
  [37, "Rb", "Rubidio"], [38, "Sr", "Stronzio"], [39, "Y", "Ittrio"], [40, "Zr", "Zirconio"],
  [41, "Nb", "Niobio"], [42, "Mo", "Molibdeno"], [43, "Tc", "Tecnezio"], [44, "Ru", "Rutenio"],
  [45, "Rh", "Rodio"], [46, "Pd", "Palladio"], [47, "Ag", "Argento"], [48, "Cd", "Cadmio"],
  [49, "In", "Indio"], [50, "Sn", "Stagno"], [51, "Sb", "Antimonio"], [52, "Te", "Tellurio"],
  [53, "I", "Iodio"], [54, "Xe", "Xenon"],
  [79, "Au", "Oro"], [82, "Pb", "Piombo"],
];

function renderElementPalette() {
  const el = $("element-palette");
  if (!el) return;
  el.innerHTML = "";
  PERIODIC_ELEMENTS.forEach(([z, symbol, name]) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = symbol;
    chip.title = `${name} (Z=${z})`;
    chip.addEventListener("click", () => {
      const box = $("ham-symbols");
      const current = box.value.split(",").map((s) => s.trim()).filter(Boolean);
      current.push(symbol);
      box.value = current.join(", ");
    });
    el.appendChild(chip);
  });
}

// This page is static -- served from GitHub Pages, from `mkdocs serve`'s
// own preview, or opened straight off disk as a downloaded copy (origin
// "null"). The local kernel (local_site/app/server.py) never binds
// anywhere but 127.0.0.1:8800 (see its own __main__ block), so that's
// always the real address, regardless of where this script itself came
// from -- there's no case anymore where the page and the kernel share an
// origin (see server.py's own module docstring: it serves no HTML at all).
const API_BASE = "http://127.0.0.1:8800";

async function api(path, opts) {
  const res = await fetch(API_BASE + path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

// Presence probe for the local kernel (server.py running on this machine).
// Published pages start locked -- the interactive panels are real, running
// code the moment they're used, so nothing runs until a real kernel answers.
async function checkKernel() {
  const root = $("de-composer-root");
  let banner = document.getElementById("de-kernel-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "de-kernel-banner";
    root.parentNode.insertBefore(banner, root);
  }

  try {
    const health = await api("/api/health");
    root.classList.remove("locked");
    banner.className = "de-kernel-banner de-kernel-ok";
    banner.innerHTML =
      `Kernel locale connesso &mdash; dense_evolution v${health.dense_evolution_version} ` +
      `su <strong>${health.hostname}</strong> &mdash; RAM: ${health.available_ram_gb} GB liberi ` +
      `di ${health.total_ram_gb} GB totali (${health.ram_percent_free}% libera).`;
    return true;
  } catch (err) {
    root.classList.add("locked");
    banner.className = "de-kernel-banner de-kernel-locked";
    banner.innerHTML =
      `<strong>Kernel locale non rilevato.</strong> Serve dense_evolution + JAX in esecuzione ` +
      `sul tuo PC. Scarica lo script di installazione (va eseguito da te: nessun sito puo' ` +
      `installare o avviare programmi sul tuo PC da solo):<br>` +
      `<a href="assets/installer/install-composer.bat" download class="btn btn-ghost">Windows (.bat)</a> ` +
      `<a href="assets/installer/install-composer.sh" download class="btn btn-ghost">macOS / Linux (.sh)</a><br><br>` +
      `Oppure a mano, in un terminale:<br>` +
      `<code>pip install dense-evolution[composer]</code><br>` +
      `<code>dense-evolution serve</code><br>` +
      `<button id="de-kernel-retry" class="btn btn-primary">Riprova</button>`;
    document.getElementById("de-kernel-retry").addEventListener("click", () => { void loadEverything(); });
    return false;
  }
}

function setStatus(text, isError) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", !!isError);
}

function buildEmptyGrid() {
  grid = Array.from({ length: nQubits }, () => Array(N_COLS).fill(null));
}

const SHORT_LABEL = { "rx": "Rx", "ry": "Ry", "rz": "Rz" };

function renderPalette() {
  const el = $("palette");
  el.innerHTML = "";
  palette.forEach((g) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.style.background = chipColor(g);
    chip.textContent = SHORT_LABEL[g.gate] || g.label;
    chip.title = g.label;
    chip.draggable = true;
    chip.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", g.id);
    });
    el.appendChild(chip);
  });
}

function renderGrid() {
  const el = $("grid");
  el.innerHTML = "";
  el.style.gridTemplateColumns = `34px repeat(${N_COLS}, 38px)`;
  for (let r = 0; r < nQubits; r++) {
    const label = document.createElement("div");
    label.className = "qlabel";
    label.textContent = `q${r}`;
    el.appendChild(label);

    const wireRow = document.createElement("div");
    wireRow.className = "wire-row";
    wireRow.style.gridColumn = `2 / span ${N_COLS}`;
    for (let c = 0; c < N_COLS; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.row = r;
      cell.dataset.col = c;
      cell.addEventListener("dragover", (e) => { e.preventDefault(); cell.classList.add("dragover"); });
      cell.addEventListener("dragleave", () => cell.classList.remove("dragover"));
      cell.addEventListener("drop", (e) => {
        e.preventDefault();
        cell.classList.remove("dragover");
        const gateId = e.dataTransfer.getData("text/plain");
        const g = palette.find((x) => x.id === gateId);
        if (!g) return;
        grid[r][c] = { kind: g.kind, gate: g.gate };
        paintCell(r, c);
        renderConnectors();
        syncQasmFromGrid();
      });
      cell.addEventListener("click", () => {
        if (grid[r][c]) {
          grid[r][c] = null;
          paintCell(r, c);
          renderConnectors();
          syncQasmFromGrid();
        }
      });
      wireRow.appendChild(cell);
    }
    el.appendChild(wireRow);
  }
  renderConnectors();
}

function paintCell(r, c) {
  const cell = document.querySelector(`.cell[data-row="${r}"][data-col="${c}"]`);
  const v = grid[r][c];
  cell.classList.remove("kind-control", "kind-target");
  if (!v) {
    cell.textContent = "";
    cell.classList.remove("filled");
    cell.style.background = "";
    cell.title = "";
    return;
  }
  cell.classList.add("filled");
  if (v.kind === "control" || v.kind === "target") cell.classList.add(`kind-${v.kind}`);
  cell.style.background = v.kind === "control" ? "var(--de-navy)"
    : v.kind === "target" ? "var(--de-navy)"
    : v.kind === "swap" ? "var(--de-gray)"
    : (COLOR_BY_GATE[v.gate] || "var(--de-gray)");
  cell.textContent = v.kind === "control" ? "●"
    : v.kind === "target" ? (v.gate === "x" ? "⊕" : v.gate.toUpperCase())
    : v.kind === "swap" ? "×"
    : (CELL_LABEL[v.gate] || v.gate.toUpperCase());
  cell.title = "click to remove";
}

// Draws the vertical line connecting a control to its target (or the two
// ends of a SWAP) in the same column, the same visual convention every
// real circuit diagram uses -- computed purely from cell geometry, no
// separate data model to keep in sync.
function renderConnectors() {
  document.querySelectorAll(".connector").forEach((el) => el.remove());
  const gridEl = $("grid");
  const cellSize = 38;
  const labelColWidth = 34;
  const topPad = 8; // .grid { padding: 0.5rem 0 } top padding
  for (let c = 0; c < N_COLS; c++) {
    const rowsInColumn = [];
    for (let r = 0; r < nQubits; r++) {
      if (grid[r][c] && grid[r][c].kind !== "single") rowsInColumn.push(r);
    }
    if (rowsInColumn.length < 2) continue;
    const top = Math.min(...rowsInColumn);
    const bottom = Math.max(...rowsInColumn);
    const line = document.createElement("div");
    line.className = "connector";
    line.style.left = `${labelColWidth + c * cellSize + cellSize / 2 - 1}px`;
    line.style.top = `${topPad + top * cellSize + cellSize / 2}px`;
    line.style.height = `${(bottom - top) * cellSize}px`;
    gridEl.appendChild(line);
  }
}

function gridToOps() {
  const ops = [];
  for (let c = 0; c < N_COLS; c++) {
    let single = null;
    const controls = [];
    const targets = [];
    const swaps = [];
    for (let r = 0; r < nQubits; r++) {
      const v = grid[r][c];
      if (!v) continue;
      if (v.kind === "single") single = { row: r, gate: v.gate };
      else if (v.kind === "control") controls.push(r);
      else if (v.kind === "target") targets.push({ row: r, gate: v.gate });
      else if (v.kind === "swap") swaps.push(r);
    }
    if (single && !controls.length && !targets.length && !swaps.length) {
      ops.push({ gate: single.gate, qubits: [single.row] });
    } else if (!single && controls.length === 1 && targets.length === 1 && !swaps.length) {
      ops.push({ gate: "c" + targets[0].gate, qubits: [controls[0], targets[0].row] });
    } else if (!single && !controls.length && !targets.length && swaps.length === 2) {
      ops.push({ gate: "swap", qubits: swaps });
    }
  }
  return ops;
}

async function syncQasmFromGrid() {
  const ops = gridToOps();
  if (!ops.length) return;
  try {
    const { qasm } = await api("/api/build_from_ops", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_qubits: nQubits, ops }),
    });
    $("qasm").value = qasm;
  } catch (err) {
    setStatus(`Errore griglia: ${err.message}`, true);
  }
}

const MOLECULE_PRESET_PREFIX = "__molecule__:";

async function loadPresetsAndPalette() {
  palette = await api("/api/palette");
  presets = await api("/api/presets");
  const sel = $("preset-select");
  sel.innerHTML = "";
  const customOpt = document.createElement("option");
  customOpt.value = "__custom__";
  customOpt.textContent = "Custom";
  sel.appendChild(customOpt);

  // Real molecular Hartree-Fock reference circuits (LiH etc.) come FIRST,
  // before the ~18 generic textbook circuits -- this is meant to be where
  // a molecule -> ansatz -> VQE chain starts, and a real molecule buried
  // after 18 unrelated entries in a plain <select> is easy to miss
  // entirely (reported directly: looked in this exact dropdown, only saw
  // the generic circuits). Picking one here uses the same real, instant
  // fast path (dashboard_core.vqe.run_vqe with n_layers=0) as picking it
  // in the Hamiltonian panel below.
  try {
    const mapping = $("ham-mapping-select") ? $("ham-mapping-select").value || "jordan_wigner" : "jordan_wigner";
    moleculeCatalog = await api(`/api/hamiltonians?mapping=${mapping}`);
    const moleculeGroup = document.createElement("optgroup");
    moleculeGroup.label = "Molecole (Hartree-Fock reale)";
    Object.keys(moleculeCatalog).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = MOLECULE_PRESET_PREFIX + name;
      opt.textContent = `[${moleculeCatalog[name].n_qubits}q] ${name}`;
      moleculeGroup.appendChild(opt);
    });
    sel.appendChild(moleculeGroup);
  } catch (err) {
    setStatus(`Errore caricamento molecole: ${err.message}`, true);
  }

  const circuitGroup = document.createElement("optgroup");
  circuitGroup.label = "Circuiti generici";
  Object.keys(presets).forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    circuitGroup.appendChild(opt);
  });
  sel.appendChild(circuitGroup);

  sel.addEventListener("change", () => {
    if (sel.value === "__custom__") return;
    if (sel.value.startsWith(MOLECULE_PRESET_PREFIX)) {
      loadMoleculeCircuit({ name: sel.value.slice(MOLECULE_PRESET_PREFIX.length) });
      return;
    }
    $("qasm").value = presets[sel.value];
    syncQubitCountFromQasm(presets[sel.value]);
  });
  $("qasm").value = presets["Bell state (2 qubit)"] || "";
  syncQubitCountFromQasm($("qasm").value);
  renderPalette();
}

// Each QASM_LIBRARY preset is a fixed circuit for a fixed qubit count
// (e.g. "Bell state" is a real 2-qubit concept, not something that scales) --
// but the #n-qubits field previously kept whatever value it had *before* the
// preset was picked, so it could show a number with no relationship to what
// would actually run (execution always uses the QASM text's own qreg count,
// never the field). Reading the real qreg size out of the QASM text and
// syncing the field to it means the field always tells the truth about what
// a Run click will actually execute, instead of silently disagreeing with it.
function syncQubitCountFromQasm(qasmText) {
  const match = /qreg\s+\w+\s*\[\s*(\d+)\s*\]/.exec(qasmText || "");
  if (!match) return;
  const n = parseInt(match[1], 10);
  if (!Number.isFinite(n) || n < 1) return;
  nQubits = n;
  $("n-qubits").value = String(n);
  buildEmptyGrid();
  renderGrid();
}

// Real Kraus-channel noise models (dense_evolution.NoiseModel) -- applied
// as an actual stochastic channel to the statevector, not a fabricated
// decay curve.
async function loadNoiseModels() {
  const models = await api("/api/noise_models");
  const sel = $("noise-model-select");
  sel.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  });
}

function renderLargeScaleResult(result) {
  // Beyond MPS's dense-contraction ceiling there is no full statevector/
  // histogram/Q-sphere to show -- only the real top-k most probable
  // states (exact beam search, see dashboard_core.engine.run_large_circuit_mps).
  $("circuit-img").src = `data:image/png;base64,${result.circuit_png}`;
  $("qsphere-img").style.display = "none";
  $("qsphere-skip-msg").textContent =
    "Nessuno statevector denso a questa scala -- vedi gli stati piu probabili nella tabella Statevector qui sotto.";
  $("bloch-img").style.display = "none";
  $("bloch-skip-msg").textContent =
    "Nessuno statevector denso a questa scala -- vedi gli stati piu probabili nella tabella Statevector qui sotto.";
  $("histogram-img").style.display = "none";
  $("histogram-img").removeAttribute("src");
  $("histogram-skip-msg").textContent =
    "Nessun istogramma denso a questa scala -- vedi gli stati piu probabili nella tabella Statevector qui sotto.";

  $("fidelity-info").textContent = "";

  const backendInfo = $("backend-info");
  backendInfo.textContent =
    `Backend: MPS, modalita large-scale (>${24} qubit, nessuno statevector denso) -- ` +
    `bond massimo usato=${result.mps_max_bond_used}, memoria stimata=${result.mps_memory_mb.toFixed(4)} MB, ` +
    `JSD medio=${result.mps_avg_jsd.toExponential(2)}. Top-${result.k_requested} stati via ricerca a fascio esatta ` +
    `(dense_evolution.mps.MPSSimulator.get_top_k_probable_states).`;

  const tbody = document.querySelector("#statevector-table tbody");
  tbody.innerHTML = "";
  result.top_k_states.forEach((row) => {
    if (row.probability <= 1e-9) return;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>|${row.state}⟩</td><td colspan="3">probabilita = ${row.probability.toFixed(6)}</td><td></td>`;
    tbody.appendChild(tr);
  });

  setStatus(`Fatto — ${result.n_qubits} qubit (large-scale MPS), top-${result.k_requested} stati trovati.`);
}

async function runCircuit() {
  setStatus("Esecuzione sul motore dense_evolution...");
  try {
    const result = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        qasm: $("qasm").value,
        shots: parseInt($("shots").value, 10),
        seed: parseInt($("seed").value, 10),
        noise_model: $("noise-model-select").value || "ideal",
        noise_p: parseFloat($("noise-p").value) || 0,
        backend: $("backend-select").value || "dense",
      }),
    });

    if (result.large_scale) {
      renderLargeScaleResult(result);
      return;
    }

    $("histogram-img").style.display = "";
    $("histogram-img").src = `data:image/png;base64,${result.histogram_png}`;
    $("histogram-skip-msg").textContent = "";
    $("circuit-img").src = `data:image/png;base64,${result.circuit_png}`;
    if (result.qsphere_png) {
      $("qsphere-img").style.display = "";
      $("qsphere-img").src = `data:image/png;base64,${result.qsphere_png}`;
      $("qsphere-skip-msg").textContent = "";
    } else {
      $("qsphere-img").style.display = "none";
      $("qsphere-skip-msg").textContent = result.qsphere_skipped_reason || "Q-sphere non disponibile.";
    }
    if (result.bloch_png) {
      $("bloch-img").style.display = "";
      $("bloch-img").src = `data:image/png;base64,${result.bloch_png}`;
      $("bloch-skip-msg").textContent = "";
    } else {
      $("bloch-img").style.display = "none";
      $("bloch-skip-msg").textContent = result.bloch_skipped_reason || "Bloch spheres non disponibili.";
    }

    const backendInfo = $("backend-info");
    if (result.backend === "mps") {
      backendInfo.textContent =
        `Backend: MPS -- bond massimo usato=${result.mps_max_bond_used}, ` +
        `memoria stimata=${result.mps_memory_mb.toFixed(4)} MB, JSD medio=${result.mps_avg_jsd.toExponential(2)}`;
    } else {
      backendInfo.textContent = "Backend: Dense (statevector completo)";
    }

    const fidelityInfo = $("fidelity-info");
    if (result.fidelity_vs_ideal !== null && result.fidelity_vs_ideal !== undefined) {
      fidelityInfo.textContent =
        `Fedelta vs stato ideale (dense_evolution.statevector_fidelity, questa traiettoria rumorosa) = ` +
        `${result.fidelity_vs_ideal.toFixed(6)}`;
    } else {
      fidelityInfo.textContent = "";
    }

    const tbody = document.querySelector("#statevector-table tbody");
    tbody.innerHTML = "";
    result.statevector.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>|${row.state}⟩</td><td>${row.re.toFixed(4)}</td>` +
        `<td>${row.im.toFixed(4)}</td><td>${row.abs.toFixed(4)}</td><td>${row.phase.toFixed(4)}</td>`;
      tbody.appendChild(tr);
    });

    setStatus(`Fatto — ${result.n_qubits} qubit, ${Object.values(result.counts).reduce((a, b) => a + b, 0)} shot reali.`);
  } catch (err) {
    setStatus(`Errore: ${err.message}`, true);
  }
}

// Real molecular Hamiltonians (PennyLane Hartree-Fock, dashboard_core.
// hamiltonians via /api/hamiltonians*) -- the FULL catalog is always
// listed (each molecule really does need a fixed qubit count, so the
// catalog can't sensibly be filtered by whatever Qubits happens to be
// set to); picking one sets Qubits and loads a real circuit instead.
let moleculeCatalog = {};

async function refreshHamiltonianCatalog() {
  const sel = $("ham-catalog-select");
  const mapping = $("ham-mapping-select").value || "jordan_wigner";
  try {
    moleculeCatalog = await api(`/api/hamiltonians?mapping=${mapping}`);
    const names = Object.keys(moleculeCatalog);
    sel.innerHTML = "";
    if (!names.length) {
      sel.innerHTML = `<option value="">— catalogo vuoto —</option>`;
      return;
    }
    names.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = `[${moleculeCatalog[name].n_qubits}q] ${name}`;
      sel.appendChild(opt);
    });
    refreshMixSelect();
    // Loading the whole catalog's real Hartree-Fock circuits/qubit counts
    // is exactly the "pick a molecule -> get a circuit" mechanic, so the
    // very first entry loads automatically instead of leaving the editor
    // on whatever preset happened to be selected before.
    if (names.length) loadMoleculeCircuit({ name: names[0] });
  } catch (err) {
    sel.innerHTML = `<option value="">errore: ${err.message}</option>`;
  }
}

// Real, instant Hartree-Fock reference circuit (dashboard_core.vqe.run_vqe
// with n_layers=0 -- a genuine fast path, not a fake shortcut: with zero
// ansatz parameters there's nothing to optimize, so this is just the
// real HF occupation loaded as X gates) -- this is the mechanic that
// turns "pick a molecule" directly into "here's a real circuit for it",
// with Qubits synced to what that molecule's Hamiltonian actually needs.
async function loadMoleculeCircuit(body) {
  const el = $("ham-result");
  el.classList.remove("error");
  el.textContent = "Costruzione Hartree-Fock in corso...";
  try {
    const data = await api("/api/vqe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, n_layers: 0, maxiter: 0 }),
    });
    if (body.name && $("preset-select").querySelector(`option[value="${MOLECULE_PRESET_PREFIX}${body.name}"]`)) {
      $("preset-select").value = MOLECULE_PRESET_PREFIX + body.name;
    } else {
      $("preset-select").value = "__custom__";
    }
    $("n-qubits").value = String(data.n_qubits);
    nQubits = data.n_qubits;
    buildEmptyGrid();
    renderGrid();
    $("qasm").value = data.qasm;

    const exactLine = data.exact_energy_hartree !== null
      ? `Energia esatta (diagonalizzazione densa) = ${data.exact_energy_hartree.toFixed(6)} Hartree\n`
      : "";
    el.textContent =
      `n_qubits = ${data.n_qubits}\n` +
      `Stato Hartree-Fock (occupazione) = ${data.hf_occupation.join("")}\n` +
      `Energia Hartree-Fock = ${data.vqe_energy_hartree.toFixed(6)} Hartree\n` +
      exactLine +
      `Circuito HF caricato nell'editor -- premi Run per eseguirlo, o usa il pannello VQE sotto per ottimizzarlo.`;
  } catch (err) {
    el.classList.add("error");
    el.textContent = `Errore: ${err.message}`;
  }
}

// Real Hamiltonian mixing (dashboard_core.hamiltonians.mix_hamiltonians via
// /api/hamiltonian/mix) -- only meaningful between two molecules with the
// same real qubit count (same electron/Hilbert space), so the second
// selector is filtered to exactly those, driven off the same
// moleculeCatalog data the main catalog dropdown already has.
function refreshMixSelect() {
  const primaryName = $("ham-catalog-select").value;
  const mixSel = $("ham-mix-select");
  mixSel.innerHTML = '<option value="">— nessuno —</option>';
  if (!primaryName || !moleculeCatalog[primaryName]) return;
  const targetQubits = moleculeCatalog[primaryName].n_qubits;
  Object.keys(moleculeCatalog).forEach((name) => {
    if (name === primaryName) return;
    if (moleculeCatalog[name].n_qubits !== targetQubits) return;
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = `[${moleculeCatalog[name].n_qubits}q] ${name}`;
    mixSel.appendChild(opt);
  });
}

async function runMixHamiltonians() {
  const el = $("ham-result");
  el.classList.remove("error");
  const nameA = $("ham-catalog-select").value;
  const nameB = $("ham-mix-select").value;
  if (!nameA || !nameB) {
    el.classList.add("error");
    el.textContent = "Seleziona sia la molecola A (catalogo sopra) che B (mix) prima di mescolare.";
    return;
  }
  el.textContent = "Calcolo Hamiltoniana mescolata (diagonalizzazione esatta)...";
  try {
    const data = await api("/api/hamiltonian/mix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name_a: nameA,
        name_b: nameB,
        weight_a: parseFloat($("ham-mix-weight-a").value) || 0,
        weight_b: parseFloat($("ham-mix-weight-b").value) || 0,
        mapping: $("ham-mapping-select").value || "jordan_wigner",
      }),
    });
    el.classList.remove("error");
    el.textContent =
      `n_qubits = ${data.n_qubits}\n` +
      `Energia A sola = ${data.energy_a.toFixed(6)} Hartree\n` +
      `Energia B sola = ${data.energy_b.toFixed(6)} Hartree\n` +
      `Energia mescolata (w_A*H_A + w_B*H_B) = ${data.energy_mixed.toFixed(6)} Hartree`;
  } catch (err) {
    el.classList.add("error");
    el.textContent = `Errore: ${err.message}`;
  }
}

// Real geometry generators (dashboard_core.hamiltonians.linear_chain_geometry
// / ring_geometry, same regular-polygon math mirrored here in JS so the
// custom-molecule textarea can be filled instantly, client-side): linear
// chain (any atom count) or regular-polygon ring (bond length = polygon
// side; 3 atoms is exactly the equilateral-triangle case, e.g. H3+'s
// real D3h geometry).
function generateGeometryTemplate(shape, nAtoms, bondLength) {
  const rows = [];
  if (shape === "linear") {
    for (let i = 0; i < nAtoms; i++) rows.push([0, 0, i * bondLength]);
  } else {
    const n = shape === "triangular" ? 3 : nAtoms;
    if (shape === "triangular" && nAtoms !== 3) {
      throw new Error(`Geometria triangolare richiede esattamente 3 atomi (simboli attuali: ${nAtoms}).`);
    }
    if (n < 3) throw new Error("L'anello richiede almeno 3 atomi.");
    const R = bondLength / (2 * Math.sin(Math.PI / n));
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * i) / n;
      rows.push([R * Math.cos(angle), R * Math.sin(angle), 0]);
    }
  }
  return rows;
}

function runGenerateGeometry() {
  const el = $("ham-result");
  try {
    const symbols = $("ham-symbols").value.split(",").map((s) => s.trim()).filter(Boolean);
    const shape = $("geom-shape-select").value;
    const bondLength = parseFloat($("geom-bond-length").value) || 1.0;
    const rows = generateGeometryTemplate(shape, symbols.length, bondLength);
    $("ham-geometry").value = rows.map((r) => r.map((x) => x.toFixed(6)).join(", ")).join("\n");
    el.classList.remove("error");
    el.textContent = `Geometria ${shape} generata per ${rows.length} atomi (legame = ${bondLength} A). Modificabile a mano prima di calcolare.`;
  } catch (err) {
    el.classList.add("error");
    el.textContent = `Errore: ${err.message}`;
  }
}

function showHamiltonianResult(data) {
  const el = $("ham-result");
  el.classList.remove("error");
  el.textContent =
    `n_qubits = ${data.n_qubits}\n` +
    `Energia di stato fondamentale (Hartree, esatta via diagonalizzazione densa) = ${data.ground_state_energy_hartree.toFixed(6)}`;
}

function showHamiltonianError(err) {
  const el = $("ham-result");
  el.classList.add("error");
  el.textContent = `Errore: ${err.message}`;
}

async function runCatalogHamiltonian() {
  const name = $("ham-catalog-select").value;
  if (!name) return;
  $("ham-result").textContent = "Calcolo Hartree-Fock + Jordan-Wigner in corso...";
  try {
    const data = await api("/api/hamiltonian/molecule", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, mapping: $("ham-mapping-select").value || "jordan_wigner" }),
    });
    showHamiltonianResult(data);
  } catch (err) {
    showHamiltonianError(err);
  }
}

async function runCustomHamiltonian() {
  const symbols = $("ham-symbols").value.split(",").map((s) => s.trim()).filter(Boolean);
  const geometry = $("ham-geometry").value.split("\n").map((line) => line.trim()).filter(Boolean)
    .map((line) => line.split(",").map((x) => parseFloat(x.trim())));
  const charge = parseInt($("ham-charge").value, 10) || 0;
  $("ham-result").textContent = "Calcolo Hartree-Fock + Jordan-Wigner in corso...";
  try {
    const data = await api("/api/hamiltonian/custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbols, geometry, charge, mapping: $("ham-mapping-select").value || "jordan_wigner" }),
    });
    showHamiltonianResult(data);
  } catch (err) {
    showHamiltonianError(err);
  }
}

// Real VQE (dense_evolution.vqe.run_vqe via /api/vqe) -- hardware-
// efficient ansatz, real Adam optimization against the molecule's real
// Hamiltonian, no fixed/precomputed angles. Loads the resulting QASM
// into the main editor so the converged circuit runs through the same
// engine as everything else on this page.
async function runVqe() {
  const el = $("vqe-result");
  el.classList.remove("error");
  const useCustom = $("vqe-source-select").value === "custom";
  const ansatzType = $("vqe-ansatz-select").value || "hardware_efficient";
  const nLayers = parseInt($("vqe-layers").value, 10) || 8;
  const maxiter = parseInt($("vqe-maxiter").value, 10) || 200;
  const stepSize = parseFloat($("vqe-step-size").value) || 0.1;
  const beta1 = parseFloat($("vqe-beta1").value);
  const beta2 = parseFloat($("vqe-beta2").value);

  const body = {
    ansatz_type: ansatzType, n_layers: nLayers, maxiter,
    step_size: stepSize,
    beta1: Number.isFinite(beta1) ? beta1 : 0.9,
    beta2: Number.isFinite(beta2) ? beta2 : 0.999,
  };
  if (useCustom) {
    body.symbols = $("ham-symbols").value.split(",").map((s) => s.trim()).filter(Boolean);
    body.geometry = $("ham-geometry").value.split("\n").map((line) => line.trim()).filter(Boolean)
      .map((line) => line.split(",").map((x) => parseFloat(x.trim())));
    body.charge = parseInt($("ham-charge").value, 10) || 0;
  } else {
    const name = $("ham-catalog-select").value;
    if (!name) {
      el.classList.add("error");
      el.textContent = "Nessuna molecola di catalogo selezionata per il qubit count corrente.";
      return;
    }
    body.name = name;
  }

  const ansatzNote = ansatzType === "uccsd"
    ? "UCCSD -- pochi parametri ma ogni iterazione e' molto piu' pesante (circuito decomposto profondo): puo' richiedere diversi minuti"
    : `hardware-efficient, ${nLayers} layer -- fino a ~2 minuti per molecole a 12 qubit`;
  el.textContent = `Ottimizzazione VQE in corso (Adam, ${ansatzNote}, fino a ${maxiter} iterazioni)...`;
  try {
    const data = await api("/api/vqe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const exactLine = data.exact_energy_hartree !== null
      ? `Energia esatta (diagonalizzazione densa) = ${data.exact_energy_hartree.toFixed(6)} Hartree\n` +
        `Errore |VQE - esatta| = ${Math.abs(data.vqe_energy_hartree - data.exact_energy_hartree).toFixed(6)} Hartree\n`
      : "";
    const layerLine = data.n_layers !== null ? `layer = ${data.n_layers}, ` : "";
    el.textContent =
      `n_qubits = ${data.n_qubits}, ansatz = ${data.ansatz_type}, ${layerLine}parametri = ${data.n_params}\n` +
      `Stato iniziale Hartree-Fock (occupazione) = ${data.hf_occupation.join("")}\n` +
      `Energia VQE convergente = ${data.vqe_energy_hartree.toFixed(6)} Hartree\n` +
      exactLine +
      `Circuito caricato nell'editor OpenQASM e aggiunto alla libreria preset qui sopra -- premi Run per eseguirlo sul motore reale.`;

    $("n-qubits").value = String(data.n_qubits);
    nQubits = data.n_qubits;
    buildEmptyGrid();
    renderGrid();
    $("qasm").value = data.qasm;

    // The whole point of running a real optimization is to end up with a
    // real circuit -- so it belongs in the same preset library every
    // other circuit lives in, not stranded in the QASM textarea only.
    const shortLabel = useCustom ? body.symbols.join("") : (body.name || "").split(" - ")[0];
    const ansatzTag = data.ansatz_type === "uccsd" ? "UCCSD" : `${data.n_layers}L`;
    const vqeLabel = `[VQE ${ansatzTag}] ${shortLabel} (${data.n_qubits}q)`;
    presets[vqeLabel] = data.qasm;
    let vqeGroup = document.querySelector('#preset-select optgroup[data-role="vqe"]');
    if (!vqeGroup) {
      vqeGroup = document.createElement("optgroup");
      vqeGroup.label = "Ansatz VQE (generati)";
      vqeGroup.dataset.role = "vqe";
      $("preset-select").appendChild(vqeGroup);
    }
    const opt = document.createElement("option");
    opt.value = vqeLabel;
    opt.textContent = vqeLabel;
    vqeGroup.appendChild(opt);
    $("preset-select").value = vqeLabel;
  } catch (err) {
    el.classList.add("error");
    el.textContent = `Errore: ${err.message}`;
  }
}

// Real Hellmann-Feynman forces / MD (dashboard_core.qmmm) -- both reuse
// whichever catalog molecule is selected above (#ham-catalog-select),
// the same molecule VQE uses. Custom (non-catalog) molecules aren't
// supported here: the real molecule catalog is what qmmm.py's real
// mass/geometry/Hamiltonian pipeline is built against.
function _selectedCatalogMoleculeName() {
  const name = $("ham-catalog-select").value;
  if (!name) {
    throw new Error("Nessuna molecola di catalogo selezionata -- scegli una molecola sopra (Hamiltonian panel).");
  }
  return name;
}

async function runQmmmForces() {
  const el = $("qmmm-result");
  el.classList.remove("error");
  try {
    const name = _selectedCatalogMoleculeName();
    el.textContent = "Calcolo forze Hellmann-Feynman reali (differenziazione PennyLane)...";
    const data = await api("/api/qmmm_forces", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const forceLines = data.symbols.map((s, i) =>
      `  ${s}${i}: [${data.forces_hartree_per_angstrom[i].map((x) => x.toFixed(6)).join(", ")}] Hartree/A`
    ).join("\n");
    el.textContent =
      `Energia = ${data.energy_hartree.toFixed(6)} Hartree\n` +
      `Forze per nucleo (Hartree/Angstrom):\n${forceLines}\n` +
      `Norma totale = ${data.force_norm.toFixed(6)} Hartree/Angstrom`;
  } catch (err) {
    el.classList.add("error");
    el.textContent = `Errore: ${err.message}`;
  }
}

async function runMdTrajectory() {
  const el = $("qmmm-result");
  el.classList.remove("error");
  try {
    const name = _selectedCatalogMoleculeName();
    const nSteps = parseInt($("md-steps").value, 10) || 20;
    const dtFs = parseFloat($("md-dt").value) || 0.5;
    const recompute = $("md-recompute").checked;
    el.textContent = recompute
      ? `Traiettoria MD ab-initio (Hartree-Fock reale ad ogni passo, ${nSteps} passi) -- puo' richiedere qualche minuto...`
      : `Traiettoria MD (stato elettronico iniziale fisso, ${nSteps} passi)...`;
    const data = await api("/api/md_trajectory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, n_steps: nSteps, dt_fs: dtFs, recompute_electronic_state: recompute }),
    });
    const last = data.step.length - 1;
    el.textContent =
      `${data.step.length} passi, ${data.time_fs[last].toFixed(2)} fs totali.\n` +
      `Energia: ${data.energy_hartree[0].toFixed(6)} -> ${data.energy_hartree[last].toFixed(6)} Hartree\n` +
      `Norma forza: ${data.force_norm[0].toFixed(6)} -> ${data.force_norm[last].toFixed(6)} Hartree/Angstrom\n` +
      `Posizione finale (Angstrom):\n` +
      data.positions_angstrom[last].map((p, i) => `  atomo ${i}: [${p.map((x) => x.toFixed(6)).join(", ")}]`).join("\n");
  } catch (err) {
    el.classList.add("error");
    el.textContent = `Errore: ${err.message}`;
  }
}

// Real Zero-Noise Extrapolation (dense_evolution.zero_noise_extrapolation,
// via dashboard_core.mitigation.run_zne_mitigation) -- reuses the current
// QASM circuit and the Noise model/Noise p already set in the toolbar.
async function runZne() {
  const el = $("zne-result");
  el.classList.remove("error");
  el.textContent = "Misura <P> su stato ideale e su canale reale (media su molte estrazioni stocastiche)...";
  try {
    const data = await api("/api/mitigate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        qasm: $("qasm").value,
        pauli_string: $("zne-pauli").value.trim().toUpperCase(),
        noise_model: $("noise-model-select").value || "ideal",
        noise_p: parseFloat($("noise-p").value) || 0,
        seed: parseInt($("seed").value, 10),
        extrapolation_method: $("zne-method-select").value || "richardson",
      }),
    });
    const factors = data.noise_factors.map((f, i) => `${f}x: ${data.noisy_expectations[i].toFixed(4)}`).join("  |  ");
    el.textContent =
      `Metodo: ${data.extrapolation_method}\n` +
      `<${data.pauli_string}> ideale       = ${data.ideal_expectation.toFixed(4)}\n` +
      `<${data.pauli_string}> con rumore   = ${factors}\n` +
      `<${data.pauli_string}> ZNE estrapolato = ${data.zne_extrapolated.toFixed(4)}\n` +
      `errore residuo |ZNE - ideale| = ${Math.abs(data.zne_extrapolated - data.ideal_expectation).toFixed(4)}  ` +
      `(vs |rumoroso@1x - ideale| = ${Math.abs(data.noisy_expectations[0] - data.ideal_expectation).toFixed(4)})`;
  } catch (err) {
    el.classList.add("error");
    el.textContent = `Errore: ${err.message}`;
  }
}

// Real density-matrix ZNE (dense_evolution.zne_density_matrix, via
// dashboard_core.mitigation.run_density_matrix_zne) -- Monte-Carlo
// density-matrix reconstruction at each noise scale, extrapolated and
// projected onto the nearest physical state, graded by real Uhlmann
// fidelity against the true ideal state.
async function runZneMatrix() {
  const el = $("zne-matrix-result");
  el.classList.remove("error");
  el.textContent = "Ricostruzione Monte-Carlo della matrice densita a piu scale di rumore...";
  try {
    const data = await api("/api/mitigate_matrix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        qasm: $("qasm").value,
        noise_model: $("noise-model-select").value || "ideal",
        noise_p: parseFloat($("noise-p").value) || 0,
        seed: parseInt($("seed").value, 10),
      }),
    });
    const improvement = data.fidelity_corrected - data.fidelity_raw;
    el.textContent =
      `Fedelta (Uhlmann) rumorosa @1x     = ${data.fidelity_raw.toFixed(4)}\n` +
      `Fedelta (Uhlmann) corretta via ZNE = ${data.fidelity_corrected.toFixed(4)}\n` +
      `Miglioramento = ${improvement >= 0 ? "+" : ""}${improvement.toFixed(4)}`;
  } catch (err) {
    el.classList.add("error");
    el.textContent = `Errore: ${err.message}`;
  }
}

// Real, per-machine RAM check (dense_evolution.chunk.SafeMemoryGuard via
// /api/system_limits) -- replaces a fixed qubit cap with whatever this
// specific machine can actually hold safely right now. Dense-backend-only:
// applyQubitLimitForBackend() is what actually sets the input's live max,
// since MPS above MPS_DENSE_CONTRACTION_LIMIT (24) isn't RAM-bound this way
// at all (see MAX_QUBITS_MPS_LARGE_SCALE above) -- using this dense figure
// for MPS too used to make the UI's own large-scale MPS mode unreachable
// whenever this machine's free RAM pushed max_qubits_dense below 24.
async function loadSystemLimits() {
  try {
    const limits = await api("/api/system_limits");
    maxQubitsDense = limits.max_qubits_dense;
    $("qubits-limit-info").textContent =
      `max ${maxQubitsDense} qubit (denso) su questo PC (${(limits.available_mb / 1024).toFixed(1)} GB liberi di ${(limits.total_mb / 1024).toFixed(1)} GB, soglia anti-OOM ${(limits.threshold_pct * 100).toFixed(0)}%) ` +
      `-- MPS large-scale (>24 qubit) non e' limitato dalla RAM allo stesso modo, fino a ${MAX_QUBITS_MPS_LARGE_SCALE} qubit testati.`;
    applyQubitLimitForBackend();
  } catch (err) {
    $("qubits-limit-info").textContent = `impossibile rilevare la RAM: ${err.message}`;
  }
}

// Sets the qubit input's live max from whichever backend is currently
// selected -- dense (and MPS at or below MPS_DENSE_CONTRACTION_LIMIT, which
// still contracts to a dense array internally) is bounded by this machine's
// real free RAM; MPS above that isn't, so it gets the much higher, engine-
// tested figure instead. Called on init and every time #backend-select
// changes.
function applyQubitLimitForBackend() {
  const backend = $("backend-select").value || "dense";
  maxQubits = backend === "mps"
    ? Math.max(maxQubitsDense, MAX_QUBITS_MPS_LARGE_SCALE)
    : maxQubitsDense;
  const input = $("n-qubits");
  input.max = String(maxQubits);
  // Clamping the displayed field without also updating nQubits/the grid
  // would recreate the exact bug this function exists to avoid elsewhere
  // (the field showing one qubit count while the grid/nQubits still reflect
  // another) -- so every place that changes what the field shows keeps all
  // three (field, nQubits, grid) in the same state, the same way the QASM
  // preset sync (syncQubitCountFromQasm) and the #n-qubits change handler
  // both already do.
  if (parseInt(input.value, 10) > maxQubits) {
    input.value = String(maxQubits);
    nQubits = maxQubits;
    buildEmptyGrid();
    renderGrid();
  }
}

// Everything here needs a real kernel to answer, so it's gated behind
// checkKernel() rather than fired unconditionally -- on a published page
// with no kernel installed yet, this would otherwise be five doomed
// fetches on every load instead of one clear locked state. Re-callable
// from the banner's Riprova button once the visitor starts the kernel,
// without re-registering any of init()'s event listeners.
async function loadEverything() {
  const connected = await checkKernel();
  if (!connected) return;
  // Sequenced, not fire-and-forget: loadPresetsAndPalette's own init sets
  // a Bell-state baseline (#qasm, #preset-select) unconditionally at its
  // end, and refreshHamiltonianCatalog's auto-load of the catalog's first
  // molecule (loadMoleculeCircuit) sets #qasm/#preset-select/#n-qubits
  // together, atomically, from one real Hartree-Fock computation. Firing
  // both concurrently (as before) raced two independent network fetches
  // against each other: whichever resolved last silently overwrote only
  // *some* of the other's fields (loadPresetsAndPalette never touches
  // #n-qubits at all), landing on a real, reproducible split state --
  // confirmed directly: #n-qubits stuck at a molecule's real qubit count
  // while #qasm/#preset-select showed the unrelated Bell-state default.
  // Awaiting them in order makes the molecule auto-load always run last
  // and set all three fields together, deterministically.
  await loadPresetsAndPalette();
  await refreshHamiltonianCatalog();
  loadNoiseModels();
  loadSystemLimits();
}

async function init() {
  buildEmptyGrid();
  renderGrid();
  renderElementPalette();
  await loadEverything();

  $("n-qubits").addEventListener("change", (e) => {
    nQubits = Math.max(1, Math.min(maxQubits, parseInt(e.target.value, 10) || 1));
    e.target.value = String(nQubits);
    buildEmptyGrid();
    renderGrid();
    // Keep the QASM text's own qreg/creg declarations in step with the
    // field the same way syncQubitCountFromQasm() keeps the field in step
    // with a picked preset's QASM -- otherwise editing this field after a
    // preset is loaded silently changes nothing about what Run actually
    // executes (the QASM text, not this field, is what's sent), recreating
    // the exact "field disagrees with what really runs" bug from the other
    // direction. Existing gate lines are left alone; only the two
    // declaration lines change, so a preset's real gates survive the edit
    // (they may now reference qubit indices outside the new range if N
    // shrank -- that's a real circuit error the backend will report, not
    // something to silently paper over here).
    const qasmField = $("qasm");
    if (qasmField.value) {
      qasmField.value = qasmField.value
        .replace(/qreg\s+(\w+)\s*\[\s*\d+\s*\]/, `qreg $1[${nQubits}]`)
        .replace(/creg\s+(\w+)\s*\[\s*\d+\s*\]/, `creg $1[${nQubits}]`);
    }
    if ($("preset-select").value !== "__custom__") $("preset-select").value = "__custom__";
  });
  $("backend-select").addEventListener("change", applyQubitLimitForBackend);
  $("clear-btn").addEventListener("click", () => {
    buildEmptyGrid();
    renderGrid();
    // syncQasmFromGrid() no-ops on an empty grid (nothing to convert), so
    // the QASM box has to be reset explicitly here or it keeps showing
    // whatever circuit was there before Clear was pressed.
    $("qasm").value = `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[${nQubits}];\ncreg c[${nQubits}];\n`;
  });
  $("run-btn").addEventListener("click", runCircuit);
  $("ham-catalog-select").addEventListener("change", () => {
    const name = $("ham-catalog-select").value;
    refreshMixSelect();
    if (name) loadMoleculeCircuit({ name });
  });
  $("ham-catalog-btn").addEventListener("click", runCatalogHamiltonian);
  $("ham-custom-btn").addEventListener("click", runCustomHamiltonian);
  $("ham-mix-btn").addEventListener("click", runMixHamiltonians);
  $("ham-mapping-select").addEventListener("change", refreshHamiltonianCatalog);
  $("geom-generate-btn").addEventListener("click", runGenerateGeometry);
  $("element-clear-btn").addEventListener("click", () => { $("ham-symbols").value = ""; });
  $("custom-load-circuit-btn").addEventListener("click", () => {
    const symbols = $("ham-symbols").value.split(",").map((s) => s.trim()).filter(Boolean);
    const geometry = $("ham-geometry").value.split("\n").map((line) => line.trim()).filter(Boolean)
      .map((line) => line.split(",").map((x) => parseFloat(x.trim())));
    const charge = parseInt($("ham-charge").value, 10) || 0;
    loadMoleculeCircuit({ symbols, geometry, charge });
  });
  $("vqe-btn").addEventListener("click", runVqe);
  $("qmmm-forces-btn").addEventListener("click", runQmmmForces);
  $("md-trajectory-btn").addEventListener("click", runMdTrajectory);
  $("vqe-ansatz-select").addEventListener("change", () => {
    // UCCSD converges in far fewer iterations than hardware-efficient,
    // which matters because each of its iterations is much more
    // expensive (deeper decomposed circuit) -- the default reflects that
    // real cost difference, not an arbitrary number.
    $("vqe-maxiter").value = $("vqe-ansatz-select").value === "uccsd" ? "60" : "200";
  });
  $("zne-btn").addEventListener("click", runZne);
  $("zne-matrix-btn").addEventListener("click", runZneMatrix);
}

init();

})();
