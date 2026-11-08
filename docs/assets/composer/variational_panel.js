// The panel "VQE e QAOA nel browser": the two variational algorithms of
// dense_evolution/backends/variational_numpy.py, run in the NumPy engine of the page.
//
// VQE: the molecules of the catalog of the dashboard (molecules.json, written from the real code),
// the hardware-efficient ansatz (a RY on every qubit, then the entangler: any of the five patterns of
// the Topology panel) and Adam. QAOA: MaxCut on the graph of a pattern. The numbers come from the
// worker; the page only draws them.
(function () {

const $ = (id) => document.getElementById(id);

// A polyline of the values of `history` (energy or F_p at each step).
function chart(history) {
  if (!history.length) return "";
  const w = 420;
  const h = 150;
  const lo = Math.min(...history);
  const hi = Math.max(...history);
  const span = hi - lo || 1;
  const pts = history.map((v, i) => {
    const x = 8 + (i / Math.max(history.length - 1, 1)) * (w - 16);
    const y = 8 + (1 - (v - lo) / span) * (h - 16);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" class="figure" role="img" aria-label="history">` +
    `<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="${pts.join(" ")}"/>` +
    `<text x="10" y="14" font-size="10" fill="currentColor">${hi.toFixed(6)}</text>` +
    `<text x="10" y="${h - 4}" font-size="10" fill="currentColor">${lo.toFixed(6)}</text></svg>`;
}

function table(rows) {
  return "<table class=\"sv-table\"><tbody>" +
    rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("") + "</tbody></table>";
}

function busy(button, out, text) {
  button.disabled = true;
  out.innerHTML = `<p class="hint">${text}</p>`;
}

async function runVqe() {
  const out = $("vqe-w-result");
  const button = $("vqe-w-run");
  try {
    const mols = await window.DEWebKernel.molecules();
    const mol = mols[$("vqe-w-mol").value];
    const pattern = $("vqe-w-pattern").value;
    const n = mol.n_qubits;
    busy(button, out, `VQE di ${n} qubit in corso nel browser (${mol.terms.length} termini di Pauli)...`);
    const res = await window.DEWebKernel.vqe({
      n_qubits: n,
      terms: mol.terms,
      layers: parseInt($("vqe-w-layers").value, 10),
      edges: window.DECircuitTools.topologyEdges(n, pattern, false, 0),
      maxiter: parseInt($("vqe-w-iter").value, 10),
      seed: parseInt($("vqe-w-seed").value, 10),
      dtype: $("vqe-w-dtype").value,
      occupation: mol.hf_occupation,
    });
    out.innerHTML = table([
      ["Energia VQE (Hartree)", res.energy.toFixed(8)],
      ["Energia Hartree-Fock", res.hf_energy.toFixed(8)],
      ["Energia esatta (diagonalizzazione, molecules.json)", mol.exact_energy.toFixed(8)],
      ["Errore sull'esatta", (res.energy - mol.exact_energy).toExponential(3)],
      ["Parametri", res.n_params],
      ["Precisione", res.dtype],
      ["Tempo", `${res.elapsed_s.toFixed(1)} s`],
    ]) + chart(res.history) +
      "<button id=\"vqe-w-load\" class=\"btn btn-ghost\">Carica il circuito nel box OpenQASM</button>";
    $("vqe-w-load").addEventListener("click", () => {
      $("qasm").value = res.qasm;
      if ($("preset-select").value !== "__custom__") $("preset-select").value = "__custom__";
      $("qasm").dispatchEvent(new Event("input", { bubbles: true }));
    });
  } catch (err) {
    out.innerHTML = `<p class="hint error">Errore: ${err.message}</p>`;
  } finally {
    button.disabled = false;
  }
}

async function runQaoa() {
  const out = $("qaoa-w-result");
  const button = $("qaoa-w-run");
  try {
    const n = parseInt($("qaoa-w-n").value, 10);
    const pattern = $("qaoa-w-pattern").value;
    const edges = window.DECircuitTools.topologyEdges(n, pattern, false, 0);
    busy(button, out, `QAOA di ${n} qubit su ${edges.length} lati in corso nel browser...`);
    const res = await window.DEWebKernel.qaoa({
      n_qubits: n,
      edges,
      layers: parseInt($("qaoa-w-p").value, 10),
      maxiter: parseInt($("qaoa-w-iter").value, 10),
      seed: parseInt($("qaoa-w-seed").value, 10),
      dtype: $("qaoa-w-dtype").value,
    });
    out.innerHTML = table([
      ["F_p (valore atteso del taglio)", res.expectation.toFixed(6)],
      ["Taglio massimo esatto", res.exact_maxcut === null ? "non calcolato (piu di 20 qubit)" : res.exact_maxcut.toFixed(6)],
      ["Rapporto", res.ratio === null ? "-" : res.ratio.toFixed(6)],
      ["gamma", res.gammas.map((x) => x.toFixed(5)).join(", ")],
      ["beta", res.betas.map((x) => x.toFixed(5)).join(", ")],
      ["Stati piu probabili", res.top_states.map(([s, q]) => `${s} (${q.toFixed(4)})`).join(", ")],
      ["Precisione", res.dtype],
      ["Tempo", `${res.elapsed_s.toFixed(1)} s`],
    ]) + chart(res.history);
  } catch (err) {
    out.innerHTML = `<p class="hint error">Errore: ${err.message}</p>`;
  } finally {
    button.disabled = false;
  }
}

async function start() {
  if (!$("vqe-w-run")) return;
  const sel = $("vqe-w-mol");
  try {
    const mols = await window.DEWebKernel.molecules();
    Object.keys(mols).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = `${name} -- ${mols[name].n_qubits} qubit, ${mols[name].terms.length} termini`;
      sel.appendChild(opt);
    });
  } catch (err) {
    $("vqe-w-result").innerHTML = `<p class="hint error">Errore: ${err.message}</p>`;
  }
  $("vqe-w-run").addEventListener("click", () => { void runVqe(); });
  $("qaoa-w-run").addEventListener("click", () => { void runQaoa(); });
}

window.addEventListener("load", () => { void start(); });

})();
