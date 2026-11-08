// How many qubits the page contracts to a statevector, from the memory of this browser.
//
// The numbers are measured, not chosen. On Chromium (headless, Kaggle) the WebAssembly memory of
// Pyodide after a run of the page on an n-qubit GHZ circuit, contracted to 2^n amplitudes, was
//   n = 20: 214.8125 MB,   n = 22: 534.625 MB,   n = 24: 1545.9375 MB.
// The line through the points 22 and 24 gives a cost per amplitude and a base; it is above the
// measure at 20 (266 MB against 215 MB), so it errs on the safe side. The rules around it are the
// ones of the engine of the repository: a floor of 16 qubits (dense_evolution.chunk.get_dynamic_chunk),
// a ceiling of 24 qubits (the contraction of the MPS stops there, tools/dashboard/core/system_limits.py)
// and a margin of 15% of the memory kept free (SafeMemoryGuard, threshold_pct = 0.15). The
// WebAssembly memory of Pyodide is 32-bit, so 4096 MB is its own ceiling.
(function () {

const MEASURED = [[22, 534.625], [24, 1545.9375]];
const FLOOR = 16;
const CEILING = 24;
const MARGIN = 0.15;
const WASM_MAX_MB = 4096;

const SLOPE_MB = (MEASURED[1][1] - MEASURED[0][1]) / (2 ** MEASURED[1][0] - 2 ** MEASURED[0][0]);
const BASE_MB = MEASURED[0][1] - SLOPE_MB * 2 ** MEASURED[0][0];

// The memory that a run of n qubits needs, in MB.
function predictedHeapMb(n) {
  return BASE_MB + SLOPE_MB * 2 ** n;
}

// deviceMemoryGb: navigator.deviceMemory (GB) or undefined when the browser does not tell. Returns
// {limit, usableMb, basis}: the qubits, the memory the page may use, and what the number rests on.
function browserQubitLimit(deviceMemoryGb) {
  if (!(deviceMemoryGb > 0)) {
    return { limit: FLOOR, usableMb: null, basis: "memoria del browser non rilevabile: minimo di " + FLOOR + " qubit" };
  }
  const usableMb = Math.min(deviceMemoryGb * 1024 * (1 - MARGIN), WASM_MAX_MB);
  let n = FLOOR;
  while (n < CEILING && predictedHeapMb(n + 1) <= usableMb) n += 1;
  return {
    limit: n,
    usableMb,
    basis: `${deviceMemoryGb} GB di memoria, ${Math.round(MARGIN * 100)}% tenuto libero, ` +
      `${Math.round(usableMb)} MB utilizzabili (WebAssembly a 32 bit: al massimo ${WASM_MAX_MB} MB)`,
  };
}

window.DEBrowserMemory = { MEASURED, FLOOR, CEILING, MARGIN, WASM_MAX_MB, predictedHeapMb, browserQubitLimit };

})();
