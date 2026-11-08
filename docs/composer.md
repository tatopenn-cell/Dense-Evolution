# Composer

Draw a circuit in [Quirk](https://algassert.com/quirk), which simulates it itself up to
16 qubits. Beyond 16 qubits, with noise, or on a local kernel, use the Dense-Evolution
engine below: the matrix product state (MPS) simulator takes the circuit as OpenQASM.

<link rel="stylesheet" href="../assets/composer/style.css" />

<h2 class="de-h">Quirk: up to 16 qubits</h2>
<iframe id="de-quirk" class="de-quirk" title="Quirk circuit editor"
  src="../assets/composer/quirk/quirk.html#circuit=%7B%22cols%22%3A%5B%5B%22H%22%5D%2C%5B%22%E2%80%A2%22%2C%22X%22%5D%5D%7D"></iframe>
<div class="de-quirk-actions">
  <button id="de-quirk-compile" class="btn btn-primary">Compila verso Dense</button>
  <button id="de-quirk-auto" class="btn btn-ghost" aria-pressed="false"
    title="Con Auto acceso il circuito passa da solo, in tutti e due i versi, a ogni modifica">Auto: spento</button>
</div>
<p class="hint">
  The editor and its simulator are Quirk by Craig Gidney (Apache-2.0), unchanged.
</p>
<div id="de-quirk-note" class="hint"></div>

<h2 class="de-h">Dense-Evolution engine: beyond 16 qubits, noise, VQE</h2>

<div id="de-composer-root">
  <div class="de-layout">
  <aside class="de-side">
  <section class="toolbar">
    <label>Preset
      <select id="preset-select"></select>
    </label>
    <label>Qubits
      <input id="n-qubits" type="number" min="1" max="20" value="2" />
      <span id="qubits-limit-info" class="hint"></span>
    </label>
    <label>Shots
      <input id="shots" type="number" min="1" max="100000" value="1000" step="100" />
    </label>
    <label>Seed
      <input id="seed" type="number" min="0" value="42" />
    </label>
    <label>Noise model
      <select id="noise-model-select"></select>
    </label>
    <label>Noise p
      <input id="noise-p" type="number" min="0" max="1" step="0.01" value="0" style="width: 4.5rem;" />
    </label>
    <label>Backend
      <select id="backend-select">
        <option value="dense">Dense</option>
        <option value="mps">MPS</option>
      </select>
    </label>
    <button id="run-btn" class="btn btn-primary">▶ Run</button>
    <span id="status" class="status"></span>
  </section>

    <div class="panel de-topology">
      <h3>Topology</h3>
      <label>Pattern
        <select id="topo-pattern">
          <option value="linear">linear (chain)</option>
          <option value="circular">circular (ring)</option>
          <option value="full">full (all-to-all)</option>
          <option value="star">star (hub)</option>
          <option value="brick">brick (brickwork)</option>
        </select>
      </label>
      <label>Qubits
        <input id="topo-n" type="number" min="2" max="64" value="5" />
      </label>
      <label>Gate
        <select id="topo-gate"><option value="cx">cx</option><option value="cz">cz</option><option value="cy">cy</option></select>
      </label>
      <label><input id="topo-reverse" type="checkbox" /> Reverse direction</label>
      <label id="topo-hub-label" hidden>Hub qubit
        <input id="topo-hub" type="number" min="0" value="0" />
      </label>
      <label><input id="topo-h" type="checkbox" checked /> Hadamard on every qubit first</label>
      <div id="topo-edges" class="hint"></div>
      <button id="topo-load" class="btn btn-primary">&rarr; Carica nel box OpenQASM</button>
    </div>

    <div class="panel de-inputs">
      <h3>Qubit che partono da |1&#x27E9;</h3>
      <label>Indici, separati da virgola
        <input id="input-ones" type="text" placeholder="per esempio 0,2" />
      </label>
      <div class="hint">Run aggiunge una riga x per ognuno, dopo le dichiarazioni, senza cambiare il testo del box.</div>
    </div>

    <div class="panel panel-code">
      <h3>OpenQASM 2.0</h3>
      <textarea id="qasm" class="code" spellcheck="false"></textarea>
      <button id="de-quirk-draw" class="btn btn-primary">Compila verso Quirk</button>
    </div>
  </aside>

  <div class="de-main">
    <section id="de-view-build" class="de-view">
      <p class="hint">
        Run sends the OpenQASM on the left to the engine chosen. Use it beyond 16 qubits (the
        presets go up to 1500), with noise, or on the local kernel. A circuit drawn in Quirk is
        copied to the box at every edit.
      </p>
    </section>

    <section id="de-view-results" class="de-view" hidden>
      <div id="backend-info" class="hint"></div>
      <div id="fidelity-info" class="hint"></div>
      <div class="de-tabs" role="tablist">
        <button class="de-tab active" data-tab="circuit" role="tab">Circuit</button>
        <button class="de-tab" data-tab="statevector" role="tab">Statevector</button>
        <button class="de-tab" data-tab="prob" role="tab">Probabilities &amp; Q-sphere</button>
        <button class="de-tab" data-tab="qubits" role="tab">Qubits</button>
        <button class="de-tab" data-tab="perqubit" role="tab">Per qubit</button>
        <button class="de-tab" data-tab="outcomes" role="tab">Outcomes</button>
      </div>
      <div class="de-tabpanel" data-tab="circuit">
        <img id="circuit-img" class="figure" alt="Circuit diagram" />
      </div>
      <div class="de-tabpanel" data-tab="statevector" hidden>
        <table id="statevector-table" class="sv-table">
          <thead><tr><th>state</th><th>re</th><th>im</th><th>|amp|</th><th>phase</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="de-tabpanel" data-tab="prob" hidden>
        <img id="histogram-img" class="figure" alt="Probabilities histogram" />
        <div id="histogram-skip-msg" class="hint"></div>
        <img id="qsphere-img" class="figure" alt="Q-sphere" />
        <div id="qsphere-skip-msg" class="hint"></div>
      </div>
      <div class="de-tabpanel" data-tab="qubits" hidden>
        <img id="bloch-img" class="figure" alt="Bloch spheres" />
        <div id="bloch-skip-msg" class="hint"></div>
      </div>
      <div class="de-tabpanel" data-tab="perqubit" hidden>
        <table id="perqubit-table" class="sv-table">
          <thead><tr><th>qubit</th><th>P(1)</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="de-tabpanel" data-tab="outcomes" hidden>
        <table id="outcome-table" class="sv-table">
          <thead><tr><th>outcome</th><th>shots</th><th>probability</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
  </div>
  </div>


  <details class="de-collapsible" open>
  <summary>VQE e QAOA nel browser (NumPy, nessun kernel)</summary>
  <section class="row row-hamiltonian">
    <div class="panel panel-hamiltonian">
      <h3>VQE &mdash; ansatz hardware-efficient su una molecola del catalogo</h3>
      <p class="hint">
        Un RY su ogni qubit, poi l'entangler scelto (gli stessi cinque schemi della scheda Topology), per i layer
        indicati; parametri trovati da Adam con la regola parameter-shift. Gli Hamiltoniani sono quelli del
        catalogo del dashboard, scritti dal codice vero (Hartree-Fock e mappa di Jordan-Wigner).
      </p>
      <div class="ham-row">
        <label>Molecola <select id="vqe-w-mol"></select></label>
        <label>Layer <input id="vqe-w-layers" type="number" min="1" max="12" value="2" style="width: 4rem;" /></label>
        <label>Entangler
          <select id="vqe-w-pattern">
            <option value="linear">linear (chain)</option>
            <option value="circular">circular (ring)</option>
            <option value="full">full (all-to-all)</option>
            <option value="star">star (hub)</option>
            <option value="brick">brick (brickwork)</option>
          </select>
        </label>
        <label>Iterazioni <input id="vqe-w-iter" type="number" min="1" value="200" style="width: 5rem;" /></label>
        <label>Seed <input id="vqe-w-seed" type="number" min="0" value="0" style="width: 4rem;" /></label>
        <label>Precisione
          <select id="vqe-w-dtype"><option value="complex64">complex64</option><option value="complex128">complex128</option></select>
        </label>
        <button id="vqe-w-run" class="btn btn-primary">Esegui VQE</button>
      </div>
      <div id="vqe-w-result" class="ham-result"></div>
    </div>
    <div class="panel panel-hamiltonian">
      <h3>QAOA &mdash; MaxCut sul grafo di uno schema</h3>
      <p class="hint">
        Il grafo ha i lati dello schema scelto, con peso 1. Angoli di partenza gamma in [0, 2 pi] e beta in [0, pi]
        (Farhi, Goldstone, Gutmann), ottimizzati da Adam con la regola parameter-shift. Il taglio massimo esatto
        si prova fino a 20 qubit.
      </p>
      <div class="ham-row">
        <label>Qubit <input id="qaoa-w-n" type="number" min="2" max="20" value="6" style="width: 4rem;" /></label>
        <label>Grafo
          <select id="qaoa-w-pattern">
            <option value="linear">linear (chain)</option>
            <option value="circular" selected>circular (ring)</option>
            <option value="full">full (all-to-all)</option>
            <option value="star">star (hub)</option>
            <option value="brick">brick (brickwork)</option>
          </select>
        </label>
        <label>p <input id="qaoa-w-p" type="number" min="1" max="8" value="2" style="width: 4rem;" /></label>
        <label>Iterazioni <input id="qaoa-w-iter" type="number" min="1" value="100" style="width: 5rem;" /></label>
        <label>Seed <input id="qaoa-w-seed" type="number" min="0" value="0" style="width: 4rem;" /></label>
        <label>Precisione
          <select id="qaoa-w-dtype"><option value="complex64">complex64</option><option value="complex128">complex128</option></select>
        </label>
        <button id="qaoa-w-run" class="btn btn-primary">Esegui QAOA</button>
      </div>
      <div id="qaoa-w-result" class="ham-result"></div>
    </div>
  </section>
  </details>

  <details class="de-collapsible">
  <summary>Materiali (tavola periodica, elementi reali Z=1..54 + Au, Pb)</summary>
  <section class="row row-hamiltonian">
    <div class="panel panel-hamiltonian">
      <h3>Materiali (tavola periodica, elementi reali Z=1..54 + Au, Pb)</h3>
      <p class="hint">
        Clicca gli elementi per assemblare la lista simboli della molecola
        custom sotto — stesso effetto di scriverli a mano, solo più
        comodo. Qualsiasi combinazione è accettata; il numero di qubit
        reale che ne risulta decide se questo simulatore riesce a
        gestirla (diagonalizzazione esatta/VQE restano nel range ~12
        qubit — un elemento pesante come il piombo darà un errore onesto
        di "troppi qubit", non un risultato finto).
      </p>
      <div id="element-palette" class="palette element-palette"></div>
      <div class="ham-row">
        <button id="element-clear-btn" class="btn btn-ghost">Svuota simboli</button>
      </div>
    </div>
  </section>
  </details>

  <details class="de-collapsible">
  <summary>Hamiltonian (real molecular, PennyLane Hartree-Fock)</summary>
  <section class="row row-hamiltonian">
    <div class="panel panel-hamiltonian">
      <h3>Hamiltonian (real molecular, PennyLane Hartree-Fock)</h3>
      <p class="hint">
        Real fermion-to-qubit Hamiltonians, computed on demand. The whole
        catalog is always listed here (each entry shows the real qubit
        count its Hamiltonian needs) — picking one sets Qubits and loads
        its real Hartree-Fock reference circuit into the editor
        automatically. Or specify any small molecule's own atoms/geometry
        directly below.
      </p>
      <div class="ham-row">
        <label>Mix con (stesso qubit count di sopra)
          <select id="ham-mix-select"><option value="">— nessuno —</option></select>
        </label>
        <label>Peso A
          <input id="ham-mix-weight-a" type="number" value="0.5" step="0.1" style="width: 4rem;" />
        </label>
        <label>Peso B
          <input id="ham-mix-weight-b" type="number" value="0.5" step="0.1" style="width: 4rem;" />
        </label>
        <button id="ham-mix-btn" class="btn btn-ghost">Mescola e calcola stato fondamentale</button>
      </div>
      <div class="ham-row">
        <label>Fermion-to-qubit mapping
          <select id="ham-mapping-select">
            <option value="jordan_wigner">Jordan-Wigner</option>
            <option value="bravyi_kitaev">Bravyi-Kitaev</option>
          </select>
        </label>
        <span class="hint">La mappatura non cambia l'energia (spettro identico) — cambia solo la rappresentazione a qubit dell'Hamiltoniana.</span>
      </div>
      <div class="ham-row">
        <label>Catalog molecule (tutte, con qubit richiesti)
          <select id="ham-catalog-select"><option value="">— caricamento... —</option></select>
        </label>
        <button id="ham-catalog-btn" class="btn btn-ghost">Compute ground state</button>
      </div>
      <div class="ham-row">
        <label>Custom molecule — symbols (comma-separated)
          <input id="ham-symbols" type="text" value="H, H" />
        </label>
        <label>Geometry (Å, one atom per line: x,y,z)
          <textarea id="ham-geometry" class="code ham-geometry" spellcheck="false">0.0, 0.0, 0.0
0.0, 0.0, 0.7414</textarea>
        </label>
        <label>Charge
          <input id="ham-charge" type="number" value="0" style="width: 4rem;" />
        </label>
        <button id="ham-custom-btn" class="btn btn-ghost">Compute ground state</button>
      </div>
      <div class="ham-row">
        <label>Genera geometria — forma
          <select id="geom-shape-select">
            <option value="linear">Lineare (catena)</option>
            <option value="triangular">Triangolare (3 atomi, D3h)</option>
            <option value="ring">Anello (poligono regolare)</option>
          </select>
        </label>
        <label>Lunghezza legame (Å)
          <input id="geom-bond-length" type="number" value="1.0" step="0.01" style="width: 5rem;" />
        </label>
        <button id="geom-generate-btn" class="btn btn-ghost">Genera geometria</button>
        <button id="custom-load-circuit-btn" class="btn btn-ghost">Carica circuito HF (custom)</button>
      </div>
      <div id="ham-result" class="ham-result"></div>
    </div>
  </section>
  </details>

  <details class="de-collapsible">
  <summary>Scan energia — curva vs lunghezza di legame (diagonalizzazione esatta)</summary>
  <section class="row row-hamiltonian">
    <div class="panel panel-hamiltonian">
      <h3>Scan energia — curva vs lunghezza di legame (diagonalizzazione esatta)</h3>
      <p class="hint">
        Ricalcola l'energia di stato fondamentale della molecola custom (i
        simboli inseriti sopra) a più lunghezze di legame in una sola
        richiesta — utile per vedere una curva di dissociazione senza
        rifare "Compute ground state" a mano punto per punto. Ogni punto è
        una diagonalizzazione esatta reale, non un'interpolazione; la forma
        è la stessa usata da "Genera geometria" sopra (lineare, triangolare
        o ad anello), riscalata alla lunghezza di legame di ciascun punto.
      </p>
      <div class="ham-row">
        <label>Forma
          <select id="scan-shape-select">
            <option value="linear">Lineare (catena)</option>
            <option value="triangular">Triangolare (3 atomi, D3h)</option>
            <option value="ring">Anello (poligono regolare)</option>
          </select>
        </label>
        <label>Legame da (Å)
          <input id="scan-bond-from" type="number" value="0.4" step="0.01" style="width: 5rem;" />
        </label>
        <label>Legame a (Å)
          <input id="scan-bond-to" type="number" value="3.0" step="0.01" style="width: 5rem;" />
        </label>
        <label>Punti
          <input id="scan-points" type="number" min="2" max="30" value="9" style="width: 4rem;" />
        </label>
        <button id="scan-btn" class="btn btn-ghost">Calcola curva</button>
      </div>
      <div id="scan-result" class="ham-result"></div>
    </div>
  </section>
  </details>

  <details class="de-collapsible">
  <summary>VQE — ansatz variazionale reale (dense_evolution.vqe)</summary>
  <section class="row row-hamiltonian">
    <div class="panel panel-hamiltonian">
      <h3>VQE — ansatz variazionale reale (dense_evolution.vqe)</h3>
      <p class="hint">
        Genera e ottimizza un vero circuito VQE per la molecola scelta sopra
        (catalogo o custom), parametri trovati da un vero Adam gradient
        descent (differenziazione adjoint, dense_evolution.vqe.run_vqe) —
        nessun angolo fisso, ogni esecuzione riottimizza da capo. Due
        ansatz reali disponibili: <strong>Hardware-efficient</strong>
        (template generico N layer RY+CNOT, tanti parametri, veloce per
        iterazione) o <strong>UCCSD</strong> (eccitazioni fermioniche
        singole/doppie reali della molecola, qml.qchem.excitations — pochi
        parametri, converge in meno iterazioni, ma ogni iterazione è molto
        più pesante perché il circuito decomposto è molto più profondo:
        per LiH/H2O conta minuti, non secondi, per iterazione). Il
        circuito risultante (tradotto in vero OpenQASM dalla decomposizione
        esatta di PennyLane, verificato bit-per-bit contro l'esecuzione su
        dense_evolution) viene caricato nell'editor principale e aggiunto
        alla libreria preset qui sopra.
      </p>
      <div class="ham-row">
        <label>Molecola (catalogo, riusa la selezione sopra) o custom (simboli/geometria sopra)
          <select id="vqe-source-select">
            <option value="catalog">Usa selezione catalogo</option>
            <option value="custom">Usa molecola custom</option>
          </select>
        </label>
        <label>Ansatz
          <select id="vqe-ansatz-select">
            <option value="hardware_efficient">Hardware-efficient</option>
            <option value="uccsd">UCCSD (eccitazioni fermioniche reali)</option>
          </select>
        </label>
        <label>Layer ansatz (solo hardware-efficient)
          <input id="vqe-layers" type="number" min="1" max="20" value="8" style="width: 4rem;" />
        </label>
        <label>Iterazioni Adam (epoche)
          <input id="vqe-maxiter" type="number" min="10" max="1000" value="200" style="width: 5rem;" />
        </label>
      </div>
      <div class="ham-row">
        <label>Adam — Learning rate
          <input id="vqe-step-size" type="number" min="0.0001" max="1" step="0.0001" value="0.1" style="width: 6rem;" />
        </label>
        <label>Adam — beta1
          <input id="vqe-beta1" type="number" min="0" max="0.999" step="0.001" value="0.9" style="width: 5.5rem;" />
        </label>
        <label>Adam — beta2
          <input id="vqe-beta2" type="number" min="0" max="0.9999" step="0.0001" value="0.999" style="width: 6rem;" />
        </label>
        <button id="vqe-btn" class="btn btn-ghost">Genera circuito VQE</button>
      </div>
      <div id="vqe-result" class="ham-result"></div>
    </div>
  </section>
  </details>

  <details class="de-collapsible">
  <summary>Forze &amp; MD — Hellmann-Feynman reale (dashboard_core.qmmm)</summary>
  <section class="row row-hamiltonian">
    <div class="panel panel-hamiltonian">
      <h3>Forze &amp; MD — Hellmann-Feynman reale (dashboard_core.qmmm)</h3>
      <p class="hint">
        Forze nucleari reali sulla molecola scelta sopra:
        F = -d&lt;&psi;|H(R)|&psi;&gt;/dR, con H(R) l'Hamiltoniana
        molecolare reale e differenziabile di PennyLane (metodo "dhf") e
        &psi; lo stato elettronico tenuto fisso (teorema di
        Hellmann-Feynman esatto, non differenze finite). La traiettoria
        MD integra queste forze reali con Velocity-Verlet classico
        (masse atomiche reali, costante di conversione derivata dalle
        costanti CODATA) — "Ricalcola stato elettronico" risolve un vero
        Hartree-Fock ad ogni passo (MD ab-initio vero, molto più
        costoso) invece di tenere fisso lo stato iniziale (approssimazione
        reale e dichiarata, valida vicino alla geometria di partenza).
      </p>
      <div class="ham-row">
        <button id="qmmm-forces-btn" class="btn btn-ghost">Calcola forze (stato attuale)</button>
        <label>Passi MD
          <input id="md-steps" type="number" min="1" max="200" value="20" style="width: 4.5rem;" />
        </label>
        <label>dt (fs)
          <input id="md-dt" type="number" min="0.01" max="5" step="0.01" value="0.5" style="width: 4.5rem;" />
        </label>
        <label>
          <input id="md-recompute" type="checkbox" style="width: auto;" />
          Ricalcola stato elettronico ad ogni passo (ab-initio vero, lento)
        </label>
        <button id="md-trajectory-btn" class="btn btn-ghost">Esegui traiettoria MD</button>
      </div>
      <div id="qmmm-result" class="ham-result"></div>
    </div>
  </section>
  </details>

  <details class="de-collapsible">
  <summary>Mitigation — Zero-Noise Extrapolation (real, dense_evolution.zero_noise_extrapolation)</summary>
  <section class="row row-mitigation">
    <div class="panel panel-mitigation">
      <h3>Mitigation — Zero-Noise Extrapolation (real, dense_evolution.zero_noise_extrapolation)</h3>
      <p class="hint">
        Measures a Pauli expectation value on the ideal state and on the
        real noise channel above (each an ensemble average over many
        stochastic Kraus draws), then extrapolates back to zero noise.
        Richardson: exact interpolation through 1×/2×/3× "Noise p".
        Polynomial (degree 2): least-squares fit through 1×..5×, trades a
        little interpolation bias for averaging down statistical noise
        across more scales. Pauli string uses dense_evolution's own qubit
        ordering (position 0 = qubit 0), length must match the circuit's
        qubit count.
      </p>
      <div class="ham-row">
        <label>Pauli string (e.g. ZZ)
          <input id="zne-pauli" type="text" value="ZZ" style="width: 6rem;" />
        </label>
        <label>Extrapolation
          <select id="zne-method-select">
            <option value="richardson">Richardson (3 scale, esatta)</option>
            <option value="polynomial">Polynomial deg. 2 (5 scale, fit)</option>
          </select>
        </label>
        <button id="zne-btn" class="btn btn-ghost">Run ZNE</button>
      </div>
      <div id="zne-result" class="ham-result"></div>
    </div>
  </section>
  </details>

  <details class="de-collapsible">
  <summary>Mitigation — density-matrix ZNE (real, dense_evolution.zne_density_matrix)</summary>
  <section class="row row-mitigation">
    <div class="panel panel-mitigation">
      <h3>Mitigation — density-matrix ZNE (real, dense_evolution.zne_density_matrix)</h3>
      <p class="hint">
        Builds a real Monte-Carlo density-matrix estimate at 1×/2×/3× the
        "Noise p" above, extrapolates to zero noise, and projects onto the
        nearest physical (positive-semidefinite) state
        (Smolin–Gambetta–Smith). Graded — never fed back in — against the
        true ideal state via real Uhlmann fidelity, so the improvement
        shown is an honest measurement, not a guaranteed number.
      </p>
      <div class="ham-row">
        <button id="zne-matrix-btn" class="btn btn-ghost">Run density-matrix ZNE</button>
      </div>
      <div id="zne-matrix-result" class="ham-result"></div>
    </div>
  </section>
  </details>
</div>

<script src="../assets/composer/web_kernel.js"></script>
<script src="../assets/composer/quirk_bridge.js"></script>
<script src="../assets/composer/circuit_tools.js"></script>
<script src="../assets/composer/variational_panel.js"></script>
<script src="../assets/composer/app.js"></script>

!!! tip "No kernel? Run the circuit in your browser"
    If no local kernel answers, the page offers **Prova nel browser (NumPy,
    senza installare)**. It loads Pyodide (Python in WebAssembly) and the NumPy
    MPS simulator
    ([`dense_evolution/backends/mps_numpy.py`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/dense_evolution/backends/mps_numpy.py)),
    so the circuit runs in this page and is sent nowhere. What it does: any
    OpenQASM circuit of the supported gates. Up to 20 qubits it answers as the local
    kernel does, with the sampled shots (same seed), the statevector and the circuit,
    histogram, Q-sphere and Bloch figures; beyond 20 qubits, up to 1500, with the most
    probable states, the bond dimension and the memory of the MPS. What it does not:
    noise models, VQE, Hamiltonians and mitigation, which need JAX and the local
    kernel below. The numbers are in single precision (`complex64`); the
    first load downloads Pyodide from jsDelivr, so it needs the network.

!!! note "Runs on your own machine, not a shared server"
    This page is static — every circuit runs on a real local kernel
    (`dense_evolution`'s own `DenseSVSimulator`, driven by `local_site/app/server.py`)
    that you run on your own PC. Nothing here is precomputed or mocked, and
    nothing you run is sent anywhere else. Grab the installer for your OS
    and run it yourself (no site can install or launch anything on your
    machine on its own) — it walks you through what it's about to do
    (install the package, optionally download an offline copy of this
    page, optionally add Desktop/Start Menu/startup shortcuts) before
    doing any of it:
    [Windows (.bat)](assets/installer/install-composer.bat){ download }
    &middot;
    [macOS / Linux (.sh)](assets/installer/install-composer.sh){ download }
    — or by hand: `pip install dense-evolution[composer]` then
    `dense-evolution serve`. Either way, reload this page once it's
    running — the banner above turns green the moment it detects your
    kernel. To remove everything the installer created, run
    [uninstall-composer.bat](assets/installer/uninstall-composer.bat){ download }
    or
    [uninstall-composer.sh](assets/installer/uninstall-composer.sh){ download }.
