# Composer

Build a circuit graphically or in OpenQASM, run it on the real
`dense_evolution.DenseSVSimulator`, and see the real statevector,
probabilities, Q-sphere and circuit diagram — the same engine documented
throughout the [API Reference](api/index.md), not a separate demo.

<link rel="stylesheet" href="../static/style.css" />

<div id="de-composer-root">
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

  <section class="row row-top">
    <div class="panel panel-palette">
      <h3>Operations</h3>
      <div id="palette" class="palette"></div>
    </div>

    <div class="panel panel-canvas">
      <div class="panel-head">
        <h3>Circuit</h3>
        <button id="clear-btn" class="btn btn-ghost">Clear grid</button>
      </div>
      <div id="grid" class="grid"></div>
    </div>

    <div class="panel panel-code">
      <h3>OpenQASM 2.0</h3>
      <textarea id="qasm" class="code" spellcheck="false"></textarea>
    </div>
  </section>

  <section class="row row-bottom">
    <div class="panel panel-probabilities">
      <h3>Probabilities</h3>
      <img id="histogram-img" class="figure" alt="Probabilities histogram" />
      <div id="histogram-skip-msg" class="hint"></div>
    </div>
    <div class="panel panel-qsphere">
      <h3>Q-sphere</h3>
      <img id="qsphere-img" class="figure" alt="Q-sphere" />
      <div id="qsphere-skip-msg" class="hint"></div>
    </div>
    <div class="panel panel-bloch">
      <h3>Bloch spheres</h3>
      <img id="bloch-img" class="figure" alt="Bloch spheres" />
      <div id="bloch-skip-msg" class="hint"></div>
    </div>
  </section>

  <section class="row row-extra">
    <div class="panel panel-circuit-diagram">
      <h3>Circuit diagram</h3>
      <img id="circuit-img" class="figure" alt="Circuit diagram" />
    </div>
    <div class="panel panel-statevector">
      <h3>Statevector</h3>
      <div id="backend-info" class="hint"></div>
      <div id="fidelity-info" class="hint"></div>
      <table id="statevector-table" class="sv-table">
        <thead><tr><th>state</th><th>re</th><th>im</th><th>|amp|</th><th>phase</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

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
        <label>Iterazioni Adam
          <input id="vqe-maxiter" type="number" min="10" max="1000" value="200" style="width: 5rem;" />
        </label>
        <button id="vqe-btn" class="btn btn-ghost">Genera circuito VQE</button>
      </div>
      <div id="vqe-result" class="ham-result"></div>
    </div>
  </section>

  <section class="row row-mitigation">
    <div class="panel panel-mitigation">
      <h3>Mitigation — Zero-Noise Extrapolation (real, dense_evolution.zero_noise_extrapolation)</h3>
      <p class="hint">
        Measures a Pauli expectation value on the ideal state and on the
        real noise channel above (1×/2×/3× its "Noise p", each an
        ensemble average over many stochastic Kraus draws), then
        Richardson-extrapolates back to zero noise. Pauli string uses
        dense_evolution's own qubit ordering (position 0 = qubit 0),
        length must match the circuit's qubit count.
      </p>
      <div class="ham-row">
        <label>Pauli string (e.g. ZZ)
          <input id="zne-pauli" type="text" value="ZZ" style="width: 6rem;" />
        </label>
        <button id="zne-btn" class="btn btn-ghost">Run ZNE</button>
      </div>
      <div id="zne-result" class="ham-result"></div>
    </div>
  </section>

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
</div>

<script src="../static/app.js"></script>

!!! note "Runs from this machine"
    This page calls a local API (`/api/run`, `/api/build_from_ops`) served
    by `local_site/app/server.py`, which runs `python server.py` on your
    own PC and executes every circuit on the real simulator — nothing here
    is precomputed or mocked.
