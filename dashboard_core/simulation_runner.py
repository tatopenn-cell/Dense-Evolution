"""
Circuit execution: run_simulation() and its internals.

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- pure move, no behavior change.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

import dense_evolution as de

from .qasm_library import QASM_LIBRARY, _run_on_mps

# Above this many basis states, shots sampling draws from only the
# top-K most probable states instead of the full 2**n_qubits distribution
# (see run_simulation) -- same idea as build_panel_mosaico's dim_vis cap.
SHOTS_SAMPLING_CAP = 4096


def estrai_valore_puro(elemento):
    if elemento is None:
        return 0
    tipo_str = str(type(elemento)).lower()
    if "builtin" in tipo_str or "method" in tipo_str or "function" in tipo_str:
        try:
            return estrai_valore_puro(elemento())
        except Exception:
            return 0
    if callable(elemento):
        try:
            return estrai_valore_puro(elemento())
        except Exception:
            pass
    if hasattr(elemento, 'index'):
        val = getattr(elemento, 'index')
        val_tipo = str(type(val)).lower()
        if "builtin" in val_tipo or "method" in val_tipo or callable(val):
            try: val = val()
            except Exception: pass
        if val is not elemento:
            return estrai_valore_puro(val)
    if hasattr(elemento, 'value'):
        val = getattr(elemento, 'value')
        val_tipo = str(type(val)).lower()
        if "builtin" in val_tipo or "method" in val_tipo or callable(val):
            try: val = val()
            except Exception: pass
        if val is not elemento:
            return estrai_valore_puro(val)
    if isinstance(elemento, str):
        try:
            if '.' in elemento:
                return float(elemento)
            return int(elemento)
        except ValueError:
            return elemento
    if isinstance(elemento, (int, float, np.integer, np.floating)):
        return elemento
    return elemento


def run_simulation(source_mode, circuit_name, qasm_text, noise_model, noise_p, shots, seed, use_float32=True, engine='dense'):
    """Adapted from core_calcolo_quantistico (dash.py:3356, canonical/later definition).

    engine: 'dense' (default, DenseSVSimulator/Chunk) or 'mps' (MPSSimulator,
    dense_evolution/mps.py -- bond-truncated, exact for low-entanglement
    circuits, capped at 24 qubits in this dashboard path; see that module's
    docstring for why the Lloyd-Max quantization from the original
    prototype was dropped)."""
    previous_x64_state = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', not use_float32)
    try:
        return _run_simulation_body(source_mode, circuit_name, qasm_text, noise_model, noise_p, shots, seed, use_float32, engine)
    finally:
        # jax_enable_x64 is a process-wide flag: without restoring it here, a float32 run here
        # silently downgrades precision for unrelated code (e.g. the Vector Healing page) that
        # runs later in the same process and never sets its own precision.
        jax.config.update('jax_enable_x64', previous_x64_state)


def _run_simulation_body(source_mode, circuit_name, qasm_text, noise_model, noise_p, shots, seed, use_float32, engine='dense'):
    if source_mode == 'Libreria Built-in' and QASM_LIBRARY:
        qasm_string = QASM_LIBRARY[circuit_name]
        nome_circuito = circuit_name
    else:
        qasm_string = qasm_text
        nome_circuito = 'Custom Workspace'

    parser = de.QASMParser()
    parsed_circuit = parser.parse(qasm_string)
    comandi_originali = parsed_circuit.ops

    try:
        n_qubits = int(parsed_circuit.n_qubits)
    except Exception:
        n_qubits = 0

    if n_qubits <= 2 or n_qubits > 34:
        max_qubit_idx = -1
        for cmd in comandi_originali:
            if isinstance(cmd, dict) and 'qubits' in cmd:
                q_list = cmd['qubits']
                if isinstance(q_list, (list, tuple, np.ndarray)):
                    for q_item in q_list:
                        val_puro = estrai_valore_puro(q_item)
                        try:
                            idx_check = int(val_puro)
                            if idx_check > max_qubit_idx and idx_check < 40:
                                max_qubit_idx = idx_check
                        except Exception:
                            pass
                else:
                    try:
                        idx_check = int(estrai_valore_puro(q_list))
                        if idx_check > max_qubit_idx and idx_check < 40:
                            max_qubit_idx = idx_check
                    except Exception:
                        pass
        n_qubits = max_qubit_idx + 1 if max_qubit_idx != -1 else 4

    for token in nome_circuito.replace('_', ' ').split():
        if 'q' in token.lower() and token.lower().replace('q', '').isdigit():
            n_qubits = int(token.lower().replace('q', ''))
        elif 'd' in token.lower() and token.lower().replace('d', '').isdigit():
            n_qubits = int(token.lower().replace('d', ''))

    # Direct DenseSVSimulator allocates 2**n_qubits complex elements
    # immediately (17 GB at 30 qubits) -- for circuits that don't fit
    # safely in RAM, de.Chunk is used instead: it collapses to a single
    # inner DenseSVSimulator when the circuit *does* fit (no behavior
    # change, verified against the pre-existing test suite), and only
    # actually splits into multiple chunks when the direct allocation
    # would risk an OOM crash. Decided by de.Chunk's own dynamic,
    # available-RAM-based sizing (dense_evolution.chunk.get_dynamic_chunk),
    # not a hardcoded qubit-count constant, so it adapts to the machine
    # it's running on rather than an arbitrary limit.
    usa_mps = engine == 'mps'
    if usa_mps and n_qubits > 24:
        raise ValueError(
            f"Motore MPS: {n_qubits} qubit superano il limite di 24 qubit per la "
            "modalita' dashboard (oltre serve il campionamento sequenziale di "
            "dense_evolution.mps.MPSSimulator.get_probabilities_sampled, non ancora "
            "collegato ai pannelli esistenti -- questi si aspettano un array denso "
            "di probabilita'). Usa il motore denso per circuiti piu' grandi, oppure "
            "MPSSimulator direttamente via script per centinaia di qubit a bassa "
            "entanglement."
        )
    usa_chunk = (not usa_mps) and de.chunk.MemoryChunker(n_qubits).num_chunks > 1
    if usa_mps:
        # Placeholder DenseSVSimulator: its .sv gets overwritten below with the
        # MPS-computed statevector. Keeping everything downstream (noise
        # injection, get_probabilities, memory_mb, VQE's res['sim'] usage) on
        # the same DenseSVSimulator interface regardless of which engine
        # produced the state -- MPS is a different way to COMPUTE the same
        # statevector, not a different downstream data shape, at this <=24
        # qubit scope.
        sim = de.DenseSVSimulator(n_qubits=n_qubits, use_gpu=False, use_float32=use_float32)
    elif usa_chunk:
        sim = de.Chunk(n_qubits=n_qubits, use_gpu=False, use_float32=use_float32)
    else:
        sim = de.DenseSVSimulator(n_qubits=n_qubits, use_gpu=False, use_float32=use_float32)

    comandi_beast_mode = []
    for cmd in comandi_originali:
        if not isinstance(cmd, dict) or 'name' not in cmd:
            continue
        nome_porta = str(cmd['name']).lower().strip()
        qubits_grezzi = cmd.get('qubits', [])
        params_grezzi = cmd.get('params', [])

        if nome_porta in ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg']:
            try:
                target = int(estrai_valore_puro(qubits_grezzi[0]))
                if target < n_qubits:
                    comandi_beast_mode.append([nome_porta, target, -1])
            except Exception: pass
        elif nome_porta in ['rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'p']:
            try:
                param = float(estrai_valore_puro(params_grezzi[0]))
                target = int(estrai_valore_puro(qubits_grezzi[0]))
                if target < n_qubits:
                    comandi_beast_mode.append([nome_porta, target, param])
            except Exception: pass
        elif nome_porta in ['cx', 'cy', 'cz', 'swap']:
            try:
                control = int(estrai_valore_puro(qubits_grezzi[0]))
                target = int(estrai_valore_puro(qubits_grezzi[1]))
                if control < n_qubits and target < n_qubits:
                    # compiler.py's documented tuple contract is (gate, control, target)
                    # for 2-qubit gates — this used to be reversed (see audit finding #1).
                    comandi_beast_mode.append([nome_porta, control, target])
            except Exception: pass
        elif nome_porta in ['ccx', 'toffoli']:
            try:
                c1 = int(estrai_valore_puro(qubits_grezzi[0]))
                c2 = int(estrai_valore_puro(qubits_grezzi[1]))
                t = int(estrai_valore_puro(qubits_grezzi[2]))
                if c1 < n_qubits and c2 < n_qubits and t < n_qubits:
                    comandi_beast_mode.append([nome_porta, c1, c2, t])
            except Exception: pass
        elif nome_porta in ['cp', 'crz']:
            try:
                param = float(estrai_valore_puro(params_grezzi[0]))
                control = int(estrai_valore_puro(qubits_grezzi[0]))
                target = int(estrai_valore_puro(qubits_grezzi[1]))
                if control < n_qubits and target < n_qubits:
                    comandi_beast_mode.append([nome_porta, control, target, param])
            except Exception: pass

    start_time = time.perf_counter()

    def _run(s, ops):
        # de.Chunk exposes run_chunk() instead of run_circuit_jit_beast_mode()
        # -- everything else about the two classes is forwarded identically
        # (get_probabilities, sv, memory_mb), so this is the only branch point.
        if isinstance(s, de.Chunk):
            s.run_chunk(ops)
        else:
            s.run_circuit_jit_beast_mode(ops)

    def _mps_statevector(ops):
        """Runs ops on a fresh MPSSimulator (dense_evolution/mps.py) and
        returns the contracted statevector as a JAX array matching sim's
        dtype -- a new MPSSimulator per call, so this is safe to call twice
        (ideal + noisy comparison) without shared mutable state."""
        mps_sim = _run_on_mps(ops, n_qubits)
        sv = mps_sim.contract_to_statevector()
        return jnp.asarray(sv, dtype=jnp.complex64 if use_float32 else jnp.complex128)

    if noise_model == 'ideal':
        if usa_mps:
            sim.sv = _mps_statevector(comandi_beast_mode)
        else:
            _run(sim, comandi_beast_mode)
        prob_ideal = sim.get_probabilities()
        prob_noisy = prob_ideal
    else:
        np.random.seed(seed)
        if usa_mps:
            sim_ideal = de.DenseSVSimulator(n_qubits=n_qubits, use_float32=use_float32)
            sim_ideal.sv = _mps_statevector(comandi_beast_mode)
        elif usa_chunk:
            sim_ideal = de.Chunk(n_qubits=n_qubits, use_float32=use_float32)
            _run(sim_ideal, comandi_beast_mode)
        else:
            sim_ideal = de.DenseSVSimulator(n_qubits=n_qubits, use_float32=use_float32)
            _run(sim_ideal, comandi_beast_mode)
        prob_ideal = sim_ideal.get_probabilities()

        if usa_mps:
            sim.sv = _mps_statevector(comandi_beast_mode)
        else:
            _run(sim, comandi_beast_mode)
        if noise_p > 0:
            # `rng=` matters here even though sim.sv is a JAX array: without
            # it, NoiseModel.apply_to_sv falls back to OS entropy for the
            # noise channel's randomness (see dense_evolution/registry.py's
            # rng/jax_key docs), so two runs with the same `seed` produced
            # different noisy results despite the seed parameter implying
            # otherwise -- found via a flaky test that called this with a
            # fixed seed and expected (and got, only sometimes) the same
            # fidelity degradation curve.
            sim.sv = de.NoiseModel.apply_to_sv(
                sv=sim.sv, n=n_qubits, model=noise_model, p=float(noise_p),
                rng=np.random.default_rng(seed),
            )
        prob_noisy = sim.get_probabilities()

    t_elapsed = time.perf_counter() - start_time

    if use_float32:
        prob = np.array(prob_noisy, dtype=np.float32)
        prob_id = np.array(prob_ideal, dtype=np.float32)
    else:
        prob = np.array(prob_noisy, dtype=np.float64)
        prob_id = np.array(prob_ideal, dtype=np.float64)

    shannon_entropy = -np.sum(prob * np.log2(prob + 1e-10))
    idx_max = np.argmax(prob)
    stato_dominante = format(idx_max, '0' + str(n_qubits) + 'b')

    fidelity_value = float(np.sum(np.sqrt(prob * prob_id)))
    noise_factor_curve = np.array([fidelity_value * (1.0 - (i * float(noise_p) / 20.0)) for i in range(100)])
    noise_factor_curve = np.clip(noise_factor_curve, 0.0, 1.0)

    # np.random.choice(len(prob), p=prob, ...) builds an internal cumulative
    # array over the full 2**n_qubits distribution -- another O(2**n) cost
    # independent of the Chunk/DenseSVSimulator split above. For huge
    # qubit counts, sample from only the top-K most probable states
    # (same truncation idea already used by build_panel_mosaico's
    # dim_vis = min(1024, ...)) instead of the full distribution.
    if len(prob) > SHOTS_SAMPLING_CAP:
        top_idx = np.argpartition(prob, -SHOTS_SAMPLING_CAP)[-SHOTS_SAMPLING_CAP:]
        top_prob = prob[top_idx]
        top_prob = top_prob / top_prob.sum()
        shots_data = top_idx[np.random.choice(len(top_idx), p=top_prob, size=shots)]
    else:
        shots_data = np.random.choice(len(prob), p=prob, size=shots)

    return {
        'prob': prob,
        'prob_ideal': prob_ideal,
        'noise_factor': noise_factor_curve,
        'fidelity': fidelity_value,
        'n_qubits': n_qubits,
        'entropy': shannon_entropy,
        'idx_max': idx_max,
        'stato_dominante': stato_dominante,
        'tempo': t_elapsed,
        'ram': sim.memory_mb(),
        'nome': nome_circuito,
        'porte_count': len(comandi_beast_mode),
        'shots_data': shots_data,
        'sim': sim,
        'parser': parser,
    }
