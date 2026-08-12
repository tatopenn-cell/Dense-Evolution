"""
Tests for dense_evolution.interop.noise_model_from_qiskit_backend --
promoted from Dense-Evolution-Discovery's Steane-code hardware bridge
script (scripts/steane_code_block5_qiskit_bridge.py), which converts a
Qiskit backend's real calibration data into a Dense-Evolution-native
noise specification.

Needs qiskit_ibm_runtime (for FakeSherbrooke, a real IBM Eagle-127
historical calibration snapshot -- offline, no account/network needed)
on top of qiskit itself -- pytest.importorskip guards both so this file
stays green in environments without either installed.

Same macOS skip as tests/test_interop.py: qiskit's own QuantumCircuit
construction is known to segfault the whole process on macOS/arm64, so
this whole file's real-circuit-touching class is skipped there rather
than defined at all (an importorskip-only guard still imports qiskit at
collection time and doesn't remove the trigger).
"""
import sys

import numpy as np
import pytest

import dense_evolution as de
from dense_evolution import interop
from dense_evolution.interop import noise_model_from_qiskit_backend
from dense_evolution.registry import NoiseModel
from dense_evolution.measurement import statevector_fidelity


class TestImportSafety:

    def test_root_import_never_fails(self):
        assert hasattr(de, 'noise_model_from_qiskit_backend')

    def test_missing_qiskit_raises_clear_importerror(self, monkeypatch):
        monkeypatch.setattr(interop, 'HAS_QISKIT', False)
        with pytest.raises(ImportError, match='qiskit'):
            noise_model_from_qiskit_backend(None)


if sys.platform == 'darwin':

    @pytest.mark.skip(reason="qiskit destabilizes the process on macOS CI runners -- see tests/test_interop.py")
    class TestCalibrationNoise:
        pass

else:

    class TestCalibrationNoise:

        qiskit = pytest.importorskip('qiskit')
        qiskit_ibm_runtime = pytest.importorskip('qiskit_ibm_runtime')

        @staticmethod
        def _backend():
            from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
            return FakeSherbrooke()

        # ── (a) real calibration data is actually extracted ──────────

        def test_extracts_real_calibration_data(self):
            backend = self._backend()
            specs = noise_model_from_qiskit_backend(backend)

            assert len(specs) > 100  # FakeSherbrooke: 127 qubits, many gate types
            for entry in specs:
                assert set(entry.keys()) == {'gate', 'qubits', 'model', 'p'}
                assert entry['model'] == 'depolarizing'
                assert 0.0 <= entry['p'] <= 1.0
                assert all(isinstance(q, int) for q in entry['qubits'])

            gate_names = {entry['gate'] for entry in specs}
            assert 'sx' in gate_names
            assert 'ecr' in gate_names
            assert 'measure' not in gate_names  # excluded by default -- readout error, not a unitary channel

            sx_q0 = [e for e in specs if e['gate'] == 'sx' and e['qubits'] == [0]]
            assert len(sx_q0) == 1
            assert sx_q0[0]['p'] == pytest.approx(0.00028775142091170115, rel=1e-9)

        def test_skip_gates_is_configurable(self):
            backend = self._backend()
            specs = noise_model_from_qiskit_backend(backend, skip_gates=frozenset())
            gate_names = {entry['gate'] for entry in specs}
            assert 'measure' in gate_names  # no longer excluded

        def test_circuit_filter_restricts_to_used_targets(self):
            from qiskit import QuantumCircuit
            backend = self._backend()
            qc = QuantumCircuit(3)
            qc.sx(0)
            qc.ecr(0, 1)

            specs = noise_model_from_qiskit_backend(backend, circuit=qc)
            targets = {(e['gate'], tuple(e['qubits'])) for e in specs}
            assert ('sx', (0,)) in targets
            assert ('ecr', (0, 1)) in targets or ('ecr', (1, 0)) in targets
            # nothing outside the circuit's own qubits/gates leaked in
            assert all(e['gate'] in ('sx', 'ecr') for e in specs)
            assert all(set(e['qubits']) <= {0, 1} for e in specs)

        def test_ecr_direction_independent_match(self):
            # calibration stores ecr(1, 0) but the circuit calls ecr(0, 1) --
            # must still match (apply_to_sv treats multi-qubit targets as
            # independent per-qubit channels, so gate direction is irrelevant).
            from qiskit import QuantumCircuit
            backend = self._backend()
            assert (1, 0) in backend.target['ecr']
            assert (0, 1) not in backend.target['ecr']

            qc = QuantumCircuit(2)
            qc.ecr(0, 1)
            specs = noise_model_from_qiskit_backend(backend, circuit=qc)
            ecr_specs = [e for e in specs if e['gate'] == 'ecr']
            assert len(ecr_specs) == 1

        # ── (b) dedup fix: many repeated gate occurrences must not blow up ──

        def test_dedup_does_not_scale_with_repeated_occurrences(self):
            # An earlier version of the promoted logic (in Discovery's
            # steane_code_block5_qiskit_bridge.py, calling qiskit_aer's
            # add_quantum_error once per gate INSTANCE) composed the same
            # Kraus channel with itself for every repeat, causing multi-GB
            # memory blowup on circuits with many repeated occurrences on
            # the same qubits. This test would have caught that: 500
            # repeated occurrences of the same two targets must still
            # collapse to exactly one spec entry each, not 500.
            from qiskit import QuantumCircuit
            backend = self._backend()
            qc = QuantumCircuit(2)
            for _ in range(500):
                qc.sx(0)
                qc.ecr(0, 1)

            specs = noise_model_from_qiskit_backend(backend, circuit=qc)
            sx_specs = [e for e in specs if e['gate'] == 'sx' and e['qubits'] == [0]]
            ecr_specs = [e for e in specs if e['gate'] == 'ecr']
            assert len(sx_specs) == 1
            assert len(ecr_specs) == 1
            assert len(specs) < 10  # bounded by unique targets, not circuit length (1000 ops)

        def test_dedup_scales_with_unique_targets_not_occurrences(self):
            # Same circuit shape, but scaled two different ways -- entry
            # count must track unique qubits touched, not op count.
            from qiskit import QuantumCircuit
            backend = self._backend()

            qc_many_repeats = QuantumCircuit(2)
            for _ in range(300):
                qc_many_repeats.sx(0)
            specs_repeats = noise_model_from_qiskit_backend(backend, circuit=qc_many_repeats)

            qc_many_qubits = QuantumCircuit(5)
            for q in range(5):
                qc_many_qubits.sx(q)
            specs_qubits = noise_model_from_qiskit_backend(backend, circuit=qc_many_qubits)

            assert len(specs_repeats) == 1
            assert len(specs_qubits) == 5

        # ── (c) applied to a Bell state, gives a sane, non-trivial fidelity ──

        def test_bell_state_fidelity_sane_and_non_trivial(self):
            from qiskit import QuantumCircuit
            from dense_evolution.interop import run_qiskit_circuit

            backend = self._backend()
            specs_full = noise_model_from_qiskit_backend(backend)
            bell_specs = [
                e for e in specs_full
                if set(e['qubits']) <= {0, 1} and e['gate'] in ('sx', 'x', 'ecr')
            ]
            assert len(bell_specs) >= 2  # at least the two sx entries, ideally + ecr

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            sim, _ = run_qiskit_circuit(qc, use_float32=False)
            ideal_sv = np.asarray(sim.get_statevector())

            n_trials = 300
            fidelities = np.empty(n_trials)
            for trial in range(n_trials):
                rng = np.random.default_rng(trial)
                noisy_sv = ideal_sv.copy()
                for entry in bell_specs:
                    noisy_sv = NoiseModel.apply_to_sv(
                        noisy_sv, 2, model=entry['model'], p=entry['p'],
                        qubits=entry['qubits'], rng=rng,
                    )
                fidelities[trial] = statevector_fidelity(noisy_sv, ideal_sv)

            mean_fid = float(np.mean(fidelities))
            assert 0.0 < mean_fid <= 1.0
            assert mean_fid < 1.0 - 1e-9  # noise actually did something, not a silent no-op
            assert mean_fid > 0.5  # calibrated error rates are ~0.1-1% per gate -- should stay high
