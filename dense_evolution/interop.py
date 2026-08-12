import sys
import warnings
from typing import Optional, Tuple
import numpy as np

from .parser import QASMParser, QASMCircuit
from .simulator import DenseSVSimulator

try:
    import qiskit.qasm2 as _qasm2
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

_macos_qiskit_warning_shown = False


def _require_qiskit():
    if not HAS_QISKIT:
        raise ImportError(
            "Qiskit interop requires the 'qiskit' package. "
            "Install it with: pip install dense-evolution[qiskit]")
    global _macos_qiskit_warning_shown
    if sys.platform == 'darwin' and not _macos_qiskit_warning_shown:
        _macos_qiskit_warning_shown = True
        warnings.warn(
            "Qiskit's own QuantumCircuit.__init__ is known to segfault the "
            "whole process on macOS/arm64 (reproduced on GitHub Actions "
            "arm64 runners, Python 3.10-3.12) -- this is an upstream Qiskit "
            "bug, not something dense-evolution can fix from its side. If "
            "you hit a crash, consider the PennyLane bridge instead "
            "(pip install dense-evolution[pennylane]), which does not "
            "construct Qiskit objects and has no known issue of this kind.",
            RuntimeWarning, stacklevel=2)


def _require_pennylane():
    if not HAS_PENNYLANE:
        raise ImportError(
            "PennyLane interop requires the 'pennylane' package. "
            "Install it with: pip install dense-evolution[pennylane]")


def _to_qiskit_bit_order(probs: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Reindex a probability array from Dense-Evolution's native MSB-first
    convention (qubit 0 = most significant bit of the index, the same
    convention as apply_gate_1q/apply_gate_2q/measure/beast-mode) to
    Qiskit's little-endian convention (qubit 0 = least significant bit),
    so the result is directly comparable to Statevector(...).probabilities().

    A plain bit-reversal permutation of the index — verified against
    Statevector.from_instruction(...).probabilities() on an asymmetric
    circuit (exact match only after this reversal, not before).
    """
    perm = [int(format(i, f'0{n_qubits}b')[::-1], 2) for i in range(2 ** n_qubits)]
    return probs[perm]


def from_qiskit(circuit) -> QASMCircuit:
    """Convert a Qiskit QuantumCircuit into a QASMCircuit via OpenQASM 2.0
    (qiskit.qasm2.dumps), reusing the existing QASMParser rather than a
    bespoke gate-by-gate translator."""
    _require_qiskit()
    qasm_str = _qasm2.dumps(circuit)
    return QASMParser().parse(qasm_str)


def _sorted_wires(wires):
    """Best-effort ascending order for a wire sequence. Falls back to the
    original order if the labels aren't mutually comparable (e.g. mixed
    str/int wire names) — better to fall back to the old touch-order
    behavior than to crash on an exotic device's wire labels."""
    try:
        return sorted(wires)
    except TypeError:
        return list(wires)


def from_pennylane(circuit, *args, **kwargs) -> QASMCircuit:
    """Convert a PennyLane QNode or QuantumTape/QuantumScript into a
    QASMCircuit via OpenQASM 2.0, reusing the existing QASMParser.

    PennyLane's own serialization API for a bare tape has changed across
    versions in an incompatible way (verified directly against both):
      - >=~0.43 (Python 3.11+ only): qml.to_openqasm(tape) returns the
        QASM string directly; QuantumTape/QuantumScript no longer has a
        to_openqasm() method at all.
      - <=0.42.x (still installed on Python 3.10, where newer PennyLane
        isn't available): qml.to_openqasm(tape) does NOT special-case a
        bare tape — it returns a QNode-oriented wrapper that crashes with
        AttributeError ('QuantumTape' object has no attribute 'func') if
        called on one. The tape's own tape.to_openqasm() method is what
        works there instead.
    So: a bare tape/QuantumScript uses its own to_openqasm() method when
    present (old API), otherwise falls through to the top-level
    qml.to_openqasm() (new API). A QNode (not a QuantumScript instance)
    always uses the top-level function, which returns a wrapper that must
    be called with the QNode's own arguments — consistent across both
    versions, this path was never the one that broke.

    WIRE ORDER: by default, both PennyLane APIs number the exported QASM
    qubits in the order wires are FIRST TOUCHED in the circuit, not by
    their actual wire index — e.g. `qml.PauliX(wires=2)` followed by
    `qml.CNOT(wires=[2, 1])` becomes `x q[0]; cx q[0],q[1];` in the
    default export, silently renumbering wire 2 -> q[0] and wire 1 -> q[1].
    Verified directly: this produced a topologically different circuit
    from the one PennyLane itself executes whenever wires aren't touched
    in ascending order (a QASMParser-based bridge has no way to recover
    the true mapping after the fact — the touch-order renumbering has
    already happened by the time QASM text exists). Both APIs accept an
    explicit `wires=` argument that forces the true wire order into the
    export instead — used here for both the QNode path (the device's own
    declared wire order) and the tape path (the tape's own wires, sorted
    ascending, since a bare tape has no device to ask).
    """
    _require_pennylane()
    if isinstance(circuit, qml.tape.QuantumScript):
        wires = _sorted_wires(circuit.wires)
        if hasattr(circuit, 'to_openqasm'):
            qasm_str = circuit.to_openqasm(wires=wires, measure_all=False)
        else:
            qasm_str = qml.to_openqasm(circuit, wires=wires, measure_all=False)
    else:
        device = getattr(circuit, 'device', None)
        wires = device.wires if device is not None else None
        result = qml.to_openqasm(circuit, wires=wires, measure_all=False)
        qasm_str = result if isinstance(result, str) else result(*args, **kwargs)
    return QASMParser().parse(qasm_str)


def run_qiskit_circuit(
    circuit,
    use_float32: bool = True,
    sim: Optional[DenseSVSimulator] = None,
) -> Tuple[DenseSVSimulator, np.ndarray]:
    """Run a Qiskit QuantumCircuit on DenseSVSimulator. Returns
    (sim, probabilities) with probabilities reordered into Qiskit's own
    little-endian bit convention, so they compare directly against
    Statevector(circuit).probabilities() — see _to_qiskit_bit_order."""
    circ = from_qiskit(circuit)
    if sim is None:
        sim = DenseSVSimulator(n_qubits=circ.n_qubits, use_float32=use_float32)
    sim.run_circuit(circ.to_tuples())
    probs = np.asarray(sim.get_probabilities())
    return sim, _to_qiskit_bit_order(probs, circ.n_qubits)


_DEFAULT_CALIBRATION_SKIP_GATES = frozenset({'measure'})


def noise_model_from_qiskit_backend(
    backend,
    circuit=None,
    skip_gates=_DEFAULT_CALIBRATION_SKIP_GATES,
) -> list:
    """Build a Dense-Evolution-native noise specification from a Qiskit
    BackendV2's own calibration data (backend.target) -- works for both
    real backends and fake/mock backends carrying a real historical
    calibration snapshot (e.g.
    qiskit_ibm_runtime.fake_provider.FakeSherbrooke), so a simulation can
    use the device's actual measured per-qubit/per-gate error rates
    instead of an idealized channel.

    Returns a list of dicts, each directly usable as the model/p/qubits
    arguments to NoiseModel.apply_to_sv:
        [{'gate': 'sx', 'qubits': [3], 'model': 'depolarizing', 'p': 0.00029},
         {'gate': 'ecr', 'qubits': [1, 0], 'model': 'depolarizing', 'p': 0.0075}, ...]
    One entry per unique (gate, qubit-target) pair found in the
    calibration data -- never duplicated by how many times a gate occurs
    in any one circuit.

    If `circuit` is given, restricts the result to the (gate, qargs)
    targets that circuit actually uses. Still exactly one entry per
    unique target regardless of how many times the circuit repeats it --
    promoted from Dense-Evolution-Discovery's Steane-code hardware bridge
    script (scripts/steane_code_block5_qiskit_bridge.py), where an
    earlier version called qiskit_aer's NoiseModel.add_quantum_error once
    per gate OCCURRENCE instead of once per unique target: AerSimulator
    composes the same Kraus channel with itself on repeated registration
    for the same target (documented semantics, not "apply once per
    instance"), and on a circuit with many repeats on the same qubits
    this blew up to multi-GB memory from the resulting Kraus-term
    combinatorial explosion before OOM-killing the process. This function
    can't reproduce that bug by construction: it walks target[gate_name]
    (already deduplicated by qargs, one InstructionProperties per target)
    and, when `circuit` is given, additionally dedupes via a `seen` set
    keyed on (gate, qargs) before ever appending a spec entry -- so
    repeated occurrences in `circuit` collapse to the same single entry.

    `measure` is excluded by default (readout error is a classical
    bit-flip-on-outcome effect measured after collapse, not a
    pre-measurement unitary channel on the state) -- pass a smaller/
    larger `skip_gates` to change that. Gates with no calibrated error
    (virtual gates like `rz`, or entries with error=None such as `delay`
    or control-flow ops like `for_loop`/`if_else`) are skipped too, since
    there is no error rate to convert into a channel.

    Every entry uses Dense-Evolution's 'depolarizing' model with the
    calibration's average gate error as `p` -- the same standard
    average-gate-infidelity-to-depolarizing-parameter approximation
    qiskit_aer.noise.NoiseModel.from_backend itself falls back to when no
    finer-grained error data is available, not a from-scratch physical
    model of this library's own invention.

    `circuit` matching is order-independent on the qubit tuple (e.g. an
    `ecr(0, 1)` instruction matches a calibration target stored as
    `(1, 0)`) -- NoiseModel.apply_to_sv itself only ever applies
    independent single-qubit channels per entry in `qubits`, never a
    genuine joint multi-qubit channel, so which qubit was "control" vs
    "target" in the original gate has no effect on the result here."""
    _require_qiskit()
    target = backend.target

    wanted = None
    if circuit is not None:
        wanted = {
            (instr.operation.name, tuple(sorted(circuit.find_bit(q).index for q in instr.qubits)))
            for instr in circuit.data
        }

    specs = []
    seen = set()
    for gate_name in target.operation_names:
        if gate_name in skip_gates:
            continue
        gate_map = target[gate_name]
        if not gate_map:
            continue
        for qargs, props in gate_map.items():
            if qargs is None or props is None or props.error is None:
                continue
            if wanted is not None and (gate_name, tuple(sorted(qargs))) not in wanted:
                continue
            key = (gate_name, qargs)
            if key in seen:
                continue
            seen.add(key)
            specs.append({
                'gate': gate_name,
                'qubits': list(qargs),
                'model': 'depolarizing',
                'p': float(props.error),
            })
    return specs


def run_pennylane_circuit(
    circuit,
    *args,
    use_float32: bool = True,
    sim: Optional[DenseSVSimulator] = None,
    **kwargs,
) -> Tuple[DenseSVSimulator, np.ndarray]:
    """Run a PennyLane QNode/tape on DenseSVSimulator. Returns
    (sim, probabilities) in Dense-Evolution's native ordering, WITHOUT any
    bit-reversal — unlike run_qiskit_circuit, because PennyLane's own wire
    convention (wire 0 = most significant) already matches Dense-Evolution's
    MSB-first convention. Do not "symmetrize" this with the Qiskit version;
    that would silently misorder circuits that are asymmetric under qubit
    reversal (verified directly: no permutation needed here, one is
    required for Qiskit — the two frameworks are genuinely different).

    NOT DIFFERENTIABLE: from_pennylane() bakes every gate parameter into a
    plain Python float inside the QASM text, so it leaves the JAX trace.
    jax.grad through this function does not raise — it silently returns
    0.0 (verified), which reads as "converged" rather than "not wired up".
    For a real gradient through a Dense-Evolution circuit, use the
    dashboard_core._vqe_energy_fn pattern instead (jax.value_and_grad over
    a jax.lax.scan template with sentinel-injected parameters)."""
    circ = from_pennylane(circuit, *args, **kwargs)
    if sim is None:
        sim = DenseSVSimulator(n_qubits=circ.n_qubits, use_float32=use_float32)
    sim.run_circuit(circ.to_tuples())
    probs = np.asarray(sim.get_probabilities())
    return sim, probs
