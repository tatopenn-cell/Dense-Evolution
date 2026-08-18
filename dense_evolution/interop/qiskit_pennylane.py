"""Qiskit / PennyLane interop bridges (`from_qiskit`, `from_pennylane`,
`run_qiskit_circuit`, `run_pennylane_circuit`), a real-calibration noise
bridge (`noise_model_from_qiskit_backend`), and a Clifford-only STIM
bridge (`to_stim`). All circuit bridges go through OpenQASM 2.0
(`qiskit.qasm2.dumps` / `qml.to_openqasm`) and this library's own
`QASMParser` -- not bespoke gate-by-gate translators, so gate coverage
matches whatever the parser/simulator already support.

**Bit-order.** Qiskit indexes probability/statevector arrays
little-endian (qubit 0 = least significant bit); Dense-Evolution indexes
MSB-first everywhere (`phys = n_qubits - 1 - qubit`, the same convention
`apply_gate_1q`/`apply_gate_2q`/`measure`/`run_circuit_jit` use).
`run_qiskit_circuit` reorders its output into Qiskit's own convention
(`_to_qiskit_bit_order`, a plain bit-reversal permutation, verified
against `Statevector.from_instruction(...).probabilities()` on an
asymmetric circuit) so it's directly comparable to
`Statevector(circuit).probabilities()`. PennyLane's own wire convention
already matches Dense-Evolution's MSB-first indexing natively --
`run_pennylane_circuit` does **not** reorder, on purpose; verified
directly that the two frameworks genuinely need different treatment
here, not just "symmetric for simplicity."

**Known limits**, inherited from the QASM2 bridge (not something this
layer works around):

- No classical control flow -- `if`/`while` and mid-circuit-measurement-
  conditioned gates are parsed out, not executed (same limitation as
  native QASM3 circuits).
- No expansion of composite/custom gates. A Qiskit call like `mcx` with
  3+ controls gets exported as a named `gate mcx { ... }` definition;
  the definition parses cleanly but the gate itself isn't a primitive
  this simulator knows how to execute, so a call to it is a silent
  no-op -- same as referencing any unrecognized gate name elsewhere.
  Stick to the gates this simulator actually implements for results you
  can trust.
- Only a plugin/backend-free bridge for circuit *execution* -- no
  `qiskit.providers.BackendV2` or PennyLane `Device` registration, so
  callers still invoke `run_qiskit_circuit`/`run_pennylane_circuit`
  explicitly rather than pointing existing framework code at a new
  backend/device string. (`noise_model_from_qiskit_backend` reads a
  `BackendV2`'s calibration data -- that's data extraction only, not
  backend/provider registration.)
- **`run_qiskit_circuit`/`run_pennylane_circuit` are not differentiable.**
  `from_pennylane`/`from_qiskit` materialize every gate parameter into a
  plain Python `float` inside the QASM text before parsing -- the value
  leaves the JAX trace entirely. `jax.grad` through `run_pennylane_circuit`
  does **not** raise: it silently returns `0.0`, which looks like
  "already converged" rather than "not wired up" (verified directly).
  For a real gradient, use `circuit_to_energy_fn` instead -- pass it the
  `QASMCircuit` that `from_qiskit`/`from_pennylane` returns, before that
  float-baking happens, and `jax.grad` works correctly.
"""
import sys
import warnings
from typing import Optional, Tuple
import numpy as np

from ..circuits.parser import QASMParser, QASMCircuit
from ..backends.statevector import DenseSVSimulator

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

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False

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


def _require_stim():
    if not HAS_STIM:
        raise ImportError(
            "STIM interop requires the 'stim' package. "
            "Install it with: pip install dense-evolution[stim]")


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


_STIM_GATE_MAP = {
    'h': 'H', 'x': 'X', 'y': 'Y', 'z': 'Z',
    's': 'S', 'sdg': 'S_DAG', 'sx': 'SQRT_X', 'id': 'I',
    'cx': 'CX', 'cy': 'CY', 'cz': 'CZ',
}


def to_stim(ops, n_qubits: int) -> 'stim.Circuit':
    """Convert a Dense-Evolution op-list circuit (e.g. [['h', 0], ['cx', 0, 1]])
    into a stim.Circuit, for cross-validation against STIM's own stabilizer
    simulator/decoder tooling. Promoted from Dense-Evolution-Discovery's
    Steane-code STIM bridge script (scripts/steane_code_block4_stim_translation.py),
    generalized from that script's Steane-specific gate set to every
    STIM-representable gate this library has.

    STIM is a STABILIZER simulator: it can only represent Clifford
    operations. Every gate below maps 1:1 onto a native STIM instruction
    (h/x/y/z/s/sdg/sx/id/cx/cy/cz); anything else in the op list --
    continuous-angle rotations (rx/ry/rz/p/phase/u1/cp/cphase/crz) or the
    non-Clifford t/tdg gates -- raises ValueError rather than being dropped
    or approximated, since a silently-wrong stabilizer circuit defeats the
    purpose of using STIM as an independent cross-check in the first place.

    A leading no-op 'I' on qubit n_qubits - 1 is emitted first so the
    returned circuit always has exactly n_qubits qubits, even if the
    highest-indexed qubit is never otherwise touched by `ops`.

    Qubit indexing matches `ops` directly (STIM has no reordering of its
    own, unlike Qiskit's little-endian convention handled by
    run_qiskit_circuit) -- but STIM's state_vector()/TableauSimulator
    output is still little-endian (qubit 0 = least significant bit),
    the same convention _to_qiskit_bit_order reorders into, so comparing
    against DenseSVSimulator's own MSB-first probabilities/statevector
    still needs that same reordering."""
    _require_stim()
    circuit = stim.Circuit()
    if n_qubits > 0:
        circuit.append('I', [n_qubits - 1])
    for op in ops:
        gate_name = op[0]
        stim_gate = _STIM_GATE_MAP.get(gate_name)
        if stim_gate is None:
            raise ValueError(
                f"Gate '{gate_name}' is not representable in STIM -- STIM is "
                "a stabilizer simulator and only supports Clifford "
                "operations (h, x, y, z, s, sdg, sx, id, cx, cy, cz here), "
                "not continuous-angle rotations (rx/ry/rz/p/phase/u1/cp/"
                "cphase/crz) or non-Clifford gates (t/tdg).")
        circuit.append(stim_gate, list(op[1:]))
    return circuit
