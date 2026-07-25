from typing import Callable, Optional, Tuple

from .parser import QASMCircuit
from .gates import GATE_IDS
from .compiler import QuantumTranspiler
from .registry import HAS_JAX, NoiseModel, NoiseSpec

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from .compiler import _compile_and_run_circuit_jit
else:
    jnp = None


#: gates that receive a value from theta — must match the n_params count
#: below exactly, or theta's allocation order desyncs from the template's
#: injection order. Kept identical to dashboard_core's own list (same
#: engine, one source of truth).
_PARAMETRIC_GATES = ('rx', 'ry', 'rz', 'u1', 'p', 'cp', 'crz')
_TWO_QUBIT_GATES = ('cx', 'cy', 'cz', 'cp', 'crz', 'swap')


def _require_jax():
    if not HAS_JAX:
        raise ImportError(
            "circuit_to_energy_fn requires JAX. "
            "Install it with: pip install dense-evolution[jax]")


def _build_template(circuit: QASMCircuit, n_qubits: int) -> "jnp.ndarray":
    """Builds the (n_ops, 4) float64 [g_id, q1, q2, sentinel] template that
    the energy function injects theta into (-1.0 in the param slot for
    gates whose value comes from theta, patched in via jnp.where inside a
    jax.lax.scan, never a Python float() — that would sever the JAX trace).

    Structural pass only: build (name, *qubits) tuples (no param values —
    QuantumTranspiler.transpile only inspects gate name/qubit-count, for
    ccx/swap decomposition), transpile once, then look up g_id per gate and
    mark parametric slots with the sentinel. ccx/toffoli decomposes into
    non-parametric gates only, so this never desyncs theta's order.

    circuit.ops qubits are always plain ints here (QASMCircuit is the
    package's own interchange type, produced by QASMParser.parse and by
    the Qiskit/PennyLane interop bridge alike) — no defensive unwrapping
    of framework-specific qubit/wire objects needed.
    """
    tuples = []
    for op in circuit.ops:
        name = str(op['name']).lower().strip()
        qubits = [int(q) for q in op.get('qubits', [])]
        if not qubits or any(q >= n_qubits for q in qubits):
            continue
        tuples.append((name, *qubits))

    target = QuantumTranspiler.transpile(tuples)

    rows = []
    for cmd in target:
        name = cmd[0].lower()
        if name not in GATE_IDS:
            continue
        g_id = float(GATE_IDS[name])
        qubits = cmd[1:]
        sentinel = -1.0 if name in _PARAMETRIC_GATES else 0.0
        if name in _TWO_QUBIT_GATES and len(qubits) >= 2:
            rows.append([g_id, float(qubits[0]), float(qubits[1]), sentinel])
        elif qubits:
            rows.append([g_id, float(qubits[0]), 0.0, sentinel])

    if not rows:
        return jnp.empty((0, 4), dtype=jnp.float64)
    return jnp.array(rows, dtype=jnp.float64)


def circuit_to_energy_fn(
    circuit: QASMCircuit, n_qubits: int
) -> Tuple[Callable, int]:
    """
    Convert a QASMCircuit into a JAX-differentiable energy function.

    circuit : QASMCircuit — from QASMParser.parse(qasm), or from the
              Qiskit/PennyLane interop bridge (from_qiskit/from_pennylane).

    Returns (energy_fn, n_params):
      energy_fn(theta, h_matrix, stato_zero=None, noise=None) ->
      (energy, statevector) is a pure JAX function, differentiable w.r.t.
      theta via jax.grad / jax.value_and_grad(energy_fn, argnums=0,
      has_aux=True). stato_zero defaults to |0...0> if not given.
      n_params is the number of parametric gates in the circuit, in the
      same order theta is injected — build theta as an array of that
      length.

      noise, when given, is a registry.NoiseSpec (a JAX PyTree) applied
      to the statevector right after the circuit and before the energy
      expectation value is computed — natively inside the same traced
      computation as theta, not as an external step the caller has to
      splice in around energy_fn themselves. Because NoiseSpec carries
      its own jax_key as a pytree leaf, the whole thing stays
      jit/grad/vmap-composable with no OS-entropy fallback and no
      external key-management workaround:

          noise = NoiseSpec(model='depolarizing', p=0.05,
                             jax_key=jax.random.PRNGKey(0))
          energy, sv = energy_fn(theta, h_matrix, noise=noise)

    This is the same engine dashboard_core.py's real VQE gradient uses
    internally (verified against finite differences, ~1e-11 agreement) —
    exposed here as public API so it's reachable without reading
    dashboard_core.py, and so circuits imported via from_qiskit/
    from_pennylane (which are NOT differentiable on their own — see
    run_pennylane_circuit's docstring) have a real way to become
    differentiable instead of just a documented dead end.
    """
    _require_jax()
    template = _build_template(circuit, n_qubits)
    n_params = sum(1 for op in circuit.ops
                   if str(op['name']).lower().strip() in _PARAMETRIC_GATES)

    def energy_fn(theta, h_matrix, stato_zero: Optional["jnp.ndarray"] = None,
                  noise: Optional["NoiseSpec"] = None):
        if stato_zero is None:
            stato_zero = jnp.zeros(2 ** n_qubits, dtype=jnp.complex128).at[0].set(1.0)

        if n_params == 0:
            # No parametric gates -> no sentinel (-1.0) rows in template, so
            # patch_and_apply below would never take its is_param branch.
            # Skip the scan entirely rather than index into an empty theta
            # array during tracing (n_params is a static Python int, fixed
            # at circuit_to_energy_fn() call time, so this branch is
            # resolved before any tracing happens — not a jax.lax.cond).
            sv = _compile_and_run_circuit_jit(stato_zero, template)
        else:
            def patch_and_apply(carry, op):
                idx = carry
                is_param = op[3] == -1.0
                final_p = jnp.where(is_param, theta[idx], op[3])
                next_idx = jnp.where(is_param, idx + jnp.int32(1), idx)
                return next_idx, jnp.array([op[0], op[1], op[2], final_p], dtype=jnp.float64)

            _, patched_ops = jax.lax.scan(patch_and_apply, jnp.int32(0), template)
            sv = _compile_and_run_circuit_jit(stato_zero, patched_ops)

        if noise is not None:
            sv = NoiseModel.apply_to_sv(
                sv, n_qubits, model=noise.model, p=noise.p,
                jax_key=noise.jax_key, qubits=list(noise.qubits) if noise.qubits else None,
            )

        energy = jnp.real(jnp.vdot(sv, h_matrix @ sv))
        return energy, sv

    return energy_fn, n_params
