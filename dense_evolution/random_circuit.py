"""
Random circuit generation for benchmarking and fuzz-testing -- the same
purpose Qiskit's `random_circuit` serves.
"""
import numpy as np

__all__ = ['random_circuit']

_1Q_STATIC = ('h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx')
_1Q_PARAMETRIC = ('rx', 'ry', 'rz')
_2Q_STATIC = ('cx', 'cz', 'cy', 'swap')
_ALL_KNOWN = set(_1Q_STATIC) | set(_1Q_PARAMETRIC) | set(_2Q_STATIC)


def random_circuit(n_qubits, n_gates, seed=None, gate_set=None, two_qubit_prob=0.4):
    """
    Build a random circuit for benchmarking or fuzz-testing.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, must be >= 1.
    n_gates : int
        Number of gate operations to generate, must be >= 0.
    seed : int | numpy.random.Generator, optional
        Seed (or an existing Generator) for reproducible circuits.
    gate_set : iterable of str, optional
        Restrict generation to this set of gate names, mixing 1- and
        2-qubit gates freely (default spans both: h, x, y, z, s, sdg, t,
        tdg, sx, rx, ry, rz, cx, cz, cy, swap). Unrecognized names raise
        immediately rather than failing later inside run_circuit.
    two_qubit_prob : float
        Probability in [0, 1] of picking a 2-qubit gate at each step.
        Ignored (forced to single-qubit gates) once n_qubits < 2, or once
        gate_set excludes every 2-qubit gate name.

    Returns
    -------
    list[tuple]
    """
    if n_qubits < 1:
        raise ValueError(f"random_circuit needs at least 1 qubit, got {n_qubits}")
    if n_gates < 0:
        raise ValueError(f"n_gates must be >= 0, got {n_gates}")
    if not (0.0 <= two_qubit_prob <= 1.0):
        raise ValueError(f"two_qubit_prob must be in [0, 1], got {two_qubit_prob}")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    if gate_set is not None:
        gate_set = list(gate_set)
        unknown = set(gate_set) - _ALL_KNOWN
        if unknown:
            raise ValueError(f"unknown gate name(s) in gate_set: {sorted(unknown)}")
        one_q_static = [g for g in gate_set if g in _1Q_STATIC]
        one_q_param = [g for g in gate_set if g in _1Q_PARAMETRIC]
        two_q = [g for g in gate_set if g in _2Q_STATIC]
    else:
        one_q_static = list(_1Q_STATIC)
        one_q_param = list(_1Q_PARAMETRIC)
        two_q = list(_2Q_STATIC) if n_qubits >= 2 else []

    one_q_pool = one_q_static + one_q_param
    if not one_q_pool and not two_q:
        raise ValueError("gate_set leaves no usable gates for this n_qubits")

    ops = []
    for _ in range(n_gates):
        use_two_qubit = (
            bool(two_q) and n_qubits >= 2
            and (not one_q_pool or rng.random() < two_qubit_prob)
        )
        if use_two_qubit:
            name = two_q[rng.integers(len(two_q))]
            a, b = rng.choice(n_qubits, size=2, replace=False)
            ops.append((name, int(a), int(b)))
        else:
            name = one_q_pool[rng.integers(len(one_q_pool))]
            q = int(rng.integers(n_qubits))
            if name in _1Q_PARAMETRIC:
                theta = float(rng.uniform(0, 2 * np.pi))
                ops.append((name, q, theta))
            else:
                ops.append((name, q))
    return ops
