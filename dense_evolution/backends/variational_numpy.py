"""QAOA and VQE on the NumPy MPS simulator, in single or double precision.

Both algorithms run on ``mps_numpy.MPSSimulatorNumpy`` and read energies with its
``pauli_expectation``. The state and the gates are in ``complex64`` by default (the
single precision of the NumPy engine), or in ``complex128``; the parameters and the
Adam moments are in the matching real type. The few scalar terms of an energy are
summed in double precision. Nothing here imports JAX, so it runs where JAX does not:
in Pyodide, or on a small flight computer with NumPy only.

QAOA follows Farhi, Goldstone and Gutmann, "A Quantum Approximate Optimization
Algorithm", arXiv:1411.4028 (2014): the cost function ``C = sum_a C_a`` (Eq. 1), the
phase-separation unitary ``U(C, g) = exp(-i g C)`` (Eq. 2), the mixing operator
``B = sum_j X_j`` and ``U(B, b) = exp(-i b B)`` (Eqs. 3 and 4), the start state
``|s>``, the uniform superposition (Eq. 5), the state
``|g, b> = U(B, b_p) U(C, g_p) ... U(B, b_1) U(C, g_1) |s>`` (Eq. 6) and the objective
``F_p(g, b) = <g, b| C |g, b>`` (Eq. 7); for MaxCut ``C_jk = (1/2)(1 - Z_j Z_k)``
(Eq. 12). For a weighted edge the term is multiplied by its weight.

VQE uses the hardware-efficient ansatz of Kandala et al., Nature 549, 242 (2017), as
``tools/dashboard/core/vqe.py`` does: the Hartree-Fock occupation as X gates, then
``layers`` times a RY rotation on every qubit and a ladder of CNOT gates; and the same
Adam loop (step 0.1, moments 0.9 and 0.999, ``eps`` 1e-8, start parameters uniform in
[-0.1, 0.1] from ``numpy.random.default_rng(seed)``). The gradient is the
parameter-shift rule: for a gate ``exp(-i t G / 2)`` with ``G`` a Pauli operator,
``dF/dt = (F(t + pi/2) - F(t - pi/2)) / 2``.

The module needs NumPy and the sibling file ``mps_numpy``. Bit strings follow the engine:
qubit 0 is the leftmost character.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from . import mps_numpy as _mn
except ImportError:
    import mps_numpy as _mn

_REAL = {np.dtype(np.complex64): np.float32, np.dtype(np.complex128): np.float64}


def _real_type(dtype):
    key = np.dtype(dtype)
    if key not in _REAL:
        raise ValueError("dtype must be np.complex64 or np.complex128")
    return _REAL[key]


def pauli_string(n: int, paulis: dict[int, str]) -> str:
    """A Pauli string of ``n`` characters with ``paulis`` ({qubit: 'X'|'Y'|'Z'}) and ``I`` elsewhere."""
    return "".join(paulis.get(q, "I") for q in range(n))


def energy(ops, n: int, terms, dtype=np.complex64, max_bond: int = 64) -> float:
    """``sum_t c_t <P_t>`` for the state that ``ops`` prepares from ``|0...0>``.

    Args:
        ops: gate tuples (the format of ``MPSSimulatorNumpy.run_circuit``).
        n: the number of qubits.
        terms: a list of ``(coefficient, pauli_string)``.
        dtype: ``np.complex64`` or ``np.complex128``.
        max_bond: the cap on the bond dimension.

    Examples:
        >>> round(energy([("x", 0)], 1, [(1.0, "Z")], np.complex128), 6)
        -1.0
    """
    _real_type(dtype)
    sim = _mn.MPSSimulatorNumpy(n, max_bond=max_bond, dtype=dtype)
    sim.run_circuit(ops)
    return float(sum(float(c) * _mn.pauli_expectation(sim, p).real for c, p in terms))


def _adam(objective, gradient, theta0, maxiter, step_size, beta1, beta2, real):
    theta = np.asarray(theta0, dtype=real).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []
    for t in range(1, maxiter + 1):
        history.append(float(objective(theta)))
        grad = np.asarray(gradient(theta), dtype=real)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = (theta - step_size * m_hat / (np.sqrt(v_hat) + real(1e-8))).astype(real)
    history.append(float(objective(theta)))
    return theta, history


def hardware_efficient_ops(params, n: int, layers: int, occupation=()) -> list:
    """The ansatz of ``tools/dashboard/core/vqe.py`` as gate tuples (``n * layers`` parameters)."""
    if len(params) != n * layers:
        raise ValueError(f"expected {n * layers} parameters, got {len(params)}")
    ops: list = [("x", q) for q, occupied in enumerate(occupation) if occupied]
    k = 0
    for _ in range(layers):
        for q in range(n):
            ops.append(("ry", q, float(params[k])))
            k += 1
        ops.extend(("cx", q, q + 1) for q in range(n - 1))
    return ops


def vqe(
    terms,
    n: int,
    layers: int = 2,
    maxiter: int = 200,
    step_size: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    seed: int = 0,
    dtype=np.complex64,
    occupation=(),
    max_bond: int = 64,
) -> dict:
    """Minimise ``sum_t c_t <P_t>`` over the hardware-efficient ansatz with Adam.

    Returns:
        A dict with ``energy`` (the last), ``params``, ``history`` (the energy before each
        step and after the last), ``dtype`` and ``n_params``.
    """
    real = _real_type(dtype)
    n_params = n * layers

    def objective(theta):
        return energy(
            hardware_efficient_ops(theta, n, layers, occupation), n, terms, dtype, max_bond
        )

    def gradient(theta):
        grad = np.zeros(n_params, dtype=np.float64)
        for k in range(n_params):
            up, down = theta.astype(np.float64), theta.astype(np.float64)
            up[k] += math.pi / 2
            down[k] -= math.pi / 2
            grad[k] = 0.5 * (objective(up) - objective(down))
        return grad

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-0.1, 0.1, size=n_params)
    theta, history = _adam(objective, gradient, theta0, maxiter, step_size, beta1, beta2, real)
    return {
        "energy": history[-1],
        "params": theta,
        "history": history,
        "dtype": str(np.dtype(dtype)),
        "n_params": n_params,
    }


def maxcut_terms(n: int, edges) -> list:
    """The cost ``C = sum w (1 - Z_j Z_k) / 2`` of ``edges`` ``[(j, k, weight)]`` as Pauli terms."""
    terms = [(0.5 * sum(w for _, _, w in edges), "I" * n)]
    terms += [(-0.5 * w, pauli_string(n, {j: "Z", k: "Z"})) for j, k, w in edges]
    return terms


def exact_maxcut(n: int, edges) -> float:
    """The best cut by trying every assignment (``n`` up to about 24)."""
    bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
    cut = np.zeros(2**n)
    for j, k, w in edges:
        cut += w * (bits[:, j] != bits[:, k])
    return float(cut.max())


def qaoa_ops(n: int, edges, gammas, betas, shift: tuple | None = None) -> list:
    """The circuit of Eq. 6: ``H`` on every qubit, then for each layer the phase separation and the mixer.

    ``exp(-i g w (1 - Z_j Z_k) / 2)`` is, up to a global phase, ``exp(+i g w Z_j Z_k / 2)``:
    CX(j, k), RZ(-g w) on k, CX(j, k). ``exp(-i b X)`` is ``RX(2 b)``.

    ``shift = ("cost" | "mixer", layer, index, delta)`` adds ``delta`` to the angle of one
    gate, for the parameter-shift rule.
    """
    ops: list = [("h", q) for q in range(n)]
    for layer, (g, b) in enumerate(zip(gammas, betas)):
        for index, (j, k, w) in enumerate(edges):
            delta = shift[3] if shift and shift[:3] == ("cost", layer, index) else 0.0
            ops += [("cx", j, k), ("rz", k, -float(g) * w + delta), ("cx", j, k)]
        for q in range(n):
            delta = shift[3] if shift and shift[:3] == ("mixer", layer, q) else 0.0
            ops.append(("rx", q, 2.0 * float(b) + delta))
    return ops


def qaoa_expectation(n: int, edges, gammas, betas, dtype=np.complex64, max_bond: int = 64) -> float:
    """``F_p(gammas, betas) = <gammas, betas| C |gammas, betas>`` (Eq. 7) for the MaxCut cost."""
    return energy(qaoa_ops(n, edges, gammas, betas), n, maxcut_terms(n, edges), dtype, max_bond)


def qaoa_gradient(n: int, edges, gammas, betas, dtype=np.complex64, max_bond: int = 64):
    """The gradient of ``F_p`` by the parameter-shift rule, one pair of circuits for each gate."""
    terms = maxcut_terms(n, edges)

    def f(shift):
        ops = qaoa_ops(n, edges, gammas, betas, shift)
        return energy(ops, n, terms, dtype, max_bond)

    half_pi = math.pi / 2
    d_gamma = np.zeros(len(gammas))
    d_beta = np.zeros(len(betas))
    for layer in range(len(gammas)):
        for index, (_, _, w) in enumerate(edges):
            d_gamma[layer] += (
                -w
                * 0.5
                * (f(("cost", layer, index, half_pi)) - f(("cost", layer, index, -half_pi)))
            )
        for q in range(n):
            d_beta[layer] += (
                2.0 * 0.5 * (f(("mixer", layer, q, half_pi)) - f(("mixer", layer, q, -half_pi)))
            )
    return d_gamma, d_beta


def qaoa(
    n: int,
    edges,
    p: int = 1,
    maxiter: int = 100,
    step_size: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    seed: int = 0,
    dtype=np.complex64,
    max_bond: int = 64,
    k: int = 8,
) -> dict:
    """Maximise ``F_p`` for MaxCut with Adam, starting from angles drawn in the ranges of the paper.

    The paper takes ``gamma`` in [0, 2 pi] and ``beta`` in [0, pi]; the start is drawn there
    with ``numpy.random.default_rng(seed)``.

    Returns:
        A dict with ``expectation`` (the final ``F_p``), ``gammas``, ``betas``, ``history``
        (``F_p`` before each step and after the last) and ``top_states`` (the ``k`` most
        probable bit strings with their probabilities, qubit 0 leftmost).
    """
    real = _real_type(dtype)
    rng = np.random.default_rng(seed)
    start = np.concatenate([rng.uniform(0, 2 * math.pi, p), rng.uniform(0, math.pi, p)])

    def split(theta):
        theta = np.asarray(theta, dtype=np.float64)
        return theta[:p], theta[p:]

    def objective(theta):
        return -qaoa_expectation(n, edges, *split(theta), dtype=dtype, max_bond=max_bond)

    def gradient(theta):
        d_gamma, d_beta = qaoa_gradient(n, edges, *split(theta), dtype=dtype, max_bond=max_bond)
        return -np.concatenate([d_gamma, d_beta])

    theta, history = _adam(objective, gradient, start, maxiter, step_size, beta1, beta2, real)
    gammas, betas = split(theta)
    sim = _mn.MPSSimulatorNumpy(n, max_bond=max_bond, dtype=dtype)
    sim.run_circuit(qaoa_ops(n, edges, gammas, betas))
    idx, probs = sim.get_top_k_probable_states(k)
    return {
        "expectation": -history[-1],
        "gammas": gammas,
        "betas": betas,
        "history": [-h for h in history],
        "top_states": [(format(int(i), f"0{n}b"), float(q)) for i, q in zip(idx, probs)],
        "dtype": str(np.dtype(dtype)),
    }
