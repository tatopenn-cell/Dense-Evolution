"""Matrix product state simulator in NumPy only.

The state is stored in the Gamma/lambda form of Vidal (arXiv:quant-ph/0301063,
section 2, equations 4-5 on page 2), and every two-qubit gate update is the
one of equations 21-27 on page 3. The module is standalone: it imports only
``heapq``, ``math``, ``warnings`` and ``numpy`` -- no ``jax``, no ``scipy``,
no ``dense_evolution``. The point is running the same algorithm inside a
browser (Pyodide), where JAX cannot be installed.

References:
    Vidal, G. (2003). Efficient classical simulation of slightly entangled
        quantum computations. arXiv:quant-ph/0301063. Equation 22 (page 3)
        defines the two-qubit tensor Theta; equations 23-27 (page 3) recover
        the new Gamma tensors and the new lambda from its SVD. Page 3 states
        that a two-qubit gate between non-adjacent qubits C and D is applied
        by "O(n) swap gates between adjacent qubits to bring C and D
        together".
"""

from __future__ import annotations

import heapq
import math
import warnings

import numpy as np


def _expi(angle: float) -> complex:
    """``exp(1j * angle)`` with the standard library only."""
    return complex(math.cos(angle), math.sin(angle))


_SQRT2_INV = 1.0 / math.sqrt(2.0)
_EXP_PI4 = _expi(math.pi / 4.0)
_EXP_M_PI4 = _expi(-math.pi / 4.0)

_1Q_STATIC = ("id", "i", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx")
_1Q_PARAM = ("rx", "ry", "rz", "p", "u1", "phase")
_2Q_STATIC = ("cx", "cy", "cz", "swap")
_2Q_PARAM = ("cp", "crz")


def gate_1q(name: str, angle: float | None = None) -> np.ndarray:
    """Return the (2, 2) complex128 matrix of a one-qubit gate.

    Args:
        name: one of ``id``, ``h``, ``x``, ``y``, ``z``, ``s``, ``sdg``,
            ``t``, ``tdg``, ``sx``, ``rx``, ``ry``, ``rz``, ``p`` (with
            its aliases ``u1`` and ``phase``), case-insensitive.
        angle: rotation angle, required for ``rx``, ``ry``, ``rz`` and
            ``p``/``u1``/``phase``, ignored for the others.

    Returns:
        The (2, 2) unitary of the gate, dtype ``complex128``.

    Raises:
        ValueError: if ``name`` is not one of the gates above, or if a
            rotation gate is called without an ``angle``.

    Examples:
        >>> import numpy as np
        >>> from dense_evolution.backends.mps_numpy import gate_1q
        >>> bool(np.allclose(gate_1q("h") @ gate_1q("h"), np.eye(2)))
        True
        >>> bool(np.allclose(gate_1q("x"), np.array([[0, 1], [1, 0]])))
        True
    """
    key = name.lower()
    if key in ("id", "i"):
        return np.eye(2, dtype=np.complex128)
    if key == "h":
        return _SQRT2_INV * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    if key == "x":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if key == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if key == "z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    if key == "s":
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    if key == "sdg":
        return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    if key == "t":
        return np.array([[1, 0], [0, _EXP_PI4]], dtype=np.complex128)
    if key == "tdg":
        return np.array([[1, 0], [0, _EXP_M_PI4]], dtype=np.complex128)
    if key == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)
    if key not in _1Q_PARAM:
        raise ValueError(f"gate_1q: unknown gate {name!r}")
    if angle is None:
        raise ValueError(f"gate_1q: gate {name!r} needs an angle")
    theta = float(angle)
    if key == "rx":
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if key == "ry":
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if key == "rz":
        return np.array(
            [[_expi(-theta / 2.0), 0], [0, _expi(theta / 2.0)]],
            dtype=np.complex128,
        )
    return np.array([[1, 0], [0, _expi(theta)]], dtype=np.complex128)


def gate_2q(name: str, angle: float | None = None) -> np.ndarray:
    """Return the (4, 4) complex128 matrix of a two-qubit gate.

    The row/column order is the tensor product ``q1 ⊗ q2`` with the first
    qubit as the most significant index, matching the convention of
    ``_mps_2q_matrix`` in ``mps.py``.

    Args:
        name: one of ``cx``, ``cy``, ``cz``, ``swap``, ``cp``, ``crz``,
            case-insensitive.
        angle: rotation angle, required for ``cp`` and ``crz``, ignored
            for the others.

    Returns:
        The (4, 4) unitary of the gate, dtype ``complex128``.

    Raises:
        ValueError: if ``name`` is not one of the gates above, or if
            ``cp``/``crz`` is called without an ``angle``.
    """
    key = name.lower()
    if key == "cx":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
        )
    if key == "cy":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=np.complex128
        )
    if key == "cz":
        return np.diag(np.array([1, 1, 1, -1], dtype=np.complex128))
    if key == "swap":
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
        )
    if key not in _2Q_PARAM:
        raise ValueError(f"gate_2q: unknown gate {name!r}")
    if angle is None:
        raise ValueError(f"gate_2q: gate {name!r} needs an angle")
    theta = float(angle)
    if key == "cp":
        return np.diag(np.array([1, 1, 1, _expi(theta)], dtype=np.complex128))
    return np.diag(np.array([1, 1, _expi(-theta / 2.0), _expi(theta / 2.0)], dtype=np.complex128))


def jsd_vectors(p: np.ndarray, q: np.ndarray) -> float:
    """Adaptive Jensen-Shannon distance used to size the truncated bond.

    The distance is scaled with ``log10(dim) / 2`` so the same JSD budget
    keeps a comparable meaning whether the local Hilbert space is small or
    large. The formula is the one of ``_jsd_vectors_jax`` in ``mps.py``,
    evaluated with NumPy.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    eps = 1e-12
    pn = p / (p.sum() + eps)
    qn = q / (q.sum() + eps)
    m = 0.5 * (pn + qn)
    mask = pn > eps
    safe = np.where(mask, np.log2(np.where(mask, pn, 1.0) / (m + eps)), 0.0)
    js = 0.5 * float(np.sum(np.where(mask, pn * safe, 0.0)))
    mask = qn > eps
    safe = np.where(mask, np.log2(np.where(mask, qn, 1.0) / (m + eps)), 0.0)
    js += 0.5 * float(np.sum(np.where(mask, qn * safe, 0.0)))
    dim = p.shape[0]
    factor = math.log10(dim) / 2.0 if dim > 1 else 1.0
    return math.sqrt(max(0.0, min(1.0, js * factor)))


def chi_search(S: np.ndarray, eps: float, jsd_budget: float, max_bond: int) -> tuple[int, float]:
    """Search the smallest bond dimension that meets the JSD budget.

    The candidates run from ``1`` to ``min(len(S), max_bond)``, the
    starting point is the count of singular values above ``eps`` clipped
    into that range, and the tail of the normalized spectrum below
    ``1e-12`` is zeroed before the JSD is computed (this is the fix of
    ``_vectorized_chi_search_jax`` in ``mps.py`` for the eager-vs-jit
    disagreement on negligible tails). When no candidate satisfies the
    budget, the largest possible ``chi`` is returned.

    Args:
        S: singular values from ``np.linalg.svd``, sorted descending.
        eps: cutoff on ``S`` that sets the starting candidate.
        jsd_budget: maximum admissible JSD.
        max_bond: largest admissible ``chi``.

    Returns:
        A pair ``(chi, jsd)`` with ``chi`` a Python ``int`` and ``jsd``
        a Python ``float``.
    """
    S = np.asarray(S, dtype=np.float64)
    n = int(S.shape[0])
    max_possible = min(n, int(max_bond))
    initial_chi = int(np.clip(int((S > eps).sum()), 1, max_possible))
    norm_full = float(np.sum(S**2)) + 1e-15
    p_full = (S**2) / norm_full
    p_full = np.where(p_full > 1e-12, p_full, 0.0)
    candidates = np.arange(1, max_possible + 1)
    idx = np.arange(n)
    trunc_mask = idx[None, :] < candidates[:, None]
    p_trunc = np.where(trunc_mask, p_full[None, :], 0.0)
    jsd_all = np.array([jsd_vectors(p_full, pt) for pt in p_trunc])
    valid = candidates >= initial_chi
    ok = (jsd_all <= jsd_budget) & valid
    if bool(ok.any()):
        first = int(np.argmax(ok))
        return int(candidates[first]), float(jsd_all[first])
    return int(max_possible), float(jsd_all[max_possible - 1])


class MPSSimulatorNumpy:
    """Matrix product state simulator with NumPy only.

    The state is stored in the canonical Gamma/lambda form of Vidal
    (arXiv:quant-ph/0301063, equations 4-5 on page 2): a list of ``n``
    three-index tensors ``Gamma[l][a, s, b]`` and ``n + 1`` vectors
    ``lambda[l]`` of Schmidt coefficients. ``gamma[l]`` has shape
    ``(chi_l, 2, chi_r)``; the two boundary lambdas are scalars.

    Every two-qubit gate update is equations 21-27 on page 3 of the same
    paper: build the tensor ``Theta`` from the middle and outer lambdas,
    apply the gate, compute the SVD, keep the leading ``chi`` Schmidt
    coefficients, renormalize and rebuild the two gammas. A two-qubit gate
    between non-adjacent qubits C and D is applied through ``O(n)`` swap
    gates between adjacent qubits (page 3 of the same paper).

    Args:
        n_qubits: number of qubits.
        max_bond: largest admissible bond dimension.
        svd_cutoff: smallest singular value that counts as nonzero; when
            ``None``, ``1e-12`` is used.
        jsd_budget: maximum admissible JSD per truncation.
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond: int = 64,
        svd_cutoff: float | None = None,
        jsd_budget: float = 1e-5,
    ) -> None:
        self.n = int(n_qubits)
        self.chi = int(max_bond)
        self.eps = float(svd_cutoff) if svd_cutoff is not None else 1e-12
        self.jsd_budget = float(jsd_budget)
        self.gammas: list[np.ndarray] = []
        for _ in range(self.n):
            g: np.ndarray = np.zeros((1, 2, 1), dtype=np.complex128)
            g[0, 0, 0] = 1.0
            self.gammas.append(g)
        self.lambdas: list[np.ndarray] = [
            np.ones(1, dtype=np.complex128) for _ in range(self.n + 1)
        ]
        self.truncation_errors: list[float] = []
        self.jsd_per_bond: list[float] = []
        self.budget_violations: int = 0

    def apply_gate_1q(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a (2, 2) unitary to ``qubit`` in place (Vidal Eq. 4)."""
        g = np.asarray(gate, dtype=np.complex128)
        self.gammas[qubit] = np.einsum("ij,ljr->lir", g, self.gammas[qubit])

    def apply_gate_2q(self, gate_2q: np.ndarray, q1: int, q2: int) -> None:
        """Apply a (4, 4) unitary to (q1, q2) with adaptive truncation.

        Follows Vidal equations 21-27 on page 3: ``Theta`` is built from
        the two outer lambdas and the middle one (equation 22), the gate
        is contracted in, the SVD of the reshaped matrix gives the new
        ``lambda`` and the two new gammas. Non-adjacent qubits go through
        a chain of swap gates, as page 3 prescribes.
        """
        g = np.asarray(gate_2q, dtype=np.complex128).reshape(2, 2, 2, 2)
        if abs(q1 - q2) != 1:
            self._apply_nonlocal_2q(g, q1, q2)
            return
        if q1 > q2:
            q1, q2 = q2, q1
            g = np.transpose(g, (1, 0, 3, 2))
        lam_l = self.lambdas[q1]
        g1 = self.gammas[q1]
        lam_m = self.lambdas[q2]
        g2 = self.gammas[q2]
        lam_r = self.lambdas[q2 + 1]

        theta = np.einsum("l,lik,k,kjr,r->lijr", lam_l, g1, lam_m, g2, lam_r)
        chi_l, d1, d2, chi_r = theta.shape
        theta_new = np.einsum("abcd,ecdf->eabf", g, theta)
        theta_mat = theta_new.reshape(chi_l * d1, d2 * chi_r)

        u_t, s_t, vh_t, trunc_err, jsd_val = self._svd_truncate(theta_mat)
        chi_new = len(s_t)

        lam_l_inv = np.where(np.abs(lam_l) > self.eps, 1.0 / lam_l, 0.0)
        lam_r_inv = np.where(np.abs(lam_r) > self.eps, 1.0 / lam_r, 0.0)

        new_g1 = np.einsum("l,lir->lir", lam_l_inv, u_t.reshape(chi_l, d1, chi_new))
        new_g2 = np.einsum("ljr,r->ljr", vh_t.reshape(chi_new, d2, chi_r), lam_r_inv)

        self.lambdas[q2] = s_t
        self.gammas[q1] = new_g1
        self.gammas[q2] = new_g2

        self.jsd_per_bond.append(float(jsd_val))

    def _apply_nonlocal_2q(self, gate_2q: np.ndarray, q1: int, q2: int) -> None:
        """Swap chain of Vidal page 3 to bring q2 next to q1."""
        if q1 > q2:
            q1, q2 = q2, q1
            gate_2q = np.transpose(gate_2q, (1, 0, 3, 2))
        swap = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
        ).reshape(2, 2, 2, 2)
        target = min(q1, q2)
        for q in range(max(q1, q2) - 1, target, -1):
            self.apply_gate_2q(swap, q, q + 1)
        self.apply_gate_2q(gate_2q, target, target + 1)
        for q in range(target + 1, max(q1, q2)):
            self.apply_gate_2q(swap, q, q + 1)

    def apply_cx(self, ctrl: int, tgt: int) -> None:
        self.apply_gate_2q(gate_2q("cx"), ctrl, tgt)

    def apply_cy(self, ctrl: int, tgt: int) -> None:
        self.apply_gate_2q(gate_2q("cy"), ctrl, tgt)

    def apply_cz(self, ctrl: int, tgt: int) -> None:
        self.apply_gate_2q(gate_2q("cz"), ctrl, tgt)

    def apply_swap(self, q1: int, q2: int) -> None:
        self.apply_gate_2q(gate_2q("swap"), q1, q2)

    def apply_cp(self, ctrl: int, tgt: int, angle: float) -> None:
        self.apply_gate_2q(gate_2q("cp", angle), ctrl, tgt)

    def apply_crz(self, ctrl: int, tgt: int, angle: float) -> None:
        self.apply_gate_2q(gate_2q("crz", angle), ctrl, tgt)

    def apply_ccx(self, c1: int, c2: int, tgt: int) -> None:
        """Toffoli via the standard T-gate decomposition (all 1q/2q gates)."""
        h = gate_1q("h")
        t = gate_1q("t")
        tdg = gate_1q("tdg")
        self.apply_gate_1q(h, tgt)
        self.apply_cx(c2, tgt)
        self.apply_gate_1q(tdg, tgt)
        self.apply_cx(c1, tgt)
        self.apply_gate_1q(t, tgt)
        self.apply_cx(c2, tgt)
        self.apply_gate_1q(tdg, tgt)
        self.apply_cx(c1, tgt)
        self.apply_gate_1q(t, c2)
        self.apply_gate_1q(t, tgt)
        self.apply_gate_1q(h, tgt)
        self.apply_cx(c1, c2)
        self.apply_gate_1q(t, c1)
        self.apply_gate_1q(tdg, c2)
        self.apply_cx(c1, c2)

    def _svd_truncate(self, theta_mat: np.ndarray):
        """SVD and adaptive truncation of Theta (Vidal, Eq. 22, page 3).

        ``Theta`` is the matrix in equation 22 of the paper; its singular
        values are the Schmidt coefficients at this cut (equations 23-27
        on page 3). The kept singular values are renormalized to unit
        norm, so the whole chain stays normalized without a global
        renormalization later.
        """
        u_t, s_t, vh_t = np.linalg.svd(theta_mat, full_matrices=False)
        chi_new, jsd_val = chi_search(s_t, self.eps, self.jsd_budget, self.chi)
        if jsd_val > self.jsd_budget:
            if self.budget_violations == 0:
                warnings.warn(
                    f"MPSSimulatorNumpy: bond dimension capped at "
                    f"max_bond={self.chi}, jsd_budget={self.jsd_budget:.1e} "
                    f"not honored (jsd={jsd_val:.2e}) -- results may be "
                    f"unreliable, consider raising max_bond.",
                    UserWarning,
                    stacklevel=2,
                )
            self.budget_violations += 1

        norm_full = math.sqrt(float(np.sum(s_t**2)) + 1e-30)
        s_norm = s_t / (norm_full + 1e-30)
        tail = s_norm[chi_new:] if len(s_t) > chi_new else np.zeros(0)
        trunc_err = math.sqrt(float(np.sum(tail**2)))
        self.truncation_errors.append(trunc_err)

        s_kept = s_t[:chi_new]
        kept_norm = math.sqrt(float(np.sum(s_kept**2)) + 1e-30)
        s_kept_renorm = s_kept / (kept_norm + 1e-30)

        return u_t[:, :chi_new], s_kept_renorm, vh_t[:chi_new, :], trunc_err, jsd_val

    def run_circuit(self, ops) -> None:
        """Apply a list of gate tuples.

        The tuples are the format of ``gate_tuples_to_qasm`` in
        ``tools/dashboard/core/qasm_library.py``: ``("h", q)``,
        ``("rx", q, angle)``, ``("cx", q1, q2)``, ``("cp", q1, q2, angle)``,
        ``("ccx", c1, c2, tgt)``.

        Raises:
            ValueError: for a gate name not in the list of gates of
                ``gate_1q`` and ``gate_2q`` plus ``ccx``.
        """
        for op in ops:
            name = op[0].lower()
            if name in _1Q_STATIC:
                self.apply_gate_1q(gate_1q(name), int(op[1]))
            elif name in _1Q_PARAM:
                self.apply_gate_1q(gate_1q(name, float(op[2])), int(op[1]))
            elif name in _2Q_STATIC:
                self.apply_gate_2q(gate_2q(name), int(op[1]), int(op[2]))
            elif name in _2Q_PARAM:
                self.apply_gate_2q(gate_2q(name, float(op[3])), int(op[1]), int(op[2]))
            elif name == "ccx":
                self.apply_ccx(int(op[1]), int(op[2]), int(op[3]))
            else:
                raise ValueError(f"run_circuit: unknown gate {op[0]!r}")

    def contract_to_statevector(self) -> np.ndarray:
        """Contract the MPS into a dense statevector (``n <= 24``).

        Raises:
            MemoryError: if ``n > 24``, where a dense statevector would
                need about ``2**n * 16`` bytes.
        """
        if self.n > 24:
            raise MemoryError(
                f"contract_to_statevector at {self.n} qubits would need "
                f"~{2 ** self.n * 16 / 1e9:.1f} GB -- use "
                f"get_top_k_probable_states instead for n > 24."
            )
        result = self.gammas[0][0, :, :]
        for i in range(1, self.n):
            lam = self.lambdas[i]
            lg = np.einsum("k,kir->kir", lam, self.gammas[i])
            result = np.tensordot(result, lg, axes=([-1], [0]))
            result = result.reshape(-1, result.shape[-1])
        sv = result[..., 0]
        norm = float(np.linalg.norm(sv))
        return sv / (norm + 1e-15)

    def get_top_k_probable_states(self, k: int = 128):
        """Greedy beam search for the top-``k`` most probable basis states.

        Args:
            k: beam width.

        Returns:
            A pair ``(indices, probabilities)``: ``indices`` is a list of
            Python ``int`` (a NumPy ``int64`` does not hold 100 qubits),
            ``probabilities`` a NumPy array, both sorted by descending
            probability. The probabilities are exact for the states found.
        """
        paths: list[tuple[int, np.ndarray]] = [(0, np.array([1.0 + 0.0j], dtype=np.complex128))]
        for i in range(self.n):
            candidates = []
            gamma = self.gammas[i]
            lam = (
                self.lambdas[i + 1]
                if (i + 1) < len(self.lambdas)
                else np.ones(gamma.shape[2], dtype=np.complex128)
            )
            for idx_p, vec_p in paths:
                for bit in (0, 1):
                    new_vec = np.einsum("l,lr->r", vec_p, gamma[:, bit, :]) * lam
                    weight = float(np.sum(np.abs(new_vec) ** 2))
                    candidates.append(((idx_p << 1) | bit, new_vec, weight))
            paths = [
                (idx, vec) for idx, vec, _ in heapq.nlargest(k, candidates, key=lambda c: c[2])
            ]

        indices = [int(p[0]) for p in paths]
        amplitudes = np.array(
            [complex(vec[0]) if len(vec) == 1 else complex(np.sum(vec)) for _, vec in paths],
            dtype=np.complex128,
        )
        probabilities = np.abs(amplitudes) ** 2
        order = np.argsort(-probabilities)
        return [indices[i] for i in order], probabilities[order]

    def bond_dimensions(self) -> list[int]:
        """List of the ``n + 1`` bond dimensions (the two ends are 1)."""
        if not self.gammas:
            return [1]
        return [int(g.shape[0]) for g in self.gammas] + [int(self.gammas[-1].shape[2])]

    def max_bond_used(self) -> int:
        return max(self.bond_dimensions())

    def total_truncation_error(self) -> float:
        if not self.truncation_errors:
            return 0.0
        return math.sqrt(sum(e * e for e in self.truncation_errors))

    def avg_jsd(self) -> float:
        if not self.jsd_per_bond:
            return 0.0
        return float(np.mean(self.jsd_per_bond))

    def memory_bytes(self) -> int:
        return int(sum(g.nbytes for g in self.gammas) + sum(lam.nbytes for lam in self.lambdas))

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)


_PAULI_MATS = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_expectation(mps: MPSSimulatorNumpy, pauli_string: str) -> complex:
    """Expectation value ``<psi|P|psi>`` of a Pauli string.

    The state is normalized by construction (every truncation renormalizes
    the kept singular values), so no division by ``<psi|psi>`` is needed.

    The string has one character per qubit (``I``, ``X``, ``Y``, ``Z``),
    with qubit 0 as the leftmost character, matching the convention of
    ``mps_pauli_expectation`` in ``mps.py``. The contraction is the
    transfer-matrix sweep of Vidal page 3: ``n`` steps, each a sum over
    the bond indices, ``O(n * chi**3)``.

    Args:
        mps: the simulator with the tensors.
        pauli_string: the Pauli string, e.g. ``"XIZ"``.

    Returns:
        The complex expectation value.

    Raises:
        ValueError: if ``pauli_string`` has a different length than
            ``mps.n``, or contains a character not in ``"IXYZ"``.
    """
    if len(pauli_string) != mps.n:
        raise ValueError(
            f"pauli_expectation: {len(pauli_string)} characters for " f"{mps.n} qubits"
        )
    v = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for i in range(mps.n):
        char = pauli_string[i].upper()
        if char not in _PAULI_MATS:
            raise ValueError(f"pauli_expectation: unknown Pauli {pauli_string[i]!r}")
        p = _PAULI_MATS[char]
        g = mps.gammas[i]
        e = np.einsum("asc,st,btd->abcd", np.conj(g), p, g)
        v = np.einsum("ab,abcd->cd", v, e)
        if i + 1 < mps.n:
            lam = mps.lambdas[i + 1]
            v = v * np.outer(np.conj(lam), lam)
    return complex(v[0, 0])
