"""
MPSSimulator - Matrix Product State statevector simulator, JAX-backed.

Ported from the "TurboQuant TUREQ MPSSimulator v8.2 MatryoshkaFlash"
prototype (private research notebook, never published as part of the
dense-evolution package). Two real bugs were found and fixed by
independent verification against DenseSVSimulator before this module
existed in its current form:

1. The original applied Lloyd-Max quantization to the SVD singular
   values on every truncation ("PolarQuantizer"). Measured a real ~0.5%
   Total Variation Distance error against DenseSVSimulator on an
   8-qubit entangling test circuit, with ZERO bond-dimension savings to
   show for it. Dropped entirely -- this module keeps only the plain
   adaptive SVD truncation (JSD-budget-driven bond dimension, the
   author's own stopping criterion -- standard SVD truncation,
   non-standard stopping metric).

2. get_top_k_probable_states (originally "_extract_top_k_paths") picked
   a single "best" bond index via argmax at each step instead of
   correctly summing over the bond dimension. Measured 0/8 correct
   states against the exact contraction on the same test circuit,
   values off by ~30x. Fixed by propagating the true partial-contraction
   vector through each bond (matches exactly, to machine precision, on
   every state it finds) -- but note it's a genuine greedy beam search,
   not an exact top-k finder: recall of the true top states grows with
   beam width k but isn't guaranteed complete for any fixed k.

Originally ported in plain numpy (matching the prototype), then
converted to jax.numpy so the core tensor contractions (einsum, SVD) run
on the same backend as the rest of dense_evolution instead of a second,
inconsistent numerics stack. Re-verified against DenseSVSimulator after
the conversion -- see test_mps.py.

Uses whatever jax_enable_x64 precision is currently active in the
process (does not toggle it itself) -- same convention as
DenseSVSimulator/Chunk, which rely on the caller (dashboard_core.py's
run_simulation) to set precision, since jax_enable_x64 is a process-wide
flag and toggling it locally would leak to unrelated code running later
in the same process.

For circuits with LOW entanglement (product states, GHZ/Bell-like chains,
shallow local circuits), the bond dimension stays small regardless of
qubit count, so this scales to hundreds of qubits where DenseSVSimulator
(or Chunk) cannot -- see get_probabilities_sampled and
get_top_k_probable_states, neither of which ever materializes a
(2**n,)-shaped array. For HIGHLY entangled circuits the bond dimension
grows and this degrades back toward the same exponential cost
DenseSVSimulator has -- it is not a universal replacement, it is
complementary.
"""

import warnings
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _jsd_vectors(p: jnp.ndarray, q: jnp.ndarray) -> float:
    """Adaptive Jensen-Shannon Distance used to size the truncated bond
    dimension: scales with log10(dim) so the same JSD budget stays
    meaningful whether the local Hilbert space is small or large."""
    eps = 1e-12
    p_norm = p / (jnp.sum(p) + eps)
    q_norm = q / (jnp.sum(q) + eps)
    m = 0.5 * (p_norm + q_norm)

    def _kl(a, b):
        mask = a > eps
        # jnp.where still traces both branches, but log(0) only ever
        # feeds into a term multiplied by 0 through the mask -- safe,
        # and avoids a Python-side branch on a traced value.
        safe_log = jnp.where(mask, jnp.log2(jnp.where(mask, a, 1.0) / (b + eps)), 0.0)
        return jnp.sum(jnp.where(mask, a * safe_log, 0.0))

    js = 0.5 * (_kl(p_norm, m) + _kl(q_norm, m))
    dim_factor = float(np.log10(len(p)) / 2.0) if len(p) > 1 else 1.0
    return float(jnp.sqrt(jnp.clip(js * dim_factor, 0.0, 1.0)))


class MPSSimulator:
    """
    Matrix Product State simulator with adaptive SVD-truncated bond
    dimension (JSD-budget driven), no lossy post-truncation quantization.
    JAX-backed core (einsum, SVD).

    Parameters
    ----------
    n_qubits   : int
    max_bond   : int   -- hard cap on bond dimension chi
    svd_cutoff : float -- singular values below this are dropped outright
    jsd_budget : float -- max tolerated Jensen-Shannon distance between the
                          full and truncated singular-value distributions
                          at each cut; chi is grown by 1 until satisfied
                          or max_bond is hit.
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond: int = 64,
        svd_cutoff: float = 1e-12,
        jsd_budget: float = 1e-5,
    ):
        self.n = n_qubits
        self.chi = max_bond
        self.eps = svd_cutoff
        self.jsd_budget = jsd_budget

        self.gammas: List[jnp.ndarray] = []
        self.lambdas: List[jnp.ndarray] = [jnp.ones(1)] * (n_qubits + 1)

        self.truncation_errors: List[float] = []
        self.jsd_per_bond: List[float] = []
        self.entanglement_entropy = np.zeros(max(n_qubits - 1, 0))
        self._bond_history: List[int] = []
        # Counts truncations where max_bond was hit before jsd_budget could
        # be satisfied -- the while loop below exits silently in that case,
        # and avg_JSD (a mean over all steps) can look deceptively low even
        # when the final contracted state is badly wrong (verified: TVD
        # ~0.97 against DenseSVSimulator on an 8-qubit/15-layer entangling
        # circuit with max_bond=2, while avg_JSD read 0.0534).
        self.budget_violations: int = 0

        for _ in range(n_qubits):
            g = jnp.zeros((1, 2, 1), dtype=jnp.complex64 if not jax.config.jax_enable_x64 else jnp.complex128)
            g = g.at[0, 0, 0].set(1.0)
            self.gammas.append(g)

    # gate 1q
    def apply_gate_1q(self, gate: jnp.ndarray, qubit: int) -> None:
        """O(chi^2) -- updates only Gamma[qubit]."""
        gate = jnp.asarray(gate)
        self.gammas[qubit] = jnp.einsum("ij,ljr->lir", gate, self.gammas[qubit])

    # core: plain adaptive SVD truncation (no quantization)
    def _svd_truncate(
        self, theta_mat: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float]:
        U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)

        mask = S > self.eps
        chi_new = max(1, min(int(jnp.sum(mask)), self.chi))
        max_possible = min(len(S), self.chi)

        norm_full = float(jnp.sum(S**2)) + 1e-15
        p_full = (S**2) / norm_full

        def _trunc_jsd(chi_try):
            p_trunc = jnp.zeros_like(p_full)
            p_trunc = p_trunc.at[:chi_try].set((S[:chi_try] ** 2) / norm_full)
            return _jsd_vectors(p_full, p_trunc)

        jsd_val = _trunc_jsd(chi_new)
        while jsd_val > self.jsd_budget and chi_new < max_possible:
            chi_new += 1
            jsd_val = _trunc_jsd(chi_new)

        if jsd_val > self.jsd_budget:
            # Loop exited because chi_new hit max_possible (== max_bond, in
            # the common case where the bond isn't already capped by the
            # SVD's own rank), not because jsd_budget was satisfied.
            if self.budget_violations == 0:
                warnings.warn(
                    f"MPSSimulator: bond dimension capped at max_bond={self.chi}, "
                    f"jsd_budget={self.jsd_budget:.1e} not honored "
                    f"(jsd={jsd_val:.2e}) -- results may be unreliable, "
                    f"consider raising max_bond.",
                    UserWarning,
                    stacklevel=2,
                )
            self.budget_violations += 1

        trunc_err = float(jnp.sqrt(jnp.sum(S[chi_new:] ** 2))) if len(S) > chi_new else 0.0
        self.truncation_errors.append(trunc_err)

        return U[:, :chi_new], S[:chi_new], Vh[:chi_new, :], trunc_err, jsd_val

    # gate 2q adjacent
    def apply_gate_2q(self, gate_2q: jnp.ndarray, q1: int, q2: int) -> None:
        """2-qubit gate with adaptive SVD truncation. O(chi^3)."""
        gate_2q = jnp.asarray(gate_2q)
        if abs(q1 - q2) != 1:
            self._apply_nonlocal_2q(gate_2q, q1, q2)
            return
        if q1 > q2:
            q1, q2 = q2, q1
            gate_2q = jnp.transpose(gate_2q, (1, 0, 3, 2))

        g1 = self.gammas[q1]
        g2 = self.gammas[q2]
        lam = self.lambdas[q2]

        theta = jnp.einsum("lik,k,kjr->lijr", g1, lam, g2)
        chiL, d1, d2, chiR = theta.shape

        theta_new = jnp.einsum("abcd,ecdf->eabf", gate_2q, theta)
        theta_mat = theta_new.reshape(chiL * d1, d2 * chiR)

        U_t, S_t, Vh_t, trunc_err, jsd_val = self._svd_truncate(theta_mat)
        chi_new = len(S_t)

        s_sq = S_t**2
        p_dist = s_sq / (jnp.sum(s_sq) + 1e-20)
        p_v = p_dist[p_dist > 1e-20]
        ee = float(-jnp.sum(p_v * jnp.log2(p_v))) if len(p_v) > 1 else 0.0
        if q1 < len(self.entanglement_entropy):
            self.entanglement_entropy[q1] = ee

        self.lambdas[q2] = S_t
        self.gammas[q1] = U_t.reshape(chiL, d1, chi_new)
        self.gammas[q2] = Vh_t.reshape(chi_new, d2, chiR)

        self._bond_history.append(chi_new)
        self.jsd_per_bond.append(jsd_val)

    def _apply_nonlocal_2q(self, gate_2q: jnp.ndarray, q1: int, q2: int) -> None:
        """Non-adjacent 2-qubit gate via a SWAP chain down to adjacent."""
        swap = jnp.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        ).reshape(2, 2, 2, 2)
        target = min(q1, q2)
        for q in range(max(q1, q2) - 1, target, -1):
            self.apply_gate_2q(swap, q, q + 1)
        self.apply_gate_2q(gate_2q, target, target + 1)
        for q in range(target + 1, max(q1, q2)):
            self.apply_gate_2q(swap, q, q + 1)

    # gate shortcuts
    def apply_cx(self, ctrl: int, tgt: int) -> None:
        cx = jnp.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        ).reshape(2, 2, 2, 2)
        self.apply_gate_2q(cx, ctrl, tgt)

    def apply_cz(self, ctrl: int, tgt: int) -> None:
        cz = jnp.diag(jnp.array([1, 1, 1, -1])).astype(complex).reshape(2, 2, 2, 2)
        self.apply_gate_2q(cz, ctrl, tgt)

    def apply_swap(self, q1: int, q2: int) -> None:
        sw = jnp.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        ).reshape(2, 2, 2, 2)
        self.apply_gate_2q(sw, q1, q2)

    def apply_ccx(self, c1: int, c2: int, tgt: int) -> None:
        """Toffoli via standard T-gate decomposition (all 1q/2q gates)."""
        inv2 = 1.0 / np.sqrt(2.0)
        h = inv2 * jnp.array([[1, 1], [1, -1]], dtype=complex)
        t = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=complex)
        tdg = jnp.array([[1, 0], [0, jnp.exp(-1j * jnp.pi / 4)]], dtype=complex)

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

    # contraction to a full statevector -- O(2**n), n <= ~24 only
    def contract_to_statevector(self) -> jnp.ndarray:
        if self.n > 24:
            raise MemoryError(
                f"contract_to_statevector at {self.n} qubits would need "
                f"~{2**self.n * 16 / 1e9:.1f} GB -- use "
                f"get_probabilities_sampled or get_top_k_probable_states "
                f"instead for n > 24."
            )
        result = self.gammas[0].squeeze(axis=0)
        for i in range(1, self.n):
            lam = self.lambdas[i]
            g = self.gammas[i]
            lg = jnp.einsum("k,kir->kir", lam, g)
            result = jnp.tensordot(result, lg, axes=([-1], [0]))
            result = result.reshape(-1, result.shape[-1])
        sv = result.squeeze(axis=-1)
        norm = jnp.linalg.norm(sv)
        return sv / (norm + 1e-15)

    # sequential sampling -- O(chi^2) per qubit per sample, never
    # materializes a (2**n,) array. The only way to get results for
    # n_qubits beyond ~24. Sampling decisions themselves stay on host
    # (np.random.Generator): each step needs a concrete probability to
    # branch on, not a traced value, so this loop isn't a jax.jit
    # candidate as a whole regardless of backend.
    def _sample_bitstring(self, rng: np.random.Generator) -> List[int]:
        bits = []
        state = jnp.ones(1, dtype=complex)
        for i in range(self.n):
            g = self.gammas[i]
            lam = self.lambdas[i + 1] if i < self.n - 1 else jnp.ones(g.shape[2])
            p0v = jnp.einsum("l,lr->r", state, g[:, 0, :])
            p1v = jnp.einsum("l,lr->r", state, g[:, 1, :])
            p0l = p0v * lam
            p1l = p1v * lam
            p0 = float(jnp.real(jnp.dot(p0l, jnp.conj(p0l))))
            p1 = float(jnp.real(jnp.dot(p1l, jnp.conj(p1l))))
            norm = p0 + p1 + 1e-15
            bit = 0 if rng.random() < p0 / norm else 1
            bits.append(bit)
            state = p0l if bit == 0 else p1l
            state = state / (jnp.linalg.norm(state) + 1e-15)
        return bits

    def get_probabilities_sampled(
        self, n_samples: int = 100_000, seed: Optional[int] = None
    ) -> dict:
        """Returns a {bitstring: empirical_probability} dict from n_samples
        sequential draws -- the only entry point safe for n_qubits > 24."""
        from collections import Counter

        rng = np.random.default_rng(seed)
        counts: Counter = Counter()
        for _ in range(n_samples):
            bits = self._sample_bitstring(rng)
            counts["".join(map(str, bits))] += 1
        return {bitstr: c / n_samples for bitstr, c in counts.items()}

    # ──────────────────────────────────────────────
    # approximate top-k extraction -- see module docstring point 2.
    # ──────────────────────────────────────────────
    def get_top_k_probable_states(self, k: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy beam search (beam width k) for approximately-most-probable
        basis states, without ever contracting to a full statevector.

        Returns (indices, probabilities): indices are computational-basis
        integers, probabilities are exact for the states found (not
        approximated), sorted descending. Recall of the TRUE top states
        improves with k but is not guaranteed for any fixed k -- see the
        module docstring."""
        paths: List[Tuple[int, jnp.ndarray]] = [(0, jnp.array([1.0 + 0.0j]))]
        for i in range(self.n):
            candidates = []
            gamma = self.gammas[i]
            lam = self.lambdas[i + 1] if (i + 1) < len(self.lambdas) else jnp.ones(gamma.shape[2])
            for idx_p, vec_p in paths:
                for bit in (0, 1):
                    new_vec = jnp.einsum("l,lr->r", vec_p, gamma[:, bit, :]) * lam
                    weight = float(jnp.sum(jnp.abs(new_vec) ** 2))
                    candidates.append(((idx_p << 1) | bit, new_vec, weight))
            candidates.sort(key=lambda c: c[2], reverse=True)
            paths = [(idx, vec) for idx, vec, _ in candidates[:k]]

        indices = np.array([p[0] for p in paths])
        amplitudes = np.array([
            complex(vec[0]) if len(vec) == 1 else complex(jnp.sum(vec))
            for _, vec in paths
        ])
        probabilities = np.abs(amplitudes) ** 2
        order = np.argsort(-probabilities)
        return indices[order], probabilities[order]

    # metrics
    def max_bond_used(self) -> int:
        return max(self._bond_history) if self._bond_history else 1

    def total_truncation_error(self) -> float:
        if not self.truncation_errors:
            return 0.0
        return float(np.sqrt(np.sum(np.array(self.truncation_errors) ** 2)))

    def avg_jsd(self) -> float:
        return float(np.mean(self.jsd_per_bond)) if self.jsd_per_bond else 0.0

    def memory_bytes(self) -> int:
        bytes_gammas = sum(g.size * g.dtype.itemsize for g in self.gammas)
        bytes_lambdas = sum(l.size * l.dtype.itemsize for l in self.lambdas)
        return int(bytes_gammas + bytes_lambdas)

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)

    def summary(self) -> str:
        ee_max = self.entanglement_entropy.max() if len(self.entanglement_entropy) else 0.0
        return (
            f"MPSSimulator | n={self.n} | chi_max={self.chi} | "
            f"chi_used={self.max_bond_used()} | mem={self.memory_mb():.3f}MB | "
            f"trunc_err={self.total_truncation_error():.2e} | "
            f"avg_JSD={self.avg_jsd():.4f} | EE_max={ee_max:.3f}b | "
            f"budget_violations={self.budget_violations}"
        )
