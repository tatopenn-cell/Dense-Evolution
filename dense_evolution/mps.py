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


def _jsd_vectors_jax(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Same Jensen-Shannon Distance as _jsd_vectors, but returns a jnp
    scalar instead of a Python float -- the float() cast in _jsd_vectors
    forces concretization, which is fine called eagerly but raises
    ConcretizationTypeError under jax.vmap/jax.jit tracing (needed by
    _vectorized_chi_search below). Same math, no behavior difference for
    eager callers, who go through _jsd_vectors instead."""
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
    # p.shape[0] is a static Python int (shape, not a value) even for a
    # traced array under vmap, so this branch is safe at trace time.
    dim_factor = float(np.log10(p.shape[0]) / 2.0) if p.shape[0] > 1 else 1.0
    return jnp.sqrt(jnp.clip(js * dim_factor, 0.0, 1.0))


def _jsd_vectors(p: jnp.ndarray, q: jnp.ndarray) -> float:
    """Adaptive Jensen-Shannon Distance used to size the truncated bond
    dimension: scales with log10(dim) so the same JSD budget stays
    meaningful whether the local Hilbert space is small or large."""
    return float(_jsd_vectors_jax(p, q))


def _vectorized_chi_search_jax(S: jnp.ndarray, eps: float, jsd_budget: float, max_bond: int):
    """JIT/vmap-compatible core of _vectorized_chi_search -- no
    float()/int()/bool() concretizing casts, returns (chi_new, jsd_val) as
    traced jnp scalars instead of Python int/float, so it's usable from
    inside a jax.lax.scan/jax.jit region (needed by _build_mps_runner's
    step function). Same algorithm _vectorized_chi_search wraps and casts
    to Python types for eager callers -- see that docstring for the
    verification this replicates the real while loop exactly."""
    n = S.shape[0]
    max_possible = min(n, max_bond)
    mask_above_eps = S > eps
    initial_chi = jnp.clip(jnp.sum(mask_above_eps), 1, max_possible)

    norm_full = jnp.sum(S ** 2) + 1e-15
    p_full = (S ** 2) / norm_full

    candidates = jnp.arange(1, max_possible + 1)
    idx = jnp.arange(n)
    trunc_mask = idx[None, :] < candidates[:, None]
    p_trunc_all = jnp.where(trunc_mask, p_full[None, :], 0.0)

    jsd_all = jax.vmap(lambda pt: _jsd_vectors_jax(p_full, pt))(p_trunc_all)

    valid_candidate = candidates >= initial_chi
    satisfies = (jsd_all <= jsd_budget) & valid_candidate
    any_satisfies = jnp.any(satisfies)
    first_idx = jnp.argmax(satisfies)
    chi_new = jnp.where(any_satisfies, candidates[first_idx], max_possible)
    jsd_val = jnp.where(any_satisfies, jsd_all[first_idx], jsd_all[max_possible - 1])
    return chi_new, jsd_val


def _vectorized_chi_search(S: jnp.ndarray, eps: float, jsd_budget: float, max_bond: int):
    """JIT/vmap-compatible replacement for the Python `while` loop in
    _svd_truncate (host-syncs every iteration today). Computes the JSD for
    every candidate truncation size in one vectorized pass instead of
    incrementing chi_new one at a time, then picks the smallest candidate
    >= the eps-cutoff-based starting point that satisfies jsd_budget --
    exactly replicating the while loop's search direction and stopping
    condition (verified: 0 mismatches in chi_new/jsd_val against the real
    while loop across 171 real _svd_truncate calls from actual entangled
    circuits at 3 different (n_qubits, max_bond, jsd_budget) configurations,
    including the budget-violation fallback path).

    Does NOT assume JSD is monotonic in the candidate size -- it restricts
    the search to candidates >= the same starting point the while loop
    starts from and takes the first (smallest) one satisfying the budget,
    same as the loop would find by incrementing, whether or not JSD happens
    to be monotonic there.

    Returns (chi_new, jsd_val) as Python int/float -- eager convenience
    wrapper around _vectorized_chi_search_jax, for callers outside a jit
    region (e.g. these same tests).
    """
    chi_new, jsd_val = _vectorized_chi_search_jax(S, eps, jsd_budget, max_bond)
    return int(chi_new), float(jsd_val)


def _expand_nonlocal_2q_positions(q1: int, q2: int):
    """Pure-Python prediction of the exact adjacent-pair call sequence
    MPSSimulator._apply_nonlocal_2q's runtime SWAP-chain dispatch produces
    for a given (q1, q2) -- moves that decision from runtime (inside the
    eager gate-application loop) to pre-compile time, so it can be used to
    build a flat op list before entering a JIT region, mirroring how
    chunk.py's _compile_multi_chunk_ops does all gate-name branching
    outside the traced kernel.

    Verified exhaustively against the real runtime dispatch: 0 mismatches
    across all (q1, q2) pairs for n_qubits in {6, 10, 15} (330 pairs).

    Returns (seq, gate_index, needs_transpose):
      seq            -- list of (a, a+1) adjacent physical-position pairs,
                         in the exact order _apply_nonlocal_2q would call
                         apply_gate_2q on them.
      gate_index      -- which position in `seq` is the REAL gate (every
                         other position is a SWAP).
      needs_transpose -- True iff the original q1 > q2, matching
                         _apply_nonlocal_2q's own q1>q2 normalization
                         (the gate tensor must be transposed via axes
                         (1, 0, 3, 2) before use at `seq[gate_index]`,
                         same as _apply_nonlocal_2q already does).
    """
    needs_transpose = q1 > q2
    target, far = (q2, q1) if needs_transpose else (q1, q2)
    seq = []
    for q in range(far - 1, target, -1):
        seq.append((q, q + 1))
    gate_index = len(seq)
    seq.append((target, target + 1))
    for q in range(target + 1, far):
        seq.append((q, q + 1))
    return seq, gate_index, needs_transpose


# ── gate-ID -> matrix construction, jax.lax.switch-based (JIT-fusion) ────
# Same gate-ID table as dense_evolution/gates.py::GATE_IDS and
# compiler.py::_apply_gate_fast_step's g_1q switch -- same formulas,
# verified to match exactly (0 error, both eagerly and under jax.vmap/jit)
# before being committed here. Unlike compiler.py's statevector-bitmask
# do_2q, MPS's own gate application is an explicit tensor contraction
# (jnp.einsum in apply_gate_2q), so the 2-qubit switch below returns the
# full (2,2,2,2) unitary tensor directly -- same matrices mps.py's own
# apply_cx/apply_cz/apply_swap already build by hand, generalized to a
# traced g_id instead of one hardcoded gate per method. id=23 (SWAP) is
# reserved-but-unused in gates.py's own table (comment there: "never
# dispatched here, QuantumTranspiler always decomposes swap into 3xCX
# first") -- MPS's own kernel claims it for real, since decomposing SWAP
# into 3 CX would mean 3 SVD truncations per swap instead of 1, defeating
# the point of keeping SWAP as its own gate for the non-adjacent-gate
# chain (see _expand_nonlocal_2q_positions above).

def _mps_1q_matrix(g_id: jnp.ndarray, param: jnp.ndarray, dtype) -> jnp.ndarray:
    """Traced gate-ID -> (2,2) unitary, ids 0-13 (I,H,X,Y,Z,S,Sdg,T,Tdg,
    Rx,Ry,Rz,P,SX) -- same formulas as compiler.py's g_1q switch."""
    inv2 = jnp.asarray(1.0 / jnp.sqrt(2.0), dtype=dtype)
    half_p = param * 0.5
    cos_p = jnp.cos(half_p).astype(dtype)
    sin_p = jnp.sin(half_p).astype(dtype)
    exp_pos = jnp.exp(1j * param).astype(dtype)
    exp_ph4 = jnp.exp(1j * jnp.pi / 4.0).astype(dtype)
    exp_mh4 = jnp.exp(-1j * jnp.pi / 4.0).astype(dtype)
    safe_gid = jnp.clip(g_id, 0, 13)
    return jax.lax.switch(
        safe_gid,
        [
            lambda _: jnp.eye(2, dtype=dtype),                                          # 0  I
            lambda _: jnp.array([[inv2, inv2], [inv2, -inv2]], dtype=dtype),             # 1  H
            lambda _: jnp.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=dtype),  # 2  X
            lambda _: jnp.array([[0.0 + 0j, -1j], [1j, 0.0 + 0j]], dtype=dtype),         # 3  Y
            lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], dtype=dtype),  # 4  Z
            lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1j]], dtype=dtype),    # 5  S
            lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1j]], dtype=dtype),   # 6  Sdg
            lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_ph4]], dtype=dtype),  # 7  T
            lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_mh4]], dtype=dtype),  # 8  Tdg
            lambda _: jnp.array([[cos_p, -1j * sin_p], [-1j * sin_p, cos_p]], dtype=dtype),  # 9  Rx
            lambda _: jnp.array([[cos_p, -sin_p], [sin_p, cos_p]], dtype=dtype),         # 10 Ry
            lambda _: jnp.array([[jnp.exp(-1j * half_p), 0.0 + 0j],
                                  [0.0 + 0j, jnp.exp(1j * half_p)]], dtype=dtype),        # 11 Rz
            lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_pos]], dtype=dtype),  # 12 P
            lambda _: jnp.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=dtype),  # 13 SX
        ],
        operand=None,
    )


def _mps_2q_matrix(g_id: jnp.ndarray, param: jnp.ndarray, dtype) -> jnp.ndarray:
    """Traced gate-ID -> (2,2,2,2) unitary tensor, ids 20-25
    (CX,CZ,CP,SWAP,CY,CRZ) -- same target gates as compiler.py's do_2q,
    built as explicit tensors (not the statevector-bitmask approach) to
    match MPS's own einsum-based 2-qubit gate application."""
    exp_pos = jnp.exp(1j * param).astype(dtype)
    exp_neg_half = jnp.exp(-1j * param * 0.5).astype(dtype)
    exp_pos_half = jnp.exp(1j * param * 0.5).astype(dtype)
    one = jnp.asarray(1.0 + 0j, dtype=dtype)
    cp_diag = jnp.stack([one, one, one, exp_pos])
    crz_diag = jnp.stack([one, one, exp_neg_half, exp_pos_half])
    safe_idx = jnp.clip(g_id - 20, 0, 5)
    mat = jax.lax.switch(
        safe_idx,
        [
            lambda _: jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=dtype),  # 20 CX
            lambda _: jnp.diag(jnp.array([1, 1, 1, -1], dtype=dtype)),                    # 21 CZ
            lambda _: jnp.diag(cp_diag),                                                   # 22 CP
            lambda _: jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype),  # 23 SWAP
            lambda _: jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=dtype),  # 24 CY
            lambda _: jnp.diag(crz_diag),                                                  # 25 CRZ
        ],
        operand=None,
    )
    return mat.reshape(2, 2, 2, 2)


def _pad_gamma(g: jnp.ndarray, max_bond: int) -> jnp.ndarray:
    """Zero-pads a (chiL, 2, chiR) gamma tensor up to (max_bond, 2,
    max_bond) -- the fixed shape run_circuit_jit's scan carry needs.
    Mathematically transparent: every contraction used here (einsum) sums
    over the padded axis, and zero entries there contribute exactly zero,
    verified directly (max singular-value error at machine epsilon,
    final-state fidelity 1.0 to 1e-13, against the real eager simulator
    across multiple (n_qubits, max_bond, jsd_budget) configurations
    including budget-violation cases)."""
    chi_l, d, chi_r = g.shape
    out = jnp.zeros((max_bond, d, max_bond), dtype=g.dtype)
    return out.at[:chi_l, :, :chi_r].set(g)


def _pad_lambda(lam: jnp.ndarray, max_bond: int) -> jnp.ndarray:
    """Zero-pads a (chi,) lambda vector up to (max_bond,)."""
    n = lam.shape[0]
    out = jnp.zeros((max_bond,), dtype=lam.dtype)
    return out.at[:n].set(lam)


def _compile_mps_ops(ops, n_qubits: int) -> List[List[float]]:
    """Pre-compile-time (pure Python, outside any jit region) translation
    of a circuit -- list of (name, *args) tuples/lists, same convention as
    DenseSVSimulator.run_circuit_jit_beast_mode -- into a flat list of
    [g_id, q1, q2, param, transpose_flag] rows for run_circuit_jit's
    jax.lax.scan kernel. Two things happen here, deliberately outside the
    jit region (same principle as chunk.py's _compile_multi_chunk_ops --
    all Python-level branching on gate identity happens before tracing
    starts, never inside the traced step function):

      1. GATE_IDS (dense_evolution/gates.py) name -> numeric id lookup.
      2. Non-adjacent 2-qubit gates, and adjacent ones with q1 > q2,
         expanded/normalized via _expand_nonlocal_2q_positions (verified
         above) into their SWAP-chain-equivalent sequence of
         adjacent-ascending rows -- moves what _apply_nonlocal_2q's
         runtime dispatch does today to pre-compile time.

    id 23 (SWAP) is used for real here (not decomposed into 3xCX like
    QuantumTranspiler.transpile would) -- see _mps_2q_matrix's docstring
    for why: decomposing would mean 3 SVD truncations per swap instead
    of 1, defeating the point of the chain in the first place.
    """
    from .compiler import QuantumTranspiler
    from .gates import GATE_IDS

    # CCX has no entry in GATE_IDS (it's not a 1- or 2-qubit gate) -- reuse
    # the same Barenco H/CX/T/Tdg decomposition compiler.py/chunk.py's own
    # QuantumTranspiler already uses, rather than inventing a second one.
    # 'swap' is deliberately NOT pre-decomposed here (see id 23 note below).
    expanded_ops = []
    for cmd in ops:
        if str(cmd[0]).lower() == 'ccx':
            expanded_ops.extend(QuantumTranspiler.decompose_toffoli(*cmd[1:4]))
        else:
            expanded_ops.append(cmd)

    rows: List[List[float]] = []
    for cmd in expanded_ops:
        name = str(cmd[0]).lower()
        if name not in GATE_IDS:
            raise ValueError(
                f"unknown gate '{cmd[0]}' -- not in GATE_IDS. run_circuit_jit "
                f"does not silently drop unrecognized gates (same policy as "
                f"run_circuit_jit_beast_mode, issue #4)."
            )
        g_id = float(GATE_IDS[name])

        if g_id < 20:
            q1 = int(cmd[1])
            param = float(cmd[2]) if len(cmd) > 2 else 0.0
            rows.append([g_id, float(q1), float(q1), param, 0.0])
            continue

        q1 = int(cmd[1])
        q2 = int(cmd[2])
        param = float(cmd[3]) if len(cmd) > 3 else 0.0

        if abs(q1 - q2) == 1:
            lo, hi = (q1, q2) if q1 < q2 else (q2, q1)
            needs_transpose = q1 > q2
            rows.append([g_id, float(lo), float(hi), param, 1.0 if needs_transpose else 0.0])
        else:
            seq, gate_index, needs_transpose = _expand_nonlocal_2q_positions(q1, q2)
            for i, (a, b) in enumerate(seq):
                if i == gate_index:
                    rows.append([g_id, float(a), float(b), param, 1.0 if needs_transpose else 0.0])
                else:
                    rows.append([23.0, float(a), float(b), 0.0, 0.0])  # SWAP

    return rows


def _build_mps_runner(n_qubits: int, max_bond: int, eps: float, jsd_budget: float):
    """Factory: builds and returns a single @jax.jit-compiled closure that
    runs an entire pre-compiled MPS circuit via jax.lax.scan -- same
    factory pattern as chunk.py's _build_multi_chunk_runner, closing over
    the static n_qubits/max_bond/eps/jsd_budget (fixed for one
    MPSSimulator instance's whole lifetime, so the returned closure is
    built once and cached on the instance, never rebuilt per call)."""

    def step(carry, row):
        gammas, lambdas = carry
        dtype = gammas.dtype

        g_id = row[0].astype(jnp.int32)
        q1 = row[1].astype(jnp.int32)
        q2 = row[2].astype(jnp.int32)
        param = row[3]
        transpose_flag = row[4] > 0.5

        def branch_1q(c):
            gammas_, lambdas_ = c
            gate_1q = _mps_1q_matrix(g_id, param, dtype)
            new_g = jnp.einsum('ij,ljr->lir', gate_1q, gammas_[q1])
            new_carry = (gammas_.at[q1].set(new_g), lambdas_)
            # Placeholder diagnostics for a 1-qubit step -- run_circuit_jit
            # filters these out by g_id (>=20) before using the per-step
            # history, same as _bond_history/jsd_per_bond only ever
            # growing on 2-qubit gates in the eager path today.
            diag = (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0, dtype=jnp.float64),
                    jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(0.0, dtype=jnp.float64))
            return new_carry, diag

        def branch_2q(c):
            # Traced unconditionally only when this branch is taken --
            # jax.lax.cond (not jnp.where) skips the untaken branch's
            # compute entirely, same choice compiler.py's do_2q makes for
            # its own outer 1q/2q split: the SVD here is real work, worth
            # skipping for the common case of a 1-qubit gate.
            gammas_, lambdas_ = c
            gate_2q = _mps_2q_matrix(g_id, param, dtype)
            gate_2q = jnp.where(transpose_flag, jnp.transpose(gate_2q, (1, 0, 3, 2)), gate_2q)

            g1 = gammas_[q1]
            g2 = gammas_[q2]
            lam = lambdas_[q2]
            theta = jnp.einsum('lik,k,kjr->lijr', g1, lam, g2)
            theta_new = jnp.einsum('abcd,ecdf->eabf', gate_2q, theta)
            theta_mat = theta_new.reshape(max_bond * 2, 2 * max_bond)

            U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)
            chi_new, jsd_val = _vectorized_chi_search_jax(S, eps, jsd_budget, max_bond)

            col_mask = jnp.arange(max_bond) < chi_new
            U_masked = jnp.where(col_mask[None, :], U[:, :max_bond], 0.0)
            S_fixed = jnp.where(col_mask, S[:max_bond], 0.0)
            Vh_masked = jnp.where(col_mask[:, None], Vh[:max_bond, :], 0.0)

            new_g1 = U_masked.reshape(max_bond, 2, max_bond)
            new_g2 = Vh_masked.reshape(max_bond, 2, max_bond)

            new_gammas = gammas_.at[q1].set(new_g1).at[q2].set(new_g2)
            new_lambdas = lambdas_.at[q2].set(S_fixed)

            # Same formulas as the eager _apply_nonlocal_2q/apply_gate_2q
            # path (trunc_err = sqrt(sum(S[chi_new:]**2)), entropy from
            # the truncated distribution), rewritten vectorized/static-
            # shape (jnp.where masking instead of Python len()/list
            # filtering) -- same idiom _jsd_vectors_jax already uses.
            trunc_err = jnp.sqrt(jnp.sum(jnp.where(jnp.arange(2 * max_bond) >= chi_new, S ** 2, 0.0)))
            p_dist = S_fixed ** 2 / (jnp.sum(S_fixed ** 2) + 1e-20)
            ee = -jnp.sum(jnp.where(p_dist > 1e-20, p_dist * jnp.log2(jnp.where(p_dist > 1e-20, p_dist, 1.0)), 0.0))

            new_carry = (new_gammas, new_lambdas)
            diag = (chi_new.astype(jnp.int32), jsd_val.astype(jnp.float64),
                    trunc_err.astype(jnp.float64), ee.astype(jnp.float64))
            return new_carry, diag

        is_2q = g_id >= 20
        new_carry, diag = jax.lax.cond(is_2q, branch_2q, branch_1q, carry)
        return new_carry, diag

    @jax.jit
    def run(gammas, lambdas, compiled_ops):
        (final_gammas, final_lambdas), diag = jax.lax.scan(step, (gammas, lambdas), compiled_ops)
        return final_gammas, final_lambdas, diag

    return run


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

        # Cached compiled closure for run_circuit_jit -- built lazily on
        # first use (self.n/self.chi/self.eps/self.jsd_budget are fixed
        # for this instance's lifetime), never rebuilt per call. Same
        # caching pattern as Chunk.__init__'s self._multi_chunk_runner.
        self._mps_runner = None

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
        """Non-adjacent 2-qubit gate via a SWAP chain down to adjacent.

        The SWAP chain always ends up applying the gate at (target, target+1)
        with target = min(q1, q2) first -- so without normalizing here, a
        caller passing q1 > q2 (e.g. apply_cx(ctrl=3, tgt=1)) would silently
        have its control/target roles swapped for asymmetric gates like CNOT.
        Same fix as apply_gate_2q's adjacent-qubit branch just above, applied
        before the swap chain runs so it always sees q1 < q2."""
        if q1 > q2:
            q1, q2 = q2, q1
            gate_2q = jnp.transpose(gate_2q, (1, 0, 3, 2))
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
        # Index [0] rather than .squeeze(axis=0)/.squeeze(axis=-1): the
        # boundary axis has size 1 in the eager (unpadded) representation
        # (squeeze and [0]-indexing are identical there), but size
        # max_bond after run_circuit_jit's fixed-shape padding -- the real
        # boundary content is always at index 0 by construction either
        # way (see _pad_gamma), so [0]-indexing is the strict
        # generalization that works for both, not a behavior change for
        # the pre-existing eager path.
        result = self.gammas[0][0, :, :]
        for i in range(1, self.n):
            lam = self.lambdas[i]
            g = self.gammas[i]
            lg = jnp.einsum("k,kir->kir", lam, g)
            result = jnp.tensordot(result, lg, axes=([-1], [0]))
            result = result.reshape(-1, result.shape[-1])
        sv = result[..., 0]
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

    # ── JIT-fused whole-circuit execution ─────────────────────────────
    def run_circuit_jit(self, ops: List) -> None:
        """Runs an entire circuit through a single jax.lax.scan-fused,
        @jax.jit-compiled kernel instead of one eager Python call per gate
        -- the eager path (apply_gate_1q/apply_gate_2q/_apply_nonlocal_2q,
        all still available and unchanged) has zero @jax.jit anywhere and
        pays a host-device sync on every 2-qubit gate's bond-dimension
        search; measured 88.9s vs Qiskit Aer's 0.64s on a 60-qubit stress
        circuit -- see README changelog for the real before/after number
        this method produces on that same circuit.

        Trade-off, explicit and intentional (not hidden): every gamma/
        lambda is kept at a fixed max_bond-padded size for the rest of
        this instance's lifetime after this call. Structurally correct
        either way (zero-padding is mathematically transparent to every
        other method here -- contract_to_statevector, get_top_k_probable_
        states, etc. all still work correctly on the padded arrays,
        verified), just not memory-minimal for genuinely low-entanglement
        circuits, which is this module's whole point for very large qubit
        counts. Use the eager methods directly instead when memory, not
        speed, is the priority -- this is an addition, not a replacement.

        ops: same convention as DenseSVSimulator.run_circuit_jit_beast_mode
        -- list of (name, *args) tuples/lists. Unlike that method, SWAP is
        never decomposed into 3xCX (kept as one real gate, see
        _compile_mps_ops's docstring for why that matters here).
        """
        compiled_rows = _compile_mps_ops(ops, self.n)
        dtype = self.gammas[0].dtype
        lambda_dtype = self.lambdas[0].dtype

        if compiled_rows:
            ops_array = jnp.array(compiled_rows, dtype=jnp.float64)
        else:
            ops_array = jnp.zeros((0, 5), dtype=jnp.float64)

        if self._mps_runner is None:
            self._mps_runner = _build_mps_runner(self.n, self.chi, self.eps, self.jsd_budget)

        gammas_padded = jnp.stack([_pad_gamma(g, self.chi).astype(dtype) for g in self.gammas])
        lambdas_padded = jnp.stack([_pad_lambda(l, self.chi).astype(lambda_dtype) for l in self.lambdas])

        final_gammas, final_lambdas, diag = self._mps_runner(gammas_padded, lambdas_padded, ops_array)

        self.gammas = [final_gammas[i] for i in range(self.n)]
        self.lambdas = [final_lambdas[i] for i in range(self.n + 1)]

        # Bookkeeping parity: jax.lax.scan's stacked per-step diagnostics
        # (diag) replace the eager path's Python list.append()s inside the
        # loop -- same final content, populated differently. Only 2-qubit
        # steps (g_id >= 20, includes SWAP -- the eager _apply_nonlocal_2q
        # path routes its SWAPs through apply_gate_2q too, so its history
        # lists grow on those as well, not just the "real" gate) count.
        if compiled_rows:
            chi_history, jsd_history, trunc_err_history, entropy_history = (
                np.asarray(diag[0]), np.asarray(diag[1]), np.asarray(diag[2]), np.asarray(diag[3]))
            g_ids = np.asarray([row[0] for row in compiled_rows])
            q1_ids = np.asarray([int(row[1]) for row in compiled_rows])
            is_2q_mask = g_ids >= 20
            for i in np.nonzero(is_2q_mask)[0]:
                chi_new = int(chi_history[i])
                jsd_val = float(jsd_history[i])
                self._bond_history.append(chi_new)
                self.jsd_per_bond.append(jsd_val)
                self.truncation_errors.append(float(trunc_err_history[i]))
                q1 = q1_ids[i]
                if q1 < len(self.entanglement_entropy):
                    self.entanglement_entropy[q1] = float(entropy_history[i])
                if jsd_val > self.jsd_budget:
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
