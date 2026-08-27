from typing import List

import jax
import jax.numpy as jnp

from ._engine_imports import QuantumTranspiler, GATE_IDS

__all__ = [
    "_build_multi_chunk_step", "_build_multi_chunk_runner",
    "_build_distributed_chunk_step", "_build_distributed_chunk_runner",
    "_compile_multi_chunk_ops",
]

# ─────────────────────────────────────────────────────────────────────────────────
# Multi-chunk JIT kernel (num_chunks > 1)
# ─────────────────────────────────────────────────────────────────────────────────

def _build_multi_chunk_step(num_chunks: int, m: int, k: int):
    """
    Build a jax.lax.scan-compatible step function operating on the
    STACKED multi-chunk representation — a (num_chunks, chunk_dim)
    array — instead of a flat (2**n_qubits,) statevector.

    Modeled directly on compiler.py's _apply_gate_fast_step (same
    [g_id, q1, q2, param] encoding via GATE_IDS, same lax.switch for the
    1-qubit matrix, same per-gate controlled-U dispatch for the 5
    two-qubit gate types) — but a gate touching a "chunk-select" qubit
    (index < m, the top m logical qubits that select WHICH chunk) mixes
    whole (chunk_dim,)-shaped ROWS instead of individual amplitudes,
    while a gate touching a "local" qubit (index >= m) mixes individual
    elements WITHIN each row in parallel. This never materializes a
    (2**n_qubits,) array — Chunk's whole reason to exist — because the
    stacked shape (num_chunks, chunk_dim) holds exactly the same total
    elements as the num_chunks separate per-chunk arrays it replaces.

    The 6 gate/qubit-location combinations below are a direct
    translation of the pre-JIT _apply_gate_multi's Python-loop formulas
    (removed once this replaced it) — verified case-by-case against
    DenseSVSimulator on non-chunked reference circuits before being
    wired in, not derived fresh. All 6 are traced unconditionally every
    step and selected via jnp.where on the runtime q1/q2 vs static m
    comparison, same "trace every branch" pattern _apply_gate_fast_step
    already uses for is_1q/is_2q and the 5 two-qubit sub-gates.
    """
    chunk_dim = 1 << k

    def step(chunks, operation):
        g_id  = operation[0].astype(jnp.int32)
        q1    = operation[1].astype(jnp.int32)
        q2    = operation[2].astype(jnp.int32)
        param = operation[3]
        dtype = chunks.dtype  # never hardcode complex128 — see the
                               # use_float32 bug this exact mistake
                               # caused once already in beast-mode.

        inv2         = jnp.asarray(1.0 / jnp.sqrt(2.0), dtype=dtype)
        half_p       = param * jnp.float64(0.5)
        cos_p        = jnp.cos(half_p).astype(dtype)
        sin_p        = jnp.sin(half_p).astype(dtype)
        exp_pos      = jnp.exp(1j * param).astype(dtype)
        exp_ph4      = jnp.exp(1j * jnp.pi / 4.0).astype(dtype)
        exp_mh4      = jnp.exp(-1j * jnp.pi / 4.0).astype(dtype)
        exp_pos_half = jnp.exp(1j * half_p).astype(dtype)
        exp_neg_half = jnp.exp(-1j * half_p).astype(dtype)

        # 1-qubit gate matrix — identical table to _apply_gate_fast_step
        safe_gid = jnp.clip(g_id, 0, 14)
        g_1q = jax.lax.switch(
            safe_gid,
            [
                lambda _: jnp.eye(2, dtype=dtype),
                lambda _: jnp.array([[inv2, inv2], [inv2, -inv2]], dtype=dtype),
                lambda _: jnp.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[0.0 + 0j, -1j], [1j, 0.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_ph4]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_mh4]], dtype=dtype),
                lambda _: jnp.array([[cos_p, -1j * sin_p], [-1j * sin_p, cos_p]], dtype=dtype),
                lambda _: jnp.array([[cos_p, -sin_p], [sin_p, cos_p]], dtype=dtype),
                lambda _: jnp.array([[jnp.exp(-1j * half_p), 0.0 + 0j], [0.0 + 0j, jnp.exp(1j * half_p)]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_pos]], dtype=dtype),
                lambda _: jnp.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=dtype),
                # 14  GPhase(alpha) = e^{i*alpha} * I -- see compiler.py's
                # _apply_gate_fast_step index 14 for the derivation; this
                # table must stay in sync with that one (see comment above).
                lambda _: jnp.array([[exp_pos, 0.0 + 0j], [0.0 + 0j, exp_pos]], dtype=dtype),
            ],
            operand=None,
        )

        # Controlled-U submatrix for the 5 two-qubit gate types (mat[2:,2:]
        # of each gate's full 4x4 form — same values _apply_gate_multi's
        # `U = mat[2:, 2:]` extracted from GATES/PARAMETRIC_GATES).
        # 20=CX->X, 21=CZ->Z, 22=CP->P(theta), 24=CY->Y, 25=CRZ->RZ(theta).
        two_q_idx = jnp.where(g_id == 20, 0,
                    jnp.where(g_id == 21, 1,
                    jnp.where(g_id == 22, 2,
                    jnp.where(g_id == 24, 3, 4))))
        U = jax.lax.switch(
            two_q_idx,
            [
                lambda _: jnp.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_pos]], dtype=dtype),
                lambda _: jnp.array([[0.0 + 0j, -1j], [1j, 0.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[exp_neg_half, 0.0 + 0j], [0.0 + 0j, exp_pos_half]], dtype=dtype),
            ],
            operand=None,
        )
        g00, g01, g10, g11 = g_1q[0, 0], g_1q[0, 1], g_1q[1, 0], g_1q[1, 1]
        u00, u01, u10, u11 = U[0, 0], U[0, 1], U[1, 0], U[1, 1]

        # ── case 1: 1-qubit gate, LOCAL qubit (q1 >= m) ─────────────
        # Same do_1q amplitude-pair math as beast-mode, applied to every
        # chunk row in parallel (gather along axis 1, broadcasts over
        # axis 0 for free).
        def case_1q_local(_c):
            local_phys = (k - 1) - (q1 - m)
            stride     = jnp.int32(1) << local_phys
            idx        = jnp.arange(chunk_dim, dtype=jnp.int32)
            idx_pair   = idx ^ stride
            mask0      = (idx & stride) == 0
            amp_pair   = _c[:, idx_pair]
            new0 = g00 * _c + g01 * amp_pair
            new1 = g10 * amp_pair + g11 * _c
            return jnp.where(mask0[None, :], new0, new1)

        # ── case 2: 1-qubit gate, CHUNK-SELECT qubit (q1 < m) ────────
        # Same math, one level up: each "amplitude" is a whole
        # (chunk_dim,)-shaped row, mixed pairwise across axis 0.
        def case_1q_chunk(_c):
            stride    = jnp.int32(1) << (m - 1 - q1)
            idxc      = jnp.arange(num_chunks, dtype=jnp.int32)
            idxc_pair = idxc ^ stride
            mask0     = (idxc & stride) == 0
            amp_pair  = _c[idxc_pair]
            new0 = g00 * _c + g01 * amp_pair
            new1 = g10 * amp_pair + g11 * _c
            return jnp.where(mask0[:, None], new0, new1)

        # ── case 3: 2-qubit, ctrl AND tgt both LOCAL ─────────────
        def case_2q_local_local(_c):
            ctrl_phys = (k - 1) - (q1 - m)
            tgt_phys  = (k - 1) - (q2 - m)
            idx       = jnp.arange(chunk_dim, dtype=jnp.int32)
            ctrl_bit  = (idx & (jnp.int32(1) << ctrl_phys)) != 0
            tgt_bit   = (idx & (jnp.int32(1) << tgt_phys)) != 0
            partner   = idx ^ (jnp.int32(1) << tgt_phys)
            amp_partner = _c[:, partner]
            new0 = u00 * _c + u01 * amp_partner
            new1 = u10 * amp_partner + u11 * _c
            after = jnp.where(tgt_bit[None, :], new1, new0)
            return jnp.where(ctrl_bit[None, :], after, _c)

        # ── case 4: ctrl CHUNK-SELECT, tgt LOCAL ───────────────
        # Whole chunks where the chunk-index's ctrl bit is set get U
        # applied as a local 1-qubit gate; the rest are untouched.
        def case_2q_ctrl_chunk_tgt_local(_c):
            ctrl_stride = jnp.int32(1) << (m - 1 - q1)
            idxc        = jnp.arange(num_chunks, dtype=jnp.int32)
            ctrl_set    = (idxc & ctrl_stride) != 0
            tgt_phys    = (k - 1) - (q2 - m)
            idxl        = jnp.arange(chunk_dim, dtype=jnp.int32)
            tgt_bit     = (idxl & (jnp.int32(1) << tgt_phys)) != 0
            partner     = idxl ^ (jnp.int32(1) << tgt_phys)
            amp_partner = _c[:, partner]
            new0 = u00 * _c + u01 * amp_partner
            new1 = u10 * amp_partner + u11 * _c
            after = jnp.where(tgt_bit[None, :], new1, new0)
            return jnp.where(ctrl_set[:, None], after, _c)

        # ── case 5: ctrl LOCAL, tgt CHUNK-SELECT ─────────────
        # Pairs of chunks get mixed, but ONLY where the local ctrl bit
        # (same position in every chunk) is set — an elementwise mask.
        def case_2q_ctrl_local_tgt_chunk(_c):
            ctrl_phys  = (k - 1) - (q1 - m)
            idxl       = jnp.arange(chunk_dim, dtype=jnp.int32)
            ctrl_bit   = (idxl & (jnp.int32(1) << ctrl_phys)) != 0
            tgt_stride = jnp.int32(1) << (m - 1 - q2)
            idxc       = jnp.arange(num_chunks, dtype=jnp.int32)
            idxc_pair  = idxc ^ tgt_stride
            is_c0      = (idxc & tgt_stride) == 0
            amp_pair   = _c[idxc_pair]
            new_c0 = u00 * _c + u01 * amp_pair
            new_c1 = u10 * amp_pair + u11 * _c
            after = jnp.where(is_c0[:, None], new_c0, new_c1)
            return jnp.where(ctrl_bit[None, :], after, _c)

        # ── case 6: ctrl AND tgt both CHUNK-SELECT ───────────
        def case_2q_both_chunk(_c):
            ctrl_stride = jnp.int32(1) << (m - 1 - q1)
            tgt_stride  = jnp.int32(1) << (m - 1 - q2)
            idxc        = jnp.arange(num_chunks, dtype=jnp.int32)
            ctrl_set    = (idxc & ctrl_stride) != 0
            idxc_pair   = idxc ^ tgt_stride
            is_c0       = (idxc & tgt_stride) == 0
            amp_pair    = _c[idxc_pair]
            new_c0 = u00 * _c + u01 * amp_pair
            new_c1 = u10 * amp_pair + u11 * _c
            after = jnp.where(is_c0[:, None], new_c0, new_c1)
            return jnp.where(ctrl_set[:, None], after, _c)

        is_2q    = g_id >= 20
        q1_chunk = q1 < m
        q2_chunk = q2 < m

        result_2q = jnp.where(
            q1_chunk & q2_chunk, case_2q_both_chunk(chunks),
            jnp.where(q1_chunk & (~q2_chunk), case_2q_ctrl_chunk_tgt_local(chunks),
            jnp.where((~q1_chunk) & q2_chunk, case_2q_ctrl_local_tgt_chunk(chunks),
                                               case_2q_local_local(chunks))))
        result_1q = jnp.where(q1_chunk, case_1q_chunk(chunks), case_1q_local(chunks))

        new_chunks = jnp.where(is_2q, result_2q, result_1q)
        return new_chunks.astype(dtype), None

    return step


# ─────────────────────────────────────────────────────────
# Distributed (multi-device) variant — one physical chunk per device
# ─────────────────────────────────────────────────────────

def _build_distributed_chunk_step(num_chunks: int, m: int, k: int, axis_name: str):
    """Same 6-case formula set as _build_multi_chunk_step, but each
    device holds exactly ONE chunk row (chunk_dim,) instead of the
    whole (num_chunks, chunk_dim) stack living on one device/process —
    issue #1: distribute chunks across a device mesh (multi-GPU/
    multi-host), not just multi-chunk within one process's RAM.

    The stacked-array formulation's "mix pairs of rows across axis 0"
    (cases 2/5/6, touching a chunk-select qubit) becomes real
    point-to-point network communication here: jax.lax.ppermute, keyed
    on the fixed XOR-stride pairing between chunk indices — the
    textbook pairwise-exchange communication pattern used by
    distributed statevector simulators (each device sends its local
    row to its stride-partner device and receives the partner's row
    back). Cases 1/3/4 need NO communication at all: case 1/3 are
    purely local (both qubits live inside this device's own chunk_dim
    index space), and case 4 (ctrl chunk-select, tgt local) is a
    decision every device can make on its OWN chunk index alone
    (whether ITS id has the ctrl bit set) — no data from any other
    device is needed to decide or to apply the local tgt gate.

    ppermute's `perm` argument is a communication topology and must be
    STATIC (known at trace time) — it cannot be built from q1/q2,
    which are traced values read from the scanned circuit array. Since
    q1/q2 only ever range over the m possible chunk-select qubit
    indices [0, m), every possible stride is enumerated as its own
    statically-built ppermute call, and jax.lax.switch (traced
    unconditionally, same "trace every branch" pattern used
    throughout this codebase) picks the right one at runtime — plus
    one extra identity branch for "no chunk-select qubit involved,
    no communication needed" (verified below to be exactly cases
    1/3/4, never 2/5/6)."""
    chunk_dim = 1 << k

    def step(local_row, operation):
        g_id  = operation[0].astype(jnp.int32)
        q1    = operation[1].astype(jnp.int32)
        q2    = operation[2].astype(jnp.int32)
        param = operation[3]
        dtype = local_row.dtype

        my_id = jax.lax.axis_index(axis_name).astype(jnp.int32)

        inv2         = jnp.asarray(1.0 / jnp.sqrt(2.0), dtype=dtype)
        half_p       = param * jnp.float64(0.5)
        cos_p        = jnp.cos(half_p).astype(dtype)
        sin_p        = jnp.sin(half_p).astype(dtype)
        exp_pos      = jnp.exp(1j * param).astype(dtype)
        exp_ph4      = jnp.exp(1j * jnp.pi / 4.0).astype(dtype)
        exp_mh4      = jnp.exp(-1j * jnp.pi / 4.0).astype(dtype)
        exp_pos_half = jnp.exp(1j * half_p).astype(dtype)
        exp_neg_half = jnp.exp(-1j * half_p).astype(dtype)

        safe_gid = jnp.clip(g_id, 0, 14)
        g_1q = jax.lax.switch(
            safe_gid,
            [
                lambda _: jnp.eye(2, dtype=dtype),
                lambda _: jnp.array([[inv2, inv2], [inv2, -inv2]], dtype=dtype),
                lambda _: jnp.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[0.0 + 0j, -1j], [1j, 0.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_ph4]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_mh4]], dtype=dtype),
                lambda _: jnp.array([[cos_p, -1j * sin_p], [-1j * sin_p, cos_p]], dtype=dtype),
                lambda _: jnp.array([[cos_p, -sin_p], [sin_p, cos_p]], dtype=dtype),
                lambda _: jnp.array([[jnp.exp(-1j * half_p), 0.0 + 0j], [0.0 + 0j, jnp.exp(1j * half_p)]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_pos]], dtype=dtype),
                lambda _: jnp.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=dtype),
                # 14  GPhase(alpha) = e^{i*alpha} * I -- must stay in sync
                # with _apply_gate_fast_step (compiler.py) and the
                # non-distributed copy of this table above.
                lambda _: jnp.array([[exp_pos, 0.0 + 0j], [0.0 + 0j, exp_pos]], dtype=dtype),
            ],
            operand=None,
        )

        two_q_idx = jnp.where(g_id == 20, 0,
                    jnp.where(g_id == 21, 1,
                    jnp.where(g_id == 22, 2,
                    jnp.where(g_id == 24, 3, 4))))
        U = jax.lax.switch(
            two_q_idx,
            [
                lambda _: jnp.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, exp_pos]], dtype=dtype),
                lambda _: jnp.array([[0.0 + 0j, -1j], [1j, 0.0 + 0j]], dtype=dtype),
                lambda _: jnp.array([[exp_neg_half, 0.0 + 0j], [0.0 + 0j, exp_pos_half]], dtype=dtype),
            ],
            operand=None,
        )
        g00, g01, g10, g11 = g_1q[0, 0], g_1q[0, 1], g_1q[1, 0], g_1q[1, 1]
        u00, u01, u10, u11 = U[0, 0], U[0, 1], U[1, 0], U[1, 1]

        is_2q    = g_id >= 20
        q1_chunk = q1 < m
        q2_chunk = q2 < m

        # ── single point-to-point exchange, done once per step ──────
        # comm_qubit selects which stride to ppermute on: q2 (tgt) for
        # any 2-qubit gate with a chunk-select target (cases 5 and 6),
        # q1 for a 1-qubit gate on a chunk-select qubit (case 2),
        # sentinel `m` (-> identity, no network traffic) for cases
        # 1/3/4, which never need another device's data.
        needs_comm_2q = is_2q & q2_chunk
        needs_comm_1q = (~is_2q) & q1_chunk
        comm_qubit = jnp.where(needs_comm_2q, q2, jnp.where(needs_comm_1q, q1, m))
        safe_comm_idx = jnp.clip(comm_qubit, 0, m)

        # NOTE: `_perm` MUST be bound as a default-argument value here
        # (evaluated eagerly, once, at lambda-creation time inside this
        # loop) rather than referencing `q`/`m` freely inside the
        # lambda body -- a free reference would be looked up at CALL
        # time via Python's normal late-binding closure semantics, by
        # which point the loop variable `q` has already reached its
        # final value (m-1) for every branch, silently making every
        # ppermute use the LAST qubit's stride regardless of which
        # branch was actually selected. Caught by exactly that
        # symptom: only the branch for q == m-1 gave correct results.
        ppermute_branches = [
            (lambda _row, _perm=[(i, i ^ (1 << (m - 1 - q))) for i in range(num_chunks)]:
                 jax.lax.ppermute(_row, axis_name, perm=_perm))
            for q in range(m)
        ] + [lambda _row: _row]  # identity: no chunk-select qubit involved
        paired_row = jax.lax.switch(safe_comm_idx, ppermute_branches, local_row)

        # ── case 1: 1-qubit, LOCAL qubit (q1 >= m) — no comm ─────────
        def case_1q_local(_row):
            local_phys = (k - 1) - (q1 - m)
            stride     = jnp.int32(1) << local_phys
            idx        = jnp.arange(chunk_dim, dtype=jnp.int32)
            idx_pair   = idx ^ stride
            mask0      = (idx & stride) == 0
            amp_pair   = _row[idx_pair]
            new0 = g00 * _row + g01 * amp_pair
            new1 = g10 * amp_pair + g11 * _row
            return jnp.where(mask0, new0, new1)

        # ── case 2: 1-qubit, CHUNK-SELECT qubit (q1 < m) ───────────
        # paired_row already fetched via ppermute above (comm_qubit=q1).
        def case_1q_chunk(_row):
            stride = jnp.int32(1) << (m - 1 - q1)
            mask0  = (my_id & stride) == 0
            new0 = g00 * _row + g01 * paired_row
            new1 = g10 * paired_row + g11 * _row
            return jnp.where(mask0, new0, new1)

        # ── case 3: 2-qubit, ctrl AND tgt both LOCAL — no comm ───────
        def case_2q_local_local(_row):
            ctrl_phys = (k - 1) - (q1 - m)
            tgt_phys  = (k - 1) - (q2 - m)
            idx       = jnp.arange(chunk_dim, dtype=jnp.int32)
            ctrl_bit  = (idx & (jnp.int32(1) << ctrl_phys)) != 0
            partner   = idx ^ (jnp.int32(1) << tgt_phys)
            amp_partner = _row[partner]
            new0 = u00 * _row + u01 * amp_partner
            new1 = u10 * amp_partner + u11 * _row
            tgt_bit = (idx & (jnp.int32(1) << tgt_phys)) != 0
            after = jnp.where(tgt_bit, new1, new0)
            return jnp.where(ctrl_bit, after, _row)

        # ── case 4: ctrl CHUNK-SELECT, tgt LOCAL — no comm needed:   ─
        # every device decides purely from its OWN chunk index (my_id)
        # whether the control bit is set, and if so applies U locally.
        def case_2q_ctrl_chunk_tgt_local(_row):
            ctrl_stride = jnp.int32(1) << (m - 1 - q1)
            ctrl_set    = (my_id & ctrl_stride) != 0
            tgt_phys    = (k - 1) - (q2 - m)
            idx         = jnp.arange(chunk_dim, dtype=jnp.int32)
            tgt_bit     = (idx & (jnp.int32(1) << tgt_phys)) != 0
            partner     = idx ^ (jnp.int32(1) << tgt_phys)
            amp_partner = _row[partner]
            new0 = u00 * _row + u01 * amp_partner
            new1 = u10 * amp_partner + u11 * _row
            after = jnp.where(tgt_bit, new1, new0)
            return jnp.where(ctrl_set, after, _row)

        # ── case 5: ctrl LOCAL, tgt CHUNK-SELECT ──────────────
        # paired_row already fetched via ppermute above (comm_qubit=q2,
        # keyed on tgt's stride). is_c0 is a per-device scalar decision
        # (which side of the tgt pairing this device is on); ctrl_bit
        # is a per-element mask within the row (elementwise, local).
        def case_2q_ctrl_local_tgt_chunk(_row):
            ctrl_phys  = (k - 1) - (q1 - m)
            idx        = jnp.arange(chunk_dim, dtype=jnp.int32)
            ctrl_bit   = (idx & (jnp.int32(1) << ctrl_phys)) != 0
            tgt_stride = jnp.int32(1) << (m - 1 - q2)
            is_c0      = (my_id & tgt_stride) == 0
            new_c0 = u00 * _row + u01 * paired_row
            new_c1 = u10 * paired_row + u11 * _row
            after  = jnp.where(is_c0, new_c0, new_c1)
            return jnp.where(ctrl_bit, after, _row)

        # ── case 6: ctrl AND tgt both CHUNK-SELECT ────────────
        # paired_row via ppermute keyed on tgt's stride (comm_qubit=q2);
        # ctrl_set and is_c0 are both per-device scalars (this device's
        # own chunk index bits) — no per-element masking needed at all.
        def case_2q_both_chunk(_row):
            ctrl_stride = jnp.int32(1) << (m - 1 - q1)
            ctrl_set    = (my_id & ctrl_stride) != 0
            tgt_stride  = jnp.int32(1) << (m - 1 - q2)
            is_c0       = (my_id & tgt_stride) == 0
            new_c0 = u00 * _row + u01 * paired_row
            new_c1 = u10 * paired_row + u11 * _row
            after  = jnp.where(is_c0, new_c0, new_c1)
            return jnp.where(ctrl_set, after, _row)

        result_2q = jnp.where(
            q1_chunk & q2_chunk, case_2q_both_chunk(local_row),
            jnp.where(q1_chunk & (~q2_chunk), case_2q_ctrl_chunk_tgt_local(local_row),
            jnp.where((~q1_chunk) & q2_chunk, case_2q_ctrl_local_tgt_chunk(local_row),
                                               case_2q_local_local(local_row))))
        result_1q = jnp.where(q1_chunk, case_1q_chunk(local_row), case_1q_local(local_row))

        new_row = jnp.where(is_2q, result_2q, result_1q)
        return new_row.astype(dtype), None

    return step


def _build_distributed_chunk_runner(num_chunks: int, m: int, k: int):
    """shard_map-wrapped runner: one chunk row per physical JAX device.
    Requires jax.device_count() >= num_chunks (v1 scope: exactly one
    chunk per device, the literal reading of issue #1 -- "distribuire
    i chunk su più device"; a hybrid scheme with several chunks per
    device is a possible future refinement, not attempted here).

    compiled_ops (the small [g_id, q1, q2, param] sequence, identical
    on every device) is replicated, not sharded -- P(None, None).
    local_row is sharded along axis 0 of the (num_chunks, chunk_dim)
    logical array, one (chunk_dim,) row per device -- P(axis_name,
    None) on input/output, so each device's shard is that one row."""
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec as P

    axis_name = 'chunks'
    step = _build_distributed_chunk_step(num_chunks, m, k, axis_name)

    devices = np.array(jax.devices()[:num_chunks])
    mesh = Mesh(devices, axis_names=(axis_name,))

    def run_local(local_shard, compiled_ops):
        # shard_map keeps the sharded axis in the local shape (size
        # num_chunks/mesh_size along axis 0 -- 1 in this v1 one-
        # chunk-per-device scope), it doesn't squeeze it away: a
        # (num_chunks, chunk_dim) input shards to (1, chunk_dim) per
        # device here, not (chunk_dim,). `step` itself works on a
        # clean (chunk_dim,) row -- squeeze going in, restore going out.
        local_row = local_shard[0]
        final_row, _ = jax.lax.scan(step, local_row, compiled_ops)
        return final_row[None, :]

    sharded_run = jax.shard_map(
        run_local,
        mesh=mesh,
        in_specs=(P(axis_name, None), P(None, None)),
        out_specs=P(axis_name, None),
        check_vma=False,
    )
    return jax.jit(sharded_run), mesh


def _build_multi_chunk_runner(num_chunks: int, m: int, k: int):
    """jax.jit-compiled (chunks, compiled_ops) -> final_chunks, closed
    over the static per-Chunk-instance geometry (num_chunks, m, k don't
    change across calls on the same instance)."""
    step = _build_multi_chunk_step(num_chunks, m, k)

    @jax.jit
    def run(chunks, compiled_ops):
        final, _ = jax.lax.scan(step, chunks, compiled_ops)
        return final

    return run


def _compile_multi_chunk_ops(circuit: List) -> "jnp.ndarray":
    """Structural + GATE_IDS compilation shared by the multi-chunk JIT
    path — same [g_id, q1, q2, param] row format as beast-mode's own
    compiled ops, built via GATE_IDS instead of the old _resolve_gate's
    GATES/PARAMETRIC_GATES lookup. This finally aligns multi-chunk's
    gate coverage with beast-mode's (both silently skip a gate name not
    in GATE_IDS — same known, tracked behavior, see issue #4 — instead
    of the old _resolve_gate's NotImplementedError for e.g. ecr/iswap)."""
    target = QuantumTranspiler.transpile(circuit)
    rows = []
    for cmd in target:
        name = cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()
        if name not in GATE_IDS:
            continue
        g_id = float(GATE_IDS[name])
        args = cmd[1:]
        if name in ('cx', 'cz', 'cp', 'cphase', 'cy', 'crz'):
            q1, q2 = float(args[0]), float(args[1])
            param = float(args[2]) if len(args) > 2 else 0.0
            rows.append([g_id, q1, q2, param])
        elif args:
            q1 = float(args[0])
            param = float(args[1]) if len(args) > 1 else 0.0
            rows.append([g_id, q1, 0.0, param])
    if not rows:
        return jnp.empty((0, 4), dtype=jnp.float64)
    return jnp.array(rows, dtype=jnp.float64)
