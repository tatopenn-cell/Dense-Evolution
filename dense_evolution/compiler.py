import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# ── optional JAX import (same pattern as registry) ────────────────────────────
try:
    from .registry import HAS_JAX
except ImportError:
    try:
        import jax
        HAS_JAX = True
    except ImportError:
        HAS_JAX = False

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────────────────────────────────────
# Gate-ID encoding (shared between _apply_gate_fast_step and the beast-mode
# command builder in the dashboard).
#
#  0  I (identity)       7  T
#  1  H                  8  Tdg
#  2  X                  9  Rx(θ)
#  3  Y                 10  Ry(θ)
#  4  Z                 11  Rz(θ)
#  5  S                 12  Phase / P(θ) / U1(θ)
#  6  Sdg               13  SX (√X)
#                       ──  2-qubit gates ──
#                       20  CX / CNOT
#                       21  CZ
#                       22  CP(θ)              — phase on |11⟩ only
#                       23  SWAP (never dispatched here — always decomposed
#                           into 3xCX by QuantumTranspiler.transpile first)
#                       24  CY
#                       25  CRZ(θ)             — phase on target conditioned
#                           on control, distinct from CP: NOT the same gate,
#                           do not merge into 22 (see apply_crz below)
# ─────────────────────────────────────────────────────────────────────────────

if HAS_JAX:

    @jax.jit
    def _apply_gate_fast_step(sv: "jnp.ndarray",
                               operation: "jnp.ndarray"):
        """
        Apply a single quantum gate to statevector *sv*.

        Parameters
        ----------
        sv        : complex128 statevector of shape (2**n,)
        operation : float64 array [g_id, q1, q2, param]
                    g_id  — gate identifier (see table above)
                    q1    — target qubit (1-qubit gates) or control qubit (2-qubit)
                    q2    — target qubit for 2-qubit gates; unused for 1-qubit
                    param — rotation angle in radians; 0.0 for non-parametric gates

        Returns
        -------
        (new_sv, None)  — compatible with jax.lax.scan
        """
        g_id  = operation[0].astype(jnp.int32)
        q1    = operation[1].astype(jnp.int32)
        q2    = operation[2].astype(jnp.int32)
        param = operation[3]
        dim   = sv.shape[0]                    # static (shape is concrete even under jit)
        n_qubits = dim.bit_length() - 1        # static; dim == 2**n_qubits always

        # Every constant below must match sv's own dtype (complex64 if
        # use_float32=True, complex128 otherwise), not a dtype hardcoded to
        # complex128 — a fixed dtype here made this function only traceable
        # for complex128 input: apply_cp (inside do_2q) multiplied by an
        # exp_pos hardcoded to complex128, while lax.cond's other branch
        # (identity, `lambda s: s`) preserved sv's real dtype, so any
        # complex64 circuit hit "cond branches must have equal output
        # types" immediately at trace time — even for circuits with only
        # 1-qubit gates, since do_2q's body (and this cond) is traced
        # unconditionally as part of the is_1q/is_2q dispatch below.
        sv_dtype = sv.dtype
        inv2    = jnp.asarray(1.0 / jnp.sqrt(2.0), dtype=sv_dtype)
        half_p  = param * jnp.float64(0.5)
        cos_p   = jnp.cos(half_p).astype(sv_dtype)
        sin_p   = jnp.sin(half_p).astype(sv_dtype)
        exp_pos = jnp.exp( 1j * param).astype(sv_dtype)
        exp_neg = jnp.exp(-1j * param).astype(sv_dtype)
        exp_ph4 = jnp.exp( 1j * jnp.pi / 4.0).astype(sv_dtype)
        exp_mh4 = jnp.exp(-1j * jnp.pi / 4.0).astype(sv_dtype)
        # half-angle phases for CRZ(θ) — same convention as the 1-qubit
        # RZ(θ) switch entry below, named here since apply_crz (do_2q) needs
        # them inside a conditional function rather than an inline literal.
        exp_pos_half = jnp.exp( 1j * half_p).astype(sv_dtype)
        exp_neg_half = jnp.exp(-1j * half_p).astype(sv_dtype)

        # ── 1-qubit gate matrix selection via lax.switch ──────────────
        # Index must be in [0, 13]; anything outside is clamped to 0 (I).
        safe_gid = jnp.clip(g_id, 0, 13)

        g_1q = jax.lax.switch(
            safe_gid,
            [
                # 0  I
                lambda _: jnp.eye(2, dtype=sv_dtype),
                # 1  H
                lambda _: jnp.array(
                    [[inv2,  inv2],
                     [inv2, -inv2]], dtype=sv_dtype),
                # 2  X
                lambda _: jnp.array(
                    [[0.0+0j, 1.0+0j],
                     [1.0+0j, 0.0+0j]], dtype=sv_dtype),
                # 3  Y
                lambda _: jnp.array(
                    [[0.0+0j, -1j],
                     [1j,      0.0+0j]], dtype=sv_dtype),
                # 4  Z
                lambda _: jnp.array(
                    [[1.0+0j,  0.0+0j],
                     [0.0+0j, -1.0+0j]], dtype=sv_dtype),
                # 5  S
                lambda _: jnp.array(
                    [[1.0+0j, 0.0+0j],
                     [0.0+0j, 1j    ]], dtype=sv_dtype),
                # 6  Sdg
                lambda _: jnp.array(
                    [[1.0+0j, 0.0+0j],
                     [0.0+0j, -1j   ]], dtype=sv_dtype),
                # 7  T
                lambda _: jnp.array(
                    [[1.0+0j, 0.0+0j],
                     [0.0+0j, exp_ph4]], dtype=sv_dtype),
                # 8  Tdg
                lambda _: jnp.array(
                    [[1.0+0j, 0.0+0j],
                     [0.0+0j, exp_mh4]], dtype=sv_dtype),
                # 9  Rx(θ)  = [[cos θ/2, -i sin θ/2], [-i sin θ/2, cos θ/2]]
                lambda _: jnp.array(
                    [[cos_p,      -1j * sin_p],
                     [-1j * sin_p, cos_p     ]], dtype=sv_dtype),
                # 10  Ry(θ) = [[cos θ/2, -sin θ/2], [sin θ/2, cos θ/2]]
                lambda _: jnp.array(
                    [[cos_p,  -sin_p],
                     [sin_p,   cos_p]], dtype=sv_dtype),
                # 11  Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
                lambda _: jnp.array(
                    [[jnp.exp(-1j * half_p), 0.0+0j            ],
                     [0.0+0j,                jnp.exp(1j * half_p)]],
                    dtype=sv_dtype),
                # 12  Phase / P(θ) / U1(θ) = [[1, 0], [0, e^{iθ}]]
                lambda _: jnp.array(
                    [[1.0+0j, 0.0+0j],
                     [0.0+0j, exp_pos]], dtype=sv_dtype),
                # 13  SX (√X) = 0.5 * [[1+i, 1-i], [1-i, 1+i]]
                lambda _: jnp.array(
                    [[0.5+0.5j, 0.5-0.5j],
                     [0.5-0.5j, 0.5+0.5j]], dtype=sv_dtype),
            ],
            operand=None,
        )

        # ── 1-qubit application ────────────────────────────────────────
        def do_1q(_sv):
            # MSB-first, matching the rest of the simulator (simulator.py's
            # own docstring/_qubit_stride_pairs: qubit 0 = most significant
            # bit, phys = n_qubits - 1 - qubit) — this used to be a raw
            # `1 << q1` (LSB-first), silently disagreeing with run_circuit()/
            # apply_gate_1q()/measure() on every multi-qubit circuit that
            # isn't symmetric under qubit reversal (verified: X on qubit 0
            # in a 3-qubit register gave index 1 here vs index 4 via
            # run_circuit() before this fix — see CHANGELOG).
            phys      = jnp.int64(n_qubits - 1) - q1.astype(jnp.int64)
            stride    = jnp.int64(1) << phys
            idx_full  = jnp.arange(dim, dtype=jnp.int64)
            mask_0    = (idx_full & stride) == 0

            # idx_0: indices where qubit q1 == 0
            # idx_1: corresponding |1⟩ partners
            # We build them without xp.where-tuple confusion:
            # any index i has its pair at i ^ stride.
            # For i in |0⟩ slots (mask_0): partner = i | stride = i ^ stride
            # For i in |1⟩ slots (¬mask_0): partner = i ^ stride  (clears bit)
            # We process all indices simultaneously using the |0⟩ slot's amplitude.
            idx_pair  = idx_full ^ stride          # each element's partner
            amp_self  = _sv[idx_full]              # a[i]
            amp_pair  = _sv[idx_pair]              # a[i ^ stride]

            # When mask_0: amp_self = a_0, amp_pair = a_1
            # new_0 = g00*a_0 + g01*a_1
            # new_1 = g10*a_0 + g11*a_1
            g00 = g_1q[0, 0];  g01 = g_1q[0, 1]
            g10 = g_1q[1, 0];  g11 = g_1q[1, 1]

            new_when_0 = g00 * amp_self + g01 * amp_pair   # result for |0⟩ slot
            new_when_1 = g10 * amp_pair + g11 * amp_self   # result for |1⟩ slot
            # NOTE: for |1⟩ slots, amp_pair is the |0⟩ amplitude and
            # amp_self is the |1⟩ amplitude — roles are swapped.

            return jnp.where(mask_0, new_when_0, new_when_1)

        # ── 2-qubit application ───────────────────────────────────────
        def do_2q(_sv):
            # Same MSB-first fix as do_1q above — ctrl/trgt are physical bit
            # positions (n_qubits-1-qubit), not raw qubit indices.
            ctrl     = jnp.int64(n_qubits - 1) - q1.astype(jnp.int64)
            trgt     = jnp.int64(n_qubits - 1) - q2.astype(jnp.int64)
            idx_full = jnp.arange(dim, dtype=jnp.int64)

            ctrl_bit_set = (idx_full & (jnp.int64(1) << ctrl)) != 0
            trgt_bit_set = (idx_full & (jnp.int64(1) << trgt)) != 0

            # CX: flip target bit when control is set
            def apply_cx(__sv):
                partner = idx_full ^ (jnp.int64(1) << trgt)
                swapped = __sv[partner]
                return jnp.where(ctrl_bit_set, swapped, __sv)

            # CZ: negate amplitude when both control and target bits are set
            def apply_cz(__sv):
                both_set = ctrl_bit_set & trgt_bit_set
                return jnp.where(both_set, -__sv, __sv)

            # CP(θ): phase kick e^{iθ} on |11⟩ component only — NOT the same
            # gate as CRZ below (see apply_crz docstring for the distinction).
            def apply_cp(__sv):
                both_set = ctrl_bit_set & trgt_bit_set
                return jnp.where(both_set, exp_pos * __sv, __sv)

            # CY: apply Y to target when control is set. Same index-pair
            # structure as apply_cx (swap target=0/target=1 partners), but
            # with Y's phase factors instead of a plain swap: Y|0⟩=i|1⟩,
            # Y|1⟩=-i|0⟩ — the new amplitude at each slot is ±i times its
            # PARTNER's amplitude (matches do_1q's own Y row-formula, since
            # Y's diagonal entries are both 0).
            def apply_cy(__sv):
                partner = idx_full ^ (jnp.int64(1) << trgt)
                swapped = __sv[partner]
                phase   = jnp.where(trgt_bit_set, 1j, -1j).astype(sv_dtype)
                return jnp.where(ctrl_bit_set, phase * swapped, __sv)

            # CRZ(θ): apply RZ(θ) to target when control is set — a DIAGONAL
            # phase conditioned on the TARGET's own bit value (e^{-iθ/2} if
            # target=0, e^{+iθ/2} if target=1), not on "both bits set" like
            # CP above. Deliberately a separate function/id from apply_cp:
            # CP and CRZ are mathematically different gates that used to be
            # silently merged under one id before this fix (CRZ wasn't
            # reachable at all — see CHANGELOG).
            def apply_crz(__sv):
                phase = jnp.where(trgt_bit_set, exp_pos_half, exp_neg_half)
                return jnp.where(ctrl_bit_set, phase * __sv, __sv)

            # SWAP: exchange amplitudes of ctrl-bit and trgt-bit positions
            def apply_swap(__sv):
                # Standard SWAP = CX(c,t) · CX(t,c) · CX(c,t)
                # Computed directly: for each (ctrl=0,trgt=1) pair with
                # the other bits identical, swap the two amplitudes.
                only_ctrl = ( ctrl_bit_set & ~trgt_bit_set)
                only_trgt = (~ctrl_bit_set &  trgt_bit_set)
                swap_mask = only_ctrl | only_trgt
                partner   = idx_full ^ (jnp.int64(1) << ctrl) ^ (jnp.int64(1) << trgt)
                return jnp.where(swap_mask, __sv[partner], __sv)

            # Dispatch on g_id: 20=CX, 21=CZ, 22=CP, 23=SWAP, 24=CY, 25=CRZ
            is_cx   = g_id == 20
            is_cz   = g_id == 21
            is_cp   = g_id == 22
            is_cy   = g_id == 24
            is_crz  = g_id == 25
            # is_swap = g_id == 23 (default branch — never actually reached,
            # see comment at the top of this file: QuantumTranspiler always
            # decomposes 'swap' into 3xCX before a gate name reaches here)

            after_cx   = jax.lax.cond(is_cx,   apply_cx,   lambda s: s, _sv)
            after_cz   = jax.lax.cond(is_cz,   apply_cz,   lambda s: s, _sv)
            after_cp   = jax.lax.cond(is_cp,   apply_cp,   lambda s: s, _sv)
            after_cy   = jax.lax.cond(is_cy,   apply_cy,   lambda s: s, _sv)
            after_crz  = jax.lax.cond(is_crz,  apply_crz,  lambda s: s, _sv)
            after_swap = apply_swap(_sv)

            # Pick the right result
            result = jnp.where(is_cx,  after_cx,
                     jnp.where(is_cz,  after_cz,
                     jnp.where(is_cp,  after_cp,
                     jnp.where(is_cy,  after_cy,
                     jnp.where(is_crz, after_crz,
                                       after_swap)))))
            return result

        # ── branch on 1-qubit vs 2-qubit ─────────────────────────────
        # g_id <= 12 → 1-qubit;  g_id >= 20 → 2-qubit.
        # Both branches must have identical output dtypes — enforced here by
        # casting both outputs to sv_dtype (sv's own dtype, complex64 or
        # complex128 depending on use_float32), not a dtype hardcoded to
        # complex128 (see note above: forcing complex128 here regardless of
        # input dtype would also break jax.lax.scan's carry-type check on
        # the very next iteration when sv started out as complex64).
        is_1q = g_id <= 13
        new_sv = jax.lax.cond(
            is_1q,
            lambda s: do_1q(s).astype(sv_dtype),
            lambda s: do_2q(s).astype(sv_dtype),
            sv,
        )
        return new_sv, None


    @jax.jit
    def _compile_and_run_circuit_jit(state_vector: "jnp.ndarray",
                                      compiled_ops:  "jnp.ndarray") -> "jnp.ndarray":
        """
        Execute a pre-compiled gate sequence on *state_vector* via jax.lax.scan.

        Parameters
        ----------
        state_vector : complex128 array of shape (2**n,)
        compiled_ops : float64 array of shape (n_gates, 4)
                       each row = [g_id, q1, q2, param]

        Returns
        -------
        Final statevector after all gates.
        """
        final_sv, _ = jax.lax.scan(_apply_gate_fast_step, state_vector, compiled_ops)
        return final_sv


# ─────────────────────────────────────────────────────────────────────────────
# QuantumTranspiler
# ─────────────────────────────────────────────────────────────────────────────

class QuantumTranspiler:
    """
    Gate-level transpiler: decomposes non-native gates into the native
    {H, T, Tdg, CX, CZ} basis and performs basic circuit optimisations.
    """

    @staticmethod
    def decompose_toffoli(c1: int, c2: int, t: int) -> List[Tuple]:
        """
        Decompose CCX (Toffoli) into 15 native gates using the
        standard T/Tdg/CX Barenco decomposition.

        Gate count: 6 CX + 7 single-qubit (H, T, Tdg) = 15 total.
        """
        return [
            ('h',   t),
            ('cx',  c2, t),  ('tdg', t),
            ('cx',  c1, t),  ('t',   t),
            ('cx',  c2, t),  ('tdg', t),
            ('cx',  c1, t),
            ('t',   c2),     ('t',   t),
            ('cx',  c1, c2), ('h',   t),
            ('t',   c1),     ('tdg', c2),
            ('cx',  c1, c2),
        ]

    @staticmethod
    def decompose_swap(q1: int, q2: int) -> List[Tuple]:
        """Decompose SWAP into 3 CX gates."""
        return [('cx', q1, q2), ('cx', q2, q1), ('cx', q1, q2)]

    @classmethod
    def transpile(cls, circuit: List[Tuple]) -> List[Tuple]:
        """
        Expand CCX → 15 native gates and SWAP → 3 CX.
        All other gates are passed through unchanged.

        Parameters
        ----------
        circuit : list of tuples (gate_name, qubit, ...)

        Returns
        -------
        Expanded circuit as a list of tuples.
        """
        out: List[Tuple] = []
        for cmd in circuit:
            name = cmd[0].lower()            # BUG FIX: was cmd.lower() on a tuple
            if name == 'ccx':
                out.extend(cls.decompose_toffoli(*cmd[1:4]))
            elif name == 'swap':
                out.extend(cls.decompose_swap(*cmd[1:3]))
            else:
                out.append(cmd)
        return out
