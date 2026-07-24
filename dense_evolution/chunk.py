import gc
import psutil
import numpy as np
from typing import List, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = None
    HAS_JAX = False

# ── Flexible import with stub fallback ──────────────────────────────────────
try:
    from simulator import DenseSVSimulator
    from compiler import QuantumTranspiler
    from gates import GATE_IDS
except ModuleNotFoundError:
    try:
        from dense_evolution.simulator import DenseSVSimulator
        from dense_evolution.compiler import QuantumTranspiler
        from dense_evolution.gates import GATE_IDS
    except ModuleNotFoundError:
        class DenseSVSimulator:  # type: ignore[no-redef]
            def __init__(self, n_qubits, **kwargs):
                self.n     = n_qubits
                self.dim   = 2 ** n_qubits
                self.dtype = np.complex128
                self.sv    = np.zeros(self.dim, dtype=self.dtype)
                self.sv[0] = 1.0
            def run_circuit_jit_beast_mode(self, circuit_slice): pass
            def memory_mb(self) -> float:
                return (self.dim * np.dtype(self.dtype).itemsize) / 1_000_000

        class QuantumTranspiler:  # type: ignore[no-redef]
            @staticmethod
            def transpile(circuit): return circuit

        GATE_IDS: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_dynamic_chunk(dtype_target) -> int:
    vm = psutil.virtual_memory()
    safe_ram = vm.available * 0.85
    if HAS_JAX and dtype_target is jnp.complex128:
        bpe = 16
    elif dtype_target is np.complex128:
        bpe = 16
    else:
        bpe = 8
    max_elements = safe_ram / bpe
    max_bits = int(np.floor(np.log2(max(max_elements, 2.0))))
    return max(16, min(max_bits, 27))


def _dtype_for_qubits(n_qubits: int):
    xp = jnp if HAS_JAX else np
    return xp.complex64 if n_qubits > 26 else xp.complex128


# ─────────────────────────────────────────────────────────────────────────────
# SafeMemoryGuard  — Anti-OOM block
# ─────────────────────────────────────────────────────────────────────────────

class MemoryPressureError(RuntimeError):
    """
    Raised when available system RAM drops below the configured safety threshold.
    Catches the condition *before* the allocator attempts and crashes with
    jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED.
    """
    pass


class SafeMemoryGuard:
    """
    Monitors system RAM before every high-memory operation and blocks execution
    if free RAM falls below ``threshold_pct`` of total physical memory.
    """

    _WARN_MULTIPLIER = 2.0

    def __init__(self, threshold_pct: float = 0.15, gc_before_check: bool = True):
        if not 0.0 < threshold_pct < 1.0:
            raise ValueError(f"threshold_pct must be in (0, 1), got {threshold_pct}")
        self.threshold_pct   = threshold_pct
        self.gc_before_check = gc_before_check
        self._total_mb       = psutil.virtual_memory().total / (1024 * 1024)

    def status(self) -> dict:
        vm = psutil.virtual_memory()
        available_mb = vm.available / (1024 * 1024)
        free_pct     = vm.available / vm.total
        return {
            "total_mb"    : self._total_mb,
            "available_mb": available_mb,
            "used_pct"    : vm.percent,
            "free_pct"    : free_pct * 100.0,
            "safe"        : free_pct >= self.threshold_pct,
        }

    def check(self, context: str = "") -> None:
        if self.gc_before_check:
            gc.collect()

        s   = self.status()
        tag = f"[{context}] " if context else ""
        free_frac = s["free_pct"] / 100.0

        if not s["safe"]:
            raise MemoryPressureError(
                f"\n{'─'*60}\n"
                f"  {tag}MEMORIA CRITICA — simulazione bloccata\n"
                f"  Disponibile : {s['available_mb']:.0f} MB  "
                f"({s['free_pct']:.1f}% libera)\n"
                f"  Soglia      : {self.threshold_pct * 100:.0f}%  "
                f"({self._total_mb * self.threshold_pct:.0f} MB)\n"
                f"  Azione      : liberare RAM o ridurre n_qubits / chunk_size.\n"
                f"{'─'*60}"
            )

        warn_threshold = self.threshold_pct * self._WARN_MULTIPLIER
        if free_frac < warn_threshold:
            print(
                f"  [WARN] {tag}RAM bassa: {s['available_mb']:.0f} MB liberi "
                f"({s['free_pct']:.1f}%) — soglia critica al "
                f"{self.threshold_pct * 100:.0f}%."
            )

    def check_allocation(self, required_mb: float, context: str = "") -> None:
        """Like check(), but sized: verifies that *required_mb* can be
        allocated while still leaving threshold_pct free afterwards —
        check() alone only looks at RAM free *right now*, independent of
        what's about to be allocated (needed for Chunk's multi-chunk path,
        where several chunk-sized simulators are held in RAM at once)."""
        if self.gc_before_check:
            gc.collect()

        s   = self.status()
        tag = f"[{context}] " if context else ""
        available_after_mb = s["available_mb"] - required_mb
        free_frac_after = available_after_mb / self._total_mb if self._total_mb > 0 else 0.0

        if available_after_mb < 0 or free_frac_after < self.threshold_pct:
            raise MemoryPressureError(
                f"\n{'─'*60}\n"
                f"  {tag}MEMORIA INSUFFICIENTE per l'allocazione richiesta\n"
                f"  Richiesti    : {required_mb:.0f} MB\n"
                f"  Disponibile  : {s['available_mb']:.0f} MB ({s['free_pct']:.1f}% libera)\n"
                f"  Dopo alloc.  : {available_after_mb:.0f} MB ({free_frac_after * 100:.1f}% libera)\n"
                f"  Soglia       : {self.threshold_pct * 100:.0f}% libera dopo l'allocazione\n"
                f"  Azione       : ridurre n_qubits o liberare RAM. Il chunking su\n"
                f"                 disco per overflow oltre la RAM disponibile non\n"
                f"                 e' implementato — vedi CHANGELOG.\n"
                f"{'─'*60}"
            )

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"SafeMemoryGuard("
            f"threshold={self.threshold_pct*100:.0f}%, "
            f"available={s['available_mb']:.0f} MB / {s['free_pct']:.1f}% free, "
            f"safe={s['safe']})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MemoryChunker  (chunk1)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryChunker:
    """
    Geometry calculator for chunked simulation.

    Attributes
    ----------
    n_qubits        int   — requested logical qubit count
    dtype                 — numpy/jax dtype for the statevector
    chunk_size_bits int   — safe qubit-width that fits in RAM
    num_chunks      int   — number of statevector chunks required
    chunk_dim       int   — 2 ** chunk_size_bits
    """

    def __init__(self, n_qubits: int):
        self.n_qubits        = n_qubits
        self.dtype           = _dtype_for_qubits(n_qubits)
        self.chunk_size_bits = get_dynamic_chunk(self.dtype)

        if self.n_qubits <= self.chunk_size_bits:
            self.num_chunks = 1
            self.chunk_dim  = 2 ** self.n_qubits
        else:
            self.num_chunks = 2 ** (self.n_qubits - self.chunk_size_bits)
            self.chunk_dim  = 2 ** self.chunk_size_bits

    def geometry(self) -> Tuple[int, int, int]:
        """(num_chunks, chunk_dim, chunk_size_bits)"""
        return self.num_chunks, self.chunk_dim, self.chunk_size_bits

    def memory_mb(self) -> float:
        """Estimated RAM per chunk in MB."""
        bpe = np.dtype(self.dtype).itemsize
        return (self.chunk_dim * bpe) / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"MemoryChunker(n_qubits={self.n_qubits}, "
            f"num_chunks={self.num_chunks}, "
            f"chunk_dim={self.chunk_dim}, "
            f"chunk_size_bits={self.chunk_size_bits}, "
            f"dtype={self.dtype}, "
            f"mem_per_chunk={self.memory_mb():.2f} MB)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-chunk JIT kernel (num_chunks > 1)
# ─────────────────────────────────────────────────────────────────────────────

if HAS_JAX:

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
            safe_gid = jnp.clip(g_id, 0, 13)
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

            # ── case 1: 1-qubit gate, LOCAL qubit (q1 >= m) ──────────────
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

            # ── case 3: 2-qubit, ctrl AND tgt both LOCAL ─────────────────
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

            # ── case 4: ctrl CHUNK-SELECT, tgt LOCAL ─────────────────────
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

            # ── case 5: ctrl LOCAL, tgt CHUNK-SELECT ─────────────────────
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

            # ── case 6: ctrl AND tgt both CHUNK-SELECT ───────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# CircuitChunker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitChunker:
    """
    Transpile a circuit once, then execute it in fixed-size gate-slices so
    XLA sees the same trace shape on every compilation.

    A SafeMemoryGuard is checked **before every slice** — if RAM drops below
    15% the current slice is aborted with MemoryPressureError before JAX
    attempts the allocation.

    Parameters
    ----------
    simulator_instance : DenseSVSimulator
        Physical simulator (sized to safe_qubits, not logical n_qubits).
    memory_threshold   : float
        Passed to SafeMemoryGuard.  Default 0.15 (15%).
    """

    def __init__(
        self,
        simulator_instance: Optional[DenseSVSimulator] = None,
        memory_threshold: float = 0.15,
    ):
        self.sim   = simulator_instance
        self._guard = SafeMemoryGuard(threshold_pct=memory_threshold)

    def split_circuit(self, circuit: List, chunk_size: int = 500) -> None:
        """
        Execute *circuit* in slices of *chunk_size* gates.

        Raises
        ------
        RuntimeError        if no simulator instance is attached.
        MemoryPressureError if RAM drops below threshold before a slice.
        """
        if self.sim is None:
            raise RuntimeError(
                "CircuitChunker: no simulator instance attached. "
                "Pass simulator_instance= at construction or assign .sim."
            )

        target: List = QuantumTranspiler.transpile(circuit)
        n_slices     = (len(target) + chunk_size - 1) // chunk_size

        for idx, i in enumerate(range(0, len(target), chunk_size)):
            # ── Anti-OOM check before every slice ───────────────────────────
            self._guard.check(f"slice {idx + 1}/{n_slices}")
            self.sim.run_circuit_jit_beast_mode(target[i : i + chunk_size])


# ─────────────────────────────────────────────────────────────────────────────
# Chunk  (chunk2 / Chunk2Incrociato)
# ─────────────────────────────────────────────────────────────────────────────

class Chunk:
    """
    Anti-OOM wrapper for large-qubit simulation.

    Does NOT subclass DenseSVSimulator directly — the parent __init__ allocates
    2**n_qubits elements immediately (17 GB for 30 qubits).

    For n_qubits <= chunk_size_bits (the RAM-safe budget): a single inner
    simulator is allocated and the logical qubit count is stored separately.

    For n_qubits > chunk_size_bits: num_chunks separate chunk_size_bits-qubit
    simulators are held in RAM simultaneously (see _dispatch_multi) — no
    disk/memmap paging, so this only covers a *moderate* overflow beyond the
    safe budget (as many chunks as actually fit in RAM at once, checked
    up front via SafeMemoryGuard.check_allocation before anything is
    allocated). Benchmark attributes (num_chunks, chunk_size_bits, dtype)
    are forwarded transparently from the embedded MemoryChunker.

    A SafeMemoryGuard fires before any simulator is instantiated
    (pre-allocation check) and is also embedded in CircuitChunker for
    per-slice protection during execution (n_qubits <= chunk_size_bits path).

    Parameters
    ----------
    n_qubits          : logical qubit count of the target circuit
    chunk_size_gates  : gate-slice size for JIT compilation (default 500)
    memory_threshold  : free-RAM fraction below which execution is blocked
                        (default 0.15 = 15%)
    use_gpu           : forwarded to DenseSVSimulator
    use_float32       : forwarded to DenseSVSimulator
    """

    def __init__(
        self,
        n_qubits: int,
        chunk_size_gates:  int   = 500,
        memory_threshold:  float = 0.15,
        use_gpu:           bool  = False,
        use_float32:       bool  = False,
    ):
        # 1. Geometry — purely RAM-based, no JAX allocation yet
        self._mem_chunker     = MemoryChunker(n_qubits)
        self._guard           = SafeMemoryGuard(threshold_pct=memory_threshold)

        # 2. Logical qubit count (for circuit parsing)
        self.n                = n_qubits
        self.chunk_size_gates = chunk_size_gates
        self._m                = n_qubits - self._mem_chunker.chunk_size_bits  # chunk-select qubit count (0 if num_chunks==1)

        if self._mem_chunker.num_chunks == 1:
            # 3a. Pre-allocation RAM check — block here rather than inside JAX
            safe_q = min(n_qubits, self._mem_chunker.chunk_size_bits)
            self._guard.check(f"Chunk.__init__ — allocating {safe_q}-qubit simulator")

            # 4a. Physical simulator sized to what RAM can actually hold
            self._inner_sim = DenseSVSimulator(
                safe_q,
                use_gpu=use_gpu,
                use_float32=use_float32,
            )
            self._chunk_sims = None
            self._multi_chunk_runner = None

            # 5a. Circuit chunker wired to the physical simulator, with same threshold
            self._circuit_chunker = CircuitChunker(
                simulator_instance=self._inner_sim,
                memory_threshold=memory_threshold,
            )
        else:
            # 3b. Sized pre-allocation check: num_chunks chunk-sized simulators
            # held in RAM at once, plus ~2 chunks of headroom for the temporary
            # arrays the cross-chunk gate-mixing math allocates at its peak.
            num_chunks   = self._mem_chunker.num_chunks
            per_chunk_mb = self._mem_chunker.memory_mb()
            required_mb  = (num_chunks + 2) * per_chunk_mb
            self._guard.check_allocation(
                required_mb,
                f"Chunk.__init__ — allocating {num_chunks} chunks of "
                f"{self._mem_chunker.chunk_size_bits} qubits each",
            )

            # 4b. num_chunks independent chunk-sized simulators. Each one's own
            # __init__ resets it to |0...0>: only chunk 0 should carry the
            # amplitude-1 seed for the LOGICAL |0...0>, the rest must start
            # at all-zero (direct .sv assignment — set_state/set_initial_state
            # reject zero-norm vectors by design).
            self._chunk_sims = [
                DenseSVSimulator(self._mem_chunker.chunk_size_bits, use_gpu=use_gpu, use_float32=use_float32)
                for _ in range(num_chunks)
            ]
            for sim in self._chunk_sims[1:]:
                sim.sv = sim.xp.zeros(self._mem_chunker.chunk_dim, dtype=sim.dtype)

            self._inner_sim        = None
            self._circuit_chunker  = None

            # 6b. JIT runner for the multi-chunk dispatch — built once here
            # since num_chunks/m/chunk_size_bits are fixed for this
            # instance's whole lifetime, reused by every run_chunk() call.
            self._multi_chunk_runner = _build_multi_chunk_runner(
                num_chunks, self._m, self._mem_chunker.chunk_size_bits)

    # ── Benchmark-facing attribute forwarding ────────────────────────────────

    @property
    def num_chunks(self) -> int:
        return self._mem_chunker.num_chunks

    @property
    def chunk_size_bits(self) -> int:
        return self._mem_chunker.chunk_size_bits

    @property
    def chunk_dim(self) -> int:
        return self._mem_chunker.chunk_dim

    @property
    def dtype(self):
        return self._mem_chunker.dtype

    @property
    def memory_geometry(self) -> MemoryChunker:
        return self._mem_chunker

    # ── Simulator-facing forwarding ──────────────────────────────────────────

    @property
    def sv(self):
        """Current statevector. For num_chunks==1, the physical (chunk-sized)
        simulator's own array. For num_chunks>1, the chunks concatenated in
        ascending order — valid because of the MSB-first correspondence
        between chunk index and the top `_m` logical qubits (see
        _dispatch_multi's docstring)."""
        if self._chunk_sims is None:
            return self._inner_sim.sv
        xp = self._chunk_sims[0].xp
        return xp.concatenate([sim.sv for sim in self._chunk_sims])

    @sv.setter
    def sv(self, value):
        """Accepts a full-length (2**n,) statevector (e.g. the output of
        NoiseModel.apply_to_sv called on `.sv`) and writes it back through
        to the physical storage -- the inner simulator directly for
        num_chunks==1, or split back into per-chunk slices (same ascending
        concatenation order as the getter) for num_chunks>1."""
        if self._chunk_sims is None:
            self._inner_sim.sv = value
            return
        chunk_dim = self._mem_chunker.chunk_dim
        xp = self._chunk_sims[0].xp
        value = xp.asarray(value)
        for i, sim in enumerate(self._chunk_sims):
            sim.sv = value[i * chunk_dim:(i + 1) * chunk_dim]

    def memory_mb(self) -> float:
        """RAM used by the physical statevector(s) in MB."""
        if self._chunk_sims is None:
            return self._inner_sim.memory_mb()
        return sum(sim.memory_mb() for sim in self._chunk_sims)

    def get_probabilities(self):
        """|amplitude|^2 for every basis state.

        num_chunks==1: forwards to the inner DenseSVSimulator for parity
        with its own get_probabilities().

        num_chunks>1: concatenates the RAW statevectors first and normalizes
        ONCE over the full array — NOT each chunk's own get_probabilities()
        (that would independently renormalize each chunk's partial mass to
        1, summing to num_chunks overall and destroying the relative
        weighting between chunks)."""
        if self._chunk_sims is None:
            return self._inner_sim.get_probabilities()
        full_sv = np.concatenate([np.array(sim.sv) for sim in self._chunk_sims])
        probs = np.abs(full_sv) ** 2
        probs = np.clip(probs, 0.0, 1.0)
        total = probs.sum()
        if total > 1e-12:
            probs /= total
        return probs

    def get_statevector(self):
        """Full complex statevector, num_qubits logical qubits long
        (2**n elements). num_chunks==1: forwards to the inner
        DenseSVSimulator. num_chunks>1: raw chunks concatenated in order
        (see `sv` property)."""
        if self._chunk_sims is None:
            return self._inner_sim.get_statevector()
        return np.concatenate([np.array(sim.sv, dtype=sim.dtype) for sim in self._chunk_sims])

    # ── Multi-chunk gate dispatch (num_chunks > 1) ───────────────────────────

    def _dispatch_multi(self, circuit: List) -> None:
        """Executes *circuit* against the num_chunks>1 chunk representation
        via one jax.lax.scan call over the whole (transpiled, GATE_IDS-
        compiled) circuit — see _build_multi_chunk_step/_build_multi_chunk_runner
        above for the kernel, and _compile_multi_chunk_ops for the encoding.

        Convention: chunk index `c` (m = self._m bits, MSB-first, same
        n-1-qubit convention as DenseSVSimulator) equals the value of the
        top m logical qubits (indices [0, m)); chunk_sims[c].sv holds the
        chunk_dim amplitudes for the remaining (local) qubits [m, n). This
        makes full_sv.reshape(num_chunks, chunk_dim)[c] == chunk_sims[c].sv
        exactly, since NumPy's row-major reshape splits a (2,)*n tensor on
        the leading axes first — i.e. the most-significant qubits, matching
        this simulator's MSB-first indexing throughout. Stacking
        chunk_sims[i].sv into one (num_chunks, chunk_dim) array before the
        scan, and unstacking after, holds exactly the same total element
        count as the num_chunks separate arrays it replaces — the anti-OOM
        property this class exists for is preserved, nothing (2**n_qubits,)
        shaped is ever materialized."""
        compiled_ops = _compile_multi_chunk_ops(circuit)
        xp = self._chunk_sims[0].xp
        stacked = xp.stack([sim.sv for sim in self._chunk_sims])
        final = self._multi_chunk_runner(stacked, compiled_ops)
        for i, sim in enumerate(self._chunk_sims):
            sim.sv = final[i]

    # ── Public API ───────────────────────────────────────────────────────────

    def run_chunk(
        self,
        circuit: List,
        chunk_size_gates: Optional[int] = None,
    ) -> None:

        if self._chunk_sims is not None:
            self._dispatch_multi(circuit)
            return
        size = chunk_size_gates if chunk_size_gates is not None else self.chunk_size_gates
        self._circuit_chunker.split_circuit(circuit, chunk_size=size)

    def __repr__(self) -> str:
        s = self._guard.status()
        safe_qubits = self._inner_sim.n if self._inner_sim is not None else self._mem_chunker.chunk_size_bits
        return (
            f"Chunk(n_qubits={self.n}, "
            f"safe_qubits={safe_qubits}, "
            f"num_chunks={self.num_chunks}, "
            f"chunk_size_bits={self.chunk_size_bits}, "
            f"dtype={self.dtype}, "
            f"mem_per_chunk={self.memory_mb():.1f} MB, "
            f"ram_free={s['free_pct']:.1f}%, "
            f"has_jax={HAS_JAX})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatibility aliases
# ─────────────────────────────────────────────────────────────────────────────
chunk1           = MemoryChunker
chunk2           = Chunk
Chunk2Incrociato = Chunk
