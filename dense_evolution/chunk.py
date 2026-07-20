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
    from gates import GATES, PARAMETRIC_GATES
except ModuleNotFoundError:
    try:
        from dense_evolution.simulator import DenseSVSimulator
        from dense_evolution.compiler import QuantumTranspiler
        from dense_evolution.gates import GATES, PARAMETRIC_GATES
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

        GATES: dict = {}
        PARAMETRIC_GATES: dict = {}

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

    def _resolve_gate(self, cmd) -> Tuple[bool, "np.ndarray", int, Optional[int]]:
        """Mirrors DenseSVSimulator.run_circuit's own name -> matrix dispatch
        (GATES / PARAMETRIC_GATES), deliberately NOT run_circuit_jit_beast_mode's
        GATE_IDS table — that table is missing cy/cp/crz (and others) and
        would silently drop them. Returns (is_2q, matrix, q1, q2_or_None).

        Only the controlled-U 2-qubit gates {cx, cz, cy, cp, crz} are
        supported here — the only 2-qubit gates that can reach execution
        after QuantumTranspiler.transpile (ccx -> 15 native gates, swap ->
        3xCX). Anything else raises rather than silently mishandling a gate
        that isn't actually controlled-U structured (e.g. swap, iswap, ecr)."""
        name = cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()
        args = cmd[1:]
        xp    = self._chunk_sims[0].xp
        dtype = self._chunk_sims[0].dtype
        controlled_u_2q = ('cx', 'cz', 'cy')
        controlled_u_2q_param = ('cp', 'crz')

        if name in GATES:
            mat = xp.array(GATES[name], dtype=dtype)
            if mat.shape == (2, 2):
                return False, mat, int(args[0]), None
            if name in controlled_u_2q:
                return True, mat, int(args[0]), int(args[1])
            raise NotImplementedError(
                f"Chunk multi-chunk dispatch only supports the controlled-U "
                f"2-qubit gates {controlled_u_2q + controlled_u_2q_param} "
                f"(everything else is decomposed by QuantumTranspiler before "
                f"reaching here) — got '{name}'."
            )

        if name in PARAMETRIC_GATES:
            if len(args) == 2:
                mat = xp.array(PARAMETRIC_GATES[name](args[1]), dtype=dtype)
                return False, mat, int(args[0]), None
            if len(args) == 3:
                if name not in controlled_u_2q_param:
                    raise NotImplementedError(
                        f"Chunk multi-chunk dispatch only supports the "
                        f"controlled-U 2-qubit parametric gates "
                        f"{controlled_u_2q_param} — got '{name}'."
                    )
                mat = xp.array(PARAMETRIC_GATES[name](args[2]), dtype=dtype)
                return True, mat, int(args[0]), int(args[1])
            if len(args) == 4:
                mat = xp.array(PARAMETRIC_GATES[name](args[1], args[2], args[3]), dtype=dtype)
                return False, mat, int(args[0]), None

        raise ValueError(f"Unknown or unparseable gate command in multi-chunk circuit: {cmd!r}")

    def _apply_gate_multi(self, is_2q: bool, mat, q1: int, q2: Optional[int]) -> None:
        """Applies one gate across self._chunk_sims. See CHANGELOG / plan for
        the full derivation. `m` = self._m = number of chunk-select qubits
        (the top m logical qubits, indices [0, m)); qubits [m, n) are local
        to a chunk, re-indexed as (q - m) within that chunk's own simulator.
        """
        m = self._m
        for q in ((q1,) if not is_2q else (q1, q2)):
            if not 0 <= q < self.n:
                raise ValueError(f"Qubit index {q} out of range [0, {self.n})")

        chunks = self._chunk_sims
        xp     = chunks[0].xp

        if not is_2q:
            if q1 >= m:
                for sim in chunks:
                    sim.apply_gate_1q(mat, q1 - m)
                return
            # chunk-select 1-qubit gate: mix whole chunk arrays pairwise
            stride = 1 << (m - 1 - q1)
            g00, g01, g10, g11 = mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]
            for c0 in range(self.num_chunks):
                if c0 & stride:
                    continue
                c1 = c0 | stride
                sv0, sv1 = chunks[c0].sv, chunks[c1].sv
                chunks[c0].sv = g00 * sv0 + g01 * sv1
                chunks[c1].sv = g10 * sv0 + g11 * sv1
            return

        # 2-qubit controlled-U: control=q1, target=q2, U = mat[2:, 2:]
        U = mat[2:, 2:]
        u00, u01, u10, u11 = U[0, 0], U[0, 1], U[1, 0], U[1, 1]
        ctrl_local = q1 >= m
        tgt_local  = q2 >= m

        if ctrl_local and tgt_local:
            for sim in chunks:
                sim.apply_gate_2q(mat, q1 - m, q2 - m)

        elif (not ctrl_local) and tgt_local:
            ctrl_stride = 1 << (m - 1 - q1)
            for c in range(self.num_chunks):
                if c & ctrl_stride:
                    chunks[c].apply_gate_1q(U, q2 - m)
                # else: control bit 0 -> identity branch, no change

        elif ctrl_local and (not tgt_local):
            k = self._mem_chunker.chunk_size_bits
            local_ctrl_stride = 1 << (k - 1 - (q1 - m))
            tgt_stride = 1 << (m - 1 - q2)
            idx  = xp.arange(self._mem_chunker.chunk_dim)
            mask = (idx & local_ctrl_stride) != 0
            for c0 in range(self.num_chunks):
                if c0 & tgt_stride:
                    continue
                c1 = c0 | tgt_stride
                sv0, sv1 = chunks[c0].sv, chunks[c1].sv  # snapshot before writing
                mix0 = u00 * sv0 + u01 * sv1
                mix1 = u10 * sv0 + u11 * sv1
                chunks[c0].sv = xp.where(mask, mix0, sv0)
                chunks[c1].sv = xp.where(mask, mix1, sv1)

        else:  # both chunk-select
            ctrl_stride = 1 << (m - 1 - q1)
            tgt_stride  = 1 << (m - 1 - q2)
            for c0 in range(self.num_chunks):
                if c0 & tgt_stride:
                    continue
                c1 = c0 | tgt_stride
                if not (c0 & ctrl_stride):
                    continue  # control bit 0 (same on c0/c1) -> identity
                sv0, sv1 = chunks[c0].sv, chunks[c1].sv
                chunks[c0].sv = u00 * sv0 + u01 * sv1
                chunks[c1].sv = u10 * sv0 + u11 * sv1

    def _dispatch_multi(self, circuit: List) -> None:
        """Executes *circuit* against the num_chunks>1 chunk representation.

        Convention: chunk index `c` (m = self._m bits, MSB-first, same
        n-1-qubit convention as DenseSVSimulator) equals the value of the
        top m logical qubits (indices [0, m)); chunk_sims[c].sv holds the
        chunk_dim amplitudes for the remaining (local) qubits [m, n). This
        makes full_sv.reshape(num_chunks, chunk_dim)[c] == chunk_sims[c].sv
        exactly, since NumPy's row-major reshape splits a (2,)*n tensor on
        the leading axes first — i.e. the most-significant qubits, matching
        this simulator's MSB-first indexing throughout.

        No chunk_size_gates batching here (unlike the num_chunks==1 path via
        CircuitChunker): each per-gate call into a chunk_size_bits-qubit
        DenseSVSimulator is already a single bounded JIT op, so there's no
        JIT-recompilation-on-varying-trace-shape problem to solve."""
        target = QuantumTranspiler.transpile(circuit)
        for cmd in target:
            is_2q, mat, q1, q2 = self._resolve_gate(cmd)
            self._apply_gate_multi(is_2q, mat, q1, q2)

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
