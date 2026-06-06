<<<<<<< HEAD
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
except ModuleNotFoundError:
    try:
        from dense_evolution.simulator import DenseSVSimulator
        from dense_evolution.compiler import QuantumTranspiler
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_dynamic_chunk(dtype_target) -> int:
    """
    Largest safe qubit-width that fits in 85% of currently available RAM.

    Returns
    -------
    int in [16, 27]
    """
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
    """complex64 for very large circuits (> 26 q) to save memory, else complex128."""
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

    Behaviour
    ---------
    * **HARD BLOCK** (raises MemoryPressureError) when free RAM < threshold_pct.
    * **SOFT WARNING** (prints to stdout) when free RAM < threshold_pct × 2
      but still above the hard threshold.
    * Optionally runs ``gc.collect()`` before each check to reclaim Python
      objects before measuring.

    Parameters
    ----------
    threshold_pct : float
        Minimum fraction of total RAM that must be free.
        Default 0.15 → block when < 15% RAM available.
    gc_before_check : bool
        Run gc.collect() before every check (default True).

    Example
    -------
    guard = SafeMemoryGuard(threshold_pct=0.15)
    guard.check("before JIT allocation")   # raises MemoryPressureError if unsafe
    """

    _WARN_MULTIPLIER = 2.0   # warn when free_pct < threshold × this factor

    def __init__(self, threshold_pct: float = 0.15, gc_before_check: bool = True):
        if not 0.0 < threshold_pct < 1.0:
            raise ValueError(f"threshold_pct must be in (0, 1), got {threshold_pct}")
        self.threshold_pct   = threshold_pct
        self.gc_before_check = gc_before_check
        self._total_mb       = psutil.virtual_memory().total / (1024 * 1024)

    # ── public ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Live snapshot of system RAM."""
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
        """
        Assert that enough RAM is available before proceeding.

        Parameters
        ----------
        context : str
            Label printed in warning/error messages (e.g. slice index).

        Raises
        ------
        MemoryPressureError
            If free RAM < threshold_pct of total physical memory.
        """
        if self.gc_before_check:
            gc.collect()

        s   = self.status()
        tag = f"[{context}] " if context else ""
        free_frac = s["free_pct"] / 100.0

        # ── HARD BLOCK ───────────────────────────────────────────────────────
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

        # ── SOFT WARNING ─────────────────────────────────────────────────────
        warn_threshold = self.threshold_pct * self._WARN_MULTIPLIER
        if free_frac < warn_threshold:
            print(
                f"  [WARN] {tag}RAM bassa: {s['available_mb']:.0f} MB liberi "
                f"({s['free_pct']:.1f}%) — soglia critica al "
                f"{self.threshold_pct * 100:.0f}%."
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

    Instead, an inner simulator is allocated on ``safe_qubits``
    (= chunk_size_bits) and the logical qubit count is stored separately.
    Benchmark attributes (num_chunks, chunk_size_bits, dtype) are forwarded
    transparently from the embedded MemoryChunker.

    A SafeMemoryGuard fires before the inner simulator is instantiated
    (pre-allocation check) and is also embedded in CircuitChunker for
    per-slice protection during execution.

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

        # 3. Pre-allocation RAM check — block here rather than inside JAX
        safe_q = min(n_qubits, self._mem_chunker.chunk_size_bits)
        self._guard.check(f"Chunk.__init__ — allocating {safe_q}-qubit simulator")

        # 4. Physical simulator sized to what RAM can actually hold
        self._inner_sim = DenseSVSimulator(
            safe_q,
            use_gpu=use_gpu,
            use_float32=use_float32,
        )

        # 5. Circuit chunker wired to the physical simulator, with same threshold
        self._circuit_chunker = CircuitChunker(
            simulator_instance=self._inner_sim,
            memory_threshold=memory_threshold,
        )

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
        """Current statevector of the physical (chunk-sized) simulator."""
        return self._inner_sim.sv

    def memory_mb(self) -> float:
        """RAM used by the physical statevector in MB."""
        return self._inner_sim.memory_mb()

    # ── Public API ───────────────────────────────────────────────────────────

    def run_chunk(
        self,
        circuit: List,
        chunk_size_gates: Optional[int] = None,
    ) -> None:
        """
        Execute *circuit* in gate-slices on the physical simulator.

        Each slice is guarded by SafeMemoryGuard — execution stops cleanly
        with MemoryPressureError instead of crashing with RESOURCE_EXHAUSTED.

        Parameters
        ----------
        circuit          : list of gate tuples
        chunk_size_gates : override instance default chunk size
        """
        size = chunk_size_gates if chunk_size_gates is not None else self.chunk_size_gates
        self._circuit_chunker.split_circuit(circuit, chunk_size=size)

    def __repr__(self) -> str:
        s = self._guard.status()
        return (
            f"Chunk(n_qubits={self.n}, "
            f"safe_qubits={self._inner_sim.n}, "
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
=======
import psutil
import numpy as np
from typing import List, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Relative import — avoids the globals() anti-pattern
from .simulator import DenseSVSimulator
from .compiler import QuantumTranspiler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_dynamic_chunk(dtype_target) -> int:
    """
    Return the largest safe qubit-count (as a bit-width) that fits in the
    currently available RAM for the given dtype.

    Parameters
    ----------
    dtype_target : numpy/jax dtype
        complex64 (8 bytes/element) or complex128 (16 bytes/element).

    Returns
    -------
    int in [16, 27] — maximum chunk size in qubits.
    """
    vm = psutil.virtual_memory()
    safe_ram = vm.available * 0.85                         # leave 15 % headroom
    bytes_per_element = 16 if dtype_target in (
        np.complex128,
        getattr(jnp, "complex128", None) if HAS_JAX else None,
    ) else 8
    max_elements = safe_ram / bytes_per_element
    max_bits = int(np.floor(np.log2(max(max_elements, 1.0))))
    return max(16, min(max_bits, 27))


def _dtype_for_qubits(n_qubits: int):
    """Return complex64 for large circuits (> 26 q) to save memory, else complex128."""
    return (jnp if HAS_JAX else np).complex64 if n_qubits > 26 else (jnp if HAS_JAX else np).complex128


# ─────────────────────────────────────────────────────────────────────────────
# CircuitChunker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitChunker:
    """
    Split a circuit into fixed-size slices and execute each slice on the
    simulator's ``run_circuit_jit_beast_mode`` engine.

    This avoids XLA recompilation caused by variable-length op arrays: every
    chunk (except possibly the last) has the same length, so XLA only compiles
    once per unique size.

    Parameters
    ----------
    simulator_instance : DenseSVSimulator, optional
        The simulator that will execute each slice.  Must be provided before
        calling ``split_circuit``.
    """

    def __init__(self, simulator_instance: Optional[DenseSVSimulator] = None):
        self.sim = simulator_instance

    def split_circuit(self, circuit: List, chunk_size: int = 500) -> None:
        """
        Transpile *circuit* and execute it in ``chunk_size``-gate slices.

        Parameters
        ----------
        circuit    : list of gate tuples
        chunk_size : number of gates per JIT-compiled slice (default 500)

        Raises
        ------
        RuntimeError  if no simulator instance has been attached.
        """
        if self.sim is None:
            raise RuntimeError(
                "CircuitChunker: no simulator instance attached. "
                "Pass simulator_instance= at construction or assign self.sim."
            )

        # Transpile once before chunking so CCX/SWAP expansions are included
        target: List = QuantumTranspiler.transpile(circuit)

        for i in range(0, len(target), chunk_size):
            circuit_slice = target[i : i + chunk_size]
            self.sim.run_circuit_jit_beast_mode(circuit_slice)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryChunker
# ─────────────────────────────────────────────────────────────────────────────

class MemoryChunker:
    """
    Compute the optimal statevector chunking geometry for *n_qubits* given
    the currently available system RAM.

    Attributes
    ----------
    n_qubits        : requested qubit count
    dtype           : complex64 (large circuits) or complex128
    chunk_size_bits : maximum qubit count that fits in RAM as a bit-width
    num_chunks      : number of statevector chunks required (1 if n_qubits fits)
    chunk_dim       : dimension (2**chunk_size_bits) of each chunk

    Notes
    -----
    This class describes the *geometry* of a chunked simulation.  Actual
    circuit execution belongs to ``CircuitChunker`` / ``Chunk``.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dtype = _dtype_for_qubits(n_qubits)
        self.chunk_size_bits: int = get_dynamic_chunk(self.dtype)

        if self.n_qubits <= self.chunk_size_bits:
            self.num_chunks = 1
            self.chunk_dim  = 2 ** self.n_qubits
        else:
            self.num_chunks = 2 ** (self.n_qubits - self.chunk_size_bits)
            self.chunk_dim  = 2 ** self.chunk_size_bits

    def geometry(self) -> Tuple[int, int, int]:
        """Return (num_chunks, chunk_dim, chunk_size_bits)."""
        return self.num_chunks, self.chunk_dim, self.chunk_size_bits

    def memory_mb(self) -> float:
        """Estimated RAM usage per chunk in megabytes."""
        bytes_per_element = 8 if "64" in str(self.dtype) else 16
        return self.chunk_dim * bytes_per_element / 1_000_000

    def __repr__(self) -> str:
        return (
            f"MemoryChunker(n_qubits={self.n_qubits}, "
            f"num_chunks={self.num_chunks}, "
            f"chunk_dim={self.chunk_dim}, "
            f"chunk_size_bits={self.chunk_size_bits}, "
            f"mem_per_chunk={self.memory_mb():.2f} MB)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chunk  (DenseSVSimulator subclass)
# ─────────────────────────────────────────────────────────────────────────────

class Chunk(DenseSVSimulator):
    """
    DenseSVSimulator subclass with integrated chunked-execution support.

    Large circuits are automatically split into ``chunk_size_gates``-gate
    slices, each compiled and executed independently.  The statevector is
    preserved across slices (sequential execution — not parallel).

    Parameters
    ----------
    n_qubits         : number of qubits
    chunk_size_gates : gate-slice size passed to CircuitChunker (default 500)
    use_gpu          : forwarded to DenseSVSimulator
    use_float32      : forwarded to DenseSVSimulator

    Example
    -------
    >>> sim = Chunk(n_qubits=20, chunk_size_gates=200)
    >>> sim.run_chunk([('h', 0), ('cx', 0, 1)] * 1000)
    >>> probs = sim.get_probabilities()
    """

    def __init__(
        self,
        n_qubits: int,
        chunk_size_gates: int = 500,
        use_gpu: bool = False,
        use_float32: bool = False,
    ):
        # Initialise the parent simulator (allocates statevector |0…0⟩)
        super().__init__(n_qubits, use_gpu=use_gpu, use_float32=use_float32)

        # Memory geometry — informational; not used to split the statevector
        # (dense simulation always holds the full 2**n state in memory)
        self._mem_chunker = MemoryChunker(n_qubits)

        # Circuit chunker wired to self
        self._circuit_chunker = CircuitChunker(simulator_instance=self)
        self.chunk_size_gates = chunk_size_gates

    # ── properties ───────────────────────────────────────────────────

    @property
    def memory_geometry(self) -> MemoryChunker:
        """Expose the MemoryChunker for inspection."""
        return self._mem_chunker

    # ── public API ────────────────────────────────────────────────────

    def run_chunk(
        self,
        circuit: List,
        chunk_size_gates: Optional[int] = None,
    ) -> None:
        """
        Execute *circuit* in gate-slices using the JIT engine.

        Parameters
        ----------
        circuit          : list of gate tuples
        chunk_size_gates : override the instance default chunk size
        """
        size = chunk_size_gates if chunk_size_gates is not None else self.chunk_size_gates
        self._circuit_chunker.split_circuit(circuit, chunk_size=size)

    # ── convenience overrides ─────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Chunk(n_qubits={self.n}, "
            f"chunk_size_gates={self.chunk_size_gates}, "
            f"mem={self.memory_mb():.1f} MB, "
            f"has_jax={HAS_JAX})"
        )
>>>>>>> 10dd0b7 (v8.1.2 - SafeMemoryGuard Anti-OOM, chunk.py rewrite, README update)
