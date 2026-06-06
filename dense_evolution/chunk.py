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