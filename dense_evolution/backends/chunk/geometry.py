from typing import Tuple

import numpy as np
import jax.numpy as jnp

from .guard import _device_memory_budget_bytes, HAS_JAX

__all__ = ["MemoryChunker", "get_dynamic_chunk"]


def get_dynamic_chunk(dtype_target) -> int:
    available_bytes, _source = _device_memory_budget_bytes()
    safe_bytes = available_bytes * 0.85
    if HAS_JAX and dtype_target is jnp.complex128:
        bpe = 16
    elif dtype_target is np.complex128:
        bpe = 16
    else:
        bpe = 8
    max_elements = safe_bytes / bpe
    max_bits = int(np.floor(np.log2(max(max_elements, 2.0))))
    return max(16, min(max_bits, 27))


def _dtype_for_qubits(n_qubits: int):
    xp = jnp if HAS_JAX else np
    return xp.complex64 if n_qubits > 26 else xp.complex128


# ─────────────────────────────────────────────────────────────────────────────────
# MemoryChunker  (chunk1)
# ─────────────────────────────────────────────────────────────────────────────────

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
        # Looked up through the package's own top-level attribute, not
        # this module's local `get_dynamic_chunk` name directly -- tests
        # (and any caller) override chunk sizing via
        # `dense_evolution.chunk.get_dynamic_chunk = ...`, the package's
        # public attribute; a deferred `from . import` re-reads that
        # attribute at call time instead of capturing this module's own
        # function object once at import time, so an override actually
        # takes effect on the next MemoryChunker() constructed.
        from . import get_dynamic_chunk as _get_dynamic_chunk
        self.n_qubits        = n_qubits
        self.dtype           = _dtype_for_qubits(n_qubits)
        self.chunk_size_bits = _get_dynamic_chunk(self.dtype)

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
