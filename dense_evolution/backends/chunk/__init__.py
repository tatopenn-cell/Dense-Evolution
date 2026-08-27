"""Anti-OOM chunked statevector simulation, split into one file per concern:

    guard.py            SafeMemoryGuard, MemoryPressureError
    geometry.py         MemoryChunker, get_dynamic_chunk (pure arithmetic, no allocation)
    kernels.py          the JAX multi-chunk / distributed-chunk compiled kernels
    circuit_chunker.py  CircuitChunker (single-chunk gate-slicing path)
    core.py             Chunk, the public anti-OOM wrapper

Re-exported here as a flat namespace identical to the pre-split module: this
package is imported by path in several places (dense_evolution.chunk's own
sys.modules self-replacement shim, tools/dashboard/core/system_limits.py) that
reach for names like `.jax`, `.np`, `.psutil`, and `.HAS_JAX` as module
attributes, not just the classes -- so `jax`/`np`/`psutil` are imported here
directly, not just via the submodules that happen to use them.
"""
import gc
import numpy as np
import jax
import jax.numpy as jnp
import psutil

from ._engine_imports import DenseSVSimulator, QuantumTranspiler, GATE_IDS
from .guard import HAS_JAX, MemoryPressureError, SafeMemoryGuard
from .geometry import MemoryChunker, get_dynamic_chunk
from .kernels import (
    _build_multi_chunk_step, _build_multi_chunk_runner,
    _build_distributed_chunk_step, _build_distributed_chunk_runner,
    _compile_multi_chunk_ops,
)
from .circuit_chunker import CircuitChunker
from .core import Chunk, chunk1, chunk2, Chunk2Incrociato

__all__ = [
    "Chunk", "MemoryChunker", "CircuitChunker", "SafeMemoryGuard",
    "MemoryPressureError", "get_dynamic_chunk",
    "chunk1", "chunk2", "Chunk2Incrociato",
]
