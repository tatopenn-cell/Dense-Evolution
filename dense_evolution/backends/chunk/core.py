from typing import List, Optional

import numpy as np

from ._engine_imports import DenseSVSimulator
from .guard import HAS_JAX, SafeMemoryGuard
from .geometry import MemoryChunker
from .circuit_chunker import CircuitChunker
from .kernels import (
    _build_multi_chunk_runner, _build_distributed_chunk_runner,
    _compile_multi_chunk_ops,
)

__all__ = ["Chunk", "chunk1", "chunk2", "Chunk2Incrociato"]


# ─────────────────────────────────────────────────────────────────────────────────
# Chunk  (chunk2 / Chunk2Incrociato)
# ─────────────────────────────────────────────────────────────────────────────────

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
    use_float32       : forwarded to DenseSVSimulator
    """

    def __init__(
        self,
        n_qubits: int,
        chunk_size_gates:  int   = 500,
        memory_threshold:  float = 0.15,
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
                DenseSVSimulator(self._mem_chunker.chunk_size_bits, use_float32=use_float32)
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

            # 7b. Distributed (multi-device) runner — built lazily on first
            # run_chunk_distributed() call, not here: it requires
            # jax.device_count() >= num_chunks, which most single-process
            # uses of Chunk will never need or satisfy.
            self._distributed_runner = None
            self._distributed_mesh   = None

    # ── Benchmark-facing attribute forwarding ─────────────────────

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

    # ── Simulator-facing forwarding ─────────────────────────────

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

    # ── Multi-chunk gate dispatch (num_chunks > 1) ───────────────────

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

    # ── Distributed multi-device gate dispatch (issue #1) ────────────

    def dispatch_distributed(self, circuit: List) -> None:
        """Executes *circuit* the same way _dispatch_multi does, but with
        each chunk pinned to its own physical JAX device instead of all
        chunks sharing one process's RAM — see
        _build_distributed_chunk_step/_build_distributed_chunk_runner for
        the kernel. Requires jax.device_count() >= num_chunks (v1 scope:
        exactly one chunk per device); raises RuntimeError otherwise
        rather than silently falling back to the single-process path,
        since that would silently give up the whole point of calling this
        method instead of run_chunk().

        The (num_chunks, chunk_dim) logical array is never materialized
        on any single device here (unlike _dispatch_multi, where it's one
        process's RAM) -- each device holds and ever sees only its own
        (chunk_dim,) row, exchanging edge data with its stride-partner
        device via jax.lax.ppermute inside the kernel, not through this
        Python method."""
        import jax
        if self._chunk_sims is None:
            raise RuntimeError(
                "dispatch_distributed() requires num_chunks > 1 "
                "(this Chunk instance fits in a single chunk)."
            )
        num_chunks = self._mem_chunker.num_chunks
        available = jax.device_count()
        if available < num_chunks:
            raise RuntimeError(
                f"dispatch_distributed() needs >= {num_chunks} JAX devices "
                f"(one per chunk), only {available} available. Force extra "
                f"CPU devices for testing via the XLA_FLAGS environment "
                f"variable: --xla_force_host_platform_device_count=N "
                f"(set before the process starts, JAX's device count is "
                f"fixed at first initialization)."
            )
        if self._distributed_runner is None:
            self._distributed_runner, self._distributed_mesh = _build_distributed_chunk_runner(
                num_chunks, self._m, self._mem_chunker.chunk_size_bits)

        compiled_ops = _compile_multi_chunk_ops(circuit)
        xp = self._chunk_sims[0].xp
        stacked = xp.stack([sim.sv for sim in self._chunk_sims])
        final = self._distributed_runner(stacked, compiled_ops)
        for i, sim in enumerate(self._chunk_sims):
            sim.sv = np.asarray(final[i])

    # ── Public API ───────────────────────────────────────────

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

    def run_chunk_distributed(self, circuit: List) -> None:
        """Like run_chunk(), but dispatches across a real JAX device mesh
        (dispatch_distributed) instead of one process's RAM — issue #1.
        Requires jax.device_count() >= num_chunks; raises RuntimeError
        otherwise (see dispatch_distributed's docstring for how to test
        this with simulated multi-device CPU)."""
        self.dispatch_distributed(circuit)

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


# ─────────────────────────────────────────────────────────────────────────────────
# Backward-compatibility aliases
# ─────────────────────────────────────────────────────────────────────────────────
chunk1           = MemoryChunker
chunk2           = Chunk
Chunk2Incrociato = Chunk
