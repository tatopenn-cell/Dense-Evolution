"""Disk-backed statevector overflow for Chunk -- Pednault et al. 2019
(arXiv:1910.09534, "Leveraging Secondary Storage to Simulate Deep
54-qubit Sycamore Circuits"): when num_chunks chunks don't all fit in
RAM at once, keep the idle ones on disk as plain .npy files and only
materialize, as a jax.Array, the small working set an individual gate
actually needs -- one chunk for a local gate, two (an XOR-stride pair)
for a chunk-mixing gate -- instead of the whole (num_chunks, chunk_dim)
stack _dispatch_multi holds in RAM for the entire circuit.

v1 scope (correctness-first, matching run_chunk_distributed's own staged
rollout): processes one chunk / one pair at a time, no batching multiple
pairs into one call for speed. This is strictly slower per gate than the
in-RAM path -- it exists to make otherwise-impossible sizes possible at
all, not to compete with it on speed. See docs/api/chunk.md for the real
distinction and a verified demo.

Every gate is one of the same 6 cases kernels.py's in-RAM
_build_multi_chunk_step already classifies. This module reuses that
classification (not a fresh derivation) via the same needs_comm split
_build_distributed_chunk_step's own docstring documents:
  - 1-qubit, q1 >= m -> LOCAL: needs only this one chunk.
  - 2-qubit, q1 >= m and q2 >= m -> LOCAL: needs only this one chunk.
  - 2-qubit, q1 < m, q2 >= m ("ctrl chunk-select, tgt local") -> LOCAL,
    but conditional on this chunk's OWN absolute index (no partner
    needed -- kernels.py's _case_2q_ctrl_chunk_tgt_local already
    computes exactly this from an explicit `idxc`).
  - everything else (1-qubit q1 < m, or 2-qubit q2 < m) -> MIX: needs
    exactly the XOR-stride partner chunk.
"""
import numpy as np
import jax.numpy as jnp

from .kernels import (
    _gate_matrix_elements, _case_1q_local, _case_2q_local_local,
    _case_2q_ctrl_chunk_tgt_local,
)

__all__ = ["partition_ops_into_phases", "run_disk_overflow_circuit", "LocalPhase", "ConditionalPhase", "MixPhase"]


class LocalPhase:
    """A maximal run of consecutive gates with q1,q2 >= m (or unused) --
    applied to each chunk independently, no partner chunk ever needed."""
    __slots__ = ("ops",)

    def __init__(self, ops):
        self.ops = ops  # list of (g_id, q1, q2, param) python tuples


class ConditionalPhase:
    """A single 2-qubit gate with ctrl chunk-select (q1 < m), tgt local
    (q2 >= m) -- applied to each chunk independently, conditioned on
    that chunk's own absolute index (see _case_2q_ctrl_chunk_tgt_local)."""
    __slots__ = ("op",)

    def __init__(self, op):
        self.op = op  # single (g_id, q1, q2, param)


class MixPhase:
    """A single gate that needs a partner chunk: 1-qubit with q1 < m, or
    2-qubit with q2 < m (covers both "ctrl local/tgt chunk" and "ctrl
    AND tgt both chunk-select"). `stride` is the mixing qubit (q1 for the
    1-qubit case, q2 for the 2-qubit cases) -- chunk i always pairs with
    chunk i ^ (1 << (m - 1 - stride))."""
    __slots__ = ("op", "stride")

    def __init__(self, op, stride):
        self.op = op
        self.stride = stride


def partition_ops_into_phases(compiled_ops, m: int):
    """compiled_ops: the (n_gates, 4) [g_id, q1, q2, param] array
    _compile_multi_chunk_ops already builds (reused as-is). Returns a
    list of LocalPhase/ConditionalPhase/MixPhase in circuit order. Pure
    Python -- runs once per run_chunk() call, not traced/jitted."""
    rows = np.asarray(compiled_ops)
    phases = []
    pending_local = []

    def flush():
        if pending_local:
            phases.append(LocalPhase(list(pending_local)))
            pending_local.clear()

    for row in rows:
        g_id, q1, q2, param = int(row[0]), int(row[1]), int(row[2]), float(row[3])
        is_2q = g_id >= 20
        if not is_2q:
            if q1 < m:
                flush()
                phases.append(MixPhase((g_id, q1, q2, param), q1))
            else:
                pending_local.append((g_id, q1, q2, param))
        elif q1 < m and q2 >= m:
            flush()
            phases.append(ConditionalPhase((g_id, q1, q2, param)))
        elif q2 < m:
            flush()
            phases.append(MixPhase((g_id, q1, q2, param), q2))
        else:
            pending_local.append((g_id, q1, q2, param))
    flush()
    return phases


def _run_local_phase_on_chunk(chunk_arr, ops, m: int, k: int):
    """chunk_arr: (chunk_dim,) array for one chunk. Applies every op via
    the same _case_1q_local/_case_2q_local_local formulas the in-RAM
    kernel uses, as a (1, chunk_dim) batch of size 1 -- both cases never
    reference the batch dimension's meaning, so this is exact, not an
    approximation of the vectorized path."""
    c = chunk_arr[None, :]
    dtype = c.dtype
    for g_id, q1, q2, param in ops:
        g00, g01, g10, g11, u00, u01, u10, u11 = _gate_matrix_elements(g_id, param, dtype)
        if g_id >= 20:
            c = _case_2q_local_local(c, u00, u01, u10, u11, q1, q2, m, k)
        else:
            c = _case_1q_local(c, g00, g01, g10, g11, q1, m, k)
    return c[0]


def _run_conditional_phase_on_chunk(chunk_arr, op, chunk_index: int, m: int, k: int):
    """A single ConditionalPhase gate, applied to one chunk whose real
    absolute index is `chunk_index` (needed for the ctrl-bit decision;
    see _case_2q_ctrl_chunk_tgt_local's own docstring)."""
    g_id, q1, q2, param = op
    dtype = chunk_arr.dtype
    _, _, _, _, u00, u01, u10, u11 = _gate_matrix_elements(g_id, param, dtype)
    c = chunk_arr[None, :]
    idxc = jnp.array([chunk_index], dtype=jnp.int32)
    c = _case_2q_ctrl_chunk_tgt_local(c, u00, u01, u10, u11, q1, q2, m, k, idxc)
    return c[0]


def _mix_pair(row_a, row_b, e00, e01, e10, e11):
    """The 2x2 amplitude-mixing algebra shared by every chunk-select-
    qubit case in kernels.py (_case_1q_chunk / _case_2q_ctrl_local_tgt_chunk
    / _case_2q_both_chunk): row_a is the row on the "mask/is_c0 = True"
    side, row_b its XOR-stride partner. Mirrors those cases' own
    new0/new1 formula exactly -- new0 = e00*_c + e01*amp_pair evaluated
    at the True-side row (_c=row_a, amp_pair=row_b); new1 = e10*amp_pair
    + e11*_c evaluated at the False-side row, where _c=row_b and
    amp_pair=row_a, i.e. new_b = e10*row_a + e11*row_b."""
    new_a = e00 * row_a + e01 * row_b
    new_b = e10 * row_a + e11 * row_b
    return new_a, new_b


def _run_mix_phase_on_pair(row_a, row_b, op, index_a: int, m: int, k: int):
    """row_a is the chunk whose bit at the gate's own mixing qubit is 0,
    row_b its XOR-stride partner (bit 1) -- the caller picks which
    loaded chunk is "a" vs "b" by checking that bit on the real absolute
    indices before calling this. `index_a` is row_a's real absolute
    chunk index (needed only for case 6's ctrl-chunk-select decision)."""
    g_id, q1, q2, param = op
    dtype = row_a.dtype
    g00, g01, g10, g11, u00, u01, u10, u11 = _gate_matrix_elements(g_id, param, dtype)
    if g_id < 20:
        # case 2: 1-qubit, chunk-select q1 -- unconditional mix.
        return _mix_pair(row_a, row_b, g00, g01, g10, g11)
    if q1 >= m:
        # case 5: ctrl LOCAL (elementwise mask within the row), tgt
        # chunk-select q2 (the mixing qubit).
        ctrl_phys = (k - 1) - (q1 - m)
        idxl = jnp.arange(1 << k, dtype=jnp.int32)
        ctrl_bit = (idxl & (jnp.int32(1) << ctrl_phys)) != 0
        new_a, new_b = _mix_pair(row_a, row_b, u00, u01, u10, u11)
        return jnp.where(ctrl_bit, new_a, row_a), jnp.where(ctrl_bit, new_b, row_b)
    # case 6: ctrl AND tgt both chunk-select; mixing qubit is q2, ctrl
    # decided once from index_a's own bit at q1 (identical for both rows
    # of the pair -- XOR-ing the tgt/q2 stride never touches the q1 bit).
    ctrl_stride = 1 << (m - 1 - q1)
    ctrl_set = (index_a & ctrl_stride) != 0
    new_a, new_b = _mix_pair(row_a, row_b, u00, u01, u10, u11)
    return (new_a, new_b) if ctrl_set else (row_a, row_b)


def run_disk_overflow_circuit(chunk_paths, compiled_ops, m: int, k: int):
    """Runs the compiled circuit against num_chunks chunks stored as
    plain .npy files at `chunk_paths` (index = real chunk index),
    mutating those files in place. Never materializes more than one
    (LocalPhase/ConditionalPhase) or two (MixPhase) chunks as jax.Array
    at once, regardless of num_chunks -- this is the whole point."""
    num_chunks = len(chunk_paths)
    phases = partition_ops_into_phases(compiled_ops, m)

    for phase in phases:
        if isinstance(phase, LocalPhase):
            for i, path in enumerate(chunk_paths):
                arr = jnp.asarray(np.load(path))
                arr = _run_local_phase_on_chunk(arr, phase.ops, m, k)
                np.save(path, np.asarray(arr))

        elif isinstance(phase, ConditionalPhase):
            for i, path in enumerate(chunk_paths):
                arr = jnp.asarray(np.load(path))
                arr = _run_conditional_phase_on_chunk(arr, phase.op, i, m, k)
                np.save(path, np.asarray(arr))

        elif isinstance(phase, MixPhase):
            stride_bits = 1 << (m - 1 - phase.stride)
            done = set()
            for i in range(num_chunks):
                if i in done:
                    continue
                j = i ^ stride_bits
                done.add(i)
                done.add(j)
                row_a = jnp.asarray(np.load(chunk_paths[i]))
                row_b = jnp.asarray(np.load(chunk_paths[j]))
                new_a, new_b = _run_mix_phase_on_pair(row_a, row_b, phase.op, i, m, k)
                np.save(chunk_paths[i], np.asarray(new_a))
                np.save(chunk_paths[j], np.asarray(new_b))
