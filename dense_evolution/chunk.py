"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.backends.chunk as part of the Phase 2 subpackage split
(see prog.txt). chunk.py was the one module left behind at the package
root when the rest of the split happened (everything else -- simulator,
compiler, gates, trotter, qec, ... -- was already moved with its own
shim); this closes that gap.

Unlike trotter.py/qec.py's shims (which re-export a short, stable public
list), dense_evolution.chunk is imported directly by module path in many
places -- tests/unit/test_chunk.py, tools/dashboard/core/system_limits.py,
research/local_site/app/server.py -- including private helpers like
_compile_multi_chunk_ops, not just the public Chunk class. Re-exporting a
curated name list would silently drop one of those on the next internal
refactor, so instead this shim replaces itself in sys.modules with the
real module object: `dense_evolution.chunk` and
`dense_evolution.backends.chunk` become the exact same module, byte for
byte, not two objects kept in sync by hand.

Import from dense_evolution.backends.chunk directly in new code.
"""
import sys as _sys

from dense_evolution.backends import chunk as _real_chunk

_sys.modules[__name__] = _real_chunk
