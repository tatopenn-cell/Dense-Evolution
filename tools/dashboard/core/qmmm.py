"""
Backward-compatibility re-export. The real implementation moved to
dense_evolution.qmmm.forces (Dense-Evolution issue #283 -- qmmm now lives
in the library, alongside real QM/MM region partitioning/embedding,
instead of being buried inside the dashboard tool). Nothing in dashboard
code needed to change: `from dashboard_core.qmmm import ...` still works.
"""
from dense_evolution.qmmm.forces import (
    ATOMIC_MASSES_AMU, compute_hellmann_feynman_forces, md_step, run_md_trajectory,
    MIN_NUCLEAR_DISTANCE_ANGSTROM,
)

__all__ = [
    'ATOMIC_MASSES_AMU', 'compute_hellmann_feynman_forces', 'md_step', 'run_md_trajectory',
    'MIN_NUCLEAR_DISTANCE_ANGSTROM',
]
