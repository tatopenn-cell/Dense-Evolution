"""
Thin Composer-kernel wrapper around dense_evolution.utils.mass_decomposition
-- see docs/api/mass_decomposition.md for the real, validated CASMI26
provenance behind this. No new chemistry logic lives here.
"""
from dataclasses import dataclass
from typing import Optional

from dense_evolution.utils.mass_decomposition import (
    build_reachable_density_fft, build_reachable_masses, density_at_mass,
    nearest_reachable_mass, parse_formula, rdbe,
)

__all__ = ['MassDecompositionResult', 'run_mass_decomposition']


@dataclass
class MassDecompositionResult:
    formula_counts: dict
    rdbe: float
    nearest_reachable_mass: Optional[float]
    density_at_target: float


def run_mass_decomposition(formula: str, target_mass: float, max_mass: Optional[float] = None) -> MassDecompositionResult:
    """Check whether `target_mass` is a chemically valid, reachable
    sub-formula mass of `formula` -- e.g. is a mass-spectrometry peak-pair
    difference a real neutral loss this precursor's own atoms can produce.

    `nearest_reachable_mass` is the exact answer (Minkowski-sum subset
    sum, RDBE-filtered for chemical validity); `density_at_target` is the
    same reachable-mass landscape via the FFT/convolution-theorem route,
    a plausibility density rather than a 0/1 answer (higher means more,
    better-supported combinations land there, not a probability).
    `max_mass` defaults to `target_mass` itself if not given, since the
    caller's own target is a natural bound on what needs to be searched.
    """
    counts = parse_formula(formula)
    bound = max_mass if max_mass is not None else target_mass
    reach = build_reachable_masses(counts, max_mass=bound)
    mass_grid, density = build_reachable_density_fft(counts, max_mass=bound)
    return MassDecompositionResult(
        formula_counts=counts,
        rdbe=rdbe(counts),
        nearest_reachable_mass=nearest_reachable_mass(target_mass, reach),
        density_at_target=density_at_mass(mass_grid, density, target_mass),
    )
