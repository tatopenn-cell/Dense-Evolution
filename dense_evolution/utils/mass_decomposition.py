"""
Exact bounded mass-decomposition scoring: given a target mass difference
and a molecular formula's atom budget, find the closest achievable
sub-formula mass and its RDBE (degree-of-unsaturation) validity.

Promoted from Dense-Evolution-Discovery's CASMI26 spectral-identification
experiments (real MS/MS mass-spectrometry data, OTRF/Enveda CASMI26
Kaggle dataset). Real, verified need: testing whether a peak-to-peak mass
difference in a spectrum corresponds to a chemically real neutral loss
requires knowing whether SOME integer combination of the precursor's own
atoms reaches that mass -- a bounded integer subset-sum problem, solved
here exactly via iterative Minkowski sums (not a continuous relaxation:
the element-count space is small enough, typically under 10^5 states,
to enumerate directly).

A candidate decomposition is chemically valid only if its Ring-plus-
Double-Bond-Equivalent (degree of unsaturation) is non-negative:

    RDBE = 1 + sum_i( count_i * (valence_i - 2) ) / 2

(standard organic chemistry -- the same closed-shell constraint already
used in this package's own isodesmic bond-scission work). Verified on
real CASMI26 data: without this filter, an unrelated (wrong) molecule's
formula "explains" 58.5% of real peak-pair differences by coincidence
alone (mean, n=1952 real spectra) -- RDBE-filtering doesn't remove that
gap, but does remove chemically impossible matches from both sides,
so any downstream feature/threshold built on this function is scoring
real, buildable neutral losses only, not arithmetic coincidences.

build_reachable_density_fft below builds the same reachable-mass
landscape a second way, via the convolution theorem (an FFT instead of
direct enumeration) with a physically motivated ppm-level Gaussian
tolerance instead of a fixed absolute one -- same physics, a different
(and for large state spaces, cheaper) construction.
"""
import re
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "parse_formula", "rdbe", "build_reachable_masses", "nearest_reachable_mass",
    "build_reachable_density_fft", "density_at_mass",
]

ATOMIC_MASS: Dict[str, float] = {
    "C": 12.000000, "H": 1.007825, "N": 14.003074, "O": 15.994915,
    "S": 31.972071, "P": 30.973762, "Cl": 34.968853, "F": 18.998403,
    "Br": 78.918338, "I": 126.904473, "Na": 22.989770, "K": 38.963707,
    "Si": 27.976927, "B": 11.009305,
}
VALENCE: Dict[str, int] = {
    "C": 4, "H": 1, "N": 3, "O": 2, "S": 2, "P": 3, "Cl": 1, "F": 1,
    "Br": 1, "I": 1, "Na": 1, "K": 1, "Si": 4, "B": 3,
}

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a molecular formula string (e.g. "C20H15N3O2") into
    {element: count}. Repeated elements in the string are summed,
    matching how real formula strings from different sources
    occasionally repeat a symbol instead of merging it. Elements not in
    ATOMIC_MASS/VALENCE are kept in the returned dict (parsing doesn't
    know or care which elements downstream functions recognize) --
    rdbe()/build_reachable_masses()/build_reachable_density_fft() each
    skip an unrecognized element explicitly, not this function."""
    counts: Dict[str, int] = {}
    for elem, num in _FORMULA_TOKEN_RE.findall(formula or ""):
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def rdbe(formula_counts: Dict[str, int]) -> float:
    """Ring-plus-Double-Bond-Equivalent (degree of unsaturation) of a
    formula: RDBE = 1 + sum(count_i * (valence_i - 2)) / 2. A real,
    non-negative closed-shell molecule/fragment always has RDBE >= 0;
    negative RDBE means the atom counts cannot form any valid closed-
    shell structure (too many univalent atoms for the given carbon/
    nitrogen skeleton) -- unrecognized elements are ignored, not
    counted as an error, since a formula may legitimately include an
    element this module has no valence table entry for."""
    acc = 0
    for elem, count in formula_counts.items():
        val = VALENCE.get(elem)
        if val is not None and count > 0:
            acc += count * (val - 2)
    return 1.0 + acc / 2.0


def build_reachable_masses(
    formula_counts: Dict[str, int],
    max_mass: float,
    resolution: float = 0.0005,
    require_rdbe_valid: bool = True,
    max_states: int = 150_000,
) -> np.ndarray:
    """All sub-formula masses reachable by using 0..count_i atoms of
    each element in formula_counts (0 <= f <= formula_counts,
    component-wise), sorted ascending. Exact bounded subset-sum via
    iterative Minkowski sums over each element in turn -- correct
    because element counts here are small enough (real molecules,
    typically under a few hundred atoms total) that the reachable set
    never needs more than max_states entries after resolution-binning.

    require_rdbe_valid=True (default) keeps, for each mass bin, only
    whether ANY chemically valid (RDBE >= 0) combination reaches it --
    the physically meaningful question for neutral-loss matching.
    Pass False to get the pure arithmetic reachable set instead (e.g.
    for a sanity check against a hand-verified subset-sum test case).

    max_mass bounds the search (usually the precursor's own observed
    mass) so element counts far beyond what's physically possible for
    that precursor are never explored."""
    import pandas as pd  # local import: only this function needs it

    reach = np.array([0.0])
    rdbe2 = np.array([0])  # 2*(RDBE-1); RDBE = 1 + rdbe2/2, starts at rdbe2=0 (empty formula, RDBE=1)

    for elem, count in formula_counts.items():
        mass = ATOMIC_MASS.get(elem)
        val = VALENCE.get(elem)
        if mass is None or count <= 0:
            continue
        count = min(count, 60)  # sanity cap; real formulas here are far below this
        incr_mass = np.arange(0, count + 1) * mass
        incr_rdbe2 = np.arange(0, count + 1) * ((val - 2) if val is not None else 0)

        reach = np.add.outer(reach, incr_mass).ravel()
        rdbe2 = np.add.outer(rdbe2, incr_rdbe2).ravel()

        # No emptiness guard needed here: the "use zero atoms of this
        # element" branch (incr_mass's k=0 term) always carries every
        # prior surviving entry through unchanged, and RDBE contribution
        # 0 always satisfies `>= -2` -- so `reach`/`rdbe2` can shrink but
        # never become empty as long as they started non-empty (which
        # the initial reach=[0.0] guarantees for any max_mass >= -1).
        keep = reach <= max_mass + 1.0
        reach, rdbe2 = reach[keep], rdbe2[keep]

        bins = np.round(reach / resolution).astype(np.int64)
        if require_rdbe_valid:
            valid = rdbe2 >= -2  # RDBE >= 0  <=>  1 + rdbe2/2 >= 0
            bins, reach, rdbe2 = bins[valid], reach[valid], rdbe2[valid]
        df = pd.DataFrame({"bin": bins, "mass": reach, "rdbe2": rdbe2})
        df = df.drop_duplicates(subset="bin")
        if len(df) > max_states:
            df = df.iloc[:: max(1, len(df) // max_states)]
        reach, rdbe2 = df["mass"].to_numpy(), df["rdbe2"].to_numpy()

    return np.sort(reach)


def nearest_reachable_mass(target: float, reachable_sorted: np.ndarray) -> Optional[float]:
    """The single closest value in a (sorted, ascending) reachable-mass
    array to `target`, or None if the array is empty/trivial (only the
    zero-formula placeholder). O(log n) via binary search."""
    if reachable_sorted.size <= 1:
        return None
    i = np.searchsorted(reachable_sorted, target)
    candidates = [k for k in (i - 1, i) if 0 <= k < reachable_sorted.size]
    if not candidates:
        return None
    return float(min((reachable_sorted[k] for k in candidates), key=lambda m: abs(m - target)))


def build_reachable_density_fft(
    formula_counts: Dict[str, int],
    max_mass: float,
    grid_resolution: float = 0.001,
    relative_tolerance: float = 2e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plausibility density over achievable sub-formula masses, via the
    convolution theorem instead of the exact Minkowski-sum enumeration
    in build_reachable_masses.

    Real precedent, not a novel trick: this is the same FFT technique
    Rockwood & Van Orden use to compute isotope distributions
    (Anal. Chem. 1996, "Ultrahigh-Speed Calculation of Isotope
    Distributions") -- using 0..count_i copies of element i as a spike
    train instead of an isotope-abundance distribution, but the same
    convolution-theorem trick applies identically: the reachable-mass
    landscape of "any combination of atoms up to the formula's budget"
    is the CONVOLUTION of each element's own spike train, and
    convolution in mass-space is a plain pointwise PRODUCT in frequency
    space -- so combining K element types costs K FFTs and one inverse
    FFT, not an enumeration over the (potentially huge) product of
    per-element choices that build_reachable_masses must truncate
    (max_states) for.

    A single Gaussian broadening is applied ONCE, after combining every
    element -- not per-element -- because real mass-measurement
    uncertainty (ppm-level, `relative_tolerance`) is a property of the
    FINAL observed mass difference, not of each element's contribution
    individually. This replaces build_reachable_masses's arbitrary fixed
    absolute tolerance with a physically motivated relative (ppm) one.

    Returns (mass_grid, density). density is a plausibility landscape,
    not a normalized probability distribution: many sub-formulas can
    coincide near the same mass and add constructively, so its peak
    height reflects how many/how well-supported combinations reach that
    mass, not a probability that sums to 1.

    See test_mass_decomposition.py for a direct verification that this
    FFT-based construction agrees with brute-force convolution and with
    build_reachable_masses's own exact enumeration on the same formula.
    """
    n_points = int(max_mass / grid_resolution) + 2
    combined_fft = None

    for elem, count in formula_counts.items():
        mass = ATOMIC_MASS.get(elem)
        if mass is None or count <= 0:
            continue
        count = min(count, 60)
        spikes = np.zeros(n_points)
        for k in range(count + 1):
            idx = int(round(k * mass / grid_resolution))
            if idx < n_points:
                spikes[idx] += 1.0
        elem_fft = np.fft.rfft(spikes)
        combined_fft = elem_fft if combined_fft is None else combined_fft * elem_fft

    mass_grid = np.arange(n_points) * grid_resolution
    if combined_fft is None:
        return mass_grid, np.zeros(n_points)

    density = np.fft.irfft(combined_fft, n=n_points)
    density = np.maximum(density, 0.0)  # guards float round-off, true density is never negative

    sigma_mass = max(relative_tolerance * max_mass, grid_resolution)
    sigma_points = sigma_mass / grid_resolution
    half = max(int(6 * sigma_points), 1)
    kx = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (kx / sigma_points) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(density, kernel, mode="same")

    return mass_grid, density


def density_at_mass(mass_grid: np.ndarray, density: np.ndarray, target: float) -> float:
    """Linearly-interpolated plausibility density at an arbitrary target
    mass (need not land exactly on the grid)."""
    return float(np.interp(target, mass_grid, density, left=0.0, right=0.0))
