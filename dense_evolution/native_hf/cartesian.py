"""Enumerating the physical (lx,ly,lz) Cartesian components of a shell.

Our integral tensors are shaped (degree+1, degree+1, degree+1, ...) for
convenience, but only the (lx,ly,lz) triples with lx+ly+lz == degree
are physical basis functions -- e.g. for a p shell (degree=1) that's
(1,0,0), (0,1,0), (0,0,1), not e.g. (1,1,0) which the tensor shape
happens to also have room for.
"""

import numpy as np


def cartesian_powers(degree: int) -> np.ndarray:
    """Returns an (M, 3) array of (lx,ly,lz) triples with lx+ly+lz == degree,
    in a fixed canonical order determined by the generator below -- not the
    common lexicographic convention some other codes use. For p (degree=1)
    that's px, pz, py, not px, py, pz (confirmed by calling this function
    directly, not assumed from the shape of the loop); for d (degree=2),
    xx, xz, xy, zz, yz, yy, not the lexicographic xx, xy, xz, yy, yz, zz.
    Internally consistent (cartesian_normalization_ratios and every caller
    in assembly.py use this exact order), but this matters for anything
    that needs to match a DIFFERENT code's AO ordering (e.g. libcint's) --
    see native_hf/libcint_bridge.py's per-degree permutation."""
    return np.array(
        [(lx, degree - lx - lz, lz) for lx in range(degree, -1, -1) for lz in range(degree - lx, -1, -1)],
        dtype=np.int32,
    )


def _double_factorial_odd(n: int) -> float:
    """(2n-1)!! for n >= 0, with the (2*0-1)!! = (-1)!! = 1 convention."""
    result = 1.0
    k = 2 * n - 1
    while k > 1:
        result *= k
        k -= 2
    return result


def cartesian_normalization_ratios(degree: int) -> np.ndarray:
    """Relative normalization of each Cartesian component of a shell of
    the given degree, relative to the (degree,0,0) component -- 1.0 for
    every component when degree<=1 (px, py, pz are equivalent by
    symmetry), but genuinely different starting at degree=2 (e.g. dxy
    needs a larger normalization constant than dxx, since <dxx|dxx> =
    3*<dxy|dxy> for the same exponent).

    Standard result for an unnormalized Cartesian Gaussian primitive
    (x-Rx)^lx (y-Ry)^ly (z-Rz)^lz exp(-a|r-R|^2): its normalization
    constant is proportional to 1/sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!), so
    this ratio -- independent of the exponent, verified numerically
    against overlap_3d's own self-overlap at several exponents -- is
    sqrt((2*degree-1)!! / ((2lx-1)!!(2ly-1)!!(2lz-1)!!)).

    Order matches cartesian_powers(degree) exactly, so callers can zip
    or elementwise-multiply the two directly."""
    powers = cartesian_powers(degree)
    reference = _double_factorial_odd(degree)
    return np.array(
        [
            np.sqrt(reference / (_double_factorial_odd(lx) * _double_factorial_odd(ly) * _double_factorial_odd(lz)))
            for lx, ly, lz in powers
        ]
    )
