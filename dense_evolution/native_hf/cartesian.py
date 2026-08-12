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
    in a fixed canonical order (x-major, matching common conventions:
    for p, that's px, py, pz)."""
    return np.array(
        [(lx, degree - lx - lz, lz) for lx in range(degree, -1, -1) for lz in range(degree - lx, -1, -1)],
        dtype=np.int32,
    )
