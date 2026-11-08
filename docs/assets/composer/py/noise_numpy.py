"""The NumPy path of dense_evolution.NoiseModel.apply_to_sv, without JAX, for the page.

dense_evolution/noise/kraus_channels.py and the channels of dense_evolution/noise/kraus/ import JAX
at the top, which the browser does not have. This file is the NumPy branch of those files, line by
line: one stochastic Kraus trajectory, one fire/no-fire draw of ``rng.random()`` per qubit (two for
the depolarizing channel), the same thresholds, the same normalisation. A test runs both with the
same seed and compares the states, so a change of a channel that is not copied here fails.
"""

import numpy as np

MODELS = ["ideal", "depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined"]

_THIRD = 1.0 / 3.0


def _pairs(dim, q):
    step = 1 << q
    all_i = np.arange(dim, dtype=np.intp)
    idx_0 = all_i[(all_i & step) == 0]
    return idx_0, idx_0 | step


def _depolarizing(sv, idx_0, idx_1, p, rng):
    r = rng.random()
    ch = rng.random()
    if r < p:
        if ch < _THIRD:
            sv[idx_0], sv[idx_1] = sv[idx_1].copy(), sv[idx_0].copy()
        elif ch < 2.0 * _THIRD:
            v0, v1 = sv[idx_0].copy(), sv[idx_1].copy()
            sv[idx_0] = -1j * v1
            sv[idx_1] = 1j * v0
        else:
            sv[idx_1] = -sv[idx_1]
    return sv


def _bitflip(sv, idx_0, idx_1, p, rng):
    if rng.random() < p:
        sv[idx_0], sv[idx_1] = sv[idx_1].copy(), sv[idx_0].copy()
    return sv


def _phaseflip(sv, idx_0, idx_1, p, rng):
    if rng.random() < p:
        sv[idx_1] = -sv[idx_1]
    return sv


def _amplitude_damping(sv, idx_0, idx_1, p, rng):
    gamma = float(np.clip(p, 0.0, 1.0))
    r = rng.random()
    v0, v1 = sv[idx_0].copy(), sv[idx_1].copy()
    p1 = float(np.clip(gamma * np.sum(np.abs(v1) ** 2), 0.0, 1.0))
    if r < p1:
        norm_decay = np.sqrt(max(p1, 1e-15))
        sv[idx_0] = v1 * np.sqrt(gamma) / norm_decay
        sv[idx_1] = 0.0
    else:
        norm_no_decay = np.sqrt(max(1.0 - p1, 1e-15))
        sv[idx_0] = v0 / norm_no_decay
        sv[idx_1] = v1 * np.sqrt(1.0 - gamma) / norm_no_decay
    return sv


def _combined(sv, idx_0, idx_1, p, rng):
    sv = _depolarizing(sv, idx_0, idx_1, p * 0.5, rng)
    return _amplitude_damping(sv, idx_0, idx_1, p * 0.333333, rng)


_CHANNELS = {
    "ideal": lambda sv, idx_0, idx_1, p, rng: sv,
    "depolarizing": _depolarizing,
    "bitflip": _bitflip,
    "phaseflip": _phaseflip,
    "amplitude_damping": _amplitude_damping,
    "combined": _combined,
}


def apply_to_sv(sv, n, model, p, rng):
    """One noisy trajectory of ``model`` with probability ``p`` on every qubit of ``sv``."""
    if model not in _CHANNELS:
        raise ValueError(f"unknown noise model {model!r}, one of {MODELS}")
    out = np.array(sv, copy=True)
    dim = len(out)
    for q in range(n):
        idx_0, idx_1 = _pairs(dim, q)
        out = _CHANNELS[model](out, idx_0, idx_1, p, rng)
    return out / (np.linalg.norm(out) + 1e-15)
