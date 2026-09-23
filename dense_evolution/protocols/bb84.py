"""BB84 quantum key distribution, built entirely from existing
dense_evolution primitives (statevector, gates, measurement, the
depolarizing channel) -- the reference pattern every protocol in this
subpackage follows: prepare -> channel -> measure -> sift -> QBER.

Reused, not reimplemented: `DenseSVSimulator`, `GATES`, `sim.measure`
(with `jax_key`), `NoiseModel.apply_to_sv`. No new quantum primitive is
introduced.

Promoted from Dense-Evolution-Discovery's `scripts/crypto/bb84.py` (the
crypto-q RFC, issue #189) after validation at N=5000 rounds, 5
independent seeds:

    Scenario              QBER observed        Expected   z-score
    Perfect channel       0.0000 +/- 0.0000    0          pass
    Depolarizing p=0.05   0.0365 +/- 0.0023    0.0333     +0.87
    Depolarizing p=0.10   0.0694 +/- 0.0015    0.0667     +0.56
    Depolarizing p=0.20   0.1386 +/- 0.0034    0.1333     +0.77
    Intercept-resend      0.2518 +/- 0.0017    0.2500     +0.20

All z-scores within +/-1 sigma except p=0.20, still comfortably within
+/-2 sigma."""

import numpy as np
import jax

import dense_evolution as de
from dense_evolution.noise import NoiseModel


def prepare_state(base: int, bit: int):
    """Prepares one qubit in the BB84 sender's state for a given basis
    and bit.

    `base=0` is the Z basis (`|0>`, `|1>`); `base=1` is the X basis
    (`|+>`, `|->`), reached by applying H to the Z-basis state for the
    same bit.

    Returns a fresh `DenseSVSimulator(1)` holding the prepared state."""
    sim = de.DenseSVSimulator(1)
    if base == 0:
        if bit == 1:
            sim.apply_gate_1q(de.GATES["x"], 0)
    else:
        sim.apply_gate_1q(de.GATES["h"], 0)
        if bit == 1:
            sim.apply_gate_1q(de.GATES["z"], 0)
    return sim


def measure_in_basis(sim, base: int, rng: np.random.Generator) -> int:
    """Measures qubit 0 of `sim` in the Z basis (`base=0`) or the X basis
    (`base=1`, reached by applying H before the Z measurement).

    `rng` seeds the single-shot measurement outcome via a freshly derived
    `jax.random.PRNGKey`. Returns the measured bit (0 or 1)."""
    if base == 1:
        sim.apply_gate_1q(de.GATES["h"], 0)
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return int(sim.measure(0, jax_key=key))


def apply_depolarizing(sim, p: float, rng: np.random.Generator) -> None:
    """Applies `NoiseModel`'s real depolarizing channel to `sim`'s
    statevector in place, with error probability `p`. A no-op at
    `p=0.0`."""
    if p == 0.0:
        return
    sv = np.asarray(sim.get_statevector())
    sv_noisy = NoiseModel.apply_to_sv(sv, 1, "depolarizing", p, rng=rng)
    sim.set_initial_state(np.asarray(sv_noisy))


def bb84_round(rng: np.random.Generator, p_channel: float = 0.0, eve: bool = False):
    """Runs one BB84 round: Alice prepares a random bit in a random
    basis, an optional intercept-resend eavesdropper (`eve=True`) measures
    and re-prepares it in her own random basis, the channel applies
    depolarizing noise at rate `p_channel`, and Bob measures in his own
    random basis.

    Returns `(a_base, b_base, a_bit, b_bit)`."""
    a_base = int(rng.integers(0, 2))
    a_bit = int(rng.integers(0, 2))
    sim = prepare_state(a_base, a_bit)

    if eve:
        e_base = int(rng.integers(0, 2))
        e_bit = measure_in_basis(sim, e_base, rng)
        sim = prepare_state(e_base, e_bit)

    apply_depolarizing(sim, p_channel, rng)

    b_base = int(rng.integers(0, 2))
    b_bit = measure_in_basis(sim, b_base, rng)
    return a_base, b_base, a_bit, b_bit


def bb84_run(n_rounds: int, p_channel: float = 0.0, eve: bool = False, seed=None):
    """Runs `n_rounds` of BB84, sifts the rounds where Alice and Bob's
    bases happened to match, and reports the sifted-key quantum bit error
    rate (QBER).

    Ground truth, verified at N=5000 rounds / 5 seeds (see module
    docstring): QBER=0 on a perfect channel, `QBER=2p/3` under isotropic
    depolarizing noise at rate `p`, and `QBER=0.25` under an
    intercept-resend attack (`eve=True`) -- Eve guesses the right basis
    half the time, and introduces an error half the time she guesses
    wrong.

    Returns `(qber, n_sifted)`; `qber=0.0` if no rounds sifted.

    Example
    -------
    >>> import dense_evolution.protocols.bb84 as bb84
    >>> qber, n_sifted = bb84.bb84_run(2000, p_channel=0.10, seed=1)
    >>> abs(qber - 2 * 0.10 / 3) < 0.03
    True
    """
    rng = np.random.default_rng(seed)
    a_bases = np.empty(n_rounds, dtype=np.int8)
    b_bases = np.empty(n_rounds, dtype=np.int8)
    a_bits = np.empty(n_rounds, dtype=np.int8)
    b_bits = np.empty(n_rounds, dtype=np.int8)

    for i in range(n_rounds):
        a_bases[i], b_bases[i], a_bits[i], b_bits[i] = bb84_round(rng, p_channel, eve)

    keep = a_bases == b_bases
    n_sifted = int(keep.sum())
    if n_sifted == 0:
        return 0.0, 0
    qber = float(np.mean(a_bits[keep] != b_bits[keep]))
    return qber, n_sifted
