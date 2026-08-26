import os
import time
from typing import Optional, List, Dict, Any
import numpy as np

import jax
import jax.numpy as jnp
HAS_JAX = True

from .kraus import ideal, depolarizing, bitflip, phaseflip, amplitude_damping, combined

__all__ = ["NoiseModel"]

_CHANNELS = {
    'ideal': ideal,
    'depolarizing': depolarizing,
    'bitflip': bitflip,
    'phaseflip': phaseflip,
    'amplitude_damping': amplitude_damping,
    'combined': combined,
}


def _fresh_rng() -> np.random.Generator:
    """
    Create a hardware-entropy-seeded RNG.
    Combines os.urandom (CSPRNG) with a high-resolution nanosecond counter
    so two calls within the same microsecond still differ.
    """
    entropy_bytes = os.urandom(8)
    entropy_int   = int.from_bytes(entropy_bytes, byteorder='big')
    ns_counter    = time.perf_counter_ns() & 0xFFFF_FFFF_FFFF_FFFF
    seed          = (entropy_int ^ ns_counter) & 0xFFFF_FFFF_FFFF_FFFF
    return np.random.default_rng(seed)


def _qubit_index_pairs(dim: int, q: int):
    """
    Return (idx_0, idx_1) — two integer arrays of length dim/2 — where
    idx_0[i] has bit q == 0 and idx_1[i] = idx_0[i] | (1 << q).
    """
    step   = 1 << q
    all_i  = np.arange(dim, dtype=np.intp)
    idx_0  = all_i[(all_i & step) == 0]          # shape: (dim//2,)
    idx_1  = idx_0 | step                          # shape: (dim//2,)
    return idx_0, idx_1


class NoiseModel:
    """
    Stochastic single-qubit Kraus channels applied directly to a statevector.
    Each channel is a separate, importable module under `dense_evolution.
    noise.kraus` (`dense_evolution.noise.kraus.depolarizing`, etc.) --
    this class is the shared dispatcher: RNG/key setup, the per-qubit
    loop, and final normalisation, common to every channel.

    All channels are mathematically correct Kraus maps:
    - trace is preserved (normalisation enforced at the end)
    - phaseflip applies Z with probability p per qubit (non-deterministic)
    - amplitude_damping applies the correct K0/K1 Kraus operators
    - combined is a true worst-case NISQ mixture of all three Pauli errors
      plus amplitude damping

    Supported models
    ----------------
    'ideal'            identity — no modification
    'depolarizing'     {√(1-p)I, √(p/3)X, √(p/3)Y, √(p/3)Z}
    'bitflip'          {√(1-p)I, √p·X}
    'phaseflip'        {√(1-p)I, √p·Z}    ← was broken, now fixed
    'amplitude_damping'{K0=diag(1,√(1-γ)), K1=[[0,√γ],[0,0]]}
    'combined'         depolarizing(p/2) + amplitude_damping(p/3), renormalised

    Every channel draws one fire/no-fire decision per qubit per shot
    (plus one Pauli choice for depolarizing/combined's depolarizing
    sub-step), applied identically across the whole statevector -- the
    same single-Pauli-per-qubit-per-shot convention STIM's
    DEPOLARIZE1(p) uses. Prior to v8.1.57, every channel instead drew
    2**(n-1) INDEPENDENT decisions per qubit per shot, one per amplitude
    pair (i.e. one per branch of the other n-1 qubits) -- inert on a
    product state, but on an entangled state it over-decohered any
    coherence-sensitive (off-diagonal) observable, up to hundreds of
    sigma vs the exact density-matrix Kraus-sum result on test cases
    (e.g. per-branch sampling dropped a measured value from 1.0 to 0.31
    at p=0.15 on one such test -- see the v8.1.57 changelog entry for
    the full reproduction).
    """

    MODELS = ['ideal', 'depolarizing', 'bitflip', 'phaseflip',
              'amplitude_damping', 'combined']

    @staticmethod
    def apply_to_sv(
        sv:       np.ndarray,
        n:        int,
        model:    str,
        p:        float,
        rng:      Optional[np.random.Generator] = None,
        qubits:   Optional[List[int]] = None,
        jax_key:  Optional[Any] = None,
    ) -> np.ndarray:
        """
        Apply a stochastic Kraus channel to statevector *sv* in-place
        (numpy path) or via functional updates (JAX path).

        Parameters
        ----------
        sv      : complex statevector of length 2**n
        n       : number of qubits
        model   : one of NoiseModel.MODELS
        p       : error probability (or damping rate γ for amplitude_damping)
        rng     : optional pre-seeded numpy Generator. Used directly when
                  *sv* is a NumPy array. When *sv* is a JAX array, `rng`
                  used to be silently ignored in favor of `jax_key` (or a
                  non-reproducible OS-entropy key if that was also None
                  -- issue #7); it is now used to *derive* a reproducible
                  jax_key (`rng.integers(...)` seeds `jax.random.PRNGKey`)
                  whenever `jax_key` isn't given explicitly, so seeding
                  `rng` has the effect a caller expects on both array
                  types instead of only on one of them.
        qubits  : subset of qubits to apply the channel to; defaults to all
        jax_key : optional JAX PRNGKey, only meaningful when *sv* is a JAX
                  array. Takes precedence over `rng` when both are given
                  (explicit key beats a derived one). Created from OS
                  entropy if neither `jax_key` nor `rng` is given.

        Returns
        -------
        Normalised statevector (same array type as input).

        Examples
        --------
        A real quantum computer is never perfect -- every gate has some chance of
        error. Once you have a statevector from running your own QASM circuit (the
        same circuit as the getting-started example), this function is how you find
        out what a noisy device would have actually given you instead.

        Start from the circuit and statevector you already have:

        >>> import numpy as np
        >>> import dense_evolution as de
        >>> qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; barrier q; cx q[0],q[1]; measure q -> c;'
        >>> circuit = de.QASMParser().parse(qasm)
        >>> sim = de.DenseSVSimulator(2)
        >>> sim.run_circuit(circuit.to_tuples())
        >>> sv = np.asarray(sim.get_statevector())

        (the `barrier` is parsed and ignored -- it never becomes a gate tuple, so it
        has no effect on the statevector, only on how the circuit reads.)

        Call `NoiseModel.apply_to_sv` on that same statevector, telling it the
        qubit count, which error model to simulate, and how strong it is:

        >>> from dense_evolution.noise import NoiseModel
        >>> rng = np.random.default_rng(0)
        >>> sv_noisy = NoiseModel.apply_to_sv(sv.copy(), 2, 'depolarizing', 0.1, rng=rng)
        >>> round(float(np.vdot(sv_noisy, sv_noisy).real), 4)  # still a valid, normalised state
        1.0

        `'depolarizing'` above is one of six models; pick any other one the same way,
        by name:

        >>> NoiseModel.MODELS
        ['ideal', 'depolarizing', 'bitflip', 'phaseflip', 'amplitude_damping', 'combined']

        `p` is that model's error probability (or damping rate for
        `'amplitude_damping'`) -- 0.1 above means each qubit has a 10% chance of a
        random Pauli error per call. Run it many times and average (see
        [Density-matrix ZNE healing](../examples.md#density-matrix-zne-healing))
        to see what a real noisy device's *typical* output looks like, not just one
        random draw.
        """
        if model == 'ideal':
            return sv
        try:
            if p <= 0.0:
                return sv
        except jax.errors.TracerBoolConversionError:
            # p is a traced value (e.g. flowing through jax.jit/vmap/grad
            # as a NoiseSpec pytree leaf) -- can't early-exit on a Python
            # bool of it. Falling through is still correct: every channel
            # below already reduces to a no-op at p=0 (`fire = r < p` is
            # always False), this skips only the eager-mode optimization,
            # not correctness.
            pass

        channel = _CHANNELS[model]
        is_jax = HAS_JAX and isinstance(sv, jnp.ndarray)
        dim    = len(sv)

        # ── RNG initialisation ────────────────────────────────────────
        if is_jax:
            if jax_key is not None:
                key = jax_key
            elif rng is not None:
                # Derive a reproducible JAX key from the caller's seeded
                # NumPy generator instead of silently ignoring it -- each
                # call advances `rng`'s state, so a fresh, identically-
                # seeded `rng` reproduces the exact same sequence of keys
                # across separate runs (same guarantee the NumPy path
                # already gives).
                key = jax.random.PRNGKey(int(rng.integers(0, 2**32 - 1)))
            else:
                seed_bytes = os.urandom(4)
                jax_seed   = int.from_bytes(seed_bytes, byteorder='big')
                jax_seed  ^= time.perf_counter_ns() & 0xFFFF_FFFF
                key = jax.random.PRNGKey(jax_seed)
        else:
            key = None
            if rng is None:
                rng = _fresh_rng()

        target_qubits = qubits if qubits is not None else list(range(n))
        sv_out = sv  # JAX: functional; NumPy: will be modified in-place copy

        if not is_jax:
            sv_out = sv.copy()  # never mutate the caller's array

        for q in target_qubits:
            idx_0, idx_1 = _qubit_index_pairs(dim, q)
            sv_out, key = channel.apply(sv_out, idx_0, idx_1, p, rng, key, is_jax)

        # ── normalise ─────────────────────────────────────────────────
        if is_jax:
            norm = jnp.linalg.norm(sv_out)
            return sv_out / (norm + 1e-15)
        else:
            norm = np.linalg.norm(sv_out)
            return sv_out / (norm + 1e-15)

    @staticmethod
    def kraus_description(model: str) -> Dict:
        """Human-readable Kraus-operator formula and physical meaning for
        one of `NoiseModel.MODELS`.

        Examples
        --------
        >>> from dense_evolution.noise import NoiseModel
        >>> NoiseModel.kraus_description('bitflip')['physical']
        'Bit flip σ_x with probability p'
        """
        desc = {
            'ideal': {
                'kraus': 1,
                'formula': 'K₀ = I',
                'physical': 'No noise',
            },
            'depolarizing': {
                'kraus': 4,
                'formula': 'K₀=√(1-p)I  K₁=√(p/3)X  K₂=√(p/3)Y  K₃=√(p/3)Z',
                'physical': 'Isotropic Pauli error — equiprobable X, Y, Z',
            },
            'bitflip': {
                'kraus': 2,
                'formula': 'K₀=√(1-p)I  K₁=√p·X',
                'physical': 'Bit flip σ_x with probability p',
            },
            'phaseflip': {
                'kraus': 2,
                'formula': 'K₀=√(1-p)I  K₁=√p·Z',
                'physical': 'Pure dephasing σ_z with probability p',
            },
            'amplitude_damping': {
                'kraus': 2,
                'formula': 'K₀=diag(1,√(1-γ))  K₁=[[0,√γ],[0,0]]',
                'physical': 'T₁ energy relaxation |1⟩→|0⟩ with rate γ',
            },
            'combined': {
                'kraus': 6,
                'formula': 'Depolarizing(p/2) ∘ AmplitudeDamping(p/3)',
                'physical': 'Worst-case NISQ: dephasing + relaxation',
            },
        }
        return desc.get(model, desc['ideal'])
