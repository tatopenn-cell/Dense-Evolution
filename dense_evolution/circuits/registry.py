import subprocess
import sys
import os
import time
import psutil
import platform
from typing import Optional, List, Dict, Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# JAX is now a mandatory dependency of dense_evolution (no numpy fallback
# detection) -- the numpy code paths below are kept as-is for reference/
# reuse, they are just never selected anymore.
import jax
import jax.numpy as jnp
HAS_JAX = True
jax.config.update("jax_enable_x64", True)


class QuantumHardwareRegistry:
    def __init__(self):
        self.processor = platform.processor()
        self.ram_total = psutil.virtual_memory().total / (1024**3)
        self.ram_avail = psutil.virtual_memory().available / (1024**3)
        self.has_jax = HAS_JAX
        self.has_gpu = self._detect_gpu()
        self.max_dense_qubits = self._get_qubit_limit()

    def _detect_gpu(self) -> bool:
        try:
            subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _get_qubit_limit(self) -> int:
        if self.ram_total >= 50: return 28
        elif self.ram_total >= 12: return 24
        return 20

    def print_diagnostics(self):
        print(f"MAX_DENSE={self.max_dense_qubits}q | JAX={self.has_jax} | GPU={self.has_gpu}")


HARDWARE_REGISTRY = QuantumHardwareRegistry()
plt.style.use('dark_background')
matplotlib.rcParams.update({
    'figure.facecolor': '#010409',
    'axes.facecolor': '#0d1117',
    'axes.edgecolor': '#21262d',
    'grid.color': '#21262d',
    'font.family': 'monospace',
    'font.size': 9,
})


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

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

    This is the correct and vectorised way to build qubit-pair indices.
    The original code used `xp.where()` which returns a *tuple*, then
    did `idx_1 = idx_0 | step` on that tuple — producing wrong indices
    for all models and making phaseflip look deterministic.
    """
    step   = 1 << q
    all_i  = np.arange(dim, dtype=np.intp)
    idx_0  = all_i[(all_i & step) == 0]          # shape: (dim//2,)
    idx_1  = idx_0 | step                          # shape: (dim//2,)
    return idx_0, idx_1


# ─────────────────────────────────────────────────────────────────────────────
# NoiseSpec — JAX PyTree wrapper for NoiseModel.apply_to_sv's parameters
# ─────────────────────────────────────────────────────────────────────────────

class NoiseSpec:
    """JAX PyTree wrapper around a NoiseModel configuration, so noise
    parameters can be threaded through jax.jit/jax.grad/jax.vmap natively
    -- e.g. as the `noise=` argument to circuit_to_energy_fn's energy_fn
    -- instead of being applied as an external, Python-side step around
    the already-traced circuit (the old way: build sv, exit the trace,
    call apply_to_sv separately).

    `model`/`qubits` are static (aux_data): they select which code path
    runs, not values to differentiate or batch over -- the same role
    `static_argnames` plays for a plain jax.jit function, but automatic
    here because it's part of the pytree structure. `p`/`jax_key` are
    pytree leaves (children): `p` can be a traced/differentiable value
    (e.g. optimizing noise strength itself), and `jax_key` flows through
    jit/vmap/scan the way any other JAX array does -- no external
    Python-level key management, no OS-entropy fallback (unlike
    apply_to_sv called standalone with jax_key=None), so a NoiseSpec's
    result is always reproducible from the key it was built with.

    `jax_key` is required (not Optional) -- the whole point of wiring
    noise into the traced computation this way is to remove the need for
    an external, ad-hoc key-management workaround; a caller who wants a
    fresh key per call should split one themselves (`jax.random.split`)
    and build a fresh NoiseSpec, the same as any other JAX-idiomatic
    stateless-key pattern.
    """
    def __init__(self, model: str, p, jax_key, qubits: Optional[List[int]] = None):
        self.model = model
        self.p = p
        self.jax_key = jax_key
        self.qubits = tuple(qubits) if qubits is not None else None

    def __repr__(self):
        return f"NoiseSpec(model={self.model!r}, p={self.p!r}, qubits={self.qubits!r})"

    def tree_flatten(self):
        children = (self.p, self.jax_key)
        aux_data = (self.model, self.qubits)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        model, qubits = aux_data
        p, jax_key = children
        return cls(model, p, jax_key, qubits)


jax.tree_util.register_pytree_node(
    NoiseSpec, NoiseSpec.tree_flatten, NoiseSpec.tree_unflatten
)


# ─────────────────────────────────────────────────────────────────────────────
# NoiseModel
# ─────────────────────────────────────────────────────────────────────────────

class NoiseModel:
    """
    Stochastic single-qubit Kraus channels applied directly to a statevector.

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
            if rng is None:
                rng = _fresh_rng()

        target_qubits = qubits if qubits is not None else list(range(n))
        sv_out = sv  # JAX: functional; NumPy: will be modified in-place copy

        if not is_jax:
            sv_out = sv.copy()  # never mutate the caller's array

        for q in target_qubits:
            # ── correct index pair construction ───────────────────────
            idx_0, idx_1 = _qubit_index_pairs(dim, q)

            # ── channel application ───────────────────────────────────
            #
            # BUG FIX (all branches below): every channel used to draw
            # `half` = 2**(n-1) INDEPENDENT fire/no-fire (and, for
            # depolarizing, Pauli-choice) decisions per qubit per shot --
            # one per computational-basis amplitude pair, i.e. one per
            # branch of the OTHER n-1 qubits -- instead of ONE decision
            # per qubit per shot applied uniformly across the whole
            # statevector, which is what a correct Kraus-channel
            # unraveling requires (the convention STIM's DEPOLARIZE1(p)/
            # X_ERROR(p)/Z_ERROR(p) use: one Pauli draw per qubit per
            # shot, applied globally). A single-qubit noise event cannot
            # be correlated with what value the *other* qubits happen to
            # hold in superposition -- that correlation is exactly what
            # per-branch-independent sampling introduces.
            #
            # On a product state this was inert (only one branch has
            # nonzero amplitude). On an entangled/superposed state it
            # acts like measuring the other qubits in the computational
            # basis before deciding the error, over-decohering any
            # observable sensitive to coherence between branches --
            # confirmed for 'depolarizing' at 17-33 sigma vs an exact
            # density-matrix Kraus-sum reference on a GHZ-4 state's XXXX
            # expectation (equivalent to a true depolarizing channel at
            # p_eff ~2.2-2.5x nominal p), and for 'bitflip' even more
            # starkly: <XXXX> must be EXACTLY invariant under a bit-flip
            # channel (X commutes with the X observable) yet the old
            # per-branch sampling dropped it from 1.0 to 0.31 at p=0.15
            # (98-210 sigma). Fixed by drawing one scalar decision per
            # qubit per shot and applying it identically to every branch
            # (idx_0/idx_1 index the qubit's own two values; the fixed
            # code touches them as whole arrays, not element-by-element).
            if model == 'depolarizing':
                # Three equiprobable Pauli errors GIVEN that the channel
                # fired -- the choice AMONG X/Y/Z must be a uniform
                # 1-in-3 pick, independent of p, so ch's thresholds are
                # fixed at 1/3 and 2/3 (see historical note below).
                THIRD = 1.0 / 3.0
                if is_jax:
                    key, sk1, sk2 = jax.random.split(key, 3)
                    r  = jax.random.uniform(sk1, shape=(), minval=0.0, maxval=1.0)
                    ch = jax.random.uniform(sk2, shape=(), minval=0.0, maxval=1.0)
                    fire   = r < p
                    x_gate = fire & (ch < THIRD)
                    y_gate = fire & (ch >= THIRD) & (ch < 2.0 * THIRD)
                    z_gate = fire & (ch >= 2.0 * THIRD)
                    v0, v1 = sv_out[idx_0], sv_out[idx_1]
                    new_v0 = jnp.where(x_gate,  v1,
                             jnp.where(y_gate, -1j * v1, v0))
                    new_v1 = jnp.where(x_gate,  v0,
                             jnp.where(y_gate,  1j * v0,
                             jnp.where(z_gate, -v1, v1)))
                    sv_out = sv_out.at[idx_0].set(new_v0)
                    sv_out = sv_out.at[idx_1].set(new_v1)
                else:
                    r  = rng.random()
                    ch = rng.random()
                    if r < p:
                        if ch < THIRD:
                            sv_out[idx_0], sv_out[idx_1] = sv_out[idx_1].copy(), sv_out[idx_0].copy()
                        elif ch < 2.0 * THIRD:
                            v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                            sv_out[idx_0] = -1j * v1
                            sv_out[idx_1] =  1j * v0
                        else:
                            sv_out[idx_1] = -sv_out[idx_1]

            elif model == 'bitflip':
                # X gate applied with probability p, once per qubit per
                # shot, uniformly across the whole statevector.
                if is_jax:
                    key, subkey = jax.random.split(key)
                    r    = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)
                    fire = r < p
                    v0, v1 = sv_out[idx_0], sv_out[idx_1]
                    sv_out = sv_out.at[idx_0].set(jnp.where(fire, v1, v0))
                    sv_out = sv_out.at[idx_1].set(jnp.where(fire, v0, v1))
                else:
                    r = rng.random()
                    if r < p:
                        sv_out[idx_0], sv_out[idx_1] = sv_out[idx_1].copy(), sv_out[idx_0].copy()

            elif model == 'phaseflip':
                # Z gate applied with probability p, once per qubit per
                # shot:
                # Z|0⟩ = |0⟩  →  no change to idx_0 amplitudes
                # Z|1⟩ = -|1⟩ →  negate idx_1 amplitudes when fired
                if is_jax:
                    key, subkey = jax.random.split(key)
                    r    = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)
                    fire = r < p
                    v1     = sv_out[idx_1]
                    sv_out = sv_out.at[idx_1].set(jnp.where(fire, -v1, v1))
                else:
                    r = rng.random()
                    if r < p:
                        sv_out[idx_1] = -sv_out[idx_1]

            elif model == 'amplitude_damping':
                # K0 = [[1, 0], [0, √(1-γ)]]  — no decay
                # K1 = [[0, √γ], [0, 0]]       — decay |1⟩ → |0⟩
                #
                # Single-trajectory ("quantum jump") unraveling: ONE
                # decay/no-decay decision per qubit per shot, using the
                # Born-rule probability aggregated over the WHOLE
                # statevector, P(K1) = <psi|K1^dagger K1|psi> =
                # γ * Σ_i |v1[i]|^2 (sum over every branch of the other
                # n-1 qubits, not per branch -- see the per-branch-
                # sampling bug note above the 'depolarizing' branch).
                # Cross-checked against John Preskill's Ph219/CS219
                # Chapter 3 (Caltech lecture notes), whose own derivation
                # of this channel (system-environment isometry + partial
                # trace) gives exactly this K0/K1 pair. If it fires, K1
                # collapses the ENTIRE qubit-q=1 branch onto q=0,
                # preserving the other qubits' relative amplitudes
                # (v1[i]/norm, not flattened to unit phase per branch --
                # the previous per-branch fix only got this right for a
                # single isolated qubit, where there is only one branch
                # to preserve). Renormalizing by the GLOBAL P(K1) (or
                # 1-P(K1) for the no-decay branch) here, not a per-branch
                # norm, is what a correct sequential multi-qubit
                # trajectory needs -- the next qubit's Born-rule
                # probability must be computed on a properly
                # renormalized state.
                gamma = float(np.clip(p, 0.0, 1.0))
                if is_jax:
                    key, subkey = jax.random.split(key)
                    r = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)
                    v0, v1 = sv_out[idx_0], sv_out[idx_1]
                    p1 = jnp.clip(gamma * jnp.sum(jnp.abs(v1) ** 2), 0.0, 1.0)
                    decay = r < p1
                    norm_decay    = jnp.sqrt(jnp.maximum(p1, 1e-15))
                    norm_no_decay = jnp.sqrt(jnp.maximum(1.0 - p1, 1e-15))
                    new_v0 = jnp.where(decay, v1 * jnp.sqrt(gamma) / norm_decay, v0 / norm_no_decay)
                    new_v1 = jnp.where(decay, 0.0 + 0j, v1 * jnp.sqrt(1.0 - gamma) / norm_no_decay)
                    sv_out = sv_out.at[idx_0].set(new_v0)
                    sv_out = sv_out.at[idx_1].set(new_v1)
                else:
                    r = rng.random()
                    v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                    p1 = float(np.clip(gamma * np.sum(np.abs(v1) ** 2), 0.0, 1.0))
                    if r < p1:
                        norm_decay = np.sqrt(max(p1, 1e-15))
                        sv_out[idx_0] = v1 * np.sqrt(gamma) / norm_decay
                        sv_out[idx_1] = 0.0
                    else:
                        norm_no_decay = np.sqrt(max(1.0 - p1, 1e-15))
                        sv_out[idx_0] = v0 / norm_no_decay
                        sv_out[idx_1] = v1 * np.sqrt(1.0 - gamma) / norm_no_decay

            elif model == 'combined':
                # Worst-case NISQ: depolarizing(p/2) then amplitude_damping(p/3)
                # applied sequentially on the same qubit, each as a single
                # per-qubit-per-shot decision (see the notes on the
                # 'depolarizing' and 'amplitude_damping' branches above --
                # this sub-channel used to share the same per-branch bug,
                # plus its own amplitude-damping sub-step never replaced
                # (only added to) the decayed |0> amplitude and never
                # renormalized the no-decay branch at all).
                p_dep   = p * 0.5
                p_damp  = p * 0.333333
                THIRD   = 1.0 / 3.0  # see the 'depolarizing' branch above for why
                                     # this must be fixed at 1/3, not p_dep/3

                # — depolarizing sub-channel —
                if is_jax:
                    key, sk1, sk2 = jax.random.split(key, 3)
                    r_dep = jax.random.uniform(sk1, shape=(), minval=0.0, maxval=1.0)
                    ch    = jax.random.uniform(sk2, shape=(), minval=0.0, maxval=1.0)
                    fire   = r_dep < p_dep
                    x_gate = fire & (ch < THIRD)
                    y_gate = fire & (ch >= THIRD) & (ch < 2.0 * THIRD)
                    z_gate = fire & (ch >= 2.0 * THIRD)
                    v0, v1 = sv_out[idx_0], sv_out[idx_1]
                    new_v0 = jnp.where(x_gate,  v1,
                             jnp.where(y_gate, -1j * v1, v0))
                    new_v1 = jnp.where(x_gate,  v0,
                             jnp.where(y_gate,  1j * v0,
                             jnp.where(z_gate, -v1, v1)))
                    sv_out = sv_out.at[idx_0].set(new_v0)
                    sv_out = sv_out.at[idx_1].set(new_v1)
                    # — amplitude_damping sub-channel, global Born rule —
                    key, sk3 = jax.random.split(key)
                    r_damp = jax.random.uniform(sk3, shape=(), minval=0.0, maxval=1.0)
                    v0, v1 = sv_out[idx_0], sv_out[idx_1]
                    p1 = jnp.clip(p_damp * jnp.sum(jnp.abs(v1) ** 2), 0.0, 1.0)
                    decay = r_damp < p1
                    norm_decay    = jnp.sqrt(jnp.maximum(p1, 1e-15))
                    norm_no_decay = jnp.sqrt(jnp.maximum(1.0 - p1, 1e-15))
                    new_v0d = jnp.where(decay, v1 * jnp.sqrt(p_damp) / norm_decay, v0 / norm_no_decay)
                    new_v1d = jnp.where(decay, 0.0 + 0j, v1 * jnp.sqrt(1.0 - p_damp) / norm_no_decay)
                    sv_out = sv_out.at[idx_0].set(new_v0d)
                    sv_out = sv_out.at[idx_1].set(new_v1d)
                else:
                    r_dep = rng.random()
                    ch    = rng.random()
                    if r_dep < p_dep:
                        if ch < THIRD:
                            sv_out[idx_0], sv_out[idx_1] = sv_out[idx_1].copy(), sv_out[idx_0].copy()
                        elif ch < 2.0 * THIRD:
                            v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                            sv_out[idx_0] = -1j * v1
                            sv_out[idx_1] =  1j * v0
                        else:
                            sv_out[idx_1] = -sv_out[idx_1]
                    r_damp = rng.random()
                    v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                    p1 = float(np.clip(p_damp * np.sum(np.abs(v1) ** 2), 0.0, 1.0))
                    if r_damp < p1:
                        norm_decay = np.sqrt(max(p1, 1e-15))
                        sv_out[idx_0] = v1 * np.sqrt(p_damp) / norm_decay
                        sv_out[idx_1] = 0.0
                    else:
                        norm_no_decay = np.sqrt(max(1.0 - p1, 1e-15))
                        sv_out[idx_0] = v0 / norm_no_decay
                        sv_out[idx_1] = v1 * np.sqrt(1.0 - p_damp) / norm_no_decay

        # ── normalise ─────────────────────────────────────────────────
        if is_jax:
            norm = jnp.linalg.norm(sv_out)
            return sv_out / (norm + 1e-15)
        else:
            norm = np.linalg.norm(sv_out)
            return sv_out / (norm + 1e-15)

    @staticmethod
    def kraus_description(model: str) -> Dict:
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
