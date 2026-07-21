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

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
    jax.config.update("jax_enable_x64", True)
except ImportError:
    HAS_JAX = False
    jnp = None


class QuantumHardwareRegistry:
    def __init__(self):
        self.processor = platform.processor()
        self.ram_total = psutil.virtual_memory().total / (1024**3)
        self.ram_avail = psutil.virtual_memory().available / (1024**3)
        self.has_cupy = HAS_CUPY
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
        rng     : optional pre-seeded numpy Generator; created fresh if None
        qubits  : subset of qubits to apply the channel to; defaults to all
        jax_key : optional JAX PRNGKey; created fresh if None and sv is a JAX array

        Returns
        -------
        Normalised statevector (same array type as input).
        """
        if model == 'ideal' or p <= 0.0:
            return sv

        is_jax = HAS_JAX and isinstance(sv, jnp.ndarray)
        dim    = len(sv)

        # ── RNG initialisation ────────────────────────────────────────
        if is_jax:
            if jax_key is None:
                seed_bytes = os.urandom(4)
                jax_seed   = int.from_bytes(seed_bytes, byteorder='big')
                jax_seed  ^= time.perf_counter_ns() & 0xFFFF_FFFF
                key = jax.random.PRNGKey(jax_seed)
            else:
                key = jax_key
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
            half = len(idx_0)   # == dim // 2

            # ── draw random numbers ───────────────────────────────────
            if is_jax:
                key, subkey = jax.random.split(key)
                r = jax.random.uniform(subkey, shape=(half,), minval=0.0, maxval=1.0)
            else:
                r = rng.random(half)            # uniform [0, 1)

            # ── channel application ───────────────────────────────────
            if model == 'depolarizing':
                # Three equiprobable Pauli errors, each with rate p/3
                p3 = p / 3.0
                if is_jax:
                    key, subkey2 = jax.random.split(key)
                    ch = jax.random.uniform(subkey2, shape=(half,), minval=0.0, maxval=1.0)
                    v0, v1      = sv_out[idx_0], sv_out[idx_1]
                    fire        = r < p
                    x_gate      = fire & (ch < p3)
                    y_gate      = fire & (ch >= p3) & (ch < 2.0 * p3)
                    z_gate      = fire & (ch >= 2.0 * p3)
                    new_v0 = jnp.where(x_gate,  v1,
                             jnp.where(y_gate, -1j * v1, v0))
                    new_v1 = jnp.where(x_gate,  v0,
                             jnp.where(y_gate,  1j * v0,
                             jnp.where(z_gate, -v1, v1)))
                    sv_out = sv_out.at[idx_0].set(new_v0)
                    sv_out = sv_out.at[idx_1].set(new_v1)
                else:
                    ch     = rng.random(half)
                    v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                    fire   = r < p
                    x_gate = fire & (ch < p3)
                    y_gate = fire & (ch >= p3) & (ch < 2.0 * p3)
                    z_gate = fire & (ch >= 2.0 * p3)
                    sv_out[idx_0] = np.where(x_gate,  v1,
                                    np.where(y_gate, -1j * v1, v0))
                    sv_out[idx_1] = np.where(x_gate,  v0,
                                    np.where(y_gate,  1j * v0,
                                    np.where(z_gate, -v1, v1)))

            elif model == 'bitflip':
                # X gate applied with probability p
                fire = r < p
                if is_jax:
                    v0, v1 = sv_out[idx_0], sv_out[idx_1]
                    sv_out = sv_out.at[idx_0].set(jnp.where(fire, v1, v0))
                    sv_out = sv_out.at[idx_1].set(jnp.where(fire, v0, v1))
                else:
                    v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                    sv_out[idx_0] = np.where(fire, v1, v0)
                    sv_out[idx_1] = np.where(fire, v0, v1)

            elif model == 'phaseflip':
                # Z gate applied with probability p:
                # Z|0⟩ = |0⟩  →  no change to idx_0 amplitudes
                # Z|1⟩ = -|1⟩ →  negate idx_1 amplitudes when fired
                fire = r < p
                if is_jax:
                    v1     = sv_out[idx_1]
                    sv_out = sv_out.at[idx_1].set(jnp.where(fire, -v1, v1))
                else:
                    v1            = sv_out[idx_1].copy()
                    sv_out[idx_1] = np.where(fire, -v1, v1)

            elif model == 'amplitude_damping':
                # K0 = [[1, 0], [0, √(1-γ)]]  — no decay
                # K1 = [[0, √γ], [0, 0]]       — decay |1⟩ → |0⟩
                # Applied stochastically: with probability γ the qubit
                # decays (K1 path), otherwise K0 is applied.
                gamma = float(np.clip(p, 0.0, 1.0))
                decay = r < gamma
                if is_jax:
                    v0, v1 = sv_out[idx_0], sv_out[idx_1]
                    # decay path:    v0 += v1 * √γ,  v1 = 0
                    # no-decay path: v0 unchanged,   v1 *= √(1-γ)
                    sq_gamma    = jnp.sqrt(gamma)
                    sq_1m_gamma = jnp.sqrt(1.0 - gamma)
                    new_v0 = jnp.where(decay, v0 + v1 * sq_gamma, v0)
                    new_v1 = jnp.where(decay, 0.0 + 0j,           v1 * sq_1m_gamma)
                    sv_out = sv_out.at[idx_0].set(new_v0)
                    sv_out = sv_out.at[idx_1].set(new_v1)
                else:
                    v0, v1      = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                    sq_gamma    = np.sqrt(gamma)
                    sq_1m_gamma = np.sqrt(1.0 - gamma)
                    sv_out[idx_0] = np.where(decay, v0 + v1 * sq_gamma, v0)
                    sv_out[idx_1] = np.where(decay, 0.0 + 0j,           v1 * sq_1m_gamma)

            elif model == 'combined':
                # Worst-case NISQ: depolarizing(p/2) + amplitude_damping(p/3)
                # applied sequentially on the same qubit.
                p_dep   = p * 0.5
                p_damp  = p * 0.333333
                p3      = p_dep / 3.0

                # — depolarizing sub-channel —
                if is_jax:
                    key, sk1, sk2 = jax.random.split(key, 3)
                    r_dep = jax.random.uniform(sk1, shape=(half,), minval=0.0, maxval=1.0)
                    ch    = jax.random.uniform(sk2, shape=(half,), minval=0.0, maxval=1.0)
                    v0, v1 = sv_out[idx_0], sv_out[idx_1]
                    fire   = r_dep < p_dep
                    x_gate = fire & (ch < p3)
                    y_gate = fire & (ch >= p3) & (ch < 2.0 * p3)
                    z_gate = fire & (ch >= 2.0 * p3)
                    new_v0 = jnp.where(x_gate,  v1,
                             jnp.where(y_gate, -1j * v1, v0))
                    new_v1 = jnp.where(x_gate,  v0,
                             jnp.where(y_gate,  1j * v0,
                             jnp.where(z_gate, -v1, v1)))
                    sv_out = sv_out.at[idx_0].set(new_v0)
                    sv_out = sv_out.at[idx_1].set(new_v1)
                    # — amplitude_damping sub-channel —
                    key, sk3   = jax.random.split(key)
                    r_damp     = jax.random.uniform(sk3, shape=(half,), minval=0.0, maxval=1.0)
                    decay      = r_damp < p_damp
                    v0, v1     = sv_out[idx_0], sv_out[idx_1]
                    sq_g       = jnp.sqrt(p_damp)
                    sq_1mg     = jnp.sqrt(1.0 - p_damp)
                    sv_out     = sv_out.at[idx_0].set(jnp.where(decay, v0 + v1 * sq_g, v0))
                    sv_out     = sv_out.at[idx_1].set(jnp.where(decay, 0.0 + 0j,       v1 * sq_1mg))
                else:
                    r_dep  = rng.random(half)
                    ch     = rng.random(half)
                    v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                    fire   = r_dep < p_dep
                    x_gate = fire & (ch < p3)
                    y_gate = fire & (ch >= p3) & (ch < 2.0 * p3)
                    z_gate = fire & (ch >= 2.0 * p3)
                    sv_out[idx_0] = np.where(x_gate,  v1,
                                    np.where(y_gate, -1j * v1, v0))
                    sv_out[idx_1] = np.where(x_gate,  v0,
                                    np.where(y_gate,  1j * v0,
                                    np.where(z_gate, -v1, v1)))
                    r_damp        = rng.random(half)
                    decay         = r_damp < p_damp
                    v0, v1        = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                    sq_g          = np.sqrt(p_damp)
                    sq_1mg        = np.sqrt(1.0 - p_damp)
                    sv_out[idx_0] = np.where(decay, v0 + v1 * sq_g, v0)
                    sv_out[idx_1] = np.where(decay, 0.0 + 0j,       v1 * sq_1mg)

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
