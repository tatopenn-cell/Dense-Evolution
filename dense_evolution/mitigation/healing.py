"""
dense_evolution.healing -- predictive-healing primitives for noisy vector
sequences (VQE/MD telemetry, quantum state trajectories).

Built empirically, one formula at a time, not derived top-down from a
named theory -- worth being precise about, since one piece of it turns
out to have a real mathematical identity worth naming rather than
leaving implicit: calculate_vettore_dinamico's core term,
log(E_B / E_A), is a log-likelihood ratio -- the same elementary
quantity Kullback-Leibler divergence (relative entropy) is built from
(D_KL(A||B) = sum_x A(x) * log(A(x)/B(x)), a probability-weighted
average of exactly this log-ratio, which calculate_vettore_dinamico
does not compute -- it uses one un-weighted log-ratio between two
scalars, not a full KL divergence over a distribution). Equivalently,
if -log(E) is read as self-information ("surprisal"), log(E_B/E_A) is
the *difference in surprisal* between the two states.

Not every formula in this module carries the same reading -- calculate_
phi_ab's blend of cosine-alignment and Euclidean-distance terms is a
geometric similarity construction, not an information-theoretic one;
naming it as such would be the overclaiming this note is trying to
avoid, not fix.
"""

import warnings

import jax
import jax.numpy as jnp
import json
from datetime import datetime, timezone
from typing import Dict, Tuple, List, Any


GLOBAL_CONSTANTS = {
    'TARGET_SIGMA_IDEALE': 10.0,
    'THRESHOLD_DELTA_PREEMP': 0.2,
    'SEMANTIC_LEARNING_RATE': 0.05,
    'V_DINAMIC_K_COEFF': 5.0,
    'V_STATIC_K_PRIME_COEFF': 1.0,
    'V_DINAMIC_MIN_EFFECTIVE_VALUE': 0.01,
    'MAX_SEMANTIC_DISTANCE': jnp.sqrt(2.0),
    'WEIGHT_SEMANTIC': 0.6,
    'WEIGHT_COHERENCE': 0.4,
    'NON_STATIC_THRESHOLD_A': 1e-2,
    'DAMPING_BOOST_ON_STASIS': 0.1,
    'EPSILON_DISSIPATION_BASE': 0.01
}

# =====================================================================
# 📊 STRATO CORE
# =====================================================================

@jax.jit
def _calculate_advanced_sigma_core(kappa: jnp.ndarray, H: jnp.ndarray, Psi: jnp.ndarray, Omega_sync: jnp.ndarray, tau_K: jnp.ndarray) -> jnp.ndarray:
    return kappa * H * Psi * Omega_sync * tau_K


def calculate_advanced_sigma(kappa: jnp.ndarray, H: jnp.ndarray, Psi: jnp.ndarray, Omega_sync: jnp.ndarray, tau_K: jnp.ndarray) -> jnp.ndarray:
    """Deprecated: kappa*H*Psi*Omega_sync*tau_K, intended as the source of
    zero_noise_extrapolation's sigma_at_base_noise (see this module's own
    docs/api/healing.md), but its 5 inputs never had a defined provenance
    in a ZNE context -- and Dense-Evolution-Discovery Experiment 35
    (scripts/zne_healing_sigma_provenance.py) has since shown that even a
    fully-designed input wouldn't matter: a permutation-test negative
    control (real sigma vs. randomly shuffled sigma) performed
    statistically identically, meaning the healing-adapted branch's
    coefficient perturbation doesn't discriminate real signal from noise
    at all. Kept for backward compatibility only (no known external
    callers found in Dense-Evolution, Dense-Evolution-Discovery, or
    Dense-Armor); will be removed in a future major version. This wrapper
    is intentionally NOT @jax.jit-decorated (unlike the private core it
    delegates to) so the warning fires on every call, not just once per
    traced input shape."""
    warnings.warn(
        "calculate_advanced_sigma is deprecated: its output was never wired into "
        "any real pipeline, and Dense-Evolution-Discovery Experiment 35 found the "
        "one place it could have been used (zero_noise_extrapolation's healing "
        "branch) does not discriminate real sigma from random noise anyway. "
        "Will be removed in a future release.",
        DeprecationWarning, stacklevel=2,
    )
    return _calculate_advanced_sigma_core(kappa, H, Psi, Omega_sync, tau_K)

@jax.jit
def calculate_phi_ab(state_A: jnp.ndarray, state_B: jnp.ndarray, ipg_vector: jnp.ndarray) -> jnp.ndarray:
    """Calcola il fattore di allineamento e coerenza spaziale Phi_AB."""
    semantic_change = state_B - state_A
    norm_change = jnp.linalg.norm(semantic_change)
    norm_ipg = jnp.linalg.norm(ipg_vector)

    alignment = jnp.where(
        (norm_change > 1e-12) & (norm_ipg > 1e-12),
        # jnp.dot on complex arrays is the bilinear (non-conjugated) product
        # and stays complex, which used to blow up jnp.clip below with
        # "ValueError: Clip received a complex value". jnp.real(jnp.vdot(..))
        # is the correct Hermitian-inner-product real part -- for real
        # inputs it reduces exactly to jnp.dot (no behavior change for
        # existing real-valued callers), and for complex inputs (e.g. a
        # genuine statevector) it gives the real alignment value this
        # function needs. Re(vdot(a,b)) == Re(vdot(b,a)) always, even
        # though vdot(a,b) != vdot(b,a) in general (they're conjugates) --
        # argument order doesn't matter here only because we take the real part.
        jnp.real(jnp.vdot(semantic_change, ipg_vector)) / (norm_change * norm_ipg),
        0.0
    )
    semantic_alignment = (alignment + 1.0) / 2.0

    distance_A_B = jnp.linalg.norm(state_A - state_B)
    coherence_component = 1.0 - (distance_A_B / GLOBAL_CONSTANTS['MAX_SEMANTIC_DISTANCE'])

    phi_ab = (semantic_alignment * GLOBAL_CONSTANTS['WEIGHT_SEMANTIC']) + (coherence_component * GLOBAL_CONSTANTS['WEIGHT_COHERENCE'])
    return jnp.clip(phi_ab, 0.0, 1.0)

@jax.jit
def calculate_vettore_dinamico(E_A: jnp.ndarray, E_B: jnp.ndarray, Phi_AB: jnp.ndarray) -> jnp.ndarray:
    """Calcola il Vettore Dinamico (V_dinamic) come variazione logaritmica differenziale energetica.

    log(E_B / E_A) is a log-likelihood ratio -- the same elementary
    quantity Kullback-Leibler divergence is built from (see this
    module's docstring for the precise distinction: this is one
    un-weighted log-ratio between two scalars, not a full KL divergence
    over a probability distribution). Equivalently, the difference in
    surprisal (-log E) between the two states."""
    valid_inputs = (E_A > 1e-12) & (E_B > 1e-12)
    ratio = jnp.where(valid_inputs, E_B / E_A, 1.0)
    log_ratio_clamped = jnp.clip(jnp.log(ratio), -5.0, 5.0)
    v_vita = GLOBAL_CONSTANTS['V_DINAMIC_K_COEFF'] * log_ratio_clamped * Phi_AB
    return jnp.where(valid_inputs, v_vita, 0.0)

@jax.jit
def calculate_vettore_statico(v_dinamic_value: jnp.ndarray) -> jnp.ndarray:
    """Calcola l'indicatore di stasi tensoriale Vettore Statico."""
    is_growing = v_dinamic_value > GLOBAL_CONSTANTS['V_DINAMIC_MIN_EFFECTIVE_VALUE']
    return GLOBAL_CONSTANTS['V_STATIC_K_PRIME_COEFF'] * (1.0 - jnp.where(is_growing, 1.0, 0.0))

@jax.jit
def calculate_delta_preemp(current_sigma: jnp.ndarray, target_sigma_ideal: float = 10.0) -> jnp.ndarray:
    """Calcola la deviazione predittiva Delta_Pre_emp normalizzata rispetto all'autostato ideale."""
    safe_target = jnp.where(target_sigma_ideal <= 0.0, 1.0, target_sigma_ideal)
    return jnp.abs(current_sigma - target_sigma_ideal) / safe_target

@jax.jit
def evaluate_phi_trigger(deterministic_dq_dt_a: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Valuta lo stato del Phi-Trigger calcolando i coefficienti di damping condizionati."""
    magnitude_change_a = jnp.abs(deterministic_dq_dt_a)
    trigger_active = magnitude_change_a > GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A']

    lambda_step = jnp.where(trigger_active, 0.05, 0.05 + GLOBAL_CONSTANTS['DAMPING_BOOST_ON_STASIS'])
    epsilon_dissip = jnp.where(trigger_active, GLOBAL_CONSTANTS['EPSILON_DISSIPATION_BASE'],
                                GLOBAL_CONSTANTS['EPSILON_DISSIPATION_BASE'] + GLOBAL_CONSTANTS['DAMPING_BOOST_ON_STASIS'])

    return jnp.where(trigger_active, 1.0, 0.0), lambda_step, epsilon_dissip

@jax.jit
def calculate_jax_reflection(coherence_values: jnp.ndarray, noise_levels: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Esegue l'aggregazione statistica spettrale (Zero-Drift) sul runtime XLA."""
    n_coh = coherence_values.shape[0]
    avg_coherence = jnp.where(n_coh > 0, jnp.mean(coherence_values), 0.0)
    var_coherence = jnp.where(n_coh > 0, jnp.var(coherence_values), 0.0)

    n_noise = noise_levels.shape[0]
    avg_noise = jnp.where(n_noise > 0, jnp.mean(noise_levels), 0.0)

    return avg_coherence, var_coherence, avg_noise

# =====================================================================
#  STRATO OPERATIVO:  LOGGING E STORICIZZAZIONE
# =====================================================================

class MemoryReflectionEngine:
    def __init__(self):
        self.memory: List[Dict[str, Any]] = []

    def record_event(self, event_type: str, value: float, description: str = ""):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "value": float(value),
            "description": description
        }
        self.memory.append(event)

    def reflect(self) -> Dict[str, Any]:
        coherence_list = [e['value'] for e in self.memory if e['type'] == 'coherence']
        interventions = [e for e in self.memory if e['type'] == 'intervention']
        noise_list = [e['value'] for e in self.memory if e['type'] == 'noise']

        j_coherence = jnp.array(coherence_list, dtype=jnp.float64) if coherence_list else jnp.array([], dtype=jnp.float64)
        j_noise = jnp.array(noise_list, dtype=jnp.float64) if noise_list else jnp.array([], dtype=jnp.float64)

        avg_coh, var_coh, avg_noise = calculate_jax_reflection(j_coherence, j_noise)

        report = {
            'average_coherence': float(avg_coh) if coherence_list else None,
            'variance_coherence': float(var_coh) if coherence_list else None,
            'interventions_count': len(interventions),
            'average_noise': float(avg_noise) if noise_list else None,
            'total_events': len(self.memory)
        }
        return report

    def export_memory(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def load_memory(self, filepath: str):
        with open(filepath, 'r') as f:
            self.memory = json.load(f)
