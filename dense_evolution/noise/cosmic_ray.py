"""A cosmic-ray/gamma-ray-induced quasiparticle burst, as a real
time-dependent noise-strength profile instead of a constant rate."""
import jax.numpy as jnp

__all__ = ["cosmic_ray_burst_profile"]


def cosmic_ray_burst_profile(time_us, baseline_gamma: float, ratio_intermediate: float = 2.5,
                              ratio_peak: float = 3.75, tau1_us: float = 3.0,
                              tau2_us: float = 300.0, tau_decay_ms: float = 25.0) -> jnp.ndarray:
    """Time-dependent decay-probability profile for a cosmic-ray/gamma-ray-
    induced quasiparticle burst: a two-stage rise (fast to
    `ratio_intermediate`x baseline, slower to `ratio_peak`x baseline) times
    a single-exponential recovery, generalized out of a fixed, paper-number
    validation (Dense-Evolution-Discovery Experiment 34, reproducing
    arXiv:2104.05219's real measured event on a 26-qubit chip).

    Feed the result to `continuous_dissipative_evolve` alongside
    `amplitude_damping_channel` (or any other single-time-varying-parameter
    channel) to inject a realistic burst into any circuit or QEC study,
    without re-deriving this shape by hand each time.

    The default ratios/timescales are the paper's own real numbers -- see
    Experiment 34's docstring for exactly which are paper-fitted (the 25ms
    decay) versus chosen to match the paper's two described rise points
    (tau1/tau2). All are overridable for a different event severity or
    device generation; `baseline_gamma` is never derived here -- pass
    whatever per-slice decay probability corresponds to your own dt/T1
    convention (see Experiment 34 for one worked example of that
    conversion).

    Parameters
    ----------
    time_us : array_like
        Time since impact, in microseconds (t=0 is the impact instant).
    baseline_gamma : float
        Undisturbed per-slice decay probability; this profile scales it
        up, it does not derive it.
    ratio_intermediate, ratio_peak : float
        Multiplier on `baseline_gamma` at the two described checkpoints
        (paper defaults 2.5=10/4, 3.75=15/4, from Fig. 3's ~10us/~1ms
        readings).
    tau1_us, tau2_us : float
        Rise timescales for the two saturating-exponential stages (paper
        defaults 3, 300 -- chosen to match its ~10us/~1ms descriptions,
        not fitted by the paper itself).
    tau_decay_ms : float
        Recovery time constant (paper default 25 -- its own fitted central
        value, real range 25-30ms across 415 events).

    Returns
    -------
    jnp.ndarray
        Per-slice decay probability at each entry of `time_us`.
    """
    time_us = jnp.asarray(time_us)
    stage1 = (ratio_intermediate - 1.0) * (1.0 - jnp.exp(-time_us / tau1_us))
    stage2 = (ratio_peak - ratio_intermediate) * (1.0 - jnp.exp(-time_us / tau2_us))
    decay = jnp.exp(-time_us / (tau_decay_ms * 1000.0))
    scaling = 1.0 + (stage1 + stage2) * decay
    return baseline_gamma * scaling
