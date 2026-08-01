"""
Synthetic corrupted-sequence generator for the Vector Healing sandbox demo
(ui_pages/vector_healing.py's Streamlit page, and launch_interactive_panel's
ipywidgets equivalent). Deliberately duplicated from that page's private
`_generate_corrupted_sequence` rather than imported -- same reasoning as
`_heal_telemetry` in interactive_panel.py: ui_pages is repo-only, not part
of the installable `dashboard_core` package, so this module can't depend
on it without breaking a bare `pip install dense-evolution`.

Pure numpy, no Streamlit import -- independently testable and reusable by
both frontends.
"""
import numpy as np

__all__ = ['generate_corrupted_sequence']


def generate_corrupted_sequence(n_steps: int, hidden_dim: int, anomaly_pct: float, rng):
    """A synthetic multi-channel sine-wave "hidden state" sequence, plus a
    corrupted copy with a fraction of steps replaced by NaN / Inf / large
    spikes -- the same three failure modes enhanced_dense_healing_hybrid
    is built to catch.

    Parameters
    ----------
    n_steps : number of timesteps / tokens.
    hidden_dim : number of channels (independent sine waves).
    anomaly_pct : percentage (0-100) of steps to corrupt.
    rng : numpy.random.Generator.

    Returns
    -------
    (ideal, corrupted) : two (n_steps, hidden_dim) float arrays.
    """
    t = np.linspace(0, 4 * np.pi, n_steps)
    freqs = rng.uniform(0.8, 1.2, size=hidden_dim)
    phases = rng.uniform(0, 2 * np.pi, size=hidden_dim)
    ideal = np.stack([np.sin(freqs[d] * t + phases[d]) for d in range(hidden_dim)], axis=1)
    corrupted = ideal + rng.normal(0, 0.05, size=ideal.shape)

    n_anomalies = max(1, int(round(n_steps * anomaly_pct / 100.0)))
    anomaly_idx = rng.choice(n_steps, size=min(n_anomalies, n_steps), replace=False)
    kinds = rng.choice(["nan", "inf", "spike"], size=len(anomaly_idx))

    for idx, kind in zip(anomaly_idx, kinds):
        dim = int(rng.integers(0, hidden_dim))
        if kind == "nan":
            corrupted[idx, dim] = np.nan
        elif kind == "inf":
            corrupted[idx, dim] = np.inf if rng.random() < 0.5 else -np.inf
        else:
            corrupted[idx, dim] = ideal[idx, dim] + rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0)

    return ideal, corrupted
