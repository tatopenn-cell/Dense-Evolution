"""
Zero-Noise Extrapolation (ZNE) sweep for the dashboard -- Phase 3 of the
dashboard refactor.

Wires dense_evolution.mitigation.zero_noise_extrapolation (added earlier
in this same development effort) into the dashboard, which until now had
no ZNE/predictive-healing feature at all despite the core package having
one. Reuses zero_noise_extrapolation exactly as-is -- no reimplementation
of the Richardson/healing math here, only the plumbing to run a circuit
at several noise scales and feed the results into it.
"""

import numpy as np

import dense_evolution as de

from .simulation_runner import run_simulation


def run_mitigation_sweep(source_mode, circuit_name, qasm_text, noise_model,
                          base_noise_p, shots, seed, use_float32=True, engine='dense',
                          noise_factors=(1.0, 2.0, 3.0),
                          healing_enabled=False, target_sigma_ideal=10.0) -> dict:
    """Runs the same circuit at several noise scales (base_noise_p * factor
    for each factor in noise_factors) and extrapolates to zero noise via
    dense_evolution.mitigation.zero_noise_extrapolation.

    Raises ValueError if noise_model=='ideal' (there is no noise to
    extrapolate away) or if healing_enabled with a noise_factors length
    other than 3 -- the only point count zero_noise_extrapolation's
    healing-adapted path supports; this is a friendlier dashboard-level
    guard around that library-level NotImplementedError, not a new rule.

    sigma_at_base_noise, when healing_enabled, is the shot-noise binomial
    sigma already used and displayed elsewhere in the dashboard
    (build_panel_overview's NISQ Shot Histogram: sigma = sqrt(shots *
    p_max * (1 - p_max)), computed here at the base (1x) noise scale) --
    a pragmatic proxy for the "coherence signal" the healing math expects,
    not a first-principles derivation (there is no VQE/QM-MM telemetry in
    a bare circuit run to derive one from more rigorously).

    Returns a dict: noise_factors, prob_per_scale, fidelity_per_scale,
    prob_zne, fidelity_zne, prob_ideal, healing_enabled, delta_preemp
    (None when healing_enabled is False), n_qubits.
    """
    if noise_model == 'ideal':
        raise ValueError(
            "Zero-Noise Extrapolation richiede un modello di rumore attivo -- "
            "non ha senso estrapolare rumore da un circuito 'ideal'."
        )
    noise_factors = tuple(float(f) for f in noise_factors)
    if healing_enabled and len(noise_factors) != 3:
        raise ValueError(
            f"L'healing predittivo richiede esattamente 3 fattori di rumore, "
            f"non {len(noise_factors)} -- dense_evolution.mitigation."
            f"zero_noise_extrapolation supporta l'healing-adapted path solo "
            f"a 3 punti."
        )

    prob_per_scale = []
    fidelity_per_scale = []
    prob_ideal = None
    n_qubits = None
    sigma_at_base = None

    for i, factor in enumerate(noise_factors):
        res = run_simulation(
            source_mode, circuit_name, qasm_text, noise_model,
            base_noise_p * factor, shots, seed, use_float32=use_float32, engine=engine,
        )
        prob_per_scale.append(res['prob'])
        fidelity_per_scale.append(res['fidelity'])
        if i == 0:
            prob_ideal = res['prob_ideal']
            n_qubits = res['n_qubits']
            if healing_enabled:
                idx_max = int(np.argmax(res['prob']))
                p_max = float(res['prob'][idx_max])
                sigma_at_base = float(np.sqrt(shots * p_max * (1.0 - p_max)))

    delta_preemp = None
    if healing_enabled:
        from dense_evolution.healing import calculate_delta_preemp
        delta_preemp = float(calculate_delta_preemp(sigma_at_base, target_sigma_ideal))

    prob_zne_raw = de.zero_noise_extrapolation(
        prob_per_scale, noise_factors,
        sigma_at_base_noise=sigma_at_base if healing_enabled else None,
        target_sigma_ideal=target_sigma_ideal,
    )
    # Post-processing on the dashboard side only -- zero_noise_extrapolation
    # itself stays a general-purpose numeric primitive that doesn't assume
    # its input is a probability distribution. Richardson extrapolation of
    # a probability vector is not guaranteed to stay non-negative or
    # normalized -- that's a real property of the technique (it's fitting
    # a polynomial through noisy points and evaluating it outside their
    # range), not a bug to hide.
    prob_zne = np.clip(np.asarray(prob_zne_raw), 0.0, None)
    prob_sum = prob_zne.sum()
    if prob_sum > 1e-12:
        prob_zne = prob_zne / prob_sum

    fidelity_zne = float(de.zero_noise_extrapolation(
        fidelity_per_scale, noise_factors,
        sigma_at_base_noise=sigma_at_base if healing_enabled else None,
        target_sigma_ideal=target_sigma_ideal,
    ))

    return {
        'noise_factors': noise_factors,
        'prob_per_scale': prob_per_scale,
        'fidelity_per_scale': fidelity_per_scale,
        'prob_zne': prob_zne,
        'fidelity_zne': fidelity_zne,
        'prob_ideal': prob_ideal,
        'healing_enabled': healing_enabled,
        'delta_preemp': delta_preemp,
        'n_qubits': n_qubits,
    }
