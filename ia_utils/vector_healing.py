import numpy as np
from scipy.ndimage import median_filter

def median_healing(vettori: np.ndarray, radius_baseline: int = None) -> (np.ndarray, int):
    """
    Applica un filtro mediano avanzato ai vettori.

    Questo metodo calcola un raggio per il filtro mediano dinamicamente, se non specificato,
    e utilizza `scipy.ndimage.median_filter` per un'applicazione efficiente. Gestisce i
    bordi della sequenza tramite padding 'nearest' e preprocessa i vettori per gestire
    valori `np.nan` e `np.inf` prima dell'applicazione del filtro.

    Args:
        vettori (np.ndarray): Array di vettori di hidden states (n_tokens, hidden_dim).
        radius_baseline (int, optional): Raggio fisso per il calcolo della mediana.
                                         Se `None`, il raggio viene calcolato dinamicamente
                                         come `min(20, max(3, n_tokens // 3))`.
                                         Defaults to None.

    Returns:
        tuple: Contiene:
            - np.ndarray: Vettori con filtro mediano applicato, della stessa shape dell'input.
            - int: Il raggio effettivamente utilizzato per il filtro mediano.
    """
    vettori = np.asarray(vettori)
    n, hidden_dim = vettori.shape

    if n == 0:
        return np.empty((0, hidden_dim)), 0

    processed_vettori = np.copy(vettori)
    processed_vettori[np.isinf(processed_vettori)] = np.nan

    # A column that's entirely NaN makes np.nanmean raise "RuntimeWarning:
    # Mean of empty slice" and return NaN for it (silently caught by the
    # next line, which zeroes it anyway) -- pre-replacing whole all-NaN
    # columns with 0.0 means nanmean never sees an empty slice, so the
    # warning never fires, with byte-identical output to before.
    all_nan_cols = np.all(np.isnan(processed_vettori), axis=0)
    safe_for_mean = np.where(all_nan_cols, 0.0, processed_vettori)
    col_means = np.nanmean(safe_for_mean, axis=0)
    processed_vettori = np.where(np.isnan(processed_vettori), col_means, processed_vettori)

    if n < 3:
        calculated_radius = 0
        window_size = 1
    elif radius_baseline is None:
        calculated_radius = min(20, max(3, n // 3))
        window_size = 2 * calculated_radius + 1
    else:
        calculated_radius = radius_baseline
        window_size = 2 * calculated_radius + 1

    window_size = max(1, min(window_size, n))

    out = median_filter(processed_vettori, size=(window_size, 1), mode='nearest')

    return out, calculated_radius

def enhanced_dense_healing_hybrid(
    vettori: np.ndarray,
    radius_baseline: int = None,
) -> (np.ndarray, dict):
    """
    Applica una strategia di healing ibrida combinando la logica di dense_evolution
    con un fallback alla mediana, decidendo dinamicamente quale approccio utilizzare.

    Questa funzione preprocessa i vettori per gestire `np.nan` e `np.inf`.
    Include telemetria dettagliata per monitorare il comportamento del processo di healing.

    Args:
        vettori (np.ndarray): Array di vettori di hidden states (n_tokens, hidden_dim).
        radius_baseline (int, optional): Raggio fisso per il calcolo delle baseline (media/mediana).
                                         Se `None`, il raggio viene calcolato dinamicamente
                                         come `min(20, max(3, n_tokens // 3))`.
                                         Defaults to None.

    Returns:
        tuple: Contiene:
            - np.ndarray: Vettori curati, della stessa shape dell'input.
            - dict: Metadati di telemetria contenenti:
                    - 'fallback_triggered' (bool): `True` solo se l'input originale conteneva
                                                   NaN/Inf E il fallback mediano è stato applicato
                                                   almeno una volta per correggerlo. Non riflette
                                                   correzioni del Phi-Trigger su dati validi ma
                                                   "staticamente" rumorosi (nessuna corruzione reale).
                    - 'adaptive_radius_used' (int): Il raggio effettivamente calcolato e applicato.
                    - 'reconstruction_error' (float): La norma media di variazione (errore di ricostruzione)
                                                      introdotta rispetto ai vettori originali (potenzialmente corrotti).
    """
    try:
        import jax.numpy as jnp
        from dense_evolution.healing import (
            calculate_phi_ab,
            calculate_vettore_dinamico,
            evaluate_phi_trigger,
            GLOBAL_CONSTANTS,
        )
    except ImportError as _import_error:
        # jax is a core dependency of dense-evolution (see pyproject.toml),
        # so this shouldn't fail on a normal install -- but ia_utils could
        # be used standalone outside the full package (e.g. a stripped or
        # vendored copy missing dense_evolution.healing, or an environment
        # missing jax), where the bare ModuleNotFoundError gives no hint
        # this module needs them.
        raise ImportError(
            "enhanced_dense_healing_hybrid requires jax and dense_evolution.healing "
            f"(failed to import: {_import_error}). Install the full dense-evolution "
            "package (which depends on jax) to use this function."
        ) from _import_error

    n, hidden_dim = vettori.shape

    if n == 0:
        return np.empty((0, hidden_dim)), {'fallback_triggered': False, 'adaptive_radius_used': 0, 'reconstruction_error': 0.0}

    # Computed on the RAW input, before any sanitization -- fallback_triggered
    # in the returned metadata is gated on this (see below), so it reflects
    # "there was genuine NaN/Inf corruption AND the median fallback fired",
    # not just "the Phi-Trigger's internal heuristic called some row static".
    # That heuristic alone also fires on structurally noisy-but-valid data
    # (e.g. pure IID random input with no coherent trend for it to recognize
    # as genuine motion) -- verified directly: clean random Gaussian input
    # with zero NaN/Inf still tripped the un-gated flag.
    had_nan_or_inf = bool(np.isnan(vettori).any() or np.isinf(vettori).any())

    processed_vettori = np.copy(vettori)
    processed_vettori[np.isinf(processed_vettori)] = np.nan

    # See median_healing's identical block above for why this avoids
    # np.nanmean's "Mean of empty slice" warning on an all-NaN column.
    all_nan_cols = np.all(np.isnan(processed_vettori), axis=0)
    safe_for_mean = np.where(all_nan_cols, 0.0, processed_vettori)
    col_means = np.nanmean(safe_for_mean, axis=0)
    processed_vettori = np.where(np.isnan(processed_vettori), col_means, processed_vettori)

    out = np.copy(processed_vettori)

    if radius_baseline is None:
        if n < 3:
            adaptive_radius_used = 0
        else:
            adaptive_radius_used = min(20, max(3, n // 3))
    else:
        adaptive_radius_used = radius_baseline

    fallback_triggered_at_all = False
    reconstruction_errors_per_step = []

    if n > 0:
        reconstruction_errors_per_step.append(np.linalg.norm(out[0] - processed_vettori[0]))
    if n > 1:
        reconstruction_errors_per_step.append(np.linalg.norm(out[1] - processed_vettori[1]))

    for i in range(2, n):
        lo = max(0, i - adaptive_radius_used)
        baseline_mean = np.mean(processed_vettori[lo:i], axis=0)

        state_A = jnp.array(baseline_mean)
        state_B = jnp.array(processed_vettori[i])

        ipg_raw = processed_vettori[i-1] - processed_vettori[i-2]
        norm_ipg_raw = np.linalg.norm(ipg_raw)
        ipg_vector = jnp.array(ipg_raw / norm_ipg_raw) if norm_ipg_raw > 1e-9 else jnp.array(ipg_raw)

        phi_ab = calculate_phi_ab(state_A, state_B, ipg_vector)
        E_A = jnp.linalg.norm(state_A)
        E_B = jnp.linalg.norm(state_B)

        v_dinamic = calculate_vettore_dinamico(E_A, E_B, phi_ab)
        trigger, _, _ = evaluate_phi_trigger(v_dinamic)

        if float(trigger) > GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A']:
            # trigger == 1.0: ciclo aperto/dinamico -> cambio genuino, si tiene il valore
            healed_vector = processed_vettori[i]
        else:
            # trigger == 0.0: ciclo chiuso/statico -> rumore, si sostituisce con la mediana locale
            healed_vector = np.median(processed_vettori[lo:i], axis=0)
            fallback_triggered_at_all = True

        out[i] = healed_vector
        reconstruction_errors_per_step.append(np.linalg.norm(out[i] - processed_vettori[i]))

    mean_reconstruction_error = np.mean(reconstruction_errors_per_step) if reconstruction_errors_per_step else 0.0

    metadata = {
        'fallback_triggered': fallback_triggered_at_all and had_nan_or_inf,
        'adaptive_radius_used': adaptive_radius_used,
        'reconstruction_error': mean_reconstruction_error,
    }

    return out, metadata
