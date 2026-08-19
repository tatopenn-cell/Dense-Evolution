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
    trigger_mode: str = 'phi',
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
        trigger_mode (str, optional): Quale meccanismo decide se un dato passo è
                                       movimento genuino (mantenuto com'è) o
                                       rumore/corruzione (sostituito con la mediana
                                       locale). Uno tra:
                                       - 'phi' (default): il Phi-Trigger originale
                                         (dense_evolution.mitigation.healing.evaluate_phi_trigger),
                                         soglia fissa |v_dinamic| > 0.01. Mantenuto
                                         come default per piena compatibilità
                                         all'indietro -- è anche il meccanismo esatto
                                         preso di mira dal red-teaming a gradiente di
                                         ia_utils.adversarial_vector_attack, dato che
                                         calculate_phi_ab/calculate_vettore_dinamico
                                         sono funzioni JAX differenziabili.
                                       - 'adaptive': trigger a deviazione locale
                                         adattiva (MAD), consapevole di NaN/Inf
                                         (Dense-Evolution-Discovery Esperimento 27).
                                         Validato: riduce il tasso di falsi positivi
                                         (sostituzioni su dati rumorosi ma non
                                         corrotti) dall'~90% al ~12%, mantenendo un
                                         tasso di rilevamento delle corruzioni reali
                                         pari o superiore al Phi-Trigger su ogni tipo
                                         testato (picchi singoli, sequenze di NaN,
                                         outlier sparsi, corruzioni combinate). Non
                                         differenziabile (usa np.median/np.std), quindi
                                         il red-teaming a gradiente non si applica
                                         allo stesso modo.
                                       Defaults to 'phi'.

    Returns:
        tuple: Contiene:
            - np.ndarray: Vettori curati, della stessa shape dell'input.
            - dict: Metadati di telemetria contenenti:
                    - 'fallback_triggered' (bool): `True` solo se l'input originale conteneva
                                                   NaN/Inf E il fallback mediano è stato applicato
                                                   almeno una volta per correggerlo. Non riflette
                                                   correzioni del trigger su dati validi ma
                                                   "staticamente" rumorosi (nessuna corruzione reale).
                    - 'adaptive_radius_used' (int): Il raggio effettivamente calcolato e applicato.
                    - 'reconstruction_error' (float): La norma media di variazione (errore di ricostruzione)
                                                      introdotta rispetto ai vettori originali (potenzialmente corrotti).
                    - 'trigger_mode' (str): Il meccanismo di trigger effettivamente usato.
    """
    if trigger_mode not in ('phi', 'adaptive'):
        raise ValueError(f"trigger_mode must be 'phi' or 'adaptive', got {trigger_mode!r}")

    if trigger_mode == 'phi':
        try:
            import jax.numpy as jnp
            from dense_evolution.mitigation.healing import (
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
        return np.empty((0, hidden_dim)), {'fallback_triggered': False, 'adaptive_radius_used': 0,
                                            'reconstruction_error': 0.0, 'trigger_mode': trigger_mode}

    # Computed on the RAW input, before any sanitization -- fallback_triggered
    # in the returned metadata is gated on this (see below), so it reflects
    # "there was genuine NaN/Inf corruption AND the median fallback fired",
    # not just "the trigger's internal heuristic called some row static".
    # The 'phi' heuristic alone also fires on structurally noisy-but-valid
    # data (e.g. pure IID random input with no coherent trend for it to
    # recognize as genuine motion) -- verified directly: clean random
    # Gaussian input with zero NaN/Inf still tripped the un-gated flag.
    had_nan_or_inf = bool(np.isnan(vettori).any() or np.isinf(vettori).any())
    # Per-row raw corruption flag ('adaptive' mode only): forces healing at
    # any row that was originally NaN/Inf, regardless of the deviation
    # statistic -- after NaN/Inf sanitization below, a corrupted row is
    # replaced by the (column-wise) global mean, which can look
    # statistically unremarkable relative to the LOCAL window and evade a
    # purely deviation-based trigger. Verified in Discovery Experiment 27:
    # the adaptive trigger without this row-level override missed 100% of
    # NaN-run corruption for exactly this reason.
    raw_nan_or_inf_row = np.isnan(vettori).any(axis=1) | np.isinf(vettori).any(axis=1)

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

    # BUG FIX (perf): baseline_mean used to be np.mean(processed_vettori[lo:i])
    # recomputed from scratch every iteration -- O(window_size) per step.
    # window_size is capped at 20 for the default adaptive radius, but an
    # explicit radius_baseline (a caller-supplied parameter, unbounded) can
    # make it grow with i, making the whole loop O(n^2) in that case. A
    # sliding-window sum (add the newly-entering element, subtract the
    # element that just fell out of the window) makes each step O(1)
    # amortized regardless of radius_baseline, at the one-time cost of a
    # single O(window_size) sum for the first window.
    window_sum = np.sum(processed_vettori[max(0, 2 - adaptive_radius_used):2], axis=0)
    window_lo = max(0, 2 - adaptive_radius_used)

    # 'adaptive' mode only: running history of each step's own local
    # deviation (norm from its window mean, normalized by sqrt(hidden_dim)),
    # reused as the recent-deviation sample for later steps' adaptive
    # threshold instead of recomputing it from scratch each time.
    deviation_history = []

    for i in range(2, n):
        lo = max(0, i - adaptive_radius_used)
        if i > 2:
            window_sum = window_sum + processed_vettori[i - 1]
            for dropped_idx in range(window_lo, lo):
                window_sum = window_sum - processed_vettori[dropped_idx]
            window_lo = lo
        baseline_mean = window_sum / (i - lo)

        if trigger_mode == 'phi':
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

            # trigger == 1.0: ciclo aperto/dinamico -> cambio genuino, si tiene il valore
            # trigger == 0.0: ciclo chiuso/statico -> rumore, si sostituisce con la mediana locale
            is_dynamic = float(trigger) > GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A']
        else:  # trigger_mode == 'adaptive'
            current_deviation = np.linalg.norm(processed_vettori[i] - baseline_mean) / np.sqrt(hidden_dim)

            recent = deviation_history[-adaptive_radius_used:] if adaptive_radius_used > 0 else []
            if len(recent) > 1:
                recent_arr = np.array(recent)
                local_median = np.median(recent_arr)
                # Median Absolute Deviation, scaled by 1.4826 to be a
                # consistent estimator of std under normality -- robust to
                # the very outliers it is meant to help detect (a raw std
                # over a window containing an outlier is itself inflated by
                # that outlier, which is exactly what let scattered-outlier
                # corruption partially evade a std-based version of this
                # trigger in Discovery Experiment 27).
                local_spread = 1.4826 * np.median(np.abs(recent_arr - local_median))
            else:
                local_median, local_spread = 0.1, 0.05
            adaptive_threshold = max(local_median + 3.5 * local_spread, 0.25)

            is_dynamic = (current_deviation < adaptive_threshold) and not raw_nan_or_inf_row[i]
            deviation_history.append(current_deviation)

        if is_dynamic:
            healed_vector = processed_vettori[i]
        else:
            healed_vector = np.median(processed_vettori[lo:i], axis=0)
            fallback_triggered_at_all = True

        out[i] = healed_vector
        reconstruction_errors_per_step.append(np.linalg.norm(out[i] - processed_vettori[i]))

    mean_reconstruction_error = np.mean(reconstruction_errors_per_step) if reconstruction_errors_per_step else 0.0

    metadata = {
        'fallback_triggered': fallback_triggered_at_all and had_nan_or_inf,
        'adaptive_radius_used': adaptive_radius_used,
        'reconstruction_error': mean_reconstruction_error,
        'trigger_mode': trigger_mode,
    }

    return out, metadata
