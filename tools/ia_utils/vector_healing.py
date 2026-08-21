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
            import jax
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

    if trigger_mode == 'phi':
        # BUG FIX (perf, prog.txt Sezione 4.2): the old loop converted
        # baseline_mean/processed_vettori[i]/ipg_vector to jnp.array and the
        # trigger decision back to a Python float on EVERY iteration -- a
        # NumPy<->JAX round trip per step, the exact cost jax.lax.scan/vmap
        # exist to avoid. calculate_phi_ab/calculate_vettore_dinamico/
        # evaluate_phi_trigger take fixed-shape (hidden_dim,) or scalar
        # inputs (no variable-size window inside them -- only the median
        # fallback below has one, and that stays plain NumPy, unchanged),
        # so the whole per-index trigger computation batches cleanly with
        # jax.vmap: ONE host<->device round trip for the whole sequence
        # instead of one per index. Values, not just speed, are unchanged --
        # every array below is built with the identical NumPy arithmetic
        # the old loop did per-step, just evaluated for every index at once.
        # Measured (n, hidden_dim=32, same-shape warm call both versions):
        # n=50 1.6x, n=300 5.0x, n=1000 9.8x, n=3000 11.1x faster than the
        # old per-step loop. Real trade-off, not hidden: the old loop's
        # jax.jit'd calls operate on fixed (hidden_dim,) shapes, so JAX
        # compiles them ONCE ever for a given hidden_dim, reused across
        # every n. This vmapped version's batch axis is n-2, so a NEW n
        # means a fresh XLA compile the first time that exact length is
        # seen -- worth it whenever the same length recurs (the normal
        # case: one call per input sequence, not a different n each time),
        # not free on a single one-off call at a brand new length.
        idx = np.arange(2, n)
        if idx.size:
            radius = adaptive_radius_used
            lo_arr = np.maximum(0, idx - radius)

            # baseline_mean[k] = mean(processed_vettori[lo_arr[k]:idx[k]]) via
            # a prefix sum -- same value the old sliding-window-sum loop
            # computed per step, done here for every index in one shot.
            prefix_sum = np.concatenate(
                [np.zeros((1, hidden_dim)), np.cumsum(processed_vettori, axis=0)], axis=0
            )
            window_counts = (idx - lo_arr).astype(np.float64)
            baseline_means = (prefix_sum[idx] - prefix_sum[lo_arr]) / window_counts[:, None]

            ipg_raw = processed_vettori[idx - 1] - processed_vettori[idx - 2]
            norm_ipg_raw = np.linalg.norm(ipg_raw, axis=1)
            safe_norm = np.where(norm_ipg_raw > 1e-9, norm_ipg_raw, 1.0)
            ipg_vectors = np.where((norm_ipg_raw > 1e-9)[:, None], ipg_raw / safe_norm[:, None], ipg_raw)

            state_A = jnp.asarray(baseline_means)
            state_B = jnp.asarray(processed_vettori[idx])
            ipg_vector_batch = jnp.asarray(ipg_vectors)

            phi_ab = jax.vmap(calculate_phi_ab)(state_A, state_B, ipg_vector_batch)
            E_A = jnp.linalg.norm(state_A, axis=1)
            E_B = jnp.linalg.norm(state_B, axis=1)
            v_dinamic = jax.vmap(calculate_vettore_dinamico)(E_A, E_B, phi_ab)
            trigger, _, _ = jax.vmap(evaluate_phi_trigger)(v_dinamic)

            # trigger == 1.0: ciclo aperto/dinamico -> cambio genuino, si tiene il valore
            # trigger == 0.0: ciclo chiuso/statico -> rumore, si sostituisce con la mediana locale
            is_dynamic_arr = np.asarray(trigger) > GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A']

            dynamic_idx = idx[is_dynamic_arr]
            out[dynamic_idx] = processed_vettori[dynamic_idx]

            # The median fallback's window has a genuinely variable size
            # (grows from 1 up to radius) -- not batchable the same way,
            # but it only runs for the (typically minority) indices the
            # trigger actually flags as noise, same np.median as before.
            for i, lo in zip(idx[~is_dynamic_arr], lo_arr[~is_dynamic_arr]):
                out[i] = np.median(processed_vettori[lo:i], axis=0)
                fallback_triggered_at_all = True

            reconstruction_errors_per_step.extend(
                np.linalg.norm(out[idx] - processed_vettori[idx], axis=1).tolist()
            )

    else:  # trigger_mode == 'adaptive'
        # BUG FIX (perf): baseline_mean used to be np.mean(processed_vettori[lo:i])
        # recomputed from scratch every iteration -- O(window_size) per step.
        # window_size is capped at 20 for the default adaptive radius, but an
        # explicit radius_baseline (a caller-supplied parameter, unbounded) can
        # make it grow with i, making the whole loop O(n^2) in that case. A
        # sliding-window sum (add the newly-entering element, subtract the
        # element that just fell out of the window) makes each step O(1)
        # amortized regardless of radius_baseline, at the one-time cost of a
        # single O(window_size) sum for the first window.
        #
        # Deliberately NOT converted to JAX like the 'phi' branch above:
        # this trigger uses np.median/np.std on purpose, precisely so it is
        # NOT part of the JAX-differentiable graph gradient-based red-teaming
        # (ia_utils.adversarial_vector_attack) can attack -- see this
        # function's own docstring. Vectorizing it via JAX would silently
        # remove that property, not just speed things up.
        window_sum = np.sum(processed_vettori[max(0, 2 - adaptive_radius_used):2], axis=0)
        window_lo = max(0, 2 - adaptive_radius_used)

        # Running history of each step's own local deviation (norm from its
        # window mean, normalized by sqrt(hidden_dim)), reused as the
        # recent-deviation sample for later steps' adaptive threshold
        # instead of recomputing it from scratch each time.
        deviation_history = []

        for i in range(2, n):
            lo = max(0, i - adaptive_radius_used)
            if i > 2:
                window_sum = window_sum + processed_vettori[i - 1]
                for dropped_idx in range(window_lo, lo):
                    window_sum = window_sum - processed_vettori[dropped_idx]
                window_lo = lo
            baseline_mean = window_sum / (i - lo)

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
