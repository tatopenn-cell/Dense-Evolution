"""
AI middleware — routes quantum-simulator telemetry through the same
NaN/Inf-safe vector-healing shield used standalone on the Vector Healing
page (ia_utils.vector_healing.enhanced_dense_healing_hybrid), so it's real
shared infrastructure rather than two disconnected demos.

Only DataFrames with a genuine (n_steps, hidden_dim) shape make sense here —
VQE telemetry (6 columns) and MD telemetry (8 columns) both qualify; a
statevector's flat probability array (2**n_qubits) does not.
"""

import pandas as pd

from ia_utils.vector_healing import enhanced_dense_healing_hybrid


def heal_telemetry(df: pd.DataFrame, median_fallback_threshold: float = 0.1):
    """Runs `df` through enhanced_dense_healing_hybrid and returns
    (healed_df, metadata). Empty/None input passes through with a neutral
    metadata dict (no NaN-safety claim to make about zero rows)."""
    if df is None or df.empty:
        return (df if df is not None else pd.DataFrame()), {
            'fallback_triggered': False,
            'adaptive_radius_used': 0,
            'reconstruction_error': 0.0,
        }

    healed_values, metadata = enhanced_dense_healing_hybrid(
        df.to_numpy(dtype=float), median_fallback_threshold=median_fallback_threshold,
    )
    healed_df = pd.DataFrame(healed_values, columns=df.columns, index=df.index)
    return healed_df, metadata
