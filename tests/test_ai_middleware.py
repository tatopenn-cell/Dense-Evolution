"""
Tests for ui_pages/ai_middleware.py — the thin adapter that routes Quantum
Simulator telemetry (VQE/MD DataFrames) through the same vector-healing
shield used standalone on the Vector Healing page. No Streamlit import
needed (heal_telemetry is pure pandas/numpy), so this doesn't add a CI
dependency beyond what test_ia_healing.py already requires.
"""

import numpy as np
import pandas as pd

from ui_pages.ai_middleware import heal_telemetry


def test_heal_telemetry_empty_df():
    healed, meta = heal_telemetry(pd.DataFrame())
    assert healed.empty
    assert meta == {'fallback_triggered': False, 'adaptive_radius_used': 0, 'reconstruction_error': 0.0}


def test_heal_telemetry_none():
    healed, meta = heal_telemetry(None)
    assert healed.empty
    assert meta['fallback_triggered'] is False


def test_heal_telemetry_clean_data_no_nan():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.normal(size=(50, 6)),
        columns=["VQE_Energy", "Entropy", "Purity", "Gradient", "Noise_Factor", "Theta_Correction"],
    )
    healed, meta = heal_telemetry(df)
    assert healed.shape == df.shape
    assert list(healed.columns) == list(df.columns)
    assert not healed.isna().any().any()
    assert np.isfinite(meta['reconstruction_error'])


def test_heal_telemetry_removes_injected_nan():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.normal(size=(40, 6)),
        columns=["VQE_Energy", "Entropy", "Purity", "Gradient", "Noise_Factor", "Theta_Correction"],
    )
    df.iloc[10, 2] = np.nan
    df.iloc[20, 4] = np.inf

    healed, meta = heal_telemetry(df)
    assert not healed.isna().any().any()
    assert np.isfinite(healed.to_numpy()).all()
    assert np.isfinite(meta['reconstruction_error'])
    assert not np.isnan(meta['reconstruction_error'])


def test_heal_telemetry_preserves_index():
    df = pd.DataFrame(
        np.random.default_rng(1).normal(size=(10, 3)),
        columns=["A", "B", "C"],
        index=range(100, 110),
    )
    healed, _ = heal_telemetry(df)
    assert list(healed.index) == list(df.index)
