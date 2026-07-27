"""
Synthetic molecular-dynamics telemetry (placeholder, not a real MD/quantum-
chemistry calculation -- documented as such since the original source).

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- pure move, no behavior change.
"""

import numpy as np
import pandas as pd


def run_md_telemetry(md_steps, md_temp):
    """Synthetic MD telemetry generator — adapted verbatim from run_md_simulation_dummy
    (dash.py:1144), explicitly a placeholder for real MD/quantum-chemistry calculations
    in the source itself, not physically simulated data."""
    data = {
        "Step": [], "Energia_VQE_Ha": [], "Entropia_von_Neumann_Bit": [],
        "Purita_Stato": [], "ID_Operatore_ADAPT": [], "Gradiente_Operatore": [],
        "Fattore_Rumore_Termico": [], "Correzione_Variazionale_Theta": [], "Gradiente_Base_Classica": []
    }

    temp_factor = md_temp / 300.0 if md_temp > 0 else 0.1
    temp_factor = np.clip(temp_factor, 0.1, 2.0)

    for step in range(md_steps):
        data["Step"].append(step)

        energy = -25.0 * np.exp(-step / (md_steps / 5.0)) * temp_factor + np.random.uniform(-0.5, 0.5)
        data["Energia_VQE_Ha"].append(energy)

        entropy = 0.5 + 0.5 * (step / md_steps) * temp_factor + np.random.uniform(-0.01, 0.01)
        data["Entropia_von_Neumann_Bit"].append(entropy)

        purity = 0.8 * np.exp(-step / (md_steps / 10.0)) / temp_factor + np.random.uniform(-0.005, 0.005)
        data["Purita_Stato"].append(purity)

        data["ID_Operatore_ADAPT"].append(np.random.randint(0, 3))
        grad = 1.5 * np.exp(-step / (md_steps / 2.0)) * temp_factor + np.random.uniform(-0.05, 0.05)
        data["Gradiente_Operatore"].append(grad)
        data["Fattore_Rumore_Termico"].append(1.0 - (step / md_steps * 0.1) * temp_factor + np.random.uniform(-0.001, 0.001))
        data["Correzione_Variazionale_Theta"].append(0.1 * np.sin(step * 0.01 * temp_factor) + np.random.uniform(-0.005, 0.005))
        data["Gradiente_Base_Classica"].append(grad * 0.8)

    df_md = pd.DataFrame(data)
    df_md.set_index("Step", inplace=True)
    corr_matrix = df_md.corr(method="pearson")
    return df_md, corr_matrix
