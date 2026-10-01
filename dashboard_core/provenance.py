"""
Real hardware-scaling benchmark scan and signed provenance export.

Split out of the former monolithic dashboard_core.py (Phase 1 of the
dashboard refactor) -- pure move, no behavior change.
"""

import hashlib
import time

import numpy as np
import pandas as pd

import dense_evolution as de


def convert_numpy_types_to_python(obj):
    """Recursively converts numpy numeric types to plain Python types so a
    structure is JSON-serializable. Adapted from
    DiagnosticTools._convert_numpy_types_to_python (dash.py:2443)."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_to_python(elem) for elem in obj]
    else:
        return obj


def run_benchmark_scan(q_range=range(2, 15, 2)) -> pd.DataFrame:
    """Real hardware scaling scan: allocates a fresh DenseSVSimulator at each
    qubit count and times a Hadamard-on-every-qubit circuit. Adapted from
    DiagnosticTools.core_trigger_benchmark (dash.py:2460, canonical version).
    Genuinely slow (dense allocation up to 2**14) — caller should run this
    behind an explicit action + spinner, not automatically."""
    import psutil

    processo_os = psutil.Process()
    ram_iniziale_rss = processo_os.memory_info().rss / (1024 ** 2)

    rows = []
    for q in q_range:
        t0 = time.perf_counter()
        test_sim = de.DenseSVSimulator(n_qubits=q)
        circuito_stress = [["h", idx, -1] for idx in range(q)]
        test_sim.run_circuit_jit_beast_mode(circuito_stress)
        t_elapsed = time.perf_counter() - t0

        ram_corrente_rss = processo_os.memory_info().rss / (1024 ** 2)
        delta_ram_rss = max(0.0, ram_corrente_rss - ram_iniziale_rss)

        rows.append({
            'Qubits': q,
            'Hilbert_Dim': 2 ** q,
            'Time_s': t_elapsed,
            'RAM_Sim_MB': test_sim.memory_mb(),
            'Delta_RAM_RSS_MB': delta_ram_rss,
        })

    return pd.DataFrame(rows)


def build_provenance_json(run_history: list) -> bytes:
    """Provenance export (metadata + run history), SHA-256-signed.
    Adapted from DiagnosticTools.core_trigger_export_json (dash.py:2511) —
    returns bytes for st.download_button instead of google.colab.files.download()."""
    import json
    import platform
    import sys
    import psutil

    try:
        py_ver = sys.version.split()[0]
    except Exception:
        py_ver = "3.x-unknown"

    serializable_runs = convert_numpy_types_to_python(run_history)

    provenance_payload = {
        "metadata": {
            "software_signature": f"dense-evolution-{de.__version__}",
            "export_timestamp_utc": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
            "execution_environment": {
                "os": platform.system(),
                "architecture": platform.machine(),
                "python_version": py_ver,
                "hardware": {
                    "cpu_cores_logical": psutil.cpu_count(logical=True),
                    "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                },
            },
        },
        "records": serializable_runs,
    }

    raw_json_bytes = json.dumps(provenance_payload, sort_keys=True, indent=4).encode('utf-8')
    sha256_hash = hashlib.sha256(raw_json_bytes).hexdigest()
    provenance_payload["metadata"]["integrity_sha256"] = sha256_hash

    return json.dumps(provenance_payload, sort_keys=True, indent=4).encode('utf-8')
