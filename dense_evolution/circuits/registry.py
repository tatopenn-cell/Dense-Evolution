import subprocess
import platform
import psutil
import matplotlib
import matplotlib.pyplot as plt

from ..config import ensure_x64

# NoiseModel/NoiseSpec moved to dense_evolution.noise -- this was the
# wrong home for them (this module is hardware-capability detection,
# an unrelated concern). Re-exported here for backward compatibility
# with existing `from dense_evolution.registry import NoiseModel`-style
# imports; new code should import from dense_evolution.noise directly.
from ..noise import NoiseModel, NoiseSpec
from ..noise.kraus_channels import HAS_JAX

__all__ = ["QuantumHardwareRegistry", "NoiseModel", "NoiseSpec", "HAS_JAX"]


class QuantumHardwareRegistry:
    def __init__(self):
        # Lazy, not at import time -- see dense_evolution/config.py.
        ensure_x64()
        self.processor = platform.processor()
        self.ram_total = psutil.virtual_memory().total / (1024**3)
        self.ram_avail = psutil.virtual_memory().available / (1024**3)
        self.has_jax = HAS_JAX
        self.has_gpu = self._detect_gpu()
        self.max_dense_qubits = self._get_qubit_limit()

    def _detect_gpu(self) -> bool:
        try:
            subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _get_qubit_limit(self) -> int:
        if self.ram_total >= 50: return 28
        elif self.ram_total >= 12: return 24
        return 20

    def print_diagnostics(self):
        print(f"MAX_DENSE={self.max_dense_qubits}q | JAX={self.has_jax} | GPU={self.has_gpu}")


# BUG FIX: this module used to instantiate a module-level
# `HARDWARE_REGISTRY = QuantumHardwareRegistry()` singleton here -- never
# referenced anywhere else in the codebase (dead code), but its __init__
# calls ensure_x64() unconditionally, so merely `import dense_evolution`
# still forced jax_enable_x64=True at import time even after PR #130
# supposedly made that lazy (see dense_evolution/config.py) -- the
# laziness moved into ensure_x64() itself, but this eager singleton
# construction defeated it one level up. Removed entirely: nothing
# needs it, and QuantumHardwareRegistry() remains available for any
# caller who actually wants one, constructed on their own schedule.
plt.style.use('dark_background')
matplotlib.rcParams.update({
    'figure.facecolor': '#010409',
    'axes.facecolor': '#0d1117',
    'axes.edgecolor': '#21262d',
    'grid.color': '#21262d',
    'font.family': 'monospace',
    'font.size': 9,
})
