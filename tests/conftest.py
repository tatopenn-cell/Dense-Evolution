"""
Shared fixtures for the whole test suite, auto-loaded by pytest for every
file in this directory -- extracted from the original monolithic
test_dense_evolution.py when it was split by module (test_simulator.py,
test_registry.py, test_compiler.py, test_parser.py, test_chunk.py,
test_healing.py, test_integration.py).
"""
import numpy as np
import pytest
import jax.numpy as jnp

from dense_evolution import DenseSVSimulator

# Patch the measure method directly within the test suite to ensure pytest uses the patched version
def patched_measure_for_tests(self, qubit_idx: int) -> int:
    """
    Misura un singolo qubit e collassa lo stato quantistico.
    """
    import numpy as np # Ensure np is available for random.choice

    if not 0 <= qubit_idx < self.n:
        raise ValueError(f"Qubit {qubit_idx} out of bounds")

    xp = self.xp
    # phys_q is used for stride calculation in NumPy/CuPy branch (LSB-first index)
    phys_q = self.n - 1 - qubit_idx
    stride = 1 << phys_q

    if xp is jnp:
        # JAX branch: Calculate probabilities by moving the correct (MSB-indexed) axis
        probs = self.xp.abs(self.sv)**2
        sv_shape = [2] * self.n
        sv_nd = probs.reshape(sv_shape)
        # FIX: Use qubit_idx directly as axis, as sv_nd is MSB-first indexed
        moved_probs = jnp.moveaxis(sv_nd, qubit_idx, 0)
        prob_0 = float(jnp.sum(moved_probs[0]))
        prob_1 = float(jnp.sum(moved_probs[1]))
    else:
        # NumPy/CuPy Stride Slicing: phys_q and stride logic correctly applied here
        sv_reshaped = self.sv.reshape(-1, 2, stride)
        prob_0 = float(xp.sum(xp.abs(sv_reshaped[:, 0, :])**2))
        prob_1 = float(xp.sum(xp.abs(sv_reshaped[:, 1, :])**2))

    total = prob_0 + prob_1
    if total > 1e-12:
        prob_0 /= total
        prob_1 /= total

    # Sampling the measurement outcome
    result = int(np.random.choice([0, 1], p=[prob_0, prob_1]))

    if xp is jnp:
        sv_shape = [2] * self.n
        sv_nd = self.sv.reshape(sv_shape)
        moved_sv = jnp.moveaxis(sv_nd, qubit_idx, 0) # FIX: Apply same correction here
        # Correctly zero out the unmeasured component (1 if result is 0, 0 if result is 1)
        moved_sv = moved_sv.at[1 - result].set(0.0)
        self.sv = jnp.moveaxis(moved_sv, 0, qubit_idx).ravel() # FIX: And here too
    else:
        sv_reshaped = self.sv.reshape(-1, 2, stride)
        # Zero out the unmeasured component
        sv_reshaped[:, 1 if result == 0 else 0, :] = 0.0
        self.sv = sv_reshaped.ravel()

    self.normalize()
    return result

# Apply the patch
DenseSVSimulator.measure = patched_measure_for_tests

# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sim2():
    """Fresh 2-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=2, use_gpu=False, use_float32=False)

@pytest.fixture
def sim3():
    """Fresh 3-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=3, use_gpu=False, use_float32=False)

@pytest.fixture
def sim4():
    """Fresh 4-qubit simulator (NumPy CPU, float64)"""
    return DenseSVSimulator(n_qubits=4, use_gpu=False, use_float32=False)
