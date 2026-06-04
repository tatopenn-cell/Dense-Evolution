import numpy as np
from typing import List, Tuple, Optional
from .registry import HAS_JAX
from .gates import GATES, PARAMETRIC_GATES, GATE_IDS
from .compiler import QuantumTranspiler

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from .compiler import _compile_and_run_circuit_jit


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (module-level, not methods — so JAX can jit them cleanly)
# ─────────────────────────────────────────────────────────────────────────────

def _qubit_stride_pairs(n: int, qubit: int):
    """
    Return (stride, outer_step, inner_step) for the MSB-first statevector
    convention used throughout this simulator.

    In MSB-first ordering qubit 0 is the *most* significant bit, so:
        physical_bit_position = n - 1 - qubit
        stride = 1 << physical_bit_position
    """
    phys   = n - 1 - qubit
    stride = 1 << phys
    return stride


def _cx_numpy(sv: np.ndarray, n: int, ctrl: int, tgt: int) -> np.ndarray:
    """
    Vectorised CX on a NumPy statevector.
    No Python loops — uses strided index arithmetic.
    """
    dim      = len(sv)
    c_stride = 1 << (n - 1 - ctrl)
    t_stride = 1 << (n - 1 - tgt)
    all_i    = np.arange(dim, dtype=np.intp)
    # Select indices where ctrl bit == 1 and tgt bit == 0
    mask     = ((all_i & c_stride) != 0) & ((all_i & t_stride) == 0)
    idx_0    = all_i[mask]
    idx_1    = idx_0 | t_stride
    sv       = sv.copy()
    sv[idx_0], sv[idx_1] = sv[idx_1].copy(), sv[idx_0].copy()
    return sv


def _cz_numpy(sv: np.ndarray, n: int, ctrl: int, tgt: int) -> np.ndarray:
    """Vectorised CZ on a NumPy statevector."""
    dim      = len(sv)
    c_stride = 1 << (n - 1 - ctrl)
    t_stride = 1 << (n - 1 - tgt)
    all_i    = np.arange(dim, dtype=np.intp)
    mask     = ((all_i & c_stride) != 0) & ((all_i & t_stride) != 0)
    sv       = sv.copy()
    sv[mask] *= -1
    return sv


# ─────────────────────────────────────────────────────────────────────────────
# DenseSVSimulator
# ─────────────────────────────────────────────────────────────────────────────

class DenseSVSimulator:
    """
    Dense statevector quantum circuit simulator.

    Qubit ordering: MSB-first (qubit 0 is the most significant bit).
    Backends: NumPy (CPU), JAX XLA JIT (CPU/GPU/TPU).

    Parameters
    ----------
    n_qubits   : number of qubits
    use_gpu    : reserved for future CuPy/JAX GPU dispatch
    use_float32: use complex64 instead of complex128
    """

    def __init__(self, n_qubits: int,
                 use_gpu:     bool = False,
                 use_float32: bool = False):
        if n_qubits < 1 or n_qubits > 34:
            raise ValueError(f"n_qubits must be in [1, 34], got {n_qubits}")
        self.n         = n_qubits
        self.dim       = 1 << n_qubits            # 2 ** n_qubits
        self.use_float32 = use_float32
        self.dtype     = np.complex64 if use_float32 else np.complex128
        self.xp        = jnp if HAS_JAX else np
        self._reset_sv()

    # ── state initialisation ─────────────────────────────────────────

    def _reset_sv(self):
        """Allocate |0...0⟩ on the active backend."""
        if HAS_JAX:
            self.sv = jnp.zeros(self.dim, dtype=self.dtype).at[0].set(1.0)
        else:
            self.sv    = np.zeros(self.dim, dtype=self.dtype)
            self.sv[0] = 1.0

    def set_initial_state(self, state: Optional[np.ndarray] = None):
        """
        Reset the simulator.

        Parameters
        ----------
        state : optional complex array of length 2**n.
                If None, resets to |0...0⟩.
                The array is normalised automatically.
        """
        if state is None:
            self._reset_sv()
            return
        state = np.asarray(state, dtype=self.dtype)
        if state.shape != (self.dim,):
            raise ValueError(
                f"State vector length {len(state)} != 2**{self.n} = {self.dim}")
        norm = np.linalg.norm(state)
        if norm < 1e-12:
            raise ValueError("Cannot set a zero-norm state vector")
        state = state / norm
        if HAS_JAX:
            self.sv = jnp.array(state)
        else:
            self.sv = state.copy()

    # Alias used by the VQE engine
    def set_state(self, state: np.ndarray):
        self.set_initial_state(state)

    # ── normalisation ─────────────────────────────────────────────────

    def normalize(self):
        norm = float(self.xp.linalg.norm(self.sv))
        if norm > 1e-12:
            if HAS_JAX:
                self.sv = self.sv / norm
            else:
                self.sv /= norm

    # ── 1-qubit gate ──────────────────────────────────────────────────

    def apply_gate_1q(self, gate: np.ndarray, qubit: int):
        """
        Apply a 2×2 unitary to *qubit* via tensor contraction.

        Uses reshape + moveaxis + matmul — fully vectorised,
        no Python loops, compatible with both NumPy and JAX.
        """
        if not 0 <= qubit < self.n:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.n})")
        gate = self.xp.array(gate, dtype=self.dtype)
        sv_nd      = self.sv.reshape([2] * self.n)
        sv_moved   = self.xp.moveaxis(sv_nd, qubit, -1)          # qubit axis → last
        flat_shape = (self.dim >> 1, 2)
        # matmul: (dim/2, 2) @ (2, 2).T  → (dim/2, 2)
        result     = self.xp.dot(sv_moved.reshape(flat_shape),
                                 gate.T)
        self.sv    = self.xp.moveaxis(
            result.reshape([2] * self.n), -1, qubit).ravel()

    # ── 2-qubit gate ──────────────────────────────────────────────────

    def apply_gate_2q(self, gate: np.ndarray, q1: int, q2: int):
        """
        Apply a 4×4 unitary to qubits (q1, q2) via tensor contraction.
        """
        if q1 == q2:
            raise ValueError("Control and target qubits must differ")
        if not (0 <= q1 < self.n and 0 <= q2 < self.n):
            raise ValueError(f"Qubit indices ({q1},{q2}) out of range [0, {self.n})")
        gate = self.xp.array(gate, dtype=self.dtype)
        sv_nd      = self.sv.reshape([2] * self.n)
        sv_moved   = self.xp.moveaxis(sv_nd, (q1, q2), (-2, -1))
        flat_shape = (self.dim >> 2, 4)
        result     = self.xp.dot(sv_moved.reshape(flat_shape),
                                 gate.reshape(4, 4).T)
        self.sv    = self.xp.moveaxis(
            result.reshape([2] * self.n), (-2, -1), (q1, q2)).ravel()

    # ── specialised 2-qubit gates ─────────────────────────────────────

    def apply_cx(self, ctrl: int, tgt: int):
        """
        CX (CNOT) gate.

        JAX path: matrix contraction via apply_gate_2q.
        NumPy path: fully vectorised index swap — no Python loops.
        """
        if ctrl == tgt:
            raise ValueError("Control and target qubits must differ")
        if not (0 <= ctrl < self.n and 0 <= tgt < self.n):
            raise ValueError(f"Qubit indices ({ctrl},{tgt}) out of range [0, {self.n})")
        if HAS_JAX:
            cx_mat = jnp.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ], dtype=self.dtype)
            self.apply_gate_2q(cx_mat, ctrl, tgt)
        else:
            self.sv = _cx_numpy(np.array(self.sv), self.n, ctrl, tgt)

    def apply_cz(self, ctrl: int, tgt: int):
        """
        CZ gate.

        JAX path: matrix contraction via apply_gate_2q.
        NumPy path: fully vectorised sign flip — no Python loops.
        """
        if ctrl == tgt:
            raise ValueError("Control and target qubits must differ")
        if not (0 <= ctrl < self.n and 0 <= tgt < self.n):
            raise ValueError(f"Qubit indices ({ctrl},{tgt}) out of range [0, {self.n})")
        if HAS_JAX:
            cz_mat = jnp.array([
                [1, 0, 0,  0],
                [0, 1, 0,  0],
                [0, 0, 1,  0],
                [0, 0, 0, -1],
            ], dtype=self.dtype)
            self.apply_gate_2q(cz_mat, ctrl, tgt)
        else:
            self.sv = _cz_numpy(np.array(self.sv), self.n, ctrl, tgt)

    def apply_rx(self, qubit: int, theta: float):
        cos, sin = np.cos(theta / 2), np.sin(theta / 2)
        mat = np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=self.dtype)
        self.apply_gate_1q(mat, qubit)

    def apply_ry(self, qubit: int, theta: float):
        cos, sin = np.cos(theta / 2), np.sin(theta / 2)
        mat = np.array([[cos, -sin], [sin, cos]], dtype=self.dtype)
        self.apply_gate_1q(mat, qubit)

    def apply_rz(self, qubit: int, theta: float):
        mat = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0,                        np.exp(1j * theta / 2)],
        ], dtype=self.dtype)
        self.apply_gate_1q(mat, qubit)

    # ── measurement ───────────────────────────────────────────────────

    def measure(self, qubit_idx: int) -> int:
        """
        Projective measurement on *qubit_idx*.

        Returns 0 or 1 and collapses the statevector.
        Uses MSB-first physical bit index: phys = n - 1 - qubit_idx.

        BUG FIX (original): the original NumPy collapse wrote
            sv_reshaped[:, 1 if result == 0 else 0, :] = 0.0
        which zeroed the *wrong* basis state (0 when result=1, 1 when result=0)
        and never normalised the JAX path.
        """
        if not 0 <= qubit_idx < self.n:
            raise ValueError(
                f"Qubit {qubit_idx} out of range [0, {self.n})")

        phys   = self.n - 1 - qubit_idx
        stride = 1 << phys

        # ── compute marginal probabilities ──────────────────────────
        if HAS_JAX:
            probs   = jnp.abs(self.sv) ** 2
            sv_nd   = probs.reshape([2] * self.n)
            mv      = jnp.moveaxis(sv_nd, phys, 0)
            prob_0  = float(jnp.sum(mv[0]))
            prob_1  = float(jnp.sum(mv[1]))
        else:
            sv_res  = self.sv.reshape(-1, 2, stride)
            prob_0  = float(np.sum(np.abs(sv_res[:, 0, :]) ** 2))
            prob_1  = float(np.sum(np.abs(sv_res[:, 1, :]) ** 2))

        total = prob_0 + prob_1
        if total < 1e-12:
            raise RuntimeError("Statevector norm is zero — cannot measure")
        prob_0 /= total
        prob_1 /= total

        result = int(np.random.choice([0, 1], p=[prob_0, prob_1]))

        # ── collapse ────────────────────────────────────────────────
        # Zero out the amplitudes corresponding to the *opposite* outcome.
        zero_slot = 1 - result     # if result=0, zero slot 1; if result=1, zero slot 0

        if HAS_JAX:
            sv_nd  = self.sv.reshape([2] * self.n)
            mv     = jnp.moveaxis(sv_nd, phys, 0)
            mv     = mv.at[zero_slot].set(0.0 + 0j)
            self.sv = jnp.moveaxis(mv, 0, phys).ravel()
        else:
            sv_res = self.sv.reshape(-1, 2, stride)
            sv_res[:, zero_slot, :] = 0.0
            self.sv = sv_res.ravel()

        self.normalize()
        return result

    # ── circuit execution ─────────────────────────────────────────────

    def run_circuit(self, circuit: List[Tuple], transpile: bool = True):
        """
        Execute a circuit given as a list of (gate_name, qubit, [qubit2], [param]).

        Parameters are always the *last* element(s) of the tuple.
        Gate names are matched case-insensitively.
        """
        target = QuantumTranspiler.transpile(circuit) if transpile else circuit
        for cmd in target:
            name = cmd[0].lower()
            args = cmd[1:]

            if name in GATES:
                mat = np.array(GATES[name], dtype=self.dtype)
                if mat.shape == (2, 2):
                    self.apply_gate_1q(mat, int(args[0]))
                else:
                    self.apply_gate_2q(mat, int(args[0]), int(args[1]))

            elif name in PARAMETRIC_GATES:
                # Tuple layout: (name, qubit, param)          for 1-qubit
                #               (name, ctrl,  tgt,   param)   for 2-qubit
                if len(args) == 2:                              # 1-qubit parametric
                    mat = np.array(PARAMETRIC_GATES[name](args[1]), dtype=self.dtype)
                    self.apply_gate_1q(mat, int(args[0]))
                elif len(args) == 3:                            # 2-qubit parametric
                    mat = np.array(PARAMETRIC_GATES[name](args[2]), dtype=self.dtype)
                    self.apply_gate_2q(mat, int(args[0]), int(args[1]))
                elif len(args) == 4:                            # u3-style (θ, φ, λ)
                    mat = np.array(
                        PARAMETRIC_GATES[name](args[1], args[2], args[3]),
                        dtype=self.dtype)
                    self.apply_gate_1q(mat, int(args[0]))

    def run_circuit_jit_beast_mode(self, circuit: List):
        """
        Execute a pre-built beast-mode circuit on the JAX JIT engine.

        Beast-mode format: each command is a list [name, q1, q2_or_param, ...]
        as produced by core_calcolo_quantistico in dash.py.

        Falls back to run_circuit if JAX is unavailable.
        """
        if not HAS_JAX:
            return self.run_circuit(circuit)

        target       = QuantumTranspiler.transpile(circuit)
        compiled_ops = []

        for cmd in target:
            name = cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()
            if name not in GATE_IDS:
                continue

            g_id = float(GATE_IDS[name])
            args = cmd[1:]

            # ── gate argument parsing ──────────────────────────────
            # 1-qubit parametric: (name, qubit, param)
            if name in ('rx', 'ry', 'rz', 'p', 'u1', 'phase'):
                q1 = float(args[0])
                p  = float(args[1]) if len(args) > 1 else 0.0
                compiled_ops.append([g_id, q1, 0.0, p])

            # 2-qubit parametric: (name, ctrl, tgt, param)
            elif name in ('cp', 'crz', 'cphase'):
                ctrl = float(args[0])
                tgt  = float(args[1]) if len(args) > 1 else 0.0
                p    = float(args[2]) if len(args) > 2 else 0.0
                compiled_ops.append([g_id, ctrl, tgt, p])

            # 2-qubit non-parametric: (name, ctrl, tgt)
            elif name in ('cx', 'cz', 'swap', 'cy'):
                ctrl = float(args[0])
                tgt  = float(args[1]) if len(args) > 1 else 0.0
                compiled_ops.append([g_id, ctrl, tgt, 0.0])

            # 1-qubit non-parametric: (name, qubit)
            else:
                q1 = float(args[0]) if args else 0.0
                compiled_ops.append([g_id, q1, 0.0, 0.0])

        if compiled_ops:
            ops_jnp = jnp.array(compiled_ops, dtype=jnp.float64)
            self.sv  = _compile_and_run_circuit_jit(self.sv, ops_jnp)

    def run_circuit_with_chunking(self, circuit: List, chunk_size: int = 500):
        """
        Execute a circuit in chunks to avoid JIT recompilation on
        large variable-length circuits.

        Each chunk is a separate _compile_and_run_circuit_jit call
        with a fixed-size ops array, allowing XLA to cache each size.
        """
        target = QuantumTranspiler.transpile(circuit)
        for i in range(0, len(target), chunk_size):
            self.run_circuit_jit_beast_mode(target[i: i + chunk_size])

    def run_parametric_batch_jit(self,
                                  base_circuit:    List,
                                  parameter_batch: np.ndarray) -> "jnp.ndarray":
        """
        Vectorised VQE parameter sweep via jax.vmap.

        For each row θ in *parameter_batch*, injects θ sequentially into
        all parametric gate slots (marker value -1.0) and runs the full
        circuit. Returns a (batch_size, 2**n) array of final statevectors.

        BUG FIX (original): the nested lax.cond for idx increment used
        `(0,)` as carry — a Python tuple, not a JAX scalar — causing
        shape errors when jit-compiled. Fixed by using a plain jnp.int32.
        """
        if not HAS_JAX:
            raise RuntimeError("run_parametric_batch_jit requires JAX")

        target       = QuantumTranspiler.transpile(base_circuit)
        compiled_ops = []

        for cmd in target:
            name = cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()
            if name not in GATE_IDS:
                continue
            g_id = float(GATE_IDS[name])
            args = cmd[1:]
            if name in ('rx', 'ry', 'rz', 'p', 'u1', 'phase'):
                compiled_ops.append([g_id, float(args[0]), 0.0, -1.0])   # -1.0 = param slot
            elif name in ('cp', 'crz', 'cphase'):
                compiled_ops.append([g_id, float(args[0]), float(args[1]) if len(args) > 1 else 0.0, -1.0])
            elif name in ('cx', 'cz', 'swap', 'cy'):
                compiled_ops.append([g_id, float(args[0]), float(args[1]) if len(args) > 1 else 0.0, 0.0])
            else:
                compiled_ops.append([g_id, float(args[0]) if args else 0.0, 0.0, 0.0])

        template    = jnp.array(compiled_ops, dtype=jnp.float64)
        init_sv     = jnp.zeros(self.dim, dtype=jnp.complex128).at[0].set(1.0)

        def simulate_single_instance(single_params: "jnp.ndarray") -> "jnp.ndarray":
            """Run one parameter vector through the circuit."""

            def patch_and_apply(carry: "jnp.ndarray",
                                op:    "jnp.ndarray"):
                """
                carry: jnp.int32 scalar — current parametric gate index.
                op:    [g_id, q1, q2, p_sentinel]
                """
                idx       = carry
                is_param  = op[3] == -1.0
                final_p   = jnp.where(is_param, single_params[idx], op[3])
                next_idx  = jnp.where(is_param, idx + jnp.int32(1), idx)
                patched   = jnp.array([op[0], op[1], op[2], final_p],
                                       dtype=jnp.float64)
                return next_idx, patched

            _, patched_ops = jax.lax.scan(
                patch_and_apply,
                jnp.int32(0),       # BUG FIX: was (0,) tuple — must be a scalar
                template,
            )
            return _compile_and_run_circuit_jit(init_sv, patched_ops)

        return jax.jit(jax.vmap(simulate_single_instance, in_axes=(0,)))(
            jnp.asarray(parameter_batch, dtype=jnp.float64)
        )

    # ── observables ───────────────────────────────────────────────────

    def get_probabilities(self) -> np.ndarray:
        """Return measurement probability distribution as a NumPy float64 array."""
        probs = np.array(self.xp.abs(self.sv) ** 2, dtype=np.float64)
        # guard against floating-point leakage outside [0, 1]
        probs = np.clip(probs, 0.0, 1.0)
        total = probs.sum()
        if total > 1e-12:
            probs /= total
        return probs

    def get_statevector(self) -> np.ndarray:
        """Return the current statevector as a NumPy complex array."""
        return np.array(self.sv, dtype=self.dtype)

    def memory_mb(self) -> float:
        """Statevector memory footprint in megabytes."""
        bytes_per_element = 8 if self.use_float32 else 16   # complex64=8, complex128=16
        return self.dim * bytes_per_element / 1_000_000
