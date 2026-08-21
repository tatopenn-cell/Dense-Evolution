import warnings
import numpy as np
from typing import List, Tuple, Optional
from ..circuits.registry import HAS_JAX
from ..circuits.gates import GATES, PARAMETRIC_GATES, GATE_IDS, _TWO_QUBIT_PARAMETRIC_GATES
from ..circuits.compiler import QuantumTranspiler

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from ..circuits.compiler import _compile_and_run_circuit_jit, _compile_and_run_circuit_jit_donated


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers 
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
    Backends: NumPy (CPU), JAX XLA JIT (CPU/GPU/TPU) -- GPU dispatch is
    automatic whenever a CUDA-enabled jaxlib is installed and a GPU is
    present (jax.devices() reports it); no flag on this class selects it,
    JAX's own default-device placement does.

    Parameters
    ----------
    n_qubits   : number of qubits
    use_float32: use complex64 instead of complex128
    """

    def __init__(self, n_qubits: int,
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

    def _check_qubit_range(self, qubit: float, context: str) -> None:
        """Validate a qubit index before it reaches the JIT-compiled fast
        paths (run_circuit_jit / run_batch_jit).

        Those paths encode qubit indices as bit-shift amounts inside
        jax.lax.scan/switch and never call apply_gate_1q/apply_gate_2q
        (which already validate) — an out-of-range index there doesn't
        raise, it silently corrupts the entire statevector to zero
        (verified: a single gate on an out-of-range qubit on an otherwise
        normalized state left get_probabilities().sum() == 0.0, no error).
        """
        qi = int(qubit)
        if not 0 <= qi < self.n:
            raise ValueError(
                f"Qubit index {qi} out of range [0, {self.n}) in {context}")

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
        """Apply a parameterized RX gate using the active backend (NumPy/JAX)."""
        cos, sin = self.xp.cos(theta / 2), self.xp.sin(theta / 2)
        mat = self.xp.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=self.dtype)
        self.apply_gate_1q(mat, qubit)

    def apply_ry(self, qubit: int, theta: float):
        """Apply a parameterized RY gate using the active backend (NumPy/JAX)."""
        cos, sin = self.xp.cos(theta / 2), self.xp.sin(theta / 2)
        mat = self.xp.array([[cos, -sin], [sin, cos]], dtype=self.dtype)
        self.apply_gate_1q(mat, qubit)

    def apply_rz(self, qubit: int, theta: float):
        """Apply a parameterized RZ gate using the active backend (NumPy/JAX)."""
        exp_neg = self.xp.exp(-1j * theta / 2)
        exp_pos = self.xp.exp(1j * theta / 2)
        mat = self.xp.array([[exp_neg, 0.0], [0.0, exp_pos]], dtype=self.dtype)
        self.apply_gate_1q(mat, qubit)

    # ── measurement ───────────────────────────────────────────────────

    def measure(self, qubit_idx: int, jax_key: Optional["jax.Array"] = None) -> int:
        """
        Projective measurement on *qubit_idx*.

        Returns 0 or 1 and collapses the statevector.
        Uses MSB-first physical bit index: phys = n - 1 - qubit_idx.

        BUG FIX (original): the original NumPy collapse wrote
            sv_reshaped[:, 1 if result == 0 else 0, :] = 0.0
        which zeroed the *wrong* basis state (0 when result=1, 1 when result=0)
        and never normalised the JAX path.

        jax_key : optional JAX PRNGKey. When given, the random outcome is
                  drawn via jax.random.choice(jax_key, ...) instead of the
                  global `np.random.choice` -- explicit, seedable, and
                  independent of NumPy's global RNG state, matching
                  registry.NoiseModel.apply_to_sv's own jax_key convention
                  (see that function's docstring for why explicit keys, not
                  hidden per-instance state, are this codebase's convention
                  for JAX-side reproducibility). Default (None) keeps the
                  original np.random.choice behavior unchanged, on both
                  backends -- this measurement's own state-collapse still
                  does concrete Python branching either way (the `result`
                  drives which basis-state slot gets zeroed), so passing a
                  key makes the *outcome* reproducible, not this method
                  jax.jit-traceable.
        """
        if not 0 <= qubit_idx < self.n:
            raise ValueError(
                f"Qubit {qubit_idx} out of range [0, {self.n})")

        # BUG FIX: the JAX branch below reshapes to a [2]*n tensor and
        # moveaxis'd -- the exact same indexing scheme apply_gate_1q
        # uses (`moveaxis(sv_nd, qubit, -1)`, qubit axis == qubit index
        # directly, no conversion). This method's JAX branch was instead
        # using `phys = n-1-qubit_idx` for that moveaxis -- correct for
        # the *NumPy* branch below (genuinely different flat/stride
        # arithmetic on a raveled array), but wrong for the JAX branch's
        # reshape-based indexing, silently reading/collapsing the WRONG
        # qubit's marginal whenever qubit_idx != n-1-qubit_idx. Verified
        # directly: X on qubit 0 of a 2-qubit register, then measure(0),
        # returned 0 instead of 1 before this fix (see tests/unit/test_simulator.py's
        # TestMeasurement class).
        phys   = self.n - 1 - qubit_idx
        stride = 1 << phys

        # ── compute marginal probabilities ──────────────────────────
        if HAS_JAX:
            probs   = jnp.abs(self.sv) ** 2
            sv_nd   = probs.reshape([2] * self.n)
            mv      = jnp.moveaxis(sv_nd, qubit_idx, 0)
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

        if jax_key is not None:
            if not HAS_JAX:
                raise ValueError("measure(jax_key=...) requires JAX to be installed.")
            result = int(jax.random.choice(jax_key, jnp.array([0, 1]), p=jnp.array([prob_0, prob_1])))
        else:
            result = int(np.random.choice([0, 1], p=[prob_0, prob_1]))

        # ── collapse ────────────────────────────────────────────────
        # Zero out the amplitudes corresponding to the *opposite* outcome.
        zero_slot = 1 - result     # if result=0, zero slot 1; if result=1, zero slot 0

        if HAS_JAX:
            sv_nd  = self.sv.reshape([2] * self.n)
            mv     = jnp.moveaxis(sv_nd, qubit_idx, 0)
            mv     = mv.at[zero_slot].set(0.0 + 0j)
            self.sv = jnp.moveaxis(mv, 0, qubit_idx).ravel()
        else:
            sv_res = self.sv.reshape(-1, 2, stride)
            sv_res[:, zero_slot, :] = 0.0
            self.sv = sv_res.ravel()

        self.normalize()
        return result

    # ── circuit execution ─────────────────────────────────────────────

    def run_circuit(self, circuit: List[Tuple], transpile: bool = True):
        target = QuantumTranspiler.transpile(circuit) if transpile else circuit

        # Auto-dispatch to the compiled path whenever every gate in this
        # circuit (post-transpile) is supported there -- measured 6x+
        # faster on realistic circuits (eager per-gate dispatch pays a
        # Python<->JAX round trip per gate; the compiled path is one XLA
        # call for the whole circuit). The point: a caller who has never
        # heard of run_circuit_jit still gets it automatically, for any
        # circuit it can actually run -- only falls through to the eager
        # loop below for the few gates GATE_IDS doesn't cover yet (see
        # circuits/gates.py's own GATE_IDS comment for exactly which).
        if HAS_JAX and all(
            (cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()) in GATE_IDS
            for cmd in target
        ):
            self.run_circuit_jit(target)
            return

        for cmd in target:
            name = cmd[0].lower()
            args = cmd[1:]

            if name in GATES:
                mat = self.xp.array(GATES[name], dtype=self.dtype)
                if mat.shape == (2, 2):
                    self.apply_gate_1q(mat, int(args[0]))
                else:
                    self.apply_gate_2q(mat, int(args[0]), int(args[1]))

            elif name in PARAMETRIC_GATES:
                # Dispatch by gate NAME, not arg count -- a 1-qubit gate
                # with 2 params (u2: qubit,phi,lam) and a 2-qubit gate
                # with 1 param (cp/crz: q1,q2,theta) both have len(args)
                # == 3, so arg-count alone is ambiguous (see
                # _TWO_QUBIT_PARAMETRIC_GATES's own comment in gates.py).
                if name in _TWO_QUBIT_PARAMETRIC_GATES:
                    mat = self.xp.array(PARAMETRIC_GATES[name](*args[2:]), dtype=self.dtype)
                    self.apply_gate_2q(mat, int(args[0]), int(args[1]))
                else:
                    mat = self.xp.array(PARAMETRIC_GATES[name](*args[1:]), dtype=self.dtype)
                    self.apply_gate_1q(mat, int(args[0]))

            else:
                raise ValueError(
                    f"unknown gate '{cmd[0]}' -- not in GATES or PARAMETRIC_GATES. "
                    f"A typo in a gate name used to be silently dropped from the "
                    f"circuit instead of raising (issue #4)."
                )



    def run_circuit_jit(self, circuit: List):

        if not HAS_JAX:
            return self.run_circuit(circuit)

        target       = QuantumTranspiler.transpile(circuit)
        compiled_ops = []

        for cmd in target:
            name = cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()
            if name not in GATE_IDS:
                raise ValueError(
                    f"unknown gate '{cmd[0]}' -- not in GATE_IDS. A typo in a gate "
                    f"name used to be silently dropped from the circuit instead of "
                    f"raising (issue #4)."
                )

            g_id = float(GATE_IDS[name])
            args = cmd[1:]

            # ── gate argument parsing ──────────────────────────────
            # 1-qubit parametric: (name, qubit, param)
            if name in ('rx', 'ry', 'rz', 'p', 'u1', 'phase', 'gphase'):
                q1 = float(args[0])
                self._check_qubit_range(q1, f"gate '{name}'")
                p  = float(args[1]) if len(args) > 1 else 0.0
                compiled_ops.append([g_id, q1, 0.0, p])

            # 2-qubit parametric: (name, ctrl, tgt, param)
            elif name in ('cp', 'crz', 'cphase'):
                ctrl = float(args[0])
                tgt  = float(args[1]) if len(args) > 1 else 0.0
                self._check_qubit_range(ctrl, f"gate '{name}' (control)")
                self._check_qubit_range(tgt, f"gate '{name}' (target)")
                p    = float(args[2]) if len(args) > 2 else 0.0
                compiled_ops.append([g_id, ctrl, tgt, p])

            # 2-qubit non-parametric: (name, ctrl, tgt)
            elif name in ('cx', 'cz', 'swap', 'cy'):
                ctrl = float(args[0])
                tgt  = float(args[1]) if len(args) > 1 else 0.0
                self._check_qubit_range(ctrl, f"gate '{name}' (control)")
                self._check_qubit_range(tgt, f"gate '{name}' (target)")
                compiled_ops.append([g_id, ctrl, tgt, 0.0])

            # 1-qubit non-parametric: (name, qubit)
            else:
                q1 = float(args[0]) if args else 0.0
                self._check_qubit_range(q1, f"gate '{name}'")
                compiled_ops.append([g_id, q1, 0.0, 0.0])

        if compiled_ops:
            ops_jnp = jnp.array(compiled_ops, dtype=jnp.float64)
            # Safe to donate self.sv here: it's rebound immediately below and
            # no code path anywhere keeps a stale reference to the old buffer
            # across this call (verified across chunked/repeated calls too,
            # see _compile_and_run_circuit_jit_donated's docstring).
            self.sv  = _compile_and_run_circuit_jit_donated(self.sv, ops_jnp)

    def run_circuit_jit_beast_mode(self, circuit: List):
        """Deprecated alias for run_circuit_jit -- kept so code written
        against any pre-8.1.46 release keeps working. Will be removed in
        a future major version; switch to run_circuit_jit."""
        warnings.warn(
            "run_circuit_jit_beast_mode is deprecated, use run_circuit_jit instead "
            "(same behavior, shorter name). This alias will be removed in a future release.",
            DeprecationWarning, stacklevel=2,
        )
        return self.run_circuit_jit(circuit)

    def run_circuit_with_chunking(self, circuit: List, chunk_size: int = 500):
        """
        Execute a circuit in chunks to avoid JIT recompilation on
        large variable-length circuits.

        Each chunk is a separate _compile_and_run_circuit_jit call
        with a fixed-size ops array, allowing XLA to cache each size.
        """
        target = QuantumTranspiler.transpile(circuit)
        for i in range(0, len(target), chunk_size):
            self.run_circuit_jit(target[i: i + chunk_size])

    def run_batch_jit(self,
                       base_circuit:    List,
                       parameter_batch: np.ndarray) -> "jnp.ndarray":

        if not HAS_JAX:
            raise RuntimeError("run_batch_jit requires JAX")

        target       = QuantumTranspiler.transpile(base_circuit)
        compiled_ops = []

        for cmd in target:
            name = cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()
            if name not in GATE_IDS:
                raise ValueError(
                    f"unknown gate '{cmd[0]}' -- not in GATE_IDS. A typo in a gate "
                    f"name used to be silently dropped from the circuit instead of "
                    f"raising (issue #4)."
                )
            g_id = float(GATE_IDS[name])
            args = cmd[1:]
            if name in ('rx', 'ry', 'rz', 'p', 'u1', 'phase'):
                q1 = float(args[0])
                self._check_qubit_range(q1, f"gate '{name}'")
                compiled_ops.append([g_id, q1, 0.0, -1.0])   # -1.0 = param slot
            elif name in ('cp', 'crz', 'cphase'):
                ctrl = float(args[0])
                tgt  = float(args[1]) if len(args) > 1 else 0.0
                self._check_qubit_range(ctrl, f"gate '{name}' (control)")
                self._check_qubit_range(tgt, f"gate '{name}' (target)")
                compiled_ops.append([g_id, ctrl, tgt, -1.0])
            elif name in ('cx', 'cz', 'swap', 'cy'):
                ctrl = float(args[0])
                tgt  = float(args[1]) if len(args) > 1 else 0.0
                self._check_qubit_range(ctrl, f"gate '{name}' (control)")
                self._check_qubit_range(tgt, f"gate '{name}' (target)")
                compiled_ops.append([g_id, ctrl, tgt, 0.0])
            else:
                q1 = float(args[0]) if args else 0.0
                self._check_qubit_range(q1, f"gate '{name}'")
                compiled_ops.append([g_id, q1, 0.0, 0.0])

        n_param_slots = sum(1 for op in compiled_ops if op[3] == -1.0)
        parameter_batch = np.asarray(parameter_batch)
        if parameter_batch.ndim != 2 or parameter_batch.shape[1] != n_param_slots:
            raise ValueError(
                f"parameter_batch has {parameter_batch.shape[-1] if parameter_batch.ndim else 0} "
                f"column(s) but base_circuit has {n_param_slots} parametric gate(s) (rx/ry/rz/p/u1/"
                f"phase/cp/crz/cphase) -- one column per parametric gate, in gate-appearance order. "
                f"A literal float passed for one of these gates is NOT exempt: it still consumes a "
                f"positional column. A mismatch here used to be clipped silently by JAX's default "
                f"out-of-bounds indexing instead of raising (issue #6)."
            )

        template    = jnp.array(compiled_ops, dtype=jnp.float64)
        # self.dtype, not a hardcoded jnp.complex128: _apply_gate_fast_step
        # (compiler.py) already derives its working dtype from the input
        # statevector specifically so use_float32=True instances run in
        # complex64 end to end -- hardcoding complex128 here silently
        # discarded that for every run_batch_jit/run_parametric_batch_jit
        # call regardless of the instance's own configured dtype.
        init_sv     = jnp.zeros(self.dim, dtype=self.dtype).at[0].set(1.0)

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

    def run_parametric_batch_jit(self, base_circuit: List, parameter_batch: np.ndarray) -> "jnp.ndarray":
        """Deprecated alias for run_batch_jit -- kept so code written
        against any pre-8.1.46 release keeps working. Will be removed in
        a future major version; switch to run_batch_jit."""
        warnings.warn(
            "run_parametric_batch_jit is deprecated, use run_batch_jit instead "
            "(same behavior, shorter name). This alias will be removed in a future release.",
            DeprecationWarning, stacklevel=2,
        )
        return self.run_batch_jit(base_circuit, parameter_batch)

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
