import gc
from typing import Tuple

import psutil

# JAX is now a mandatory dependency (see registry.py) -- kept as an
# import-time flag for the dual-path code below, which stays as-is.
import jax
HAS_JAX = True

__all__ = ["MemoryPressureError", "SafeMemoryGuard"]


def _device_memory_budget_bytes(device=None) -> Tuple[float, str]:
    """Real, dynamic memory budget -- reads the ACTIVE compute device's
    own memory via JAX's device.memory_stats() (bytes_limit -
    bytes_in_use) when available, falling back to host RAM via
    psutil.virtual_memory() otherwise (some backends, e.g. plain CPU,
    return None from memory_stats() -- confirmed directly, so on CPU
    this is identical to the old psutil-only behavior). get_dynamic_chunk
    and SafeMemoryGuard used to read psutil.virtual_memory()
    UNCONDITIONALLY, which is only correct when the compute device IS the
    host (true on CPU, false on GPU/TPU) -- sizing chunks off host RAM
    while the data actually lives in smaller GPU VRAM is exactly why
    Chunk's own README benchmark ("~2GB constant RAM regardless of qubit
    count") holds on CPU but silently mis-sized chunks on GPU, causing
    MemoryPressureError/real OOM well before the GPU's own VRAM was
    actually exhausted. Verified on a real Colab T4 GPU: before this fix,
    chunk_size_bits was computed from host RAM (often >11GB free on
    Colab) while the data lived in the T4's 11-15GB VRAM pool -- after
    this fix, device.memory_stats() correctly reports VRAM instead.

    Returns (available_bytes, source_description) -- the raw number, not
    a safety-margined one; callers apply their own margin."""
    if device is None:
        device = jax.devices()[0] if HAS_JAX else None
    stats = device.memory_stats() if device is not None and hasattr(device, "memory_stats") else None
    if stats and stats.get("bytes_limit"):
        available = stats["bytes_limit"] - stats.get("bytes_in_use", 0)
        return float(available), f"device.memory_stats() on {device}"
    vm = psutil.virtual_memory()
    return float(vm.available), "psutil (host RAM) -- device.memory_stats() unavailable"


def _device_total_bytes(device=None) -> float:
    """Companion to _device_memory_budget_bytes for the TOTAL (not
    available) capacity -- used by SafeMemoryGuard to compute free_pct
    against the right pool (device VRAM on GPU/TPU, host RAM on CPU)."""
    if device is None:
        device = jax.devices()[0] if HAS_JAX else None
    stats = device.memory_stats() if device is not None and hasattr(device, "memory_stats") else None
    if stats and stats.get("bytes_limit"):
        return float(stats["bytes_limit"])
    return float(psutil.virtual_memory().total)


# ─────────────────────────────────────────────────────────────────────────────────
# SafeMemoryGuard  — Anti-OOM block
# ─────────────────────────────────────────────────────────────────────────────────

class MemoryPressureError(RuntimeError):
    """
    Raised when available system RAM drops below the configured safety threshold.
    Catches the condition *before* the allocator attempts and crashes with
    jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED.
    """
    pass


class SafeMemoryGuard:
    """
    Monitors system RAM before every high-memory operation and blocks execution
    if free RAM falls below ``threshold_pct`` of total physical memory.
    """

    _WARN_MULTIPLIER = 2.0

    def __init__(self, threshold_pct: float = 0.15, gc_before_check: bool = True):
        if not 0.0 < threshold_pct < 1.0:
            raise ValueError(f"threshold_pct must be in (0, 1), got {threshold_pct}")
        self.threshold_pct   = threshold_pct
        self.gc_before_check = gc_before_check
        # Device VRAM total on GPU/TPU, host RAM total on CPU (identical
        # to the old psutil-only value there) -- see
        # _device_memory_budget_bytes's docstring for why this matters.
        self._total_mb       = _device_total_bytes() / (1024 * 1024)

    def status(self) -> dict:
        available_bytes, _source = _device_memory_budget_bytes()
        available_mb = available_bytes / (1024 * 1024)
        free_pct     = available_mb / self._total_mb if self._total_mb > 0 else 0.0
        return {
            "total_mb"    : self._total_mb,
            "available_mb": available_mb,
            "used_pct"    : (1.0 - free_pct) * 100.0,
            "free_pct"    : free_pct * 100.0,
            "safe"        : free_pct >= self.threshold_pct,
        }

    def check(self, context: str = "") -> None:
        if self.gc_before_check:
            gc.collect()

        s   = self.status()
        tag = f"[{context}] " if context else ""
        free_frac = s["free_pct"] / 100.0

        if not s["safe"]:
            raise MemoryPressureError(
                f"\n{'─'*60}\n"
                f"  {tag}MEMORIA CRITICA — simulazione bloccata\n"
                f"  Disponibile : {s['available_mb']:.0f} MB  "
                f"({s['free_pct']:.1f}% libera)\n"
                f"  Soglia      : {self.threshold_pct * 100:.0f}%  "
                f"({self._total_mb * self.threshold_pct:.0f} MB)\n"
                f"  Azione      : liberare RAM o ridurre n_qubits / chunk_size.\n"
                f"{'─'*60}"
            )

        warn_threshold = self.threshold_pct * self._WARN_MULTIPLIER
        if free_frac < warn_threshold:
            print(
                f"  [WARN] {tag}RAM bassa: {s['available_mb']:.0f} MB liberi "
                f"({s['free_pct']:.1f}%) — soglia critica al "
                f"{self.threshold_pct * 100:.0f}%."
            )

    def check_allocation(self, required_mb: float, context: str = "") -> None:
        """Like check(), but sized: verifies that *required_mb* can be
        allocated while still leaving threshold_pct free afterwards —
        check() alone only looks at RAM free *right now*, independent of
        what's about to be allocated (needed for Chunk's multi-chunk path,
        where several chunk-sized simulators are held in RAM at once)."""
        if self.gc_before_check:
            gc.collect()

        s   = self.status()
        tag = f"[{context}] " if context else ""
        available_after_mb = s["available_mb"] - required_mb
        free_frac_after = available_after_mb / self._total_mb if self._total_mb > 0 else 0.0

        if available_after_mb < 0 or free_frac_after < self.threshold_pct:
            import socket
            raise MemoryPressureError(
                f"\n{'─'*60}\n"
                f"  {tag}MEMORIA INSUFFICIENTE per l'allocazione richiesta\n"
                f"  Macchina     : {socket.gethostname()} ({self._total_mb / 1024:.1f} GB RAM totali)\n"
                f"  Richiesti    : {required_mb:.0f} MB\n"
                f"  Disponibile  : {s['available_mb']:.0f} MB ({s['free_pct']:.1f}% libera)\n"
                f"  Dopo alloc.  : {available_after_mb:.0f} MB ({free_frac_after * 100:.1f}% libera)\n"
                f"  Soglia       : {self.threshold_pct * 100:.0f}% libera dopo l'allocazione\n"
                f"  Azione       : ridurre n_qubits o liberare RAM. Il chunking su\n"
                f"                 disco per overflow oltre la RAM disponibile non\n"
                f"                 e' implementato — vedi CHANGELOG.\n"
                f"{'─'*60}"
            )

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"SafeMemoryGuard("
            f"threshold={self.threshold_pct*100:.0f}%, "
            f"available={s['available_mb']:.0f} MB / {s['free_pct']:.1f}% free, "
            f"safe={s['safe']})"
        )
