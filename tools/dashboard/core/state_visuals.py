"""
Native statevector visualizations -- histogram, Q-sphere, per-qubit Bloch
spheres -- pure matplotlib/numpy, no Qiskit anywhere. Replaces
qiskit.visualization.{plot_histogram, plot_state_qsphere,
plot_bloch_multivector}, the last three real Qiskit call sites in this
project's dashboard (the circuit diagram itself was already native, see
circuit_diagram.py's own docstring for why that mattered on macOS).
Qiskit's own instability on macOS CI runners was never proven scoped to
just QuantumCircuit -- no reason the plotting functions built on top of
it are exempt -- so removing them entirely, not just making qiskit
optional, is the actual fix.

All inputs use Qiskit's little-endian bit convention (qubit 0 = least
significant bit of the statevector index), matching
dashboard_core.engine's own SimulationResult.statevector -- the same
convention plot_state_qsphere/plot_bloch_multivector assumed.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

__all__ = ['native_histogram_figure', 'native_qsphere_figure', 'native_bloch_multivector_figure']

_BLOCH_COLOR = "#4589ff"
_QSPHERE_LINE_COLOR = "#8d8d8d"


def native_histogram_figure(counts: dict):
    """Bar chart of shot counts, bitstrings sorted ascending -- the same
    information qiskit.visualization.plot_histogram showed, built from
    nothing but the counts dict itself.

    Examples
    --------
    >>> from dashboard_core.state_visuals import native_histogram_figure
    >>> fig = native_histogram_figure({'00': 512, '11': 488})
    >>> type(fig).__name__
    'Figure'
    """
    states = sorted(counts.keys())
    values = [counts[s] for s in states]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(states)), 4))
    bars = ax.bar(states, values, color="#648fff")
    ax.set_ylabel("Counts")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    for bar, v in zip(bars, values):
        pct = 100 * v / total if total else 0
        ax.annotate(
            f"{v}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8,
        )
    if len(states) > 8:
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout()
    return fig


def _reduced_density_matrix(statevector: np.ndarray, n_qubits: int, qubit: int) -> np.ndarray:
    """Partial trace over every qubit except `qubit`, in Qiskit's
    little-endian convention (qubit 0 = LSB of the flat index).
    np.reshape's default C order puts the LSB on the *last* tensor axis,
    so qubit q lives on axis (n_qubits-1-q) -- moved to the front, then
    the remaining axes are flattened into one "everything else" axis and
    traced out via M @ M^dagger. Verified directly against a Bell pair:
    each qubit's reduced state comes back exactly I/2 (maximally mixed),
    the known analytic answer for a maximally entangled 2-qubit state.
    """
    tensor = statevector.reshape((2,) * n_qubits)
    axis = n_qubits - 1 - qubit
    m = np.moveaxis(tensor, axis, 0).reshape(2, -1)
    return m @ m.conj().T


def _bloch_vector(rho: np.ndarray) -> np.ndarray:
    """(bx, by, bz) from a single-qubit density matrix, via
    rho = 0.5*(I + bx*X + by*Y + bz*Z) inverted: bx=2*Re(rho01),
    by=-2*Im(rho01), bz=rho00-rho11."""
    bx = 2 * np.real(rho[0, 1])
    by = -2 * np.imag(rho[0, 1])
    bz = np.real(rho[0, 0] - rho[1, 1])
    return np.array([bx, by, bz])


def _draw_bloch_sphere(ax, vector, title):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="#d0d0d0", linewidth=0.4)
    ax.plot([-1, 1], [0, 0], [0, 0], color="#d0d0d0", linewidth=0.6)
    ax.plot([0, 0], [-1, 1], [0, 0], color="#d0d0d0", linewidth=0.6)
    ax.plot([0, 0], [0, 0], [-1, 1], color="#d0d0d0", linewidth=0.6)
    ax.text(0, 0, 1.15, "|0⟩", ha="center", fontsize=8)
    ax.text(0, 0, -1.3, "|1⟩", ha="center", fontsize=8)
    ax.quiver(0, 0, 0, *vector, color=_BLOCH_COLOR, linewidth=2, arrow_length_ratio=0.15)
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()


def native_bloch_multivector_figure(statevector: np.ndarray):
    """One Bloch sphere per qubit, each from that qubit's own reduced
    density matrix (partial trace over every other qubit) -- the same
    per-qubit view qiskit.visualization.plot_bloch_multivector showed,
    computed directly from the real statevector, no Qiskit involved.

    Examples
    --------
    >>> import numpy as np
    >>> from dashboard_core.state_visuals import native_bloch_multivector_figure
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    >>> fig = native_bloch_multivector_figure(bell)
    >>> type(fig).__name__
    'Figure'
    """
    n_qubits = int(np.log2(len(statevector)))
    fig = plt.figure(figsize=(3 * n_qubits, 3.2))
    for q in range(n_qubits):
        rho = _reduced_density_matrix(statevector, n_qubits, q)
        vector = _bloch_vector(rho)
        ax = fig.add_subplot(1, n_qubits, q + 1, projection="3d")
        _draw_bloch_sphere(ax, vector, f"qubit {q}")
    fig.tight_layout()
    return fig


def native_qsphere_figure(statevector: np.ndarray, prob_threshold: float = 1e-3):
    """Q-sphere: every basis state with non-negligible probability placed
    on a sphere by Hamming weight (latitude -- |00...0> at the north pole,
    |11...1> at the south pole, states of equal weight sharing a ring),
    marker size by probability, marker color by phase (cyclic colormap,
    matching Qiskit's own convention) -- the same encoding
    qiskit.visualization.plot_state_qsphere used, built directly from the
    statevector's own amplitudes.

    Examples
    --------
    >>> import numpy as np
    >>> from dashboard_core.state_visuals import native_qsphere_figure
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    >>> fig = native_qsphere_figure(bell)
    >>> type(fig).__name__
    'Figure'
    """
    n_qubits = int(np.log2(len(statevector)))
    probs = np.abs(statevector) ** 2
    phases = np.angle(statevector)

    # Vectorized threshold: a Python-level loop over `range(len(statevector))`
    # is O(2**n_qubits) pure-Python overhead regardless of how many states
    # actually pass -- measured directly, this dominated runtime at 24
    # qubits (dense's own practical ceiling) even though only a handful of
    # states end up plotted. np.where does the same filtering in C.
    indices = np.where(probs > prob_threshold)[0]
    if len(indices) == 0:
        # A state spread thin enough that nothing clears prob_threshold
        # (e.g. a wide random state at high qubit count) -- show the
        # single most probable state rather than an empty sphere, so
        # there's always at least one real point plotted.
        indices = np.array([int(np.argmax(probs))])
    weights = np.array([bin(int(i)).count("1") for i in indices], dtype=int)

    # phi (azimuthal position within a weight's ring) needs each point's
    # rank among same-weight points, in index order -- np.unique's
    # return_inverse plus a per-group running count gets that without a
    # Python-level dict-of-lists pass.
    order = np.argsort(weights, kind="stable")
    sorted_idx, sorted_w = indices[order], weights[order]
    group_sizes = np.bincount(sorted_w, minlength=n_qubits + 1)
    # rank-within-group for each entry in sorted order: 0,1,2,... resets
    # to 0 at each group boundary. offsets[w] is where weight-w's group
    # starts in the sorted array (sum of every smaller group's size);
    # subtracting it from the running global index 0..N-1 gives each
    # point's position within its own group, e.g. weights [0,1,1,1,2,2,2,3]
    # -> ranks [0,0,1,2,0,1,2,0] -- verified by hand before use here.
    offsets = np.concatenate([[0], np.cumsum(group_sizes)[:-1]])
    rank_within_group = np.arange(len(sorted_idx)) - offsets[sorted_w]
    counts_per_point = group_sizes[sorted_w]

    theta = np.pi * sorted_w / n_qubits if n_qubits > 0 else np.zeros_like(sorted_w, dtype=float)
    phi = 2 * np.pi * rank_within_group / counts_per_point
    xs = np.sin(theta) * np.cos(phi)
    ys = np.sin(theta) * np.sin(phi)
    zs = np.cos(theta)

    sorted_probs = probs[sorted_idx]
    sorted_phases = phases[sorted_idx]
    cmap = plt.get_cmap("hsv")
    colors = cmap((sorted_phases % (2 * np.pi)) / (2 * np.pi))
    sizes = 80 + 1500 * sorted_probs

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:36j, 0:np.pi:18j]
    ax.plot_wireframe(
        np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
        color="#e0e0e0", linewidth=0.3,
    )

    # One Line3DCollection for every "stem" (instead of one ax.plot() call
    # per point) and one ax.scatter() call for every marker (instead of
    # one per point) -- measured directly: per-point matplotlib artist
    # calls, not the math above, were the real cost of this function (a
    # wide random 10-qubit state with 300+ states over prob_threshold took
    # ~1.4s with the per-point version, regardless of whether text labels
    # were also being drawn).
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    segments = [[(0, 0, 0), (x, y, z)] for x, y, z in zip(xs, ys, zs)]
    linewidths = 1.0 + 3 * sorted_probs
    ax.add_collection3d(Line3DCollection(segments, colors=_QSPHERE_LINE_COLOR, linewidths=linewidths))
    ax.scatter(xs, ys, zs, s=sizes, color=colors, edgecolor="black", linewidth=0.5, zorder=5)

    # Per-point text labels are unreadable past a few dozen overlapping
    # entries anyway (a wide random state can push 300+ states over
    # prob_threshold) -- dropped above this count, keeping the dots/
    # lines/colors (the actual data) intact.
    _MAX_LABELS = 32
    if len(sorted_idx) <= _MAX_LABELS:
        for idx, x, y, z in zip(sorted_idx, xs, ys, zs):
            label = format(int(idx), f"0{n_qubits}b")
            ax.text(x * 1.15, y * 1.15, z * 1.15, f"|{label}⟩", fontsize=7, ha="center")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.set_title("Q-sphere (size = probability, color = phase)", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2 * np.pi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])
    cbar.set_label("phase", fontsize=8)

    fig.tight_layout()
    return fig
