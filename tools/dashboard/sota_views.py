"""
Views from the circuit-visualization literature, computed from the engine's
statevector (the engine itself is not modified):

  step_states / difference_rows  state vector difference highlighting, gate by
                                 gate (McGuffin and Robert, arXiv:2510.00895)
  half_matrix_figure             pairwise concurrence and ZZ correlation in a
                                 triangular half-matrix (same paper)
  qubit_table                    per-qubit probability, purity and Bloch
                                 components (same paper, Sec. 1)
  ket_string                     ket notation of the state (Migdal et al.,
                                 Quantum Flytrap, arXiv:2203.13300)

Qubit numbering follows the dashboard (Qiskit convention: qubit 0 is the
rightmost bit of a basis label).
"""
import matplotlib.pyplot as plt
import numpy as np

import dense_evolution as de

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _rho(sv, n, qubits):
    psi = np.asarray(sv).reshape([2] * n)
    axes = [n - 1 - q for q in qubits]
    rest = [a for a in range(n) if a not in axes]
    m = np.transpose(psi, axes + rest).reshape(2 ** len(qubits), -1)
    return m @ m.conj().T


def qubit_table(sv, n):
    rows = []
    for q in range(n):
        r = _rho(sv, n, [q])
        rows.append({
            "qubit": q,
            "P(1)": float(r[1, 1].real),
            "purity": float(np.trace(r @ r).real),
            "<X>": float(np.trace(r @ _X).real),
            "<Y>": float(np.trace(r @ _Y).real),
            "<Z>": float(np.trace(r @ _Z).real),
        })
    return rows


def concurrence(rho):
    yy = np.kron(_Y, _Y)
    r = rho @ yy @ rho.conj() @ yy
    lam = np.sqrt(np.clip(np.sort(np.linalg.eigvals(r).real)[::-1], 0, None))
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def zz_correlation(sv, n, i, j):
    r2 = _rho(sv, n, [i, j])
    zi = float(np.trace(_rho(sv, n, [i]) @ _Z).real)
    zj = float(np.trace(_rho(sv, n, [j]) @ _Z).real)
    return float(np.trace(r2 @ np.kron(_Z, _Z)).real) - zi * zj


def half_matrix_figure(sv, n):
    conc = np.full((n, n), np.nan)
    corr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i):
            conc[i, j] = concurrence(_rho(sv, n, [i, j]))
            corr[i, j] = zz_correlation(sv, n, i, j)
    fig, axes = plt.subplots(1, 2, figsize=(2.2 * n + 2, 1.1 * n + 1.6))
    for ax, data, title, cmap, lim in ((axes[0], conc, "Concurrence", "viridis", (0, 1)),
                                       (axes[1], corr, "ZZ correlation", "coolwarm", (-1, 1))):
        im = ax.imshow(data, cmap=cmap, vmin=lim[0], vmax=lim[1])
        for i in range(n):
            for j in range(i):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmap == "viridis" and data[i, j] < 0.6 else "black")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("qubit")
        ax.set_ylabel("qubit")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def ket_string(sv, n, tol=1e-9, max_terms=32):
    terms = []
    for i, a in enumerate(np.asarray(sv)):
        if abs(a) > tol:
            c = f"{a.real:+.3f}" if abs(a.imag) < tol else f"({a.real:+.3f}{a.imag:+.3f}i)"
            terms.append(f"{c}|{format(i, f'0{n}b')}⟩")
    more = f" + … ({len(terms) - max_terms} more)" if len(terms) > max_terms else ""
    return " ".join(terms[:max_terms]) + more


def step_states(ops, n):
    states = []
    for k in range(len(ops) + 1):
        sim = de.DenseSVSimulator(n)
        if k:
            sim.run_circuit_jit(list(ops[:k]))
        s = np.asarray(sim.get_statevector()).reshape([2] * n)
        states.append(np.transpose(s, list(range(n))[::-1]).reshape(-1))
    return states


def difference_rows(prev, cur, n, tol=1e-9):
    rows = []
    for i, (a, b) in enumerate(zip(prev, cur)):
        if abs(a) < tol and abs(b) < tol:
            continue
        if abs(a) < tol:
            mark = "appears"
        elif abs(b) < tol:
            mark = "vanishes"
        elif abs(abs(b) - abs(a)) > tol:
            mark = "grows" if abs(b) > abs(a) else "shrinks"
        elif abs(np.angle(b / a)) > 1e-6:
            mark = "phase changes"
        else:
            mark = "unchanged"
        rows.append({"state": format(i, f"0{n}b"), "before": f"{a.real:+.4f}{a.imag:+.4f}i",
                     "after": f"{b.real:+.4f}{b.imag:+.4f}i", "change": mark})
    return rows
