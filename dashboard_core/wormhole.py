"""
Traversable-wormhole-inspired quantum teleportation (Gao-Jafferis-Wall
theory), via a binary sparse Sachdev-Ye-Kitaev (SYK) model -- the real
protocol from arXiv:2604.10090 ("Quantum simulation of
traversable-wormhole-inspired quantum teleportation in a chaotic binary
sparse SYK model", 2026), a real hardware reproduction this module
mirrors as an exact/Trotterized simulation.

Ported from research/wormhole_syk.py once the reproduction was verified
end to end (see that file's git history and research/wormhole_syk.md for
the full derivation, the no-signaling-theorem explanation of why an
earlier decorative "Traversable Wormhole (BGQ)" circuit could never have
worked, and every verification step: Majorana anticommutation, SYK
Hermiticity, Bell-pair/GHZ mutual information, the paper's own
instance-selection criterion, Trotter-vs-exact convergence). The generic
building blocks (Majorana JW mapping, partial trace / entropy / mutual
information, Trotterized gate-circuit evolution) live in the
`dense_evolution` package proper (`dense_evolution.fermions`, `.entropy`,
`.trotter`); only what's genuinely SYK/wormhole-specific -- the sparse
Hamiltonian construction, the paper's commuting-pair selection criterion,
and the two-sided teleportation protocol itself -- lives here.

Two evolution backends, both real gate circuits or exact matrix math run
through dense_evolution.DenseSVSimulator (never Qiskit, never a mock):
`run_wormhole_protocol` evolves via exact matrix exponentiation
(eigendecomposition-based, cheap and exact -- the paper's own hardware
run is validated against exactly this kind of baseline);
`run_wormhole_protocol_trotter` evolves via a real Trotterized gate
circuit (`dense_evolution.trotter_evolve_ops`), closer to what actual
hardware executes -- verified in research/wormhole_syk.py to reproduce
the exact backend's result closely at the known signal peak (seed 61,
t0=0.3, t1=0.60: I(mu=+12)=0.01301 vs exact 0.01326, I(mu=-12)=0.01821
vs exact 0.01793).

Central, honest finding carried over from the research reproduction: the
sign-dependent teleportation signal (mutual information between a
reference qubit P and a qubit read out from the R register, higher for
one coupling sign than the other) is real but realization-dependent -- a
uniformly-random draw of which SYK terms to keep does not reliably show
it. `select_good_instance` reproduces the paper's own fix: screen many
candidate seeds by their exact commuting/anticommuting term-pair count
and keep the one closest to the paper's own ratio (34 commuting / 11
anticommuting among 45 pairs, for their K=10 instance), rather than
trusting an arbitrary seed. For N=8 Majorana/side, seed 61 is an exact
match and gives the cleanest signal found (see research/wormhole_syk.md).
"""

import itertools

import numpy as np

import dense_evolution as de
from dense_evolution import mutual_information, majorana_pauli_terms, trotter_evolve_ops

__all__ = [
    'build_sparse_syk_terms', 'commuting_pair_count', 'select_good_instance',
    'run_wormhole_protocol', 'run_wormhole_protocol_trotter',
]


def _multiply_pauli_dicts(dicts):
    """Multiply several single-qubit-Pauli dicts together, per qubit,
    tracking the i^k phase from same-qubit Pauli products (XY=iZ etc.).
    Returns (phase, merged_dict)."""
    mul = {
        ('X', 'X'): (1, None), ('Y', 'Y'): (1, None), ('Z', 'Z'): (1, None),
        ('X', 'Y'): (1j, 'Z'), ('Y', 'X'): (-1j, 'Z'),
        ('Y', 'Z'): (1j, 'X'), ('Z', 'Y'): (-1j, 'X'),
        ('Z', 'X'): (1j, 'Y'), ('X', 'Z'): (-1j, 'Y'),
    }
    merged, phase = {}, 1.0
    for d in dicts:
        for q, p in d.items():
            if q not in merged:
                merged[q] = p
            else:
                ph, newp = mul[(merged[q], p)]
                phase *= ph
                if newp is None:
                    del merged[q]
                else:
                    merged[q] = newp
    return phase, merged


def _embed(mode_index, n_qubits_side, offset):
    """A single side's Majorana term, shifted into a larger joint
    register by `offset` qubits (0 for the L side, n_qubits_side for R)."""
    _, pdict = majorana_pauli_terms(mode_index, n_qubits_side)
    return {q + offset: p for q, p in pdict.items()}


# --------------------------------------------------------------------------
# Binary sparse SYK Hamiltonian (arXiv:2604.10090's construction)
# --------------------------------------------------------------------------

def build_sparse_syk_terms(n_majorana, k_terms, J, seed):
    """K of the C(n_majorana,4) four-Majorana products chi_i*chi_j*chi_k*chi_l
    (i<j<k<l), each coefficient +-J/sqrt(k_terms) with a random sign --
    the paper's K=10, J=sqrt(2) definition for N=8. The bare product of
    4 Majoranas is already Hermitian (reversing 4 anticommuting factors
    takes C(4,2)=6 transpositions, (-1)^6=+1 -- no extra i factor needed).
    Returns (n_qubits, terms) where terms is ready for
    dense_evolution.pauli_hamiltonian_to_matrix.
    """
    n_qubits = n_majorana // 2
    all_quads = list(itertools.combinations(range(1, n_majorana + 1), 4))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_quads), size=k_terms, replace=False)
    coupling = J / np.sqrt(k_terms)
    terms = []
    for idx in chosen:
        quad = all_quads[idx]
        sign = rng.choice([-1.0, 1.0])
        dicts = [majorana_pauli_terms(m, n_qubits)[1] for m in quad]
        phase, merged = _multiply_pauli_dicts(dicts)
        terms.append((sign * coupling * phase, merged))
    return n_qubits, terms


def commuting_pair_count(terms, n_qubits):
    """Exact commuting/anticommuting pair count among a set of terms
    (each term's operator, ignoring its coefficient) -- the paper's own
    instance-selection diagnostic (their chosen K=10 instance: 34
    commuting / 11 anticommuting, out of C(10,2)=45 pairs)."""
    matrices = [de.pauli_hamiltonian_to_matrix([(1.0, t[1])], n_qubits) for t in terms]
    commuting = anticommuting = 0
    for a, b in itertools.combinations(range(len(matrices)), 2):
        comm = matrices[a] @ matrices[b] - matrices[b] @ matrices[a]
        if np.max(np.abs(comm)) < 1e-9:
            commuting += 1
        else:
            anticommuting += 1
    return commuting, anticommuting


def select_good_instance(n_majorana, k_terms, J, n_candidates=200, target_commuting=34):
    """Screen n_candidates random seeds by their exact commuting-pair
    count (cheap: only needs the k_terms operators' own small matrices,
    not the full protocol simulation) and return the seed whose count is
    closest to target_commuting -- the paper's selection criterion,
    applied here rather than trusting an arbitrary single seed."""
    n_qubits = n_majorana // 2
    best_seed, best_diff = None, None
    for seed in range(n_candidates):
        _, terms = build_sparse_syk_terms(n_majorana, k_terms, J, seed)
        c, _ = commuting_pair_count(terms, n_qubits)
        diff = abs(c - target_commuting)
        if best_diff is None or diff < best_diff:
            best_seed, best_diff = seed, diff
    return best_seed


# --------------------------------------------------------------------------
# The full protocol
# --------------------------------------------------------------------------

def _evolve(state, eigvals, eigvecs, t):
    """exp(-i*H*t) @ state via a precomputed eigendecomposition -- cheap
    to reuse across a t-sweep instead of re-diagonalizing every call."""
    coeffs = eigvecs.conj().T @ state
    return eigvecs @ (coeffs * np.exp(-1j * eigvals * t))


def _protocol_layout(n_majorana, k_terms, J, seed):
    """Shared register layout + Hamiltonian/coupling terms for both
    evolution backends below: L, R = SYK registers; P, Q = reference and
    message qubits. terms_full is H_L+H_R's terms (on the joint
    register); v_terms is the L-R coupling V = (1/(4*n_majorana)) *
    sum_j chi_L^j chi_R^j (a bare, disjoint-qubit union of each side's
    Majorana term -- no extra phase needed, since chi_L^j and chi_R^j
    act on entirely different qubits and their tensor product is
    automatically Hermitian whenever each factor is)."""
    n_side = n_majorana // 2
    n_full = 2 * n_side + 2
    L, R = list(range(n_side)), list(range(n_side, 2 * n_side))
    P, Q = 2 * n_side, 2 * n_side + 1

    _, terms_l_raw = build_sparse_syk_terms(n_majorana, k_terms, J, seed)
    terms_l = [(c, dict(pd)) for c, pd in terms_l_raw]
    _, terms_r_raw = build_sparse_syk_terms(n_majorana, k_terms, J, seed)
    terms_r = [(c, {q + n_side: p for q, p in pd.items()}) for c, pd in terms_r_raw]
    terms_full = terms_l + terms_r

    v_terms = []
    for j in range(1, n_majorana + 1):
        merged = _embed(j, n_side, 0)
        merged.update(_embed(j, n_side, n_side))
        v_terms.append((1.0 / (4 * n_majorana), merged))

    return n_side, n_full, L, R, P, Q, terms_full, v_terms


def _initial_state_ops(n_side, L, R, P, Q, with_message):
    """TFD at beta=0 (n_side Bell pairs L_i-R_i) + a separate P,Q
    reference/message Bell pair, with Q swapped into L[0] as the
    injected message when with_message=True."""
    ops = []
    for i in range(n_side):
        ops += [('h', L[i]), ('cx', L[i], R[i])]
    ops += [('h', P), ('cx', P, Q)]
    if with_message:
        ops.append(('swap', Q, L[0]))
    return ops


def run_wormhole_protocol(n_majorana, k_terms, J, mu, t0, t1, seed, with_message):
    """Exact-evolution backend: TFD/message setup -> evolve under H_L+H_R
    for t0 (exact matrix exponential, eigendecomposition-based) ->
    coupling exp(i*mu*V) (exact) -> evolve for t1 (exact) -> mutual
    information between P and R[0].

    mu<0 vs mu>0 is the "traversable" vs "non-traversable" sign in the
    paper's own convention, though note a given random SYK realization's
    sign doesn't have to line up with theirs -- what matters is that
    *some* consistent sign shows the enhancement (mu=-12 does, for the
    seed 61 instance -- see research/wormhole_syk.md).
    """
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(n_majorana, k_terms, J, seed)
    H = de.pauli_hamiltonian_to_matrix(terms_full, n_full)
    eigvals, eigvecs = np.linalg.eigh(H)
    V = de.pauli_hamiltonian_to_matrix(v_terms, n_full)
    v_eigvals, v_eigvecs = np.linalg.eigh(V)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, with_message))
    sv = sim.get_statevector()

    sv = _evolve(sv, eigvals, eigvecs, t0)
    sv = _evolve(sv, v_eigvals, v_eigvecs, mu)
    sv = _evolve(sv, eigvals, eigvecs, t1)
    return mutual_information(sv, n_full, [P], [R[0]])


def run_wormhole_protocol_trotter(n_majorana, k_terms, J, mu, t0, t1, seed, with_message,
                                   n_steps_evolution=8, n_steps_coupling=16):
    """Gate-circuit backend: identical protocol to run_wormhole_protocol,
    but every evolution step is a real Trotterized circuit
    (dense_evolution.trotter_evolve_ops) run through
    DenseSVSimulator.run_circuit, instead of exact matrix exponentiation
    -- closer to what real hardware would execute. At the known peak
    (seed 61, t0=0.3, t1=0.60, ~6300 real two-qubit-gate circuit) this
    reproduces the exact backend's result closely: I(mu=+12)=0.01301 vs
    exact 0.01326, I(mu=-12)=0.01821 vs exact 0.01793 -- the
    sign-dependent asymmetry survives with a real gate circuit, it isn't
    an artifact of the exact-evolution shortcut."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(n_majorana, k_terms, J, seed)

    ops = _initial_state_ops(n_side, L, R, P, Q, with_message)
    ops += trotter_evolve_ops(terms_full, t0, n_steps_evolution)
    ops += trotter_evolve_ops(v_terms, mu, n_steps_coupling)
    ops += trotter_evolve_ops(terms_full, t1, n_steps_evolution)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(ops)
    sv = sim.get_statevector()
    return mutual_information(sv, n_full, [P], [R[0]])
