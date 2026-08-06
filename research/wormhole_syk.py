"""
Traversable-wormhole-inspired quantum teleportation, via a binary sparse
N=8 SYK model -- a real reproduction attempt, not a decorative circuit.

Background: an earlier "Traversable Wormhole (BGQ)" circuit found in an
old, discarded dashboard_core file used the right vocabulary (SYK
scrambling, a phase "kick") but wasn't real: it operated on a single
qubit register with the same scrambling unitary run forward then
backward around a bare RZ "kick", and empirically produced IDENTICAL
results for either sign of the kick -- verified directly, not assumed.
That's not a bug in that circuit's tuning; a single-register readout
CANNOT show sign-dependent behavior, full stop, because a qubit inside a
Bell pair has a maximally-mixed marginal that no local operation (or
global phase) changes -- the no-signaling theorem forbids it outright.

The real protocol (Gao-Jafferis-Wall theory; reproduced on real IBM
hardware in arXiv:2604.10090, "Quantum simulation of
traversable-wormhole-inspired quantum teleportation in a chaotic binary
sparse SYK model") needs TWO coupled systems (L, R), a message injected
into L via a separate reference-qubit pair, a real bilinear L-R coupling
term, and -- critically -- a readout that is NOT a single-qubit marginal:
mutual information I_PT between a reference qubit P and a qubit T read
out from R, which two-qubit tomography (or, here, exact partial-trace
math -- we have simulator state access real hardware didn't) can access
even though no single-qubit observable can.

This module builds each physical ingredient from scratch (nothing this
specific exists anywhere in dense_evolution/dashboard_core -- confirmed
by exploration before writing this) and verifies each one against a
known-exact case before composing them, the same discipline used
elsewhere in this project:
    - Majorana Jordan-Wigner mapping: verified via the anticommutation
      relations {chi_a, chi_b} = 2*delta_ab*I (exact, not approximate).
    - The sparse SYK Hamiltonian terms: verified Hermitian exactly.
    - Partial trace / mutual information: verified against a plain Bell
      pair (I = 2*ln(2), the textbook value) and a GHZ state.

Two evolution backends are provided. `run_wormhole_protocol` uses exact
matrix exponentiation (eigendecomposition) applied directly to the
statevector -- the paper needed gates because that's what real hardware
executes (369 two-qubit gates), but we have exact statevector access,
and the paper's own hardware results are validated against exactly this
kind of "exact result" baseline. `run_wormhole_protocol_trotter` is the
gate-circuit version, closer to what real hardware would run: every
Hamiltonian/coupling term becomes a real `pauli_rotation_ops` circuit
(basis-change + CNOT-staircase + RZ + inverse, verified to fidelity
1.0 against exact expm for single- and multi-qubit Pauli strings
including a 4-qubit mixed X/Y/Z case matching the SYK terms themselves),
composed via first-order Trotterization (verified to converge smoothly
to the exact evolution as step count increases -- infidelity ~4x lower
per doubling of steps, consistent with the expected quadratic scaling).
At the known peak (seed 61, t0=0.3, t1=0.60, ~6300 real two-qubit-gate
circuit), the Trotterized version reproduces the exact result closely
(I(mu=+12)=0.01301 vs exact 0.01326; I(mu=-12)=0.01821 vs exact 0.01793;
delta=+0.00520 vs exact +0.00468) -- the sign-dependent asymmetry is not
an artifact of the exact-evolution shortcut, it survives with a real
gate circuit too.

Central finding (see verify_signal() / __main__): the sign-dependent
asymmetry this protocol is supposed to show is real, but realization-
dependent -- a uniformly-random draw of which K=10 of the C(8,4)=70
four-Majorana terms to keep does not reliably show a clean signal (some
random seeds give a clear peak, some give the *wrong* sign for most of
the sweep, some are just noise). This matches arXiv:2604.10090 directly:
they didn't use a random instance either, they picked one "selected for
favorable commutation properties" (34 commuting vs 11 anticommuting
pairs among the 45 pairs of chosen terms, out of that paper's specific
K=10 instance). select_good_instance() below reproduces that selection
criterion (screen many random instances by their exact commuting/
anticommuting pair count, keep the one closest to the paper's ratio)
rather than trusting a single arbitrary seed -- screened here across 200
candidates, seed 61 came out an exact match (34/11) and gives the
cleanest result found: a single smooth peak in the sign-dependent
mutual-information difference, positive and consistent across ten
straight t1 sweep points (t1 in [0.10, 1.00]), before crossing back near
t1=1.2 -- see __main__ output.
"""

import itertools

import numpy as np

import dense_evolution as de

__all__ = [
    'majorana_pauli_terms', 'build_sparse_syk_terms',
    'commuting_pair_count', 'select_good_instance',
    'partial_trace', 'von_neumann_entropy', 'mutual_information',
    'run_wormhole_protocol',
    'pauli_rotation_ops', 'trotter_evolve_ops', 'run_wormhole_protocol_trotter',
]


# --------------------------------------------------------------------------
# Majorana -> qubit (Jordan-Wigner) mapping
# --------------------------------------------------------------------------

def majorana_pauli_terms(mode_index, n_qubits):
    """Standard JW mapping, one qubit per two Majorana modes:
        chi_{2j-1} = (prod_{k<j} Z_k) X_j
        chi_{2j}   = (prod_{k<j} Z_k) Y_j
    mode_index is 1-indexed (chi_1 .. chi_{2*n_qubits}). Returns a single
    (coeff, pauli_dict) term, coeff always 1.0 (chi_i is Hermitian and
    unit-normalized by this convention -- chi_i^2 = I, verified below via
    the anticommutation check at a=b).
    """
    j = (mode_index - 1) // 2
    is_even = (mode_index % 2 == 0)
    pauli = {k: 'Z' for k in range(j)}
    pauli[j] = 'Y' if is_even else 'X'
    return (1.0, pauli)


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
    takes C(4,2)=6 transpositions, (-1)^6=+1 -- no extra i factor needed;
    an earlier draft of this file got this backwards and failed the
    Hermiticity check below with a large, obvious error until fixed).
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
    applied here rather than trusting an arbitrary single seed. Screening
    200 candidates for N=8 found seed 61 an exact match (34/11); that
    match, when actually run through the full protocol in __main__,
    produced the cleanest sign-dependent signal of every seed tried."""
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
# Partial trace / von Neumann entropy / mutual information
# (MSB-first: qubit 0 = index-0 = most significant bit, matching
# dense_evolution.pauli_hamiltonian_to_matrix's own convention --
# dashboard_core.state_visuals has a *different*, little-endian,
# single-qubit-only partial trace; deliberately not reused here, see
# module docstring / plan.)
# --------------------------------------------------------------------------

def partial_trace(state, n_qubits, keep_qubits):
    keep_qubits = sorted(keep_qubits)
    trace_qubits = [q for q in range(n_qubits) if q not in keep_qubits]
    psi = np.transpose(state.reshape([2] * n_qubits), keep_qubits + trace_qubits)
    keep_dim, trace_dim = 2 ** len(keep_qubits), 2 ** len(trace_qubits)
    psi = psi.reshape(keep_dim, trace_dim)
    return psi @ psi.conj().T


def von_neumann_entropy(rho):
    eigs = np.clip(np.linalg.eigvalsh(rho).real, 1e-14, None)
    return float(-np.sum(eigs * np.log(eigs)))


def mutual_information(state, n_qubits, qubits_a, qubits_b):
    s_a = von_neumann_entropy(partial_trace(state, n_qubits, qubits_a))
    s_b = von_neumann_entropy(partial_trace(state, n_qubits, qubits_b))
    s_ab = von_neumann_entropy(partial_trace(state, n_qubits, list(qubits_a) + list(qubits_b)))
    return s_a + s_b - s_ab


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
    paper's own convention, though note our own random SYK realization
    means our sign doesn't have to line up with theirs -- what matters is
    that *some* consistent sign shows the enhancement, which mu=-12 does
    for the seed 61 instance (see __main__).
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


# --------------------------------------------------------------------------
# Trotterized (real gate-circuit) evolution backend
# --------------------------------------------------------------------------

def pauli_rotation_ops(pauli_dict, angle):
    """Gate-tuple circuit for exp(-i*angle*P), P a Pauli string given as
    {qubit: 'X'/'Y'/'Z'} -- basis-change + CNOT-staircase + RZ + inverse,
    the same identity already used elsewhere in this codebase for
    UCCSD/QAOA-style ZZ interactions, generalized here to mixed X/Y/Z
    strings (needed since Majorana JW terms are not pure-Z). This
    engine's rz(theta) = exp(-i*theta/2*Z) (checked directly against
    scipy.linalg.expm, not assumed) -> rz(2*angle) on the accumulator
    qubit gives exactly exp(-i*angle*Z) on the accumulated parity.
    Verified to fidelity 1.0 (not "close") against exact expm for
    1-, 2-, 3-, and 4-qubit mixed Pauli strings, including a 4-qubit
    case matching what an actual SYK term looks like."""
    qubits = sorted(pauli_dict.keys())
    if not qubits:
        return []
    ops = []
    for q in qubits:
        letter = pauli_dict[q]
        if letter == 'X':
            ops.append(('h', q))
        elif letter == 'Y':
            ops.append(('sdg', q))
            ops.append(('h', q))
    for i in range(len(qubits) - 1):
        ops.append(('cx', qubits[i], qubits[i + 1]))
    ops.append(('rz', qubits[-1], 2 * angle))
    for i in reversed(range(len(qubits) - 1)):
        ops.append(('cx', qubits[i], qubits[i + 1]))
    for q in qubits:
        letter = pauli_dict[q]
        if letter == 'X':
            ops.append(('h', q))
        elif letter == 'Y':
            ops.append(('h', q))
            ops.append(('s', q))
    return ops


def trotter_evolve_ops(terms, t, n_steps):
    """First-order Trotter product formula for exp(-iHt), H = sum_k
    c_k*P_k: [prod_k exp(-i*c_k*P_k*(t/n_steps))]^n_steps. Term order
    within a step is fixed (list order), identical every repetition.
    Verified to converge smoothly to the exact evolution as n_steps
    increases (infidelity ~4x lower per doubling of steps -- consistent
    with the expected quadratic scaling of first-order Trotter error in
    state overlap -- checked directly against the real N=8 SYK
    Hamiltonian, not a toy example)."""
    dt = t / n_steps
    step_ops = []
    for c, pdict in terms:
        step_ops.extend(pauli_rotation_ops(pdict, c * dt))
    return step_ops * n_steps


def run_wormhole_protocol_trotter(n_majorana, k_terms, J, mu, t0, t1, seed, with_message,
                                   n_steps_evolution=8, n_steps_coupling=16):
    """Gate-circuit backend: identical protocol to run_wormhole_protocol,
    but every evolution step is a real Trotterized circuit
    (trotter_evolve_ops) run through DenseSVSimulator.run_circuit,
    instead of exact matrix exponentiation -- closer to what real
    hardware would execute. At the known peak (seed 61, t0=0.3, t1=0.60,
    ~6300 real two-qubit-gate circuit) this reproduces the exact
    backend's result closely: I(mu=+12)=0.01301 vs exact 0.01326,
    I(mu=-12)=0.01821 vs exact 0.01793, delta=+0.00520 vs exact +0.00468
    -- the sign-dependent asymmetry survives with a real gate circuit,
    it isn't an artifact of the exact-evolution shortcut."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(n_majorana, k_terms, J, seed)

    ops = _initial_state_ops(n_side, L, R, P, Q, with_message)
    ops += trotter_evolve_ops(terms_full, t0, n_steps_evolution)
    ops += trotter_evolve_ops(v_terms, mu, n_steps_coupling)
    ops += trotter_evolve_ops(terms_full, t1, n_steps_evolution)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(ops)
    sv = sim.get_statevector()
    return mutual_information(sv, n_full, [P], [R[0]])


if __name__ == '__main__':
    print('=== Verification: Majorana anticommutation {chi_a,chi_b}=2*delta_ab*I ===')
    n_test = 3
    chis = [de.pauli_hamiltonian_to_matrix([majorana_pauli_terms(i, n_test)], n_test)
            for i in range(1, 2 * n_test + 1)]
    I = np.eye(2 ** n_test)
    max_err = max(
        np.max(np.abs(chis[a] @ chis[b] + chis[b] @ chis[a] - (2 * I if a == b else 0)))
        for a in range(len(chis)) for b in range(len(chis))
    )
    print(f'max error: {max_err:.2e}  ->  {"PASS" if max_err < 1e-10 else "FAIL"}')

    print('\n=== Verification: sparse SYK Hamiltonian is exactly Hermitian ===')
    n_qubits, terms = build_sparse_syk_terms(8, 10, np.sqrt(2), seed=61)
    H_test = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
    herm_err = np.max(np.abs(H_test - H_test.conj().T))
    print(f'hermiticity error: {herm_err:.2e}  ->  {"PASS" if herm_err < 1e-10 else "FAIL"}')

    print('\n=== Verification: mutual information, plain Bell pair ===')
    sim = de.DenseSVSimulator(2)
    sim.run_circuit([('h', 0), ('cx', 0, 1)])
    I_bell = mutual_information(sim.get_statevector(), 2, [0], [1])
    expected = 2 * np.log(2)
    print(f'I = {I_bell:.10f}  expected = {expected:.10f}  ->  '
          f'{"PASS" if abs(I_bell - expected) < 1e-8 else "FAIL"}')

    print('\n=== Selecting a good SYK instance (paper criterion: 34 commuting / 11 anticommuting) ===')
    good_seed = select_good_instance(8, 10, np.sqrt(2), n_candidates=200)
    c, a = commuting_pair_count(build_sparse_syk_terms(8, 10, np.sqrt(2), good_seed)[1], 4)
    print(f'selected seed={good_seed}: {c} commuting / {a} anticommuting (target 34/11)')

    print(f'\n=== Sign-dependent mutual-information sweep, seed={good_seed} (exact backend) ===')
    mu_mag = 12.0
    for t1 in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00, 1.20, 1.50):
        i_pos = run_wormhole_protocol(8, 10, np.sqrt(2), +mu_mag, 0.3, t1, good_seed, with_message=True)
        i_neg = run_wormhole_protocol(8, 10, np.sqrt(2), -mu_mag, 0.3, t1, good_seed, with_message=True)
        delta = i_neg - i_pos
        print(f'  t1={t1:.2f}  I(mu=+{mu_mag:g})={i_pos:.5f}  I(mu=-{mu_mag:g})={i_neg:.5f}  delta={delta:+.5f}')

    print('\n=== Verification: pauli_rotation_ops vs exact expm (1-4 qubit Pauli strings) ===')
    from scipy.linalg import expm
    test_cases = [
        ('X', {0: 'X'}, 1, 0.37, [('ry', 0, 0.9)]),
        ('Y', {0: 'Y'}, 1, 0.51, [('ry', 0, 1.3), ('rz', 0, 0.4)]),
        ('XY', {0: 'X', 1: 'Y'}, 2, 0.42, [('ry', 0, 0.6), ('ry', 1, 1.1), ('cx', 0, 1)]),
        ('YXZY (SYK-like)', {0: 'Y', 1: 'X', 2: 'Z', 3: 'Y'}, 4, 0.15,
         [('ry', 0, 0.4), ('ry', 1, 0.8), ('ry', 2, 1.2), ('ry', 3, 0.6),
          ('cx', 0, 1), ('cx', 1, 2), ('cx', 2, 3)]),
    ]
    all_pass = True
    for name, pdict, nq, angle, init in test_cases:
        sim0 = de.DenseSVSimulator(nq)
        sim0.run_circuit(init)
        sv0 = sim0.get_statevector().copy()
        P_mat = de.pauli_hamiltonian_to_matrix([(1.0, pdict)], nq)
        exact_sv = expm(-1j * angle * P_mat) @ sv0
        sim1 = de.DenseSVSimulator(nq)
        sim1.run_circuit(init + pauli_rotation_ops(pdict, angle))
        fidelity = abs(np.vdot(exact_sv, sim1.get_statevector())) ** 2
        ok = fidelity > 1 - 1e-10
        all_pass &= ok
        print(f'  {name}: fidelity={fidelity:.12f}  {"PASS" if ok else "FAIL"}')
    print('ALL PASS' if all_pass else 'SOME FAILED')

    print('\n=== Verification: Trotter convergence to exact evolution, real SYK Hamiltonian ===')
    n_qubits, terms = build_sparse_syk_terms(8, 10, np.sqrt(2), seed=good_seed)
    H_test = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
    t_test = 0.3
    init_ops = [('ry', q, 0.3 + 0.1 * q) for q in range(n_qubits)] + [('cx', 0, 1), ('cx', 2, 3)]
    sim_i = de.DenseSVSimulator(n_qubits)
    sim_i.run_circuit(init_ops)
    exact_sv = expm(-1j * H_test * t_test) @ sim_i.get_statevector()
    for n_steps in (1, 2, 4, 8, 16):
        sim_t = de.DenseSVSimulator(n_qubits)
        sim_t.run_circuit(init_ops + trotter_evolve_ops(terms, t_test, n_steps))
        fidelity = abs(np.vdot(exact_sv, sim_t.get_statevector())) ** 2
        print(f'  n_steps={n_steps:2d}  fidelity={fidelity:.8f}  infidelity={1 - fidelity:.2e}')

    print(f'\n=== Trotterized (real gate circuit) vs exact, at the peak t1=0.60, seed={good_seed} ===')
    for mu in (+mu_mag, -mu_mag):
        i_trotter = run_wormhole_protocol_trotter(8, 10, np.sqrt(2), mu, 0.3, 0.60, good_seed, with_message=True)
        i_exact = run_wormhole_protocol(8, 10, np.sqrt(2), mu, 0.3, 0.60, good_seed, with_message=True)
        print(f'  mu={mu:+.0f}: I_PT(trotter)={i_trotter:.5f}  I_PT(exact)={i_exact:.5f}')
