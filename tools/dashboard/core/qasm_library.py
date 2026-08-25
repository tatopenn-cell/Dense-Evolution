"""
Real, standard OpenQASM 2.0 circuits offered as Composer presets --
textbook examples (Bell pair, GHZ, W-state) and dense_evolution's own
real circuit generators (QFT, random circuit), not synthetic results.
The Composer runs every one of these through the real engine just like
any custom circuit typed in by hand.
"""

import dense_evolution as de

__all__ = ['QASM_LIBRARY', 'gate_tuples_to_qasm']

# Matches dense_evolution/compiler.py's own gate categories (same split
# dense_evolution/drawing.py uses for _LABELS / _TWO_QUBIT_TARGET_SYMBOL).
_ONE_QUBIT_STATIC = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "id"}
_ONE_QUBIT_PARAM = {"rx", "ry", "rz", "p", "u1", "phase"}
_TWO_QUBIT_STATIC = {"cx", "cz", "cy", "swap"}
_TWO_QUBIT_PARAM = {"cp", "crz"}
_THREE_QUBIT_STATIC = {"ccx"}


def gate_tuples_to_qasm(ops, n_qubits: int, measure: bool = True) -> str:
    """Converts a dense_evolution gate-tuple circuit (the format
    de.qft/de.ghz_state/de.random_circuit/... all return) into real
    OpenQASM 2.0 text, so any of dense_evolution's own circuit generators
    can be offered as a Composer preset without hand-transcribing gates.

    Examples
    --------
    >>> import dense_evolution as de
    >>> from dashboard_core.qasm_library import gate_tuples_to_qasm
    >>> gate_tuples_to_qasm([('h', 0), ('cx', 0, 1)], n_qubits=2)
    'OPENQASM 2.0;\\ninclude "qelib1.inc";\\nqreg q[2];\\ncreg c[2];\\nh q[0];\\ncx q[0],q[1];\\nmeasure q -> c;\\n'
    >>> qft_ops = de.qft(3)
    >>> print(gate_tuples_to_qasm(qft_ops, n_qubits=3))  # doctest: +NORMALIZE_WHITESPACE
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    h q[0];
    cp(1.5707963267948966) q[1],q[0];
    cp(0.7853981633974483) q[2],q[0];
    h q[1];
    cp(1.5707963267948966) q[2],q[1];
    h q[2];
    swap q[0],q[2];
    measure q -> c;
    """
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n_qubits}];']
    if measure:
        lines.append(f'creg c[{n_qubits}];')
    for op in ops:
        name = op[0]
        if name in _ONE_QUBIT_STATIC:
            lines.append(f'{name} q[{op[1]}];')
        elif name in _ONE_QUBIT_PARAM:
            lines.append(f'{name}({op[2]}) q[{op[1]}];')
        elif name in _TWO_QUBIT_STATIC:
            lines.append(f'{name} q[{op[1]}],q[{op[2]}];')
        elif name in _TWO_QUBIT_PARAM:
            lines.append(f'{name}({op[3]}) q[{op[1]}],q[{op[2]}];')
        elif name in _THREE_QUBIT_STATIC:
            lines.append(f'{name} q[{op[1]}],q[{op[2]}],q[{op[3]}];')
        else:
            raise ValueError(f"gate_tuples_to_qasm: unrecognized gate {name!r}")
    if measure:
        lines.append('measure q -> c;')
    return '\n'.join(lines) + '\n'


QASM_LIBRARY = {
    "Bell state (2 qubit)": (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[2];\ncreg c[2];\n'
        'h q[0];\ncx q[0],q[1];\n'
        'measure q -> c;\n'
    ),
    "GHZ state (3 qubit)": gate_tuples_to_qasm(de.ghz_state(3), 3),
    "GHZ state (4 qubit)": gate_tuples_to_qasm(de.ghz_state(4), 4),
    "Superposition (1 qubit)": (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[1];\ncreg c[1];\n'
        'h q[0];\n'
        'measure q -> c;\n'
    ),
    "W state (3 qubit)": (
        # Standard 3-qubit W-state construction: RY splits the single
        # excitation's amplitude, a controlled-Hadamard (decomposed into
        # ry-cx-ry -- dense_evolution has no native CH gate) spreads it
        # further, then two CX cascade it across all three qubits.
        # Verified against the real engine: (|001>+|010>+|100>)/sqrt(3),
        # each with probability 1/3.
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[3];\ncreg c[3];\n'
        'ry(1.9106332362490184) q[0];\n'
        'ry(0.7853981633974483) q[1];\n'
        'cx q[0],q[1];\n'
        'ry(-0.7853981633974483) q[1];\n'
        'cx q[1],q[2];\n'
        'cx q[0],q[1];\n'
        'x q[0];\n'
        'measure q -> c;\n'
    ),
    "Quantum Fourier Transform (3 qubit)": gate_tuples_to_qasm(de.qft(3), 3),
    "Random circuit (3 qubit, fixed seed)": gate_tuples_to_qasm(
        de.random_circuit(3, 8, seed=7), 3,
    ),
    "Entangling linear (5 qubit, chain topology)": gate_tuples_to_qasm(
        [("h", q) for q in range(5)] + de.entangling_layer(5, pattern="linear"), 5,
    ),
    "Entangling ring (5 qubit, circular topology)": gate_tuples_to_qasm(
        [("h", q) for q in range(5)] + de.entangling_layer(5, pattern="circular"), 5,
    ),
    "Entangling full (5 qubit, all-to-all topology)": gate_tuples_to_qasm(
        [("h", q) for q in range(5)] + de.entangling_layer(5, pattern="full"), 5,
    ),
    "Entangling star (5 qubit, hub topology)": gate_tuples_to_qasm(
        [("h", q) for q in range(5)] + de.entangling_layer(5, pattern="star"), 5,
    ),
    "Entangling brick (5 qubit, brickwork topology)": gate_tuples_to_qasm(
        [("h", q) for q in range(5)] + de.entangling_layer(5, pattern="brick"), 5,
    ),
    "GHZ state (8 qubit)": gate_tuples_to_qasm(de.ghz_state(8), 8),
    "Random circuit (5 qubit, fixed seed)": gate_tuples_to_qasm(
        de.random_circuit(5, 12, seed=13), 5,
    ),
    "Grover search (3 qubit, target |111>)": (
        # Real Grover's algorithm: uniform superposition, a CCZ oracle
        # (via H-CCX-H) marking |111>, then the standard diffuser. One
        # iteration is optimal for N=8/1 marked state. Verified against
        # the real engine: P(|111>) = 0.78125, matching the theoretical
        # sin^2(3*arcsin(1/sqrt(8))) amplitude exactly; all other 7
        # states share the remaining probability uniformly.
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[3];\ncreg c[3];\n'
        'h q[0];\nh q[1];\nh q[2];\nbarrier q;\n'
        'h q[2];\nccx q[0],q[1],q[2];\nh q[2];\nbarrier q;\n'
        'h q[0];\nh q[1];\nh q[2];\n'
        'x q[0];\nx q[1];\nx q[2];\n'
        'h q[2];\nccx q[0],q[1],q[2];\nh q[2];\n'
        'x q[0];\nx q[1];\nx q[2];\n'
        'h q[0];\nh q[1];\nh q[2];\n'
        'barrier q;\nmeasure q -> c;\n'
    ),
    "Deutsch-Jozsa (balanced oracle, 2+1 qubit)": (
        # Real Deutsch-Jozsa: oracle f(x0,x1) = x0 XOR x1 (balanced).
        # Verified: the input register (q0,q1) never measures |00>, the
        # correct signature of a balanced function (a constant function
        # would collapse the input register to |00> with certainty).
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[3];\ncreg c[3];\n'
        'h q[0];\nh q[1];\nx q[2];\nh q[2];\n'
        'cx q[0],q[2];\ncx q[1],q[2];\n'
        'h q[0];\nh q[1];\n'
        'measure q -> c;\n'
    ),
    "Bernstein-Vazirani (secret string 101, 3+1 qubit)": (
        # Real Bernstein-Vazirani: oracle f(x) = x . s with s = 101.
        # Verified: input register (q0,q1,q2) measures |101> -- the
        # secret string -- with certainty, in a single query.
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[4];\ncreg c[4];\n'
        'h q[0];\nh q[1];\nh q[2];\nx q[3];\nh q[3];\n'
        'cx q[0],q[3];\ncx q[2],q[3];\n'
        'h q[0];\nh q[1];\nh q[2];\n'
        'measure q -> c;\n'
    ),
    "Toffoli / CCX (T-gate decomposition, 3 qubit)": (
        # Standard Nielsen & Chuang Toffoli-from-Clifford+T decomposition
        # (6 CX + 7 T/Tdg + 2 H) -- verified against the engine's native
        # ccx gate on all 8 computational-basis inputs, exact match
        # (probability 1.0 on the correct output state) every time.
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[3];\ncreg c[3];\n'
        'h q[2];\n'
        'cx q[1],q[2];\ntdg q[2];\n'
        'cx q[0],q[2];\nt q[2];\n'
        'cx q[1],q[2];\ntdg q[2];\n'
        'cx q[0],q[2];\n'
        't q[1];\nt q[2];\nh q[2];\n'
        'cx q[0],q[1];\nt q[0];\ntdg q[1];\n'
        'cx q[0],q[1];\n'
        'measure q -> c;\n'
    ),
    "Quantum Teleportation (deferred measurement, 3 qubit)": (
        # Real teleportation of an arbitrary state prepared on q0 (here
        # ry(0.9), swap in any single-qubit state to teleport a different
        # one) onto q2, via a Bell pair on q1/q2. Uses the principle of
        # deferred measurement (Nielsen & Chuang Sec. 4.4): Alice's
        # classically-controlled X/Z corrections become coherent
        # controlled gates applied before any measurement, since this
        # engine's QASM parser doesn't execute if(c==N)-conditioned gates
        # (see README "Interop" known limits) -- a naive textbook
        # if(c[1]==1) x q[2]; if(c[0]==1) z q[2]; version would silently
        # no-op here. Verified against the real engine: the reduced
        # density matrix on q2 (partial trace over q0,q1) matches the
        # state originally prepared on q0 with fidelity 1.0000000000,
        # checked across 5 random states with nontrivial complex phase,
        # not just this one example angle.
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[3];\ncreg c[3];\n'
        'ry(0.9) q[0];\n'
        'h q[1];\ncx q[1],q[2];\n'
        'cx q[0],q[1];\nh q[0];\n'
        'cx q[1],q[2];\ncz q[0],q[2];\n'
        'barrier q;\nmeasure q -> c;\n'
    ),
    "Quantum Phase Estimation (3-qubit counting, T-gate)": (
        # Real QPE estimating the phase of the T gate (eigenvalue
        # e^(i*pi/4), i.e. phase = 1/8) on its |1> eigenstate, using 3
        # counting qubits (q0,q1,q2) + 1 target (q3). 1/8 = 0.001 in
        # binary, exactly representable in 3 bits, so a correct QPE gives
        # one deterministic outcome rather than a spread -- the cleanest
        # possible check. The controlled-U^(2^k) power assignment
        # (q0->k=2, q1->k=1, q2->k=0) and the inverse-QFT gate sequence
        # (swap first, then the H/CP cascade) were both determined
        # empirically against this engine's specific MSB-first qubit
        # convention -- a naive textbook assignment/ordering measured a
        # spread-out, non-deterministic distribution here; the inverse
        # QFT step is transcribed exactly from dense_evolution.qft(3,
        # inverse=True), not hand re-derived. Verified: measuring q0q1q2
        # gives |001> (=1/8) with probability 1.0000, not merely the most
        # likely outcome.
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[4];\ncreg c[3];\n'
        'x q[3];\n'
        'h q[0];\nh q[1];\nh q[2];\n'
        'cp(pi) q[0],q[3];\ncp(pi/2) q[1],q[3];\ncp(pi/4) q[2],q[3];\n'
        'swap q[0],q[2];\n'
        'h q[2];\ncp(-pi/2) q[2],q[1];\n'
        'h q[1];\ncp(-pi/4) q[2],q[0];\ncp(-pi/2) q[1],q[0];\n'
        'h q[0];\n'
        'barrier q;\n'
        'measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n'
    ),
}
