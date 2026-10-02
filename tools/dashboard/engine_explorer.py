"""
Every public function of dense_evolution, callable from the Dashboard. Each
parameter is filled from the current circuit when its name says what it is
(statevector, density matrix, QASM, gate list, number of qubits, stabilizers,
Pauli string); numbers, strings and booleans become inputs with their
defaults. Functions that need any other object are listed but not callable.
"""
import dataclasses
import inspect

import numpy as np

import dense_evolution as de

STATE = {"sv", "statevector", "state", "psi", "state_vector", "vector", "sv0"}
RHO = {"rho", "density_matrix", "dm", "rho0"}
QASM = {"qasm", "qasm_text", "qasm_str", "qasm_string"}
OPS = {"ops", "circuit", "gates", "circuit_ops", "gate_list"}
NQ = {"n_qubits", "num_qubits", "nq"}


def functions():
    names = []
    for name in getattr(de, "__all__", []):
        obj = getattr(de, name, None)
        if callable(obj) and not inspect.isclass(obj):
            names.append(name)
    return sorted(names)


def plan(name):
    params = []
    for p in inspect.signature(getattr(de, name)).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        k = p.name.lower()
        d = p.default
        if k in STATE:
            kind = "state"
        elif k in RHO:
            kind = "rho"
        elif k in QASM:
            kind = "qasm"
        elif k in OPS:
            kind = "ops"
        elif k in NQ:
            kind = "n"
        elif k == "stabilizers":
            kind = "stabilizers"
        elif k in ("pauli_string", "pauli", "pauli_error", "pauli_str"):
            kind = "pauli"
        elif d is not p.empty and isinstance(d, (bool, int, float, str)):
            kind = type(d).__name__
        elif d is not p.empty:
            kind = "default"
        else:
            kind = "unsupported"
        params.append((p.name, kind, d))
    return params


def callable_from_ui(name):
    return all(kind != "unsupported" for _, kind, _ in plan(name))


def build_args(name, result, qasm, values):
    args = {}
    for pname, kind, d in plan(name):
        if kind == "state":
            args[pname] = np.asarray(result.statevector)
        elif kind == "rho":
            s = np.asarray(result.statevector)
            args[pname] = np.outer(s, s.conj())
        elif kind == "qasm":
            args[pname] = qasm
        elif kind == "ops":
            args[pname] = list(result.ops)
        elif kind == "n":
            args[pname] = result.n_qubits
        elif kind == "stabilizers":
            args[pname] = values.get(pname, ["ZZ" + "I" * (result.n_qubits - 2)] if result.n_qubits >= 2 else ["Z"])
        elif kind == "pauli":
            args[pname] = values.get(pname, "Z" * result.n_qubits)
        elif kind in ("bool", "int", "float", "str"):
            args[pname] = values.get(pname, d)
        first = False
    return args


def _rho_one(s, n):
    m = np.asarray(s).reshape(2, -1) if n == 1 else np.moveaxis(np.asarray(s).reshape([2] * n), n - 1, 0).reshape(2, -1)
    return m @ m.conj().T


def to_display(out):
    if dataclasses.is_dataclass(out) and not isinstance(out, type):
        out = dataclasses.asdict(out)
    if isinstance(out, np.ndarray):
        return np.round(out, 6).tolist() if out.size <= 4096 else f"array {out.shape}"
    if isinstance(out, dict):
        return {str(k): to_display(v) for k, v in out.items()}
    if isinstance(out, (list, tuple)):
        return [to_display(v) for v in out][:256]
    if isinstance(out, (np.floating, np.integer)):
        return out.item()
    if isinstance(out, complex):
        return str(out)
    return out if isinstance(out, (int, float, str, bool, type(None))) else repr(out)[:500]


import ast

ARRAYS = {"expectation_values", "noise_factors", "values", "lambdas", "time_us", "Ls", "S",
          "event_times", "window_sizes", "kxyz", "k", "k_start", "k_end", "sigma_at_base_noise"}
QEC_STABS = ["ZZI", "IZZ"]
EXAMPLES = {
    "gamma": "0.1", "p": "0.1", "material": "'Si'", "k_start": "(0.0, 0.0, 0.0)", "k_end": "(1.0, 0.0, 0.0)",
    "observed_syndrome": "(1, 0)", "heralded_qubits": "[0]",
    "expectation_values": "[0.9, 0.81, 0.73]", "noise_factors": "[1.0, 2.0, 3.0]",
    "values": "[0.9, 0.81, 0.73]", "lambdas": "[1.0, 2.0, 3.0]", "degree": "2",
    "sigma_at_base_noise": "0.1", "target_sigma_ideal": "10.0",
    "Ls": "[1, 2, 3, 4]", "S": "[0.222, 0.427, 0.516, 0.543]", "time_us": "[0.0, 10.0, 1000.0]", "baseline_gamma": "0.01",
    "event_times": "[0.1, 0.5, 0.9, 1.3, 2.0, 2.2, 3.1, 3.5, 4.0, 4.4, 5.2, 5.9]",
    "window_sizes": "[0.5, 1.0, 2.0]", "q": "1", "r": "2", "s": "3", "theta": "0.1",
    "electrons": "2", "n_sites": "2", "t": "1.0", "U": "2.0", "mode_index": "1",
    "factors": "[(1.0, 'XI'), (1.0, 'ZI')]", "qubits_a": "[0]", "qubits_b": "[1]",
    "measured_bits": "'0000000'", "coset_a": "['0000000', '1111000']", "coset_b": "['1111111', '0000111']",
    "base_p": "0.01", "factor": "2.0", "freq": "1.0", "amp": "0.5", "keep_qubits": "[0]",
    "p1": "'XI'", "p2": "'ZI'", "terms": "[(1.0, 'ZZ'), (0.5, 'XI')]", "pauli_dict": "{0: 'X', 1: 'Z'}",
    "angle": "0.3", "n_gates": "5", "n_shots": "100", "element_a": "'Si'", "element_b": "'Si'",
    "bond_length_angstrom": "2.35", "kxyz": "(0.0, 0.0, 0.0)", "lx": "2", "ly": "2",
    "mode_indices": "[1, 2]", "n_steps": "2", "k": "(0.0, 0.0, 0.0)", "cation": "'Ga'", "anion": "'As'",
    "lattice_constant_angstrom": "5.65",
}
EXAMPLES_BY_FUNC = {
    ("double_excitation_ops", "p"): "0", ("single_excitation_ops", "p"): "0",
    ("trotter_evolve_ops", "t"): "0.5",
    ("trotter_evolve_ops", "terms"): "[(1.0, {0: 'Z', 1: 'Z'}), (0.5, {0: 'X'})]",
}
N_OVERRIDE = {"central_charge": 8}
FIRST_ARG = {"from_qiskit": "qiskit", "run_qiskit_circuit": "qiskit", "from_pennylane": "pennylane",
             "run_pennylane_circuit": "pennylane", "circuit_to_energy_fn": "parsed"}


def _first_arg(kind, qasm, n):
    if kind == "qiskit":
        import qiskit.qasm2
        return qiskit.qasm2.loads(qasm)
    if kind == "pennylane":
        import pennylane as qml
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def circ():
            qml.from_qasm(qasm)()
            return qml.state()
        return circ
    return de.QASMParser().parse(qasm)
NOT_FROM_UI = {"channel_fn", "hamiltonian_fn", "backend", "observable_fn", "psi0", "coeffs_t", "dt", "params_t"}


def plan2(name):
    out = []
    for pname, kind, d in plan(name):
        k = pname.lower()
        if k.startswith("statevector"):
            kind = "state"
        elif k == "rho_at_scales":
            kind = "rho_scales"
        elif k.startswith("rho") and kind not in ("state",):
            kind = "rho"
        elif kind == "unsupported" and pname not in NOT_FROM_UI:
            kind = "literal"
        out.append((pname, kind, d))
    return out


def connectable(name):
    return all(kind != "unsupported" for _, kind, _ in plan2(name))


def example(name, pname):
    return EXAMPLES_BY_FUNC.get((name, pname), EXAMPLES.get(pname, ""))


def build_args2(name, result, qasm, texts=None, values=None):
    texts, values = texts or {}, values or {}
    params = plan2(name)
    qec = any(p == "stabilizers" for p, _, _ in params)
    args = {}
    s = np.asarray(result.statevector)
    rho = np.outer(s, s.conj())
    first = True
    for pname, kind, d in params:
        if first and name in FIRST_ARG:
            args[pname] = _first_arg(FIRST_ARG[name], qasm, result.n_qubits)
        elif kind == "state":
            args[pname] = s
        elif kind == "rho":
            args[pname] = _rho_one(s, result.n_qubits) if name == "amplitude_damping_channel" else rho
        elif kind == "rho_scales":
            nf = [1.0, 2.0, 3.0]
            args[pname] = np.stack([np.asarray(de.global_depolarizing_channel(rho, 0.05 * f)) for f in nf])
        elif kind == "qasm":
            args[pname] = qasm
        elif kind == "ops":
            args[pname] = list(result.ops)
        elif kind == "n":
            args[pname] = N_OVERRIDE.get(name, len(QEC_STABS[0]) if qec else result.n_qubits)
        elif kind == "stabilizers":
            args[pname] = QEC_STABS
        elif kind == "pauli":
            args[pname] = values.get(pname, "Z" * (len(QEC_STABS[0]) if qec else result.n_qubits))
        elif kind == "literal":
            v = ast.literal_eval(texts.get(pname, example(name, pname)))
            args[pname] = np.asarray(v, dtype=float) if pname.lower() in ARRAYS else v
        elif kind in ("bool", "int", "float", "str"):
            args[pname] = values.get(pname, d)
    return args
