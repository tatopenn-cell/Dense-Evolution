"""
A handful of real, standard OpenQASM 2.0 circuits offered as sidebar
presets -- textbook examples (Bell pair, GHZ state), not synthetic
results. The dashboard runs every one of these through the real engine
just like any custom circuit the user types in.
"""

QASM_LIBRARY = {
    "Bell state (2 qubit)": (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[2];\ncreg c[2];\n'
        'h q[0];\ncx q[0],q[1];\n'
        'measure q -> c;\n'
    ),
    "GHZ state (3 qubit)": (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[3];\ncreg c[3];\n'
        'h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n'
        'measure q -> c;\n'
    ),
    "Superposition (1 qubit)": (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[1];\ncreg c[1];\n'
        'h q[0];\n'
        'measure q -> c;\n'
    ),
}

__all__ = ['QASM_LIBRARY']
