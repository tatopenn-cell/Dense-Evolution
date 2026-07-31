# QFT (Quantum Fourier Transform)

The standard H + controlled-phase cascade plus trailing qubit-order swap, returned as a
gate-tuple list. Verified against the brute-force analytic DFT matrix (max error 1.4e-15
across 1-4 qubits) and a QFT-then-inverse-QFT round trip.

::: dense_evolution.qft
