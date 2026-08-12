"""
Native, JAX-vectorized ab-initio Hartree-Fock engine for Dense-Evolution.

Computes molecular one- and two-electron integrals (overlap, kinetic,
nuclear attraction, electron repulsion) via the Obara-Saika recursion
[Obara, Saika, J. Chem. Phys. 84, 3963 (1986)], batched with jax.lax.scan
so the whole shell-quartet loop compiles to a single XLA program instead
of looping in the Python interpreter.

This module exists because PennyLane's own differentiable Hartree-Fock
solver (qml.qchem, method="dhf") builds these same integrals through a
Python-level loop wrapped in its autograd-tracing numpy layer -- correct,
but roughly 100x slower than a properly vmap/jit-compiled implementation
for anything beyond a couple of atoms (profiled on Si2/STO-3G: 482 of
483 total seconds were spent in that loop). The second-quantization and
Jordan-Wigner mapping steps that come after the integrals are already
fast in PennyLane, so this module only replaces the integral/SCF stage
and hands its output to qml.qchem.observable_hf.fermionic_observable.

Design and algorithm choice were informed by studying lowdanie/hartree-fock-solver
("slaterform", Apache-2.0) as a reference for how to structure the
Obara-Saika recursion with jax.lax.scan, and by PennyLane's own white
paper (Delgado et al., "Differentiable quantum computational chemistry
with PennyLane", arXiv:2111.09967). No source code from either project
is copied here.
"""
