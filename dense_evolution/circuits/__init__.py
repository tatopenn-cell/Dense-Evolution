"""Circuits subpackage: gate registry, parsing, compilation, topology."""
from .gates import GATES, PARAMETRIC_GATES, GATE_IDS
from .parser import QASMParser, QASMCircuit
from .compiler import QuantumTranspiler
from .registry import HAS_JAX, NoiseModel, NoiseSpec, QuantumHardwareRegistry
from .topology import entangling_layer, VALID_PATTERNS
from .qft import qft
from .arithmetic import add_registers, subtract_registers, add_constant
from .random_circuit import random_circuit
from .trotter import pauli_rotation_ops, trotter_evolve_ops

__all__ = [
    "GATES", "PARAMETRIC_GATES", "GATE_IDS",
    "QASMParser", "QASMCircuit",
    "QuantumTranspiler",
    "HAS_JAX", "NoiseModel", "NoiseSpec", "QuantumHardwareRegistry",
    "entangling_layer", "VALID_PATTERNS",
    "qft",
    "add_registers", "subtract_registers", "add_constant",
    "random_circuit",
    "pauli_rotation_ops", "trotter_evolve_ops",
]
