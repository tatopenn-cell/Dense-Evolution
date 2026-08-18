"""Circuits subpackage: gate registry, parsing, compilation, topology."""
from .gates import GATES, PARAMETRIC_GATES, GATE_IDS
from .parser import QASMParser, QASMCircuit
from .compiler import QuantumTranspiler
from .registry import HAS_JAX, NoiseModel, NoiseSpec, QuantumHardwareRegistry
from .topology import entangling_layer, VALID_PATTERNS
from .qft import qft
from .trotter import pauli_rotation_ops, trotter_evolve_ops

__all__ = [
    "GATES", "PARAMETRIC_GATES", "GATE_IDS",
    "QASMParser", "QASMCircuit",
    "QuantumTranspiler",
    "HAS_JAX", "NoiseModel", "NoiseSpec", "QuantumHardwareRegistry",
    "entangling_layer", "VALID_PATTERNS",
    "qft",
    "pauli_rotation_ops", "trotter_evolve_ops",
]
