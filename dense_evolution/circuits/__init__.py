"""Circuits subpackage: gate registry, parsing, compilation, topology."""
from .gates import GATES, PARAMETRIC_GATES, GATE_IDS
from .parser import QASMParser, QASMCircuit
from .compiler import QuantumTranspiler
from .registry import HAS_JAX, NoiseModel, NoiseSpec, QuantumHardwareRegistry
from .topology import entangling_layer, VALID_PATTERNS
from .qft import qft
from .arithmetic import (
    add_registers, subtract_registers, add_constant,
    compare_registers, compare_constant,
    add_registers_mod, add_constant_mod, multiply_add_mod, multiply_mod, power_mod,
)
from .arithmetic_qasm import (
    cuccaro_adder_qasm, draper_adder_qasm, constant_adder_qasm, modular_constant_adder_qasm,
    cmult_mod_qasm, controlled_ua_qasm,
)
from .shor import shor_order_finding
from .postselect import postselect
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
    "compare_registers", "compare_constant",
    "add_registers_mod", "add_constant_mod", "multiply_add_mod", "multiply_mod", "power_mod",
    "cuccaro_adder_qasm", "draper_adder_qasm", "constant_adder_qasm", "modular_constant_adder_qasm",
    "cmult_mod_qasm", "controlled_ua_qasm", "shor_order_finding",
    "postselect",
    "random_circuit",
    "pauli_rotation_ops", "trotter_evolve_ops",
]
