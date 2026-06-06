"""
Dense-Evolution
High-performance quantum statevector simulator optimized for NISQ circuits.
"""

from .simulator import DenseSVSimulator
from .parser import QASMParser, QASMCircuit
from .compiler import QuantumTranspiler
from .registry import NoiseModel, QuantumHardwareRegistry
from .gates import GATES, PARAMETRIC_GATES, GATE_IDS

__version__ = "8.1.2"
