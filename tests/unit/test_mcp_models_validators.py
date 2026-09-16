"""Regression tests for the cross-field @model_validator checks added to
tools/mcp_server/models.py (issue #258 point 8) -- pure Pydantic schema
tests, no kernel/MCP fixture needed, unlike tests/integration/test_mcp_server.py."""
import pytest
from pydantic import ValidationError

from mcp_server.models import RunCircuitInput, RunVqeInput

BELL_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
"""


def test_run_vqe_accepts_name_only():
    RunVqeInput(name="H2")


def test_run_vqe_accepts_custom_molecule_only():
    RunVqeInput(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 0.7414]])


def test_run_vqe_rejects_both_name_and_custom_molecule():
    with pytest.raises(ValidationError, match="not both"):
        RunVqeInput(name="H2", symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 0.7]])


def test_run_vqe_rejects_neither_name_nor_custom_molecule():
    with pytest.raises(ValidationError, match="either"):
        RunVqeInput()


def test_run_vqe_rejects_symbols_without_geometry():
    with pytest.raises(ValidationError, match="both"):
        RunVqeInput(symbols=["H", "H"])


def test_run_circuit_accepts_ideal_with_zero_noise_p():
    RunCircuitInput(qasm=BELL_QASM)  # defaults: noise_model='ideal', noise_p=0.0


def test_run_circuit_accepts_real_noise_model_with_noise_p():
    RunCircuitInput(qasm=BELL_QASM, noise_model="depolarizing", noise_p=0.05)


def test_run_circuit_rejects_noise_p_with_ideal_noise_model():
    with pytest.raises(ValidationError, match="noise_model='ideal'"):
        RunCircuitInput(qasm=BELL_QASM, noise_model="ideal", noise_p=0.1)
