import subprocess
import sys

_BLOCKER_PREAMBLE = """
import sys, importlib.abc

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == root or name.startswith(root + ".") for root in {blocked!r}):
            raise ModuleNotFoundError(f"No module named '{{name}}'")
        return None

sys.meta_path.insert(0, _Blocker())
"""


def _run_blocked(blocked_roots, body):
    script = _BLOCKER_PREAMBLE.format(blocked=list(blocked_roots)) + body
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result


def test_all_submodules_importable_with_only_base_dependencies():
    body = """
import pkgutil, importlib
import dense_evolution as pkg
failed = []
for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    try:
        importlib.import_module(m.name)
    except ModuleNotFoundError as e:
        failed.append((m.name, str(e)))
if failed:
    print("FAILED:", failed)
    sys.exit(1)
print("OK")
"""
    result = _run_blocked(["pennylane", "qiskit", "stim", "pymatching"], body)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_hf_bridge_importable_without_pennylane():
    body = """
import dense_evolution.native_hf.bridge as bridge
assert bridge.qml is None
print("OK")
"""
    result = _run_blocked(["pennylane"], body)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_hf_bridge_raises_clear_error_without_pennylane():
    body = """
import dense_evolution.native_hf.bridge as bridge
try:
    bridge.build_qubit_hamiltonian(
        atomic_numbers=[1, 1],
        geometry_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        n_electrons=2,
    )
except ModuleNotFoundError as e:
    assert "pip install dense-evolution[pennylane]" in str(e), str(e)
    print("OK")
    sys.exit(0)
print("did not raise")
sys.exit(1)
"""
    result = _run_blocked(["pennylane"], body)
    assert result.returncode == 0, result.stdout + result.stderr
