import subprocess
import sys
import textwrap


def test_mitigation_entry_points_enable_x64_in_a_fresh_process():
    code = textwrap.dedent("""
        import numpy as np, jax
        from dense_evolution.mitigation import project_to_physical, zne_density_matrix_jit
        assert not jax.config.read("jax_enable_x64")
        P = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
        r = np.array([0.3, -0.9, 0.8])
        rho = 0.5 * (np.eye(2) + sum(c * p for c, p in zip(r, P)))
        u = r / np.linalg.norm(r)
        ref = 0.5 * (np.eye(2) + sum(c * p for c, p in zip(u, P)))
        out = np.asarray(project_to_physical(rho))
        assert out.dtype == np.complex128
        assert np.abs(out - ref).max() < 1e-14, np.abs(out - ref).max()
        stack = np.stack([rho, rho, rho])
        assert np.asarray(zne_density_matrix_jit(stack, np.array([1.0, 2.0, 3.0]))).dtype == np.complex128
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
