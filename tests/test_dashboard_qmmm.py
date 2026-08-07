"""
Unit tests for dashboard_core/qmmm.py -- real Hellmann-Feynman forces
(PennyLane's differentiable "dhf" Hartree-Fock) and the real
Velocity-Verlet MD step built on them. Checks the physics is actually
correct (near-zero force at a real equilibrium geometry, a stronger
restoring force away from it, forces that change step-to-step during an
MD trajectory), not just that the functions run without raising.
"""
import numpy as np
import pytest

from dashboard_core.qmmm import (
    ATOMIC_MASSES_AMU, ACCEL_CONVERSION, MIN_NUCLEAR_DISTANCE_ANGSTROM,
    compute_hellmann_feynman_forces, md_step, run_md_trajectory,
    _assert_no_nuclear_collision,
)

H2_EQUILIBRIUM = "H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]"


class TestComputeHellmannFeynmanForces:

    def test_rejects_unknown_molecule(self):
        with pytest.raises(ValueError):
            compute_hellmann_feynman_forces("not a real molecule")

    def test_h2_force_small_at_real_equilibrium_geometry(self):
        # Recorded 2026-08-05: 0.0154 Hartree/Angstrom (small, clamped-
        # nucleus residual -- the electronic state is evaluated at fixed
        # geometry, so this isn't exactly zero, but it is small).
        result = compute_hellmann_feynman_forces(H2_EQUILIBRIUM)
        assert result["force_norm"] == pytest.approx(0.0154, abs=2e-3)
        assert result["energy_hartree"] == pytest.approx(-1.137270, abs=1e-5)

    def test_h2_force_much_larger_when_stretched(self):
        eq = compute_hellmann_feynman_forces(H2_EQUILIBRIUM)
        stretched_geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]
        stretched = compute_hellmann_feynman_forces(H2_EQUILIBRIUM, geometry=stretched_geometry)
        assert stretched["force_norm"] > 5 * eq["force_norm"]

    def test_force_points_toward_shorter_bond_when_stretched(self):
        # A stretched H2 bond should feel a restoring force pulling the
        # second atom back toward the first (negative z-component).
        stretched_geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]
        result = compute_hellmann_feynman_forces(H2_EQUILIBRIUM, geometry=stretched_geometry)
        forces = np.array(result["forces_hartree_per_angstrom"])
        assert forces[1, 2] < 0
        assert forces[0, 2] > 0

    def test_explicit_geometry_overrides_catalog_default(self):
        default_result = compute_hellmann_feynman_forces(H2_EQUILIBRIUM)
        moved_result = compute_hellmann_feynman_forces(
            H2_EQUILIBRIUM, geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]])
        assert default_result["positions_angstrom"] != moved_result["positions_angstrom"]
        assert default_result["force_norm"] != pytest.approx(moved_result["force_norm"])


class TestAccelConversion:

    def test_derived_from_codata_not_hand_picked(self):
        # Verified 2026-08-05 against scipy.constants directly.
        assert ACCEL_CONVERSION == pytest.approx(0.262550, abs=1e-5)


class TestMdStep:

    def test_zero_force_leaves_velocity_unchanged_constant_position_drift(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]])
        velocities = np.zeros_like(positions)
        forces = np.zeros_like(positions)
        new_pos, new_vel, accel = md_step(positions, velocities, forces, ['H', 'H'], dt_fs=0.5)
        assert np.allclose(accel, 0.0)
        assert np.allclose(new_vel, 0.0)
        assert np.allclose(new_pos, positions)

    def test_real_mass_scales_acceleration_inversely(self):
        positions = np.zeros((2, 3))
        velocities = np.zeros((2, 3))
        forces = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        _, _, accel_h = md_step(positions, velocities, forces, ['H', 'H'], dt_fs=0.5)
        _, _, accel_he = md_step(positions, velocities, forces, ['He', 'He'], dt_fs=0.5)
        # Heavier atom (He, ~4x H's mass) accelerates less under the same force.
        assert np.linalg.norm(accel_he) < np.linalg.norm(accel_h)
        expected_ratio = ATOMIC_MASSES_AMU['H'] / ATOMIC_MASSES_AMU['He']
        assert np.linalg.norm(accel_he) / np.linalg.norm(accel_h) == pytest.approx(expected_ratio, rel=1e-6)


class TestRunMdTrajectory:

    def test_rejects_unknown_molecule(self):
        with pytest.raises(ValueError):
            run_md_trajectory("not a real molecule", n_steps=1)

    def test_forces_and_energy_change_across_real_steps(self):
        # Regression check for a real bug found 2026-08-05: an earlier
        # version never passed the trajectory's updated positions back
        # into compute_hellmann_feynman_forces, so every step silently
        # recomputed forces at the same fixed catalog geometry --
        # force_norm and energy were identical at every step. Verifies
        # that's fixed: the trajectory must show real step-to-step change.
        trajectory = run_md_trajectory(H2_EQUILIBRIUM, n_steps=5, dt_fs=0.5)
        assert len(set(trajectory["force_norm"])) > 1, \
            "force_norm identical at every step -- geometry isn't being updated"
        assert len(set(trajectory["energy_hartree"])) > 1, \
            "energy identical at every step -- geometry isn't being updated"

    def test_starts_at_real_catalog_equilibrium_geometry_at_rest(self):
        trajectory = run_md_trajectory(H2_EQUILIBRIUM, n_steps=1, dt_fs=0.5)
        first_positions = np.array(trajectory["positions_angstrom"][0])
        assert first_positions[1, 2] == pytest.approx(0.7414, abs=1e-6)

    def test_time_axis_matches_dt(self):
        trajectory = run_md_trajectory(H2_EQUILIBRIUM, n_steps=4, dt_fs=0.25)
        assert trajectory["time_fs"] == [0.0, 0.25, 0.5, 0.75]


class TestAssertNoNuclearCollision:
    """Fast, direct tests for the collision safety check run_md_trajectory
    calls after every step -- synthetic positions, no real Hartree-Fock
    calculation needed to verify the guard itself is correct."""

    def test_real_h2_equilibrium_geometry_passes(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]])
        _assert_no_nuclear_collision(positions, step=0, dt_fs=0.5)  # must not raise

    def test_exactly_at_threshold_passes(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MIN_NUCLEAR_DISTANCE_ANGSTROM]])
        _assert_no_nuclear_collision(positions, step=0, dt_fs=0.5)  # must not raise

    def test_below_threshold_raises(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]])
        with pytest.raises(RuntimeError, match="diverged"):
            _assert_no_nuclear_collision(positions, step=3, dt_fs=50.0)

    def test_error_message_includes_step_and_dt(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])
        with pytest.raises(RuntimeError, match=r"step 4.*dt_fs=50\.0"):
            _assert_no_nuclear_collision(positions, step=3, dt_fs=50.0)

    def test_checks_closest_pair_among_more_than_two_atoms(self):
        # Two atoms far apart, one pair dangerously close -- must still catch it.
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.05],
        ])
        with pytest.raises(RuntimeError):
            _assert_no_nuclear_collision(positions, step=0, dt_fs=1.0)

    def test_single_atom_never_raises(self):
        positions = np.array([[0.0, 0.0, 0.0]])
        _assert_no_nuclear_collision(positions, step=0, dt_fs=1000.0)  # nothing to compare
