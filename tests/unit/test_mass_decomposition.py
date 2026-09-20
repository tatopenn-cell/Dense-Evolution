"""
Unit tests for dense_evolution/utils/mass_decomposition.py.
"""
import numpy as np
import pytest

from dense_evolution.utils.mass_decomposition import (
    parse_formula, rdbe, build_reachable_masses, nearest_reachable_mass,
    build_reachable_density_fft, density_at_mass, ATOMIC_MASS,
)


class TestParseFormula:

    def test_simple_formula(self):
        assert parse_formula("C20H15N3O2") == {"C": 20, "H": 15, "N": 3, "O": 2}

    def test_single_atom_no_count_means_one(self):
        assert parse_formula("H2O") == {"H": 2, "O": 1}

    def test_empty_string(self):
        assert parse_formula("") == {}

    def test_none_input_does_not_raise(self):
        assert parse_formula(None) == {}

    def test_repeated_element_is_summed(self):
        assert parse_formula("C2C3") == {"C": 5}


class TestRdbe:

    def test_water_is_zero_rdbe(self):
        # H2O: 1 + (2*(1-2) + 1*(2-2))/2 = 1 + (-2)/2 = 0 -- a real closed-shell molecule
        assert rdbe({"H": 2, "O": 1}) == pytest.approx(0.0)

    def test_methane_is_zero_rdbe(self):
        # CH4: 1 + (1*(4-2) + 4*(1-2))/2 = 1 + (2-4)/2 = 0
        assert rdbe({"C": 1, "H": 4}) == pytest.approx(0.0)

    def test_benzene_is_four_rdbe(self):
        # C6H6: 1 + (6*2 + 6*(-1))/2 = 1 + 6/2 = 4 -- real value (3 double bonds + 1 ring)
        assert rdbe({"C": 6, "H": 6}) == pytest.approx(4.0)

    def test_empty_formula_is_one(self):
        assert rdbe({}) == pytest.approx(1.0)

    def test_unrecognized_element_ignored_not_erroring(self):
        assert rdbe({"Xx": 5}) == pytest.approx(1.0)


class TestBuildReachableMasses:

    def test_contains_zero(self):
        reach = build_reachable_masses({"C": 2, "H": 4}, max_mass=50.0)
        assert reach[0] == pytest.approx(0.0)

    def test_contains_full_formula_mass(self):
        # C2H4 (ethylene): full formula itself must be reachable (all atoms used)
        formula = {"C": 2, "H": 4}
        full_mass = 2 * 12.0 + 4 * 1.007825
        reach = build_reachable_masses(formula, max_mass=full_mass + 1.0)
        nearest = nearest_reachable_mass(full_mass, reach)
        assert nearest == pytest.approx(full_mass, abs=1e-3)

    def test_water_loss_reachable_from_glucose(self):
        # Glucose C6H12O6 must be able to reach an H2O (18.0106) sub-formula
        formula = parse_formula("C6H12O6")
        reach = build_reachable_masses(formula, max_mass=200.0)
        nearest = nearest_reachable_mass(18.010565, reach)
        assert nearest is not None
        assert abs(nearest - 18.010565) < 0.001

    def test_rdbe_filter_removes_invalid_combination(self):
        # A formula with only 1 carbon and 0 hydrogens: C alone (no H) has
        # RDBE = 1 + 1*(4-2)/2 = 2, which IS valid (e.g. an isolated carbene-
        # like fragment is not realistic, but the arithmetic RDBE constraint
        # alone doesn't forbid it) -- use a case where RDBE genuinely goes
        # negative: many halogens with no carbon skeleton to attach to.
        formula = {"Cl": 4}  # 1 + 4*(1-2)/2 = 1 - 2 = -1 -- RDBE < 0, invalid
        reach_unfiltered = build_reachable_masses(formula, max_mass=200.0, require_rdbe_valid=False)
        reach_filtered = build_reachable_masses(formula, max_mass=200.0, require_rdbe_valid=True)
        full_mass = 4 * 34.968853
        assert nearest_reachable_mass(full_mass, reach_unfiltered) is not None
        # the RDBE-invalid full combination must NOT survive filtering
        nearest_filtered = nearest_reachable_mass(full_mass, reach_filtered)
        assert nearest_filtered is None or abs(nearest_filtered - full_mass) > 0.01

    def test_empty_formula_returns_trivial_array(self):
        reach = build_reachable_masses({}, max_mass=100.0)
        assert nearest_reachable_mass(50.0, reach) is None


class TestNearestReachableMass:

    def test_exact_match(self):
        reach = np.array([0.0, 10.0, 20.0, 30.0])
        assert nearest_reachable_mass(20.0, reach) == pytest.approx(20.0)

    def test_nearest_below_and_above(self):
        reach = np.array([0.0, 10.0, 20.0])
        assert nearest_reachable_mass(14.0, reach) == pytest.approx(10.0)
        assert nearest_reachable_mass(16.0, reach) == pytest.approx(20.0)

    def test_trivial_array_returns_none(self):
        assert nearest_reachable_mass(5.0, np.array([0.0])) is None

    def test_empty_array_returns_none(self):
        assert nearest_reachable_mass(5.0, np.array([])) is None


class TestBuildReachableDensityFft:

    def test_matches_brute_force_convolution_on_tiny_case(self):
        # Direct verification of the convolution theorem usage itself,
        # not just "it runs": for {C:1, H:1}, brute-force the reachable
        # spike positions by hand (0, mC, mH, mC+mH) and compare against
        # the FFT-based construction's peaks at the same masses.
        formula = {"C": 1, "H": 1}
        max_mass = 20.0
        resolution = 0.001
        mass_grid, density = build_reachable_density_fft(
            formula, max_mass, grid_resolution=resolution, relative_tolerance=1e-4
        )

        expected_masses = [0.0, ATOMIC_MASS["C"], ATOMIC_MASS["H"],
                            ATOMIC_MASS["C"] + ATOMIC_MASS["H"]]
        for m in expected_masses:
            assert density_at_mass(mass_grid, density, m) > 0.05, (
                f"expected density at real combination mass {m}, got near zero"
            )

        # a mass far from any real combination must be near-zero
        assert density_at_mass(mass_grid, density, 8.5) < 0.01

    def test_agrees_with_exact_enumeration_ranking(self):
        # On the same real formula (glucose C6H12O6), the FFT density's
        # peak location near the water-loss mass must agree with the
        # exact method's own nearest-reachable-mass answer.
        formula = parse_formula("C6H12O6")
        max_mass = 200.0
        reach_exact = build_reachable_masses(formula, max_mass=max_mass)
        exact_nearest = nearest_reachable_mass(18.010565, reach_exact)
        assert exact_nearest is not None

        mass_grid, density = build_reachable_density_fft(
            formula, max_mass, grid_resolution=0.001, relative_tolerance=2e-5
        )
        d_at_true_loss = density_at_mass(mass_grid, density, exact_nearest)
        d_at_unrelated = density_at_mass(mass_grid, density, exact_nearest + 5.5)
        assert d_at_true_loss > d_at_unrelated

    def test_empty_formula_returns_zero_density(self):
        mass_grid, density = build_reachable_density_fft({}, max_mass=50.0)
        assert np.all(density == 0.0)

    def test_wider_relative_tolerance_broadens_peak(self):
        formula = {"C": 1, "H": 4}  # methane
        max_mass = 20.0
        target = ATOMIC_MASS["C"] + 4 * ATOMIC_MASS["H"]  # full formula mass
        _, density_narrow = build_reachable_density_fft(
            formula, max_mass, grid_resolution=0.001, relative_tolerance=1e-5
        )
        _, density_wide = build_reachable_density_fft(
            formula, max_mass, grid_resolution=0.001, relative_tolerance=1e-3
        )
        # a point 0.05 Da off the true mass: wider tolerance should support it more
        grid_narrow, _ = build_reachable_density_fft(formula, max_mass, grid_resolution=0.001)
        off_target = target + 0.05
        d_narrow = density_at_mass(grid_narrow, density_narrow, off_target)
        d_wide = density_at_mass(grid_narrow, density_wide, off_target)
        assert d_wide > d_narrow


class TestDensityAtMass:

    def test_interpolates_between_grid_points(self):
        grid = np.array([0.0, 1.0, 2.0])
        density = np.array([0.0, 10.0, 0.0])
        assert density_at_mass(grid, density, 0.5) == pytest.approx(5.0)

    def test_outside_grid_returns_zero(self):
        grid = np.array([0.0, 1.0, 2.0])
        density = np.array([1.0, 2.0, 3.0])
        assert density_at_mass(grid, density, -5.0) == 0.0
        assert density_at_mass(grid, density, 50.0) == 0.0
