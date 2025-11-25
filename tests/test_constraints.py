"""Tests for the unified constraint system."""

import numpy as np
import pytest

from quantum_portfolio_optimizer.core.constraints import (
    BoundsConstraint,
    ConstraintManager,
    EqualityConstraint,
    InequalityConstraint,
    evaluate_constraints,
)


class TestEqualityConstraint:
    """Tests for budget-style equality constraints."""

    def test_is_satisfied_exact_match(self):
        """Exact match should satisfy constraint."""
        constraint = EqualityConstraint(target=1.0)
        allocations = np.array([0.3, 0.3, 0.4])
        assert constraint.is_satisfied(allocations)

    def test_is_satisfied_within_tolerance(self):
        """Value within tolerance should satisfy constraint."""
        constraint = EqualityConstraint(target=1.0)
        allocations = np.array([0.5, 0.5 + 1e-7])  # Sum = 1.0000001
        assert constraint.is_satisfied(allocations, atol=1e-6)

    def test_is_not_satisfied_outside_tolerance(self):
        """Value outside tolerance should not satisfy constraint."""
        constraint = EqualityConstraint(target=1.0)
        allocations = np.array([0.3, 0.3])  # Sum = 0.6
        assert not constraint.is_satisfied(allocations)

    def test_violation_computes_difference(self):
        """Violation should be absolute difference from target."""
        constraint = EqualityConstraint(target=1.0)
        allocations = np.array([0.7])
        assert constraint.violation(allocations) == pytest.approx(0.3)

    def test_apply_penalty_modifies_offset(self):
        """Penalty should add target^2 to offset."""
        constraint = EqualityConstraint(target=2.0, penalty_strength=1.0)
        linear = np.zeros(2)
        quadratic = np.zeros((2, 2))
        weights = np.array([1.0, 1.0])
        offset = constraint.apply_penalty(linear, quadratic, 0.0, weights, [0, 1])
        assert offset == pytest.approx(4.0)  # target^2 = 4

    def test_apply_penalty_empty_indices(self):
        """Empty indices should return unchanged offset."""
        constraint = EqualityConstraint(target=1.0)
        linear = np.zeros(2)
        quadratic = np.zeros((2, 2))
        weights = np.array([1.0, 1.0])
        offset = constraint.apply_penalty(linear, quadratic, 5.0, weights, [])
        assert offset == 5.0


class TestInequalityConstraint:
    """Tests for upper-bound inequality constraints."""

    def test_is_satisfied_under_limit(self):
        """Value under limit should satisfy constraint."""
        constraint = InequalityConstraint(limit=1.0)
        allocations = np.array([0.3, 0.3])
        assert constraint.is_satisfied(allocations)

    def test_is_satisfied_at_limit(self):
        """Value at limit should satisfy constraint."""
        constraint = InequalityConstraint(limit=1.0)
        allocations = np.array([0.5, 0.5])
        assert constraint.is_satisfied(allocations)

    def test_is_not_satisfied_over_limit(self):
        """Value over limit should not satisfy constraint."""
        constraint = InequalityConstraint(limit=1.0)
        allocations = np.array([0.8, 0.8])  # Sum = 1.6
        assert not constraint.is_satisfied(allocations)

    def test_violation_returns_excess(self):
        """Violation should be excess over limit."""
        constraint = InequalityConstraint(limit=1.0)
        allocations = np.array([0.8, 0.8])  # Sum = 1.6
        assert constraint.violation(allocations) == pytest.approx(0.6)

    def test_violation_zero_when_satisfied(self):
        """Violation should be zero when satisfied."""
        constraint = InequalityConstraint(limit=1.0)
        allocations = np.array([0.3, 0.3])
        assert constraint.violation(allocations) == 0.0


class TestBoundsConstraint:
    """Tests for per-variable bounds constraints."""

    def test_is_satisfied_within_bounds(self):
        """Values within bounds should satisfy constraint."""
        constraint = BoundsConstraint(lower=0.0, upper=1.0)
        allocations = np.array([0.0, 0.5, 1.0])
        assert constraint.is_satisfied(allocations)

    def test_is_not_satisfied_below_lower(self):
        """Values below lower bound should not satisfy constraint."""
        constraint = BoundsConstraint(lower=0.0, upper=1.0)
        allocations = np.array([-0.1, 0.5])
        assert not constraint.is_satisfied(allocations)

    def test_is_not_satisfied_above_upper(self):
        """Values above upper bound should not satisfy constraint."""
        constraint = BoundsConstraint(lower=0.0, upper=1.0)
        allocations = np.array([0.5, 1.1])
        assert not constraint.is_satisfied(allocations)

    def test_violation_computes_total_out_of_bounds(self):
        """Violation should sum out-of-bounds amounts."""
        constraint = BoundsConstraint(lower=0.0, upper=1.0)
        allocations = np.array([-0.2, 1.3])  # -0.2 below, 0.3 above
        assert constraint.violation(allocations) == pytest.approx(0.5)

    def test_project_clips_to_bounds(self):
        """Project should clip values to bounds."""
        constraint = BoundsConstraint(lower=0.0, upper=1.0)
        allocations = np.array([-0.2, 0.5, 1.3])
        projected = constraint.project(allocations)
        np.testing.assert_array_almost_equal(projected, [0.0, 0.5, 1.0])


class TestConstraintManager:
    """Tests for the constraint manager."""

    def test_add_and_evaluate(self):
        """Should add constraints and evaluate them."""
        manager = ConstraintManager()
        manager.add("budget", EqualityConstraint(target=1.0))
        manager.add("bounds", BoundsConstraint(lower=0.0, upper=1.0))

        allocations = np.array([0.5, 0.5])
        results = manager.evaluate_all(allocations)

        assert "budget" in results
        assert "bounds" in results
        assert results["budget"][0] is True  # satisfied
        assert results["bounds"][0] is True  # satisfied

    def test_all_satisfied_returns_true(self):
        """all_satisfied should return True when all constraints pass."""
        manager = ConstraintManager()
        manager.add("budget", EqualityConstraint(target=1.0))

        allocations = np.array([0.5, 0.5])
        assert manager.all_satisfied(allocations)

    def test_all_satisfied_returns_false(self):
        """all_satisfied should return False when any constraint fails."""
        manager = ConstraintManager()
        manager.add("budget", EqualityConstraint(target=1.0))

        allocations = np.array([0.3, 0.3])  # Sum = 0.6
        assert not manager.all_satisfied(allocations)

    def test_total_violation(self):
        """total_violation should sum all violations."""
        manager = ConstraintManager()
        manager.add("budget", EqualityConstraint(target=1.0))
        manager.add("max", InequalityConstraint(limit=0.5))

        allocations = np.array([0.4, 0.4])  # Sum = 0.8, budget violation = 0.2, max violation = 0.3
        violation = manager.total_violation(allocations)
        assert violation == pytest.approx(0.5)  # 0.2 + 0.3

    def test_remove_constraint(self):
        """Should be able to remove constraints."""
        manager = ConstraintManager()
        manager.add("budget", EqualityConstraint(target=1.0))
        manager.remove("budget")

        allocations = np.array([0.3])
        assert manager.all_satisfied(allocations)  # No constraints left

    def test_apply_all_penalties(self):
        """Should apply all constraint penalties."""
        manager = ConstraintManager()
        manager.add("budget", EqualityConstraint(target=1.0, penalty_strength=10.0))

        linear = np.zeros(2)
        quadratic = np.zeros((2, 2))
        weights = np.array([0.5, 0.5])

        offset = manager.apply_all(linear, quadratic, 0.0, weights)

        # offset should include target^2 * penalty = 1.0 * 10.0 = 10.0
        assert offset == pytest.approx(10.0)


class TestBackwardCompatibility:
    """Tests for backward compatibility with portfolio_constraints.py."""

    def test_evaluate_constraints_function(self):
        """evaluate_constraints should work like the old function."""
        from quantum_portfolio_optimizer.core.constraints import (
            BudgetConstraint,
            AllocationBounds,
        )

        constraints = {
            "budget": BudgetConstraint(limit=1.0),  # InequalityConstraint
            "bounds": AllocationBounds(lower=0.0, upper=1.0),  # BoundsConstraint
        }

        allocations = np.array([0.3, 0.3, 0.3])
        results = evaluate_constraints(allocations, constraints)

        assert results["budget"] is True
        assert results["bounds"] is True

    def test_budget_constraint_alias(self):
        """BudgetConstraint alias should work."""
        from quantum_portfolio_optimizer.core.constraints import BudgetConstraint

        constraint = BudgetConstraint(limit=1.0)
        assert constraint.is_satisfied(np.array([0.5, 0.4]))
        assert not constraint.is_satisfied(np.array([0.8, 0.8]))

    def test_allocation_bounds_alias(self):
        """AllocationBounds alias should work."""
        from quantum_portfolio_optimizer.core.constraints import AllocationBounds

        bounds = AllocationBounds(lower=0.0, upper=1.0)
        assert bounds.is_satisfied(np.array([0.5]))
        assert not bounds.is_satisfied(np.array([1.5]))
