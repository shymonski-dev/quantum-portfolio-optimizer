"""Unified constraint system for portfolio QUBO construction and evaluation.

This module provides both:
1. QUBO penalty application during construction (for quantum optimization)
2. Post-hoc solution validation (for interpreting results)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class Constraint(ABC):
    """Abstract base class for QUBO constraints."""

    @abstractmethod
    def apply_penalty(
        self,
        linear: np.ndarray,
        quadratic: np.ndarray,
        offset: float,
        weights: np.ndarray,
        indices: List[int],
    ) -> float:
        """Apply quadratic penalty to QUBO matrices.

        The penalty is applied in-place to linear and quadratic arrays.

        Args:
            linear: Linear coefficients array (modified in-place).
            quadratic: Quadratic coefficients matrix (modified in-place).
            offset: Current constant offset.
            weights: Variable weights for allocation values.
            indices: Variable indices this constraint applies to.

        Returns:
            Updated offset value.
        """
        pass

    @abstractmethod
    def is_satisfied(self, allocations: np.ndarray, atol: float = 1e-6) -> bool:
        """Check if allocations satisfy this constraint.

        Args:
            allocations: Allocation values to check.
            atol: Absolute tolerance for equality comparisons.

        Returns:
            True if constraint is satisfied.
        """
        pass

    @abstractmethod
    def violation(self, allocations: np.ndarray) -> float:
        """Compute the constraint violation magnitude.

        Args:
            allocations: Allocation values to check.

        Returns:
            Non-negative violation amount (0 if satisfied).
        """
        pass


@dataclass
class EqualityConstraint(Constraint):
    """Equality constraint: (sum_i w_i * x_i - target)^2 penalty.

    Used for budget constraints where total allocation must equal target.
    """

    target: float
    penalty_strength: float = 1.0
    name: str = "equality"

    def apply_penalty(
        self,
        linear: np.ndarray,
        quadratic: np.ndarray,
        offset: float,
        weights: np.ndarray,
        indices: List[int],
    ) -> float:
        """Apply (sum - target)^2 penalty to QUBO."""
        if not indices:
            return offset

        w = weights[indices]
        offset += self.penalty_strength * self.target**2

        for i, idx_i in enumerate(indices):
            weight_i = w[i]
            # Linear term: λ * w_i^2 - 2λ * target * w_i
            linear[idx_i] += self.penalty_strength * weight_i**2
            linear[idx_i] -= 2 * self.penalty_strength * self.target * weight_i

            # Quadratic terms: 2λ * w_i * w_j for i != j
            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                weight_j = w[j]
                coeff = 2 * self.penalty_strength * weight_i * weight_j
                quadratic[idx_i, idx_j] += coeff
                quadratic[idx_j, idx_i] += coeff

        return offset

    def is_satisfied(self, allocations: np.ndarray, atol: float = 1e-6) -> bool:
        """Check if total allocation equals target."""
        return bool(np.abs(allocations.sum() - self.target) <= atol)

    def violation(self, allocations: np.ndarray) -> float:
        """Return absolute difference from target."""
        return float(np.abs(allocations.sum() - self.target))


@dataclass
class InequalityConstraint(Constraint):
    """Inequality constraint: penalizes (sum_i w_i * x_i - limit)^2 when exceeded.

    Used for maximum allocation constraints.
    """

    limit: float
    penalty_strength: float = 1.0
    name: str = "inequality"

    def apply_penalty(
        self,
        linear: np.ndarray,
        quadratic: np.ndarray,
        offset: float,
        weights: np.ndarray,
        indices: List[int],
    ) -> float:
        """Apply (sum - limit)^2 penalty to QUBO.

        Note: For binary variables, this penalizes deviation from limit.
        For upper-bound only constraints, use slack variables or reformulation.
        """
        if not indices:
            return offset

        w = weights[indices]
        offset += self.penalty_strength * self.limit**2

        for i, idx_i in enumerate(indices):
            weight_i = w[i]
            linear[idx_i] += self.penalty_strength * weight_i**2
            linear[idx_i] -= 2 * self.penalty_strength * self.limit * weight_i

            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                weight_j = w[j]
                coeff = 2 * self.penalty_strength * weight_i * weight_j
                quadratic[idx_i, idx_j] += coeff
                quadratic[idx_j, idx_i] += coeff

        return offset

    def is_satisfied(self, allocations: np.ndarray, atol: float = 1e-6) -> bool:
        """Check if total allocation is within limit."""
        return bool(allocations.sum() <= self.limit + atol)

    def violation(self, allocations: np.ndarray) -> float:
        """Return excess over limit (0 if within limit)."""
        return float(max(0.0, allocations.sum() - self.limit))


@dataclass
class BoundsConstraint(Constraint):
    """Per-variable bounds constraint: lower <= x_i <= upper.

    Used for allocation bounds on individual assets.
    """

    lower: float = 0.0
    upper: float = 1.0
    penalty_strength: float = 1.0
    name: str = "bounds"

    def apply_penalty(
        self,
        linear: np.ndarray,
        quadratic: np.ndarray,
        offset: float,
        weights: np.ndarray,
        indices: List[int],
    ) -> float:
        """Apply bounds penalty to QUBO.

        Note: For binary QUBO variables (x_i in {0,1}), bounds are naturally
        enforced. This penalty is for soft enforcement on decoded allocations.
        """
        # For binary QUBO, bounds are implicit. This method supports
        # future extensions with continuous relaxations.
        return offset

    def is_satisfied(self, allocations: np.ndarray, atol: float = 1e-6) -> bool:
        """Check if all allocations are within bounds."""
        return bool(
            np.all(allocations >= self.lower - atol)
            and np.all(allocations <= self.upper + atol)
        )

    def violation(self, allocations: np.ndarray) -> float:
        """Return total out-of-bounds violation."""
        lower_violation = np.sum(np.maximum(0.0, self.lower - allocations))
        upper_violation = np.sum(np.maximum(0.0, allocations - self.upper))
        return float(lower_violation + upper_violation)

    def project(self, allocations: np.ndarray) -> np.ndarray:
        """Project allocations into valid bounds."""
        return np.clip(allocations, self.lower, self.upper)


@dataclass
class ConstraintManager:
    """Manages multiple constraints for QUBO construction and evaluation.

    Example usage:
        manager = ConstraintManager()
        manager.add("budget", EqualityConstraint(target=1.0, penalty_strength=100))
        manager.add("asset_0_max", InequalityConstraint(limit=0.5, penalty_strength=50))

        # During QUBO construction:
        offset = manager.apply_all(linear, quadratic, offset, weights, index_map)

        # During solution evaluation:
        results = manager.evaluate_all(allocations)
    """

    constraints: Dict[str, Constraint] = field(default_factory=dict)
    _index_map: Dict[str, List[int]] = field(default_factory=dict)

    def add(
        self,
        name: str,
        constraint: Constraint,
        indices: Optional[List[int]] = None,
    ) -> None:
        """Add a constraint to the manager.

        Args:
            name: Unique identifier for this constraint.
            constraint: The constraint object.
            indices: Variable indices this constraint applies to.
                     If None, constraint applies to all variables.
        """
        self.constraints[name] = constraint
        if indices is not None:
            self._index_map[name] = indices

    def remove(self, name: str) -> None:
        """Remove a constraint by name."""
        self.constraints.pop(name, None)
        self._index_map.pop(name, None)

    def apply_all(
        self,
        linear: np.ndarray,
        quadratic: np.ndarray,
        offset: float,
        weights: np.ndarray,
        default_indices: Optional[List[int]] = None,
    ) -> float:
        """Apply all constraint penalties to QUBO matrices.

        Args:
            linear: Linear coefficients (modified in-place).
            quadratic: Quadratic coefficients (modified in-place).
            offset: Current constant offset.
            weights: Variable weights for allocation values.
            default_indices: Default indices if not specified per constraint.

        Returns:
            Updated offset value.
        """
        if default_indices is None:
            default_indices = list(range(len(linear)))

        for name, constraint in self.constraints.items():
            indices = self._index_map.get(name, default_indices)
            offset = constraint.apply_penalty(
                linear, quadratic, offset, weights, indices
            )

        return offset

    def evaluate_all(
        self,
        allocations: np.ndarray,
        atol: float = 1e-6,
    ) -> Dict[str, Tuple[bool, float]]:
        """Evaluate all constraints against given allocations.

        Args:
            allocations: Allocation values to check.
            atol: Tolerance for satisfaction checks.

        Returns:
            Dictionary mapping constraint name to (is_satisfied, violation).
        """
        results = {}
        for name, constraint in self.constraints.items():
            satisfied = constraint.is_satisfied(allocations, atol=atol)
            violation = constraint.violation(allocations)
            results[name] = (satisfied, violation)
        return results

    def all_satisfied(self, allocations: np.ndarray, atol: float = 1e-6) -> bool:
        """Check if all constraints are satisfied."""
        return all(
            constraint.is_satisfied(allocations, atol=atol)
            for constraint in self.constraints.values()
        )

    def total_violation(self, allocations: np.ndarray) -> float:
        """Compute sum of all constraint violations."""
        return sum(
            constraint.violation(allocations)
            for constraint in self.constraints.values()
        )


# Backward compatibility aliases for portfolio_constraints.py migration
BudgetConstraint = InequalityConstraint
AllocationBounds = BoundsConstraint


def evaluate_constraints(
    allocations: np.ndarray,
    constraints: Dict[str, Constraint],
) -> Dict[str, bool]:
    """Evaluate multiple constraints (backward compatibility function).

    Args:
        allocations: Allocation values to check.
        constraints: Dictionary of constraints to evaluate.

    Returns:
        Dictionary mapping constraint name to satisfaction status.
    """
    return {
        name: constraint.is_satisfied(allocations)
        for name, constraint in constraints.items()
    }
