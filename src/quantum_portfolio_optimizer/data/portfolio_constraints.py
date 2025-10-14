"""Constraint helper objects for portfolio construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BudgetConstraint:
    limit: float

    def penalty(self, allocations: np.ndarray) -> float:
        excess = allocations.sum() - self.limit
        return float(max(0.0, excess))

    def is_satisfied(self, allocations: np.ndarray, atol: float = 1e-6) -> bool:
        return allocations.sum() <= self.limit + atol


@dataclass
class AllocationBounds:
    lower: float
    upper: float

    def project(self, allocations: np.ndarray) -> np.ndarray:
        return np.clip(allocations, self.lower, self.upper, dtype=float)

    def is_satisfied(self, allocations: np.ndarray, atol: float = 1e-6) -> bool:
        return bool(np.all(allocations >= self.lower - atol) and np.all(allocations <= self.upper + atol))


def evaluate_constraints(
    allocations: np.ndarray,
    constraints: Dict[str, BudgetConstraint | AllocationBounds],
) -> Dict[str, bool]:
    return {
        name: constraint.is_satisfied(allocations)
        if hasattr(constraint, "is_satisfied")
        else True
        for name, constraint in constraints.items()
    }
