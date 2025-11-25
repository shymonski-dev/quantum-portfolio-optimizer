"""Core optimisation primitives."""

from .qubo_formulation import PortfolioQUBO, QUBOProblem
from .vqe_solver import PortfolioVQESolver, VQEResult
from .qaoa_solver import PortfolioQAOASolver, QAOAResult, get_qaoa_circuit_depth
from .constraints import (
    Constraint,
    ConstraintManager,
    EqualityConstraint,
    InequalityConstraint,
    BoundsConstraint,
)

__all__ = [
    "PortfolioQUBO",
    "QUBOProblem",
    "PortfolioVQESolver",
    "VQEResult",
    "PortfolioQAOASolver",
    "QAOAResult",
    "get_qaoa_circuit_depth",
    # Constraint system
    "Constraint",
    "ConstraintManager",
    "EqualityConstraint",
    "InequalityConstraint",
    "BoundsConstraint",
]
