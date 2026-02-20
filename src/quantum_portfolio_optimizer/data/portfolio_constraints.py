"""DEPRECATED: Use quantum_portfolio_optimizer.core.constraints instead.

This module is deprecated and will be removed in a future version.
All constraint classes have been moved to the core.constraints module.

Migration guide:
    # Old import (deprecated)
    from quantum_portfolio_optimizer.data.portfolio_constraints import BudgetConstraint

    # New import (recommended)
    from quantum_portfolio_optimizer.core.constraints import InequalityConstraint as BudgetConstraint
"""

import warnings

from quantum_portfolio_optimizer.core.constraints import (
    BudgetConstraint,
    AllocationBounds,
    evaluate_constraints,
)

warnings.warn(
    "quantum_portfolio_optimizer.data.portfolio_constraints is deprecated. "
    "Use quantum_portfolio_optimizer.core.constraints instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BudgetConstraint", "AllocationBounds", "evaluate_constraints"]
