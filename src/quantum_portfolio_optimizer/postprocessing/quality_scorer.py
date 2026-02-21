"""Solution quality scoring for quantum portfolio optimization.

This module provides configurable quality scoring for portfolio solutions,
comparing quantum results against classical baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from quantum_portfolio_optimizer.exceptions import InvalidWeightsError


@dataclass
class ScoringWeights:
    """Configurable weights for solution quality scoring.

    All weights are normalized to sum to 1.0 during scoring.

    Attributes:
        sharpe_weight: Weight for Sharpe ratio component (0-1)
        feasibility_weight: Weight for budget constraint satisfaction (0-1)
        return_weight: Weight for expected return component (0-1)
        vs_classical_weight: Weight for comparison vs Markowitz baseline (0-1)
    """

    sharpe_weight: float = 0.4
    feasibility_weight: float = 0.2
    return_weight: float = 0.2
    vs_classical_weight: float = 0.2

    def __post_init__(self):
        """Validate weights are non-negative."""
        weights = [
            self.sharpe_weight,
            self.feasibility_weight,
            self.return_weight,
            self.vs_classical_weight,
        ]
        if any(w < 0 for w in weights):
            raise InvalidWeightsError(
                {
                    "sharpe": self.sharpe_weight,
                    "feasibility": self.feasibility_weight,
                    "return": self.return_weight,
                    "vs_classical": self.vs_classical_weight,
                },
                "All weights must be non-negative",
            )
        if sum(weights) == 0:
            raise InvalidWeightsError(
                {
                    "sharpe": self.sharpe_weight,
                    "feasibility": self.feasibility_weight,
                    "return": self.return_weight,
                    "vs_classical": self.vs_classical_weight,
                },
                "At least one weight must be positive",
            )

    def normalized(self) -> Dict[str, float]:
        """Return normalized weights that sum to 1.0."""
        total = (
            self.sharpe_weight
            + self.feasibility_weight
            + self.return_weight
            + self.vs_classical_weight
        )
        return {
            "sharpe": self.sharpe_weight / total,
            "feasibility": self.feasibility_weight / total,
            "return": self.return_weight / total,
            "vs_classical": self.vs_classical_weight / total,
        }


@dataclass
class QualityScore:
    """Result of solution quality scoring.

    Attributes:
        total_score: Combined weighted score (0-100)
        component_scores: Individual component scores
        grade: Letter grade (A-F)
        summary: Human-readable summary
        details: Additional scoring details
    """

    total_score: float
    component_scores: Dict[str, float]
    grade: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "total_score": round(self.total_score, 1),
            "component_scores": {
                k: round(v, 1) for k, v in self.component_scores.items()
            },
            "grade": self.grade,
            "summary": self.summary,
            "details": self.details,
        }


def _score_to_grade(score: float) -> str:
    """Convert numeric score (0-100) to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def _score_sharpe_ratio(sharpe: float, risk_free_rate: float = 0.02) -> float:
    """Score Sharpe ratio on 0-100 scale.

    Scoring:
        - Sharpe < 0: 0-20 (poor, negative risk-adjusted returns)
        - Sharpe 0-0.5: 20-50 (below average)
        - Sharpe 0.5-1.0: 50-70 (acceptable)
        - Sharpe 1.0-2.0: 70-90 (good)
        - Sharpe > 2.0: 90-100 (excellent)
    """
    if sharpe < 0:
        return max(0, 20 + sharpe * 20)  # Negative sharpe scores 0-20
    elif sharpe < 0.5:
        return 20 + sharpe * 60  # 0 to 0.5 maps to 20-50
    elif sharpe < 1.0:
        return 50 + (sharpe - 0.5) * 40  # 0.5 to 1.0 maps to 50-70
    elif sharpe < 2.0:
        return 70 + (sharpe - 1.0) * 20  # 1.0 to 2.0 maps to 70-90
    else:
        return min(100, 90 + (sharpe - 2.0) * 5)  # >2.0 approaches 100


def _score_feasibility(
    allocations: np.ndarray,
    budget: float,
    tolerance: float = 0.01,
) -> float:
    """Score budget constraint feasibility on 0-100 scale.

    100 = perfect budget satisfaction
    Decreases as allocation sum deviates from budget
    """
    total_allocation = np.sum(allocations)
    deviation = abs(total_allocation - budget) / budget

    if deviation <= tolerance:
        return 100.0
    elif deviation <= 0.05:
        return 100 - (deviation - tolerance) * 1000  # Steep penalty
    elif deviation <= 0.20:
        return max(50, 95 - deviation * 100)
    else:
        return max(0, 50 - (deviation - 0.20) * 200)


def _score_expected_return(
    expected_return: float,
    benchmark_return: float = 0.10,  # 10% annual benchmark
) -> float:
    """Score expected return on 0-100 scale.

    Scoring relative to a benchmark return (default 10% annual).
    """
    if expected_return <= 0:
        return max(0, 30 + expected_return * 100)  # Negative returns score 0-30
    elif expected_return < benchmark_return:
        ratio = expected_return / benchmark_return
        return 30 + ratio * 40  # Below benchmark: 30-70
    elif expected_return < benchmark_return * 2:
        excess = (expected_return - benchmark_return) / benchmark_return
        return 70 + excess * 30  # Above benchmark: 70-100
    else:
        return 100.0


def _score_vs_classical(
    quantum_sharpe: float,
    classical_sharpe: float,
    quantum_return: float,
    classical_return: float,
) -> float:
    """Score quantum solution vs classical Markowitz baseline.

    100 = quantum matches or beats classical on both metrics
    50 = quantum matches classical
    0 = quantum significantly worse than classical
    """
    # Sharpe comparison (weight: 60%)
    if classical_sharpe != 0:
        sharpe_ratio = quantum_sharpe / classical_sharpe
    else:
        sharpe_ratio = 1.0 if quantum_sharpe >= 0 else 0.5

    if sharpe_ratio >= 1.0:
        sharpe_score = min(100, 70 + (sharpe_ratio - 1.0) * 30)
    else:
        sharpe_score = max(0, sharpe_ratio * 70)

    # Return comparison (weight: 40%)
    if classical_return > 0:
        return_ratio = quantum_return / classical_return
    else:
        return_ratio = 1.0 if quantum_return >= 0 else 0.5

    if return_ratio >= 1.0:
        return_score = min(100, 70 + (return_ratio - 1.0) * 30)
    else:
        return_score = max(0, return_ratio * 70)

    return 0.6 * sharpe_score + 0.4 * return_score


def score_solution(
    allocations: np.ndarray,
    expected_return: float,
    portfolio_risk: float,
    classical_return: Optional[float] = None,
    classical_risk: Optional[float] = None,
    budget: float = 1.0,
    risk_free_rate: float = 0.02,
    weights: Optional[ScoringWeights] = None,
) -> QualityScore:
    """Score a portfolio solution quality.

    Args:
        allocations: Portfolio allocation weights (should sum to budget)
        expected_return: Expected portfolio return (annualized)
        portfolio_risk: Portfolio risk/volatility (annualized std)
        classical_return: Classical Markowitz expected return (optional)
        classical_risk: Classical Markowitz risk (optional)
        budget: Target budget for allocation (default 1.0)
        risk_free_rate: Risk-free rate for Sharpe calculation
        weights: Scoring weights configuration

    Returns:
        QualityScore with total score, component scores, and grade
    """
    if weights is None:
        weights = ScoringWeights()

    normalized_weights = weights.normalized()

    # Calculate Sharpe ratio
    if portfolio_risk > 0:
        sharpe = (expected_return - risk_free_rate) / portfolio_risk
    else:
        sharpe = 0.0

    # Calculate classical Sharpe if available
    if (
        classical_risk is not None
        and classical_risk > 0
        and classical_return is not None
    ):
        classical_sharpe = (classical_return - risk_free_rate) / classical_risk
    else:
        classical_sharpe = sharpe  # Default to same as quantum

    # Score each component
    sharpe_score = _score_sharpe_ratio(sharpe, risk_free_rate)
    feasibility_score = _score_feasibility(allocations, budget)
    return_score = _score_expected_return(expected_return)

    # Score vs classical (if available)
    if classical_return is not None and classical_risk is not None:
        vs_classical_score = _score_vs_classical(
            sharpe, classical_sharpe, expected_return, classical_return
        )
    else:
        vs_classical_score = 70.0  # Default neutral score when no baseline

    # Calculate weighted total
    component_scores = {
        "sharpe": sharpe_score,
        "feasibility": feasibility_score,
        "return": return_score,
        "vs_classical": vs_classical_score,
    }

    total_score = (
        normalized_weights["sharpe"] * sharpe_score
        + normalized_weights["feasibility"] * feasibility_score
        + normalized_weights["return"] * return_score
        + normalized_weights["vs_classical"] * vs_classical_score
    )

    grade = _score_to_grade(total_score)

    # Generate summary
    if total_score >= 80:
        summary = "Excellent solution with strong risk-adjusted returns"
    elif total_score >= 60:
        summary = "Good solution meeting most quality criteria"
    elif total_score >= 40:
        summary = "Acceptable solution with room for improvement"
    else:
        summary = "Solution needs optimization - consider adjusting parameters"

    return QualityScore(
        total_score=total_score,
        component_scores=component_scores,
        grade=grade,
        summary=summary,
        details={
            "sharpe_ratio": round(sharpe, 3),
            "classical_sharpe": round(classical_sharpe, 3)
            if classical_sharpe != sharpe
            else None,
            "allocation_sum": round(float(np.sum(allocations)), 4),
            "budget_deviation": round(abs(float(np.sum(allocations)) - budget), 4),
            "weights_used": normalized_weights,
        },
    )
