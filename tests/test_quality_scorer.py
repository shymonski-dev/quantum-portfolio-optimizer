"""Tests for solution quality scoring."""

import numpy as np
import pytest

from quantum_portfolio_optimizer.postprocessing.quality_scorer import (
    ScoringWeights,
    QualityScore,
    score_solution,
    _score_to_grade,
    _score_sharpe_ratio,
    _score_feasibility,
    _score_expected_return,
    _score_vs_classical,
)
from quantum_portfolio_optimizer.exceptions import InvalidWeightsError


class TestScoringWeights:
    """Test ScoringWeights dataclass."""

    def test_default_weights(self):
        """Default weights should be set correctly."""
        weights = ScoringWeights()

        assert weights.sharpe_weight == 0.4
        assert weights.feasibility_weight == 0.2
        assert weights.return_weight == 0.2
        assert weights.vs_classical_weight == 0.2

    def test_custom_weights(self):
        """Should accept custom weights."""
        weights = ScoringWeights(
            sharpe_weight=0.5,
            feasibility_weight=0.3,
            return_weight=0.1,
            vs_classical_weight=0.1,
        )

        assert weights.sharpe_weight == 0.5
        assert weights.feasibility_weight == 0.3

    def test_negative_weight_raises_error(self):
        """Negative weights should raise InvalidWeightsError."""
        with pytest.raises(InvalidWeightsError):
            ScoringWeights(sharpe_weight=-0.1)

    def test_all_zero_weights_raises_error(self):
        """All-zero weights should raise InvalidWeightsError."""
        with pytest.raises(InvalidWeightsError):
            ScoringWeights(
                sharpe_weight=0,
                feasibility_weight=0,
                return_weight=0,
                vs_classical_weight=0,
            )

    def test_normalized_sums_to_one(self):
        """Normalized weights should sum to 1.0."""
        weights = ScoringWeights(
            sharpe_weight=2.0,
            feasibility_weight=1.0,
            return_weight=1.0,
            vs_classical_weight=1.0,
        )

        normalized = weights.normalized()

        assert sum(normalized.values()) == pytest.approx(1.0)
        assert normalized["sharpe"] == pytest.approx(0.4)

    def test_normalized_preserves_ratios(self):
        """Normalization should preserve weight ratios."""
        weights = ScoringWeights(
            sharpe_weight=4.0,
            feasibility_weight=2.0,
            return_weight=2.0,
            vs_classical_weight=2.0,
        )

        normalized = weights.normalized()

        # sharpe is twice feasibility
        assert normalized["sharpe"] == pytest.approx(2 * normalized["feasibility"])


class TestScoreToGrade:
    """Test grade conversion."""

    def test_a_grade(self):
        """Score >= 90 should be A."""
        assert _score_to_grade(90) == "A"
        assert _score_to_grade(95) == "A"
        assert _score_to_grade(100) == "A"

    def test_b_grade(self):
        """Score 80-89 should be B."""
        assert _score_to_grade(80) == "B"
        assert _score_to_grade(85) == "B"
        assert _score_to_grade(89.9) == "B"

    def test_c_grade(self):
        """Score 70-79 should be C."""
        assert _score_to_grade(70) == "C"
        assert _score_to_grade(75) == "C"

    def test_d_grade(self):
        """Score 60-69 should be D."""
        assert _score_to_grade(60) == "D"
        assert _score_to_grade(65) == "D"

    def test_f_grade(self):
        """Score < 60 should be F."""
        assert _score_to_grade(59) == "F"
        assert _score_to_grade(30) == "F"
        assert _score_to_grade(0) == "F"


class TestScoreSharpeRatio:
    """Test Sharpe ratio scoring."""

    def test_negative_sharpe_low_score(self):
        """Negative Sharpe ratio should score low."""
        score = _score_sharpe_ratio(-0.5)
        assert 0 <= score <= 20

    def test_zero_sharpe_low_score(self):
        """Zero Sharpe ratio should score ~20."""
        score = _score_sharpe_ratio(0)
        assert score == pytest.approx(20)

    def test_moderate_sharpe_moderate_score(self):
        """Sharpe of 0.5-1.0 should score 50-70."""
        score = _score_sharpe_ratio(0.75)
        assert 50 <= score <= 70

    def test_good_sharpe_good_score(self):
        """Sharpe of 1.0-2.0 should score 70-90."""
        score = _score_sharpe_ratio(1.5)
        assert 70 <= score <= 90

    def test_excellent_sharpe_high_score(self):
        """Sharpe > 2.0 should score 90+."""
        score = _score_sharpe_ratio(2.5)
        assert score >= 90

    def test_score_capped_at_100(self):
        """Score should not exceed 100."""
        score = _score_sharpe_ratio(10.0)
        assert score <= 100


class TestScoreFeasibility:
    """Test feasibility scoring."""

    def test_perfect_feasibility(self):
        """Allocations summing to budget should score 100."""
        allocations = np.array([0.25, 0.25, 0.25, 0.25])
        score = _score_feasibility(allocations, budget=1.0)
        assert score == 100.0

    def test_within_tolerance(self):
        """Slight deviation within tolerance should score 100."""
        allocations = np.array([0.26, 0.25, 0.25, 0.245])  # Sum = 1.005
        score = _score_feasibility(allocations, budget=1.0, tolerance=0.01)
        assert score == 100.0

    def test_small_deviation_penalty(self):
        """Small deviation should have steep penalty."""
        allocations = np.array([0.3, 0.3, 0.3, 0.3])  # Sum = 1.2
        score = _score_feasibility(allocations, budget=1.0)
        assert 50 <= score < 100

    def test_large_deviation_low_score(self):
        """Large deviation should score low."""
        allocations = np.array([0.5, 0.5, 0.5, 0.5])  # Sum = 2.0
        score = _score_feasibility(allocations, budget=1.0)
        assert score < 50


class TestScoreExpectedReturn:
    """Test expected return scoring."""

    def test_negative_return_low_score(self):
        """Negative returns should score low."""
        score = _score_expected_return(-0.05)
        assert score < 30

    def test_zero_return_score(self):
        """Zero return should score 30."""
        score = _score_expected_return(0)
        assert score == pytest.approx(30)

    def test_below_benchmark_moderate_score(self):
        """Returns below benchmark should score 30-70."""
        score = _score_expected_return(0.05, benchmark_return=0.10)
        assert 30 < score < 70

    def test_at_benchmark_score(self):
        """Returns at benchmark should score 70."""
        score = _score_expected_return(0.10, benchmark_return=0.10)
        assert score == pytest.approx(70)

    def test_above_benchmark_high_score(self):
        """Returns above benchmark should score > 70."""
        score = _score_expected_return(0.15, benchmark_return=0.10)
        assert score > 70


class TestScoreVsClassical:
    """Test scoring vs classical baseline."""

    def test_equal_performance_moderate_score(self):
        """Equal to classical should score around 70."""
        score = _score_vs_classical(
            quantum_sharpe=1.0,
            classical_sharpe=1.0,
            quantum_return=0.10,
            classical_return=0.10,
        )
        assert 65 <= score <= 75

    def test_better_than_classical_high_score(self):
        """Better than classical should score high."""
        score = _score_vs_classical(
            quantum_sharpe=1.5,
            classical_sharpe=1.0,
            quantum_return=0.15,
            classical_return=0.10,
        )
        assert score > 75

    def test_worse_than_classical_low_score(self):
        """Worse than classical should score lower."""
        score = _score_vs_classical(
            quantum_sharpe=0.5,
            classical_sharpe=1.0,
            quantum_return=0.05,
            classical_return=0.10,
        )
        assert score < 65


class TestScoreSolution:
    """Test full solution scoring."""

    def test_good_solution_high_score(self):
        """Well-balanced solution should score well."""
        allocations = np.array([0.25, 0.25, 0.25, 0.25])
        result = score_solution(
            allocations=allocations,
            expected_return=0.12,
            portfolio_risk=0.08,
            budget=1.0,
        )

        assert result.total_score > 60
        assert result.grade in ["A", "B", "C"]

    def test_poor_solution_low_score(self):
        """Poor solution should score low."""
        allocations = np.array([1.0, 1.0, 0.0, 0.0])  # Over budget
        result = score_solution(
            allocations=allocations,
            expected_return=-0.05,  # Negative return
            portfolio_risk=0.20,  # High risk
            budget=1.0,
        )

        assert result.total_score < 50

    def test_result_contains_all_components(self):
        """Result should contain all component scores."""
        allocations = np.array([0.5, 0.5])
        result = score_solution(
            allocations=allocations,
            expected_return=0.10,
            portfolio_risk=0.10,
        )

        assert "sharpe" in result.component_scores
        assert "feasibility" in result.component_scores
        assert "return" in result.component_scores
        assert "vs_classical" in result.component_scores

    def test_result_has_details(self):
        """Result should include calculation details."""
        allocations = np.array([0.5, 0.5])
        result = score_solution(
            allocations=allocations,
            expected_return=0.10,
            portfolio_risk=0.10,
        )

        assert "sharpe_ratio" in result.details
        assert "allocation_sum" in result.details
        assert "weights_used" in result.details

    def test_custom_weights_affect_score(self):
        """Custom weights should change the final score."""
        allocations = np.array([0.5, 0.5])

        # Score with default weights
        result_default = score_solution(
            allocations=allocations,
            expected_return=0.10,
            portfolio_risk=0.10,
        )

        # Score with all weight on feasibility
        result_feasibility = score_solution(
            allocations=allocations,
            expected_return=0.10,
            portfolio_risk=0.10,
            weights=ScoringWeights(
                sharpe_weight=0,
                feasibility_weight=1.0,
                return_weight=0,
                vs_classical_weight=0,
            ),
        )

        # Feasibility should be 100, so feasibility-weighted score should be higher
        assert result_feasibility.total_score == pytest.approx(100.0)
        assert result_feasibility.total_score > result_default.total_score

    def test_with_classical_baseline(self):
        """Score should incorporate classical baseline when provided."""
        allocations = np.array([0.5, 0.5])

        result = score_solution(
            allocations=allocations,
            expected_return=0.10,
            portfolio_risk=0.10,
            classical_return=0.08,
            classical_risk=0.12,
        )

        # Better than classical should boost vs_classical component
        assert result.component_scores["vs_classical"] > 70


class TestQualityScoreToDict:
    """Test QualityScore serialization."""

    def test_to_dict_structure(self):
        """to_dict should return proper structure."""
        score = QualityScore(
            total_score=85.5,
            component_scores={"sharpe": 90.0, "feasibility": 80.0},
            grade="B",
            summary="Good solution",
            details={"sharpe_ratio": 1.5},
        )

        d = score.to_dict()

        assert d["total_score"] == 85.5
        assert d["grade"] == "B"
        assert d["summary"] == "Good solution"
        assert "sharpe" in d["component_scores"]
        assert "sharpe_ratio" in d["details"]

    def test_to_dict_rounds_values(self):
        """to_dict should round numeric values."""
        score = QualityScore(
            total_score=85.5555,
            component_scores={"test": 90.1234},
            grade="B",
            summary="Test",
        )

        d = score.to_dict()

        assert d["total_score"] == 85.6
        assert d["component_scores"]["test"] == 90.1
