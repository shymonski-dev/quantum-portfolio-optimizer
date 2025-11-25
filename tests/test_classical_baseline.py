"""Tests for classical Markowitz baseline module."""

import numpy as np
import pytest

from quantum_portfolio_optimizer.benchmarks.classical_baseline import (
    BaselineResult,
    markowitz_baseline,
)


class TestMarkowitzBaseline:
    """Test the Markowitz baseline optimizer."""

    def test_basic_optimization(self):
        """Basic optimization should return valid result."""
        expected_returns = [0.1, 0.15, 0.12]
        covariance = [
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.01],
            [0.02, 0.01, 0.06],
        ]

        result = markowitz_baseline(expected_returns, covariance)

        assert isinstance(result, BaselineResult)
        assert result.success
        assert len(result.allocations) == 3
        assert np.isclose(sum(result.allocations), 1.0, atol=1e-6)
        assert all(0 <= w <= 1 for w in result.allocations)

    def test_budget_constraint(self):
        """Allocations should sum to budget."""
        expected_returns = [0.1, 0.2]
        covariance = [[0.04, 0.01], [0.01, 0.09]]

        result = markowitz_baseline(expected_returns, covariance, budget=1.0)

        assert np.isclose(sum(result.allocations), 1.0, atol=1e-6)

    def test_custom_budget(self):
        """Custom budget should be respected."""
        expected_returns = [0.1, 0.2]
        covariance = [[0.04, 0.01], [0.01, 0.09]]

        result = markowitz_baseline(expected_returns, covariance, budget=0.5)

        assert np.isclose(sum(result.allocations), 0.5, atol=1e-6)

    def test_high_risk_aversion_favors_low_variance(self):
        """High risk aversion should favor lower variance assets."""
        # Asset 0: low return, low variance; Asset 1: high return, high variance
        expected_returns = [0.05, 0.20]
        covariance = [[0.01, 0.0], [0.0, 0.25]]

        # Very high risk aversion
        result_high = markowitz_baseline(
            expected_returns, covariance, risk_aversion=10.0
        )
        # Low risk aversion
        result_low = markowitz_baseline(
            expected_returns, covariance, risk_aversion=0.01
        )

        # High risk aversion should allocate more to asset 0 (lower variance)
        assert result_high.allocations[0] > result_low.allocations[0]

    def test_low_risk_aversion_favors_high_return(self):
        """Low risk aversion should favor higher return assets."""
        # Asset 0: low return; Asset 1: high return
        expected_returns = [0.05, 0.20]
        covariance = [[0.04, 0.01], [0.01, 0.04]]  # Same variance

        result = markowitz_baseline(
            expected_returns, covariance, risk_aversion=0.01
        )

        # Low risk aversion should favor high return asset
        assert result.allocations[1] > result.allocations[0]

    def test_bounds_respected(self):
        """Allocation bounds should be respected."""
        expected_returns = [0.1, 0.2, 0.15]
        covariance = [
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.01],
            [0.02, 0.01, 0.06],
        ]

        result = markowitz_baseline(
            expected_returns, covariance, bounds=(0.1, 0.5)
        )

        assert all(0.1 - 1e-6 <= w <= 0.5 + 1e-6 for w in result.allocations)

    def test_equal_assets_equal_weights(self):
        """Identical assets should get equal weights."""
        expected_returns = [0.1, 0.1, 0.1]
        covariance = [
            [0.04, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ]

        result = markowitz_baseline(expected_returns, covariance)

        # All weights should be approximately equal
        assert np.allclose(result.allocations, [1/3, 1/3, 1/3], atol=0.05)

    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio should be calculated correctly."""
        expected_returns = [0.1, 0.15]
        covariance = [[0.04, 0.01], [0.01, 0.06]]

        result = markowitz_baseline(
            expected_returns, covariance, risk_free_rate=0.02
        )

        # Manual Sharpe calculation
        weights = result.allocations
        port_return = sum(w * r for w, r in zip(weights, expected_returns))
        port_variance = np.array(weights) @ np.array(covariance) @ np.array(weights)
        expected_sharpe = (port_return - 0.02) / np.sqrt(port_variance)

        assert np.isclose(result.sharpe_ratio, expected_sharpe, atol=1e-4)

    def test_expected_return_calculation(self):
        """Expected return should be weighted sum."""
        expected_returns = [0.1, 0.2]
        covariance = [[0.04, 0.01], [0.01, 0.09]]

        result = markowitz_baseline(expected_returns, covariance)

        # Manual calculation
        expected = sum(w * r for w, r in zip(result.allocations, expected_returns))
        assert np.isclose(result.expected_return, expected, atol=1e-6)

    def test_variance_calculation(self):
        """Portfolio variance should be calculated correctly."""
        expected_returns = [0.1, 0.2]
        covariance = [[0.04, 0.01], [0.01, 0.09]]

        result = markowitz_baseline(expected_returns, covariance)

        # Manual calculation
        w = np.array(result.allocations)
        cov = np.array(covariance)
        expected_var = w @ cov @ w
        assert np.isclose(result.variance, expected_var, atol=1e-6)

    def test_single_asset(self):
        """Single asset should get 100% allocation."""
        expected_returns = [0.1]
        covariance = [[0.04]]

        result = markowitz_baseline(expected_returns, covariance)

        assert np.isclose(result.allocations[0], 1.0, atol=1e-6)

    def test_many_assets(self):
        """Should handle larger number of assets."""
        n = 10
        np.random.seed(42)
        expected_returns = np.random.uniform(0.05, 0.20, n).tolist()

        # Generate positive semi-definite covariance matrix
        A = np.random.randn(n, n) * 0.1
        covariance = (A @ A.T + np.eye(n) * 0.01).tolist()

        result = markowitz_baseline(expected_returns, covariance)

        assert result.success
        assert len(result.allocations) == n
        assert np.isclose(sum(result.allocations), 1.0, atol=1e-5)


class TestBaselineResultDataclass:
    """Test the BaselineResult dataclass."""

    def test_dataclass_fields(self):
        """BaselineResult should have all expected fields."""
        result = BaselineResult(
            allocations=np.array([0.5, 0.5]),
            expected_return=0.15,
            variance=0.04,
            sharpe_ratio=1.5,
            success=True,
            message="Optimization successful",
        )

        assert hasattr(result, 'allocations')
        assert hasattr(result, 'expected_return')
        assert hasattr(result, 'variance')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_dimension_mismatch_raises_error(self):
        """Mismatched dimensions should raise ValueError."""
        expected_returns = [0.1, 0.2, 0.15]
        covariance = [[0.04, 0.01], [0.01, 0.09]]  # 2x2 instead of 3x3

        with pytest.raises(ValueError, match="dimensions"):
            markowitz_baseline(expected_returns, covariance)

    def test_zero_variance_asset(self):
        """Should handle asset with zero variance."""
        expected_returns = [0.1, 0.2]
        covariance = [[0.0, 0.0], [0.0, 0.04]]

        result = markowitz_baseline(expected_returns, covariance)

        # Should still produce a valid result
        assert np.isclose(sum(result.allocations), 1.0, atol=1e-5)

    def test_negative_returns(self):
        """Should handle negative expected returns."""
        expected_returns = [-0.05, 0.10]
        covariance = [[0.04, 0.01], [0.01, 0.09]]

        result = markowitz_baseline(expected_returns, covariance)

        # Should favor the positive return asset
        assert result.allocations[1] > result.allocations[0]

    def test_high_correlation(self):
        """Should handle highly correlated assets."""
        expected_returns = [0.1, 0.12]
        # Very high correlation
        covariance = [[0.04, 0.039], [0.039, 0.04]]

        result = markowitz_baseline(expected_returns, covariance)

        assert result.success
        assert np.isclose(sum(result.allocations), 1.0, atol=1e-5)
