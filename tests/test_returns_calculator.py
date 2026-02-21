"""Tests for returns calculator module."""

import numpy as np
import pandas as pd
import pytest

from quantum_portfolio_optimizer.data.returns_calculator import (
    calculate_logarithmic_returns,
    calculate_rolling_covariance,
)


class TestLogarithmicReturns:
    """Test logarithmic returns calculation."""

    def test_basic_calculation(self):
        """Basic log returns should be calculated correctly."""
        prices = np.array([100, 110, 105, 115])
        returns = calculate_logarithmic_returns(prices.reshape(-1, 1))

        # Manual calculation: log(110/100), log(105/110), log(115/105)
        expected = np.array(
            [[np.log(110 / 100)], [np.log(105 / 110)], [np.log(115 / 105)]]
        )
        np.testing.assert_array_almost_equal(returns, expected)

    def test_multiple_assets(self):
        """Should handle multiple assets correctly."""
        prices = np.array(
            [
                [100, 50],
                [110, 55],
                [105, 52],
            ]
        )
        returns = calculate_logarithmic_returns(prices)

        assert returns.shape == (2, 2)  # 3 prices -> 2 returns
        np.testing.assert_almost_equal(returns[0, 0], np.log(110 / 100))
        np.testing.assert_almost_equal(returns[0, 1], np.log(55 / 50))

    def test_dataframe_input(self):
        """Should accept pandas DataFrame input."""
        df = pd.DataFrame({"AAPL": [100, 110, 105], "MSFT": [50, 55, 52]})
        returns = calculate_logarithmic_returns(df)

        assert returns.shape == (2, 2)
        assert isinstance(returns, np.ndarray)

    def test_zero_price_raises_error(self):
        """Zero prices should raise ValueError."""
        prices = np.array([100, 0, 105])
        with pytest.raises(ValueError, match="positive"):
            calculate_logarithmic_returns(prices.reshape(-1, 1))

    def test_negative_price_raises_error(self):
        """Negative prices should raise ValueError."""
        prices = np.array([100, -50, 105])
        with pytest.raises(ValueError, match="positive"):
            calculate_logarithmic_returns(prices.reshape(-1, 1))

    def test_single_price_returns_empty(self):
        """Single price should return empty array."""
        prices = np.array([[100]])
        returns = calculate_logarithmic_returns(prices)
        assert len(returns) == 0

    def test_constant_prices_zero_return(self):
        """Constant prices should give zero returns."""
        prices = np.array([100, 100, 100, 100]).reshape(-1, 1)
        returns = calculate_logarithmic_returns(prices)
        np.testing.assert_array_almost_equal(returns, np.zeros((3, 1)))

    def test_doubling_price_log2(self):
        """Doubling prices should give ln(2) return."""
        prices = np.array([100, 200]).reshape(-1, 1)
        returns = calculate_logarithmic_returns(prices)
        np.testing.assert_almost_equal(returns[0, 0], np.log(2))


class TestRollingCovariance:
    """Test rolling covariance calculation."""

    def test_basic_covariance(self):
        """Basic covariance calculation should work."""
        returns = np.array(
            [
                [0.01, 0.02],
                [0.02, 0.01],
                [-0.01, -0.02],
                [-0.02, -0.01],
            ]
        )
        cov = calculate_rolling_covariance(returns)

        assert cov.shape == (2, 2)
        # Should be symmetric
        np.testing.assert_almost_equal(cov[0, 1], cov[1, 0])
        # Variance should be positive
        assert cov[0, 0] > 0
        assert cov[1, 1] > 0

    def test_window_larger_than_data(self):
        """Should use all data when window > data length."""
        returns = np.array(
            [
                [0.01, 0.02],
                [0.02, 0.01],
            ]
        )
        cov = calculate_rolling_covariance(returns, window=30)

        assert cov.shape == (2, 2)
        # Should still produce valid covariance

    def test_window_equals_data(self):
        """Should work when window equals data length."""
        returns = np.array(
            [
                [0.01, 0.02],
                [0.02, 0.01],
                [-0.01, -0.02],
            ]
        )
        cov = calculate_rolling_covariance(returns, window=3)

        assert cov.shape == (2, 2)

    def test_uses_last_window_periods(self):
        """Should use only last window periods."""
        # First part: high correlation
        # Second part: low correlation
        returns1 = np.array([[0.1, 0.1], [-0.1, -0.1]] * 5)
        returns2 = np.array([[0.1, -0.1], [-0.1, 0.1]] * 5)
        returns = np.vstack([returns1, returns2])

        # With small window, should only see low correlation data
        cov_small = calculate_rolling_covariance(returns, window=5)
        # With large window, sees both
        cov_large = calculate_rolling_covariance(returns, window=20)

        # Correlation from small window should be more negative
        assert cov_small[0, 1] < cov_large[0, 1]

    def test_dataframe_input(self):
        """Should accept pandas DataFrame input."""
        df = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.02, 0.01, -0.02]})
        cov = calculate_rolling_covariance(df)

        assert cov.shape == (2, 2)
        assert isinstance(cov, np.ndarray)

    def test_single_asset(self):
        """Should work with single asset."""
        returns = np.array([[0.01], [0.02], [-0.01], [-0.02]])
        cov = calculate_rolling_covariance(returns)

        # Should be a 1x1 matrix (variance)
        assert cov.shape == () or cov.shape == (1, 1) or isinstance(cov, float)

    def test_positive_semidefinite(self):
        """Covariance matrix should be positive semi-definite."""
        np.random.seed(42)
        returns = np.random.randn(100, 3) * 0.02
        cov = calculate_rolling_covariance(returns)

        # All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical error


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_prices(self):
        """Should handle very small prices."""
        prices = np.array([1e-10, 2e-10, 1.5e-10]).reshape(-1, 1)
        returns = calculate_logarithmic_returns(prices)
        assert not np.any(np.isnan(returns))
        assert not np.any(np.isinf(returns))

    def test_very_large_prices(self):
        """Should handle very large prices."""
        prices = np.array([1e10, 1.1e10, 1.05e10]).reshape(-1, 1)
        returns = calculate_logarithmic_returns(prices)
        assert not np.any(np.isnan(returns))
        assert not np.any(np.isinf(returns))

    def test_mixed_scale_prices(self):
        """Should handle prices at different scales."""
        prices = np.array(
            [
                [100, 10000],
                [110, 11000],
                [105, 10500],
            ]
        )
        returns = calculate_logarithmic_returns(prices)
        # Returns should be similar for proportional changes
        np.testing.assert_almost_equal(returns[:, 0], returns[:, 1])
