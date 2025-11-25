"""Characterization tests for unused time_window parameter.

Bug Location: src/quantum_portfolio_optimizer/data/returns_calculator.py:10
Bug: `time_window` parameter is defined but never used in the function body.

The docstring says "Rolling window for calculations (default 30 days)" but
the parameter has no effect on the calculation.

These tests document the bug. After fix, a deprecation warning should be issued.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from quantum_portfolio_optimizer.data.returns_calculator import (
    calculate_logarithmic_returns,
    calculate_rolling_covariance,
)


class TestTimeWindowParameterBug:
    """Tests documenting the unused time_window parameter bug."""

    def test_time_window_has_no_effect(self):
        """BUG: time_window parameter should affect calculation but doesn't.

        This test documents the current buggy behavior where time_window
        is completely ignored.
        """
        prices = np.array([
            [100.0, 50.0],
            [102.0, 51.0],
            [104.0, 52.0],
            [106.0, 53.0],
            [108.0, 54.0],
            [110.0, 55.0],
        ])

        # Calculate with different time_window values
        returns_window_30 = calculate_logarithmic_returns(prices, time_window=30)
        returns_window_2 = calculate_logarithmic_returns(prices, time_window=2)
        returns_window_100 = calculate_logarithmic_returns(prices, time_window=100)

        # BUG: All results are identical regardless of time_window
        # This should NOT be the case - different windows should give different results
        assert np.allclose(returns_window_30, returns_window_2), (
            "Bug: time_window should affect results but doesn't"
        )
        assert np.allclose(returns_window_30, returns_window_100), (
            "Bug: time_window should affect results but doesn't"
        )

    def test_time_window_non_default_should_warn(self):
        """After fix: non-default time_window should emit DeprecationWarning.

        This test will FAIL before the fix (no warning) and PASS after.
        """
        prices = np.array([
            [100.0, 50.0],
            [102.0, 51.0],
            [104.0, 52.0],
        ])

        # After fix, using time_window != 30 should warn
        with pytest.warns(DeprecationWarning, match="time_window"):
            calculate_logarithmic_returns(prices, time_window=10)

    def test_time_window_default_no_warning(self):
        """Using default time_window=30 should not emit warning."""
        prices = np.array([
            [100.0, 50.0],
            [102.0, 51.0],
            [104.0, 52.0],
        ])

        # Default value should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # This should NOT raise any warning
            result = calculate_logarithmic_returns(prices, time_window=30)
            assert result is not None


class TestLogarithmicReturnsCorrectness:
    """Tests for correct logarithmic returns calculation."""

    def test_logarithmic_returns_formula(self):
        """Verify log returns formula: μ = log(P_{t+1}/P_t)."""
        prices = np.array([
            [100.0],
            [110.0],  # 10% increase
            [99.0],   # ~10% decrease
        ])

        returns = calculate_logarithmic_returns(prices)

        # log(110/100) ≈ 0.0953
        # log(99/110) ≈ -0.1054
        assert returns.shape == (2, 1)
        assert returns[0, 0] == pytest.approx(np.log(110/100))
        assert returns[1, 0] == pytest.approx(np.log(99/110))

    def test_logarithmic_returns_multiple_assets(self):
        """Test with multiple assets."""
        prices = np.array([
            [100.0, 50.0, 200.0],
            [105.0, 52.0, 190.0],
        ])

        returns = calculate_logarithmic_returns(prices)

        assert returns.shape == (1, 3)
        assert returns[0, 0] == pytest.approx(np.log(105/100))
        assert returns[0, 1] == pytest.approx(np.log(52/50))
        assert returns[0, 2] == pytest.approx(np.log(190/200))

    def test_logarithmic_returns_with_dataframe(self):
        """Test that pandas DataFrame input works."""
        df = pd.DataFrame({
            'AAPL': [100.0, 105.0, 110.0],
            'GOOG': [50.0, 51.0, 52.0],
        })

        returns = calculate_logarithmic_returns(df)

        assert returns.shape == (2, 2)
        assert isinstance(returns, np.ndarray)

    def test_logarithmic_returns_rejects_non_positive_prices(self):
        """Prices must be positive for log calculation."""
        prices_with_zero = np.array([[100.0], [0.0], [105.0]])
        prices_with_negative = np.array([[100.0], [-5.0], [105.0]])

        with pytest.raises(ValueError, match="positive"):
            calculate_logarithmic_returns(prices_with_zero)

        with pytest.raises(ValueError, match="positive"):
            calculate_logarithmic_returns(prices_with_negative)


class TestRollingCovariance:
    """Tests for rolling covariance calculation."""

    def test_rolling_covariance_uses_window(self):
        """Rolling covariance should actually use the window parameter."""
        # Create returns with distinct early and late patterns
        np.random.seed(42)
        early_returns = np.random.randn(50, 2) * 0.01  # Low volatility
        late_returns = np.random.randn(50, 2) * 0.05   # High volatility
        returns = np.vstack([early_returns, late_returns])

        # Full covariance vs windowed
        full_cov = calculate_rolling_covariance(returns, window=100)
        late_cov = calculate_rolling_covariance(returns, window=30)

        # Late window should show higher variance due to higher volatility
        # (This is the expected behavior - window should matter)
        assert late_cov[0, 0] > full_cov[0, 0] * 1.5, (
            "Rolling window should capture recent higher volatility"
        )

    def test_rolling_covariance_small_data(self):
        """When data < window, use all available data."""
        returns = np.array([
            [0.01, 0.02],
            [0.02, 0.01],
            [-0.01, -0.02],
        ])

        cov = calculate_rolling_covariance(returns, window=100)

        # Should use all 3 data points
        assert cov.shape == (2, 2)
        assert np.allclose(cov, np.cov(returns.T))
