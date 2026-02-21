"""Tests for data fetcher module including input validation."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from quantum_portfolio_optimizer.data.data_fetcher import (
    fetch_stock_data,
    validate_inputs,
)
from quantum_portfolio_optimizer.exceptions import (
    InsufficientDataError,
    InvalidDateRangeError,
    InvalidTickerError,
    MarketDataError,
)


class TestTickerValidation:
    """Test ticker symbol validation."""

    def test_empty_tickers_raises_error(self):
        """Empty ticker list should raise InvalidTickerError."""
        with pytest.raises(InvalidTickerError, match="At least one ticker"):
            validate_inputs([], "2024-01-01", "2024-06-01")

    def test_too_many_tickers_raises_error(self):
        """More than 25 tickers should raise InvalidTickerError."""
        tickers = [f"T{i:02d}" for i in range(26)]
        with pytest.raises(InvalidTickerError, match="Maximum 25 tickers"):
            validate_inputs(tickers, "2024-01-01", "2024-06-01")

    def test_invalid_ticker_format_special_chars(self):
        """Tickers with invalid special characters should fail."""
        with pytest.raises(InvalidTickerError, match="AAPL!"):
            validate_inputs(["AAPL!", "MSFT"], "2024-01-01", "2024-06-01")

    def test_invalid_ticker_format_too_long(self):
        """Tickers longer than 10 characters should fail."""
        with pytest.raises(InvalidTickerError, match="VERYLONGTICKER"):
            validate_inputs(["VERYLONGTICKER"], "2024-01-01", "2024-06-01")

    def test_invalid_ticker_format_spaces(self):
        """Tickers with spaces should fail."""
        with pytest.raises(InvalidTickerError, match="AA PL"):
            validate_inputs(["AA PL"], "2024-01-01", "2024-06-01")

    def test_duplicate_tickers_raises_error(self):
        """Duplicate tickers should raise InvalidTickerError."""
        with pytest.raises(InvalidTickerError, match="Duplicate"):
            validate_inputs(["AAPL", "MSFT", "AAPL"], "2024-01-01", "2024-06-01")

    def test_valid_tickers_pass(self):
        """Valid ticker formats should pass validation."""
        # Standard tickers
        validate_inputs(["AAPL", "MSFT", "GOOGL"], "2024-01-01", "2024-06-01")
        # With dots (e.g., BRK.B)
        validate_inputs(["BRK.B", "BF.A"], "2024-01-01", "2024-06-01")
        # With hyphens
        validate_inputs(["BRK-B"], "2024-01-01", "2024-06-01")

    def test_lowercase_tickers_converted(self):
        """Lowercase tickers should be matched case-insensitively."""
        # The validation uses .upper() so lowercase should work
        validate_inputs(["aapl", "msft"], "2024-01-01", "2024-06-01")


class TestDateValidation:
    """Test date validation."""

    def test_invalid_start_date_format(self):
        """Invalid start date format should raise InvalidDateRangeError."""
        with pytest.raises(InvalidDateRangeError, match="Invalid start date"):
            validate_inputs(["AAPL"], "01-01-2024", "2024-06-01")

    def test_invalid_end_date_format(self):
        """Invalid end date format should raise InvalidDateRangeError."""
        with pytest.raises(InvalidDateRangeError, match="Invalid end date"):
            validate_inputs(["AAPL"], "2024-01-01", "06/01/2024")

    def test_start_after_end_raises_error(self):
        """Start date after end date should raise InvalidDateRangeError."""
        with pytest.raises(InvalidDateRangeError, match="must be before"):
            validate_inputs(["AAPL"], "2024-06-01", "2024-01-01")

    def test_start_equals_end_raises_error(self):
        """Start date equal to end date should raise InvalidDateRangeError."""
        with pytest.raises(InvalidDateRangeError, match="must be before"):
            validate_inputs(["AAPL"], "2024-06-01", "2024-06-01")

    def test_future_end_date_raises_error(self):
        """Future end date should raise InvalidDateRangeError."""
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        with pytest.raises(InvalidDateRangeError, match="future"):
            validate_inputs(["AAPL"], "2024-01-01", future)

    def test_date_range_too_short(self):
        """Date range shorter than 5 days should raise InvalidDateRangeError."""
        with pytest.raises(InvalidDateRangeError, match="at least 5 days"):
            validate_inputs(["AAPL"], "2024-01-01", "2024-01-03")

    def test_valid_date_range_passes(self):
        """Valid date range should pass."""
        validate_inputs(["AAPL"], "2023-01-01", "2023-12-31")


class TestFetchStockDataWithMocks:
    """Test fetch_stock_data with mocked OpenBB."""

    @patch("quantum_portfolio_optimizer.data.data_fetcher.obb")
    def test_empty_response_raises_error(self, mock_obb):
        """Empty DataFrame from OpenBB should raise MarketDataError."""
        mock_obb.equity.price.historical.return_value.to_df.return_value = pd.DataFrame()
        with pytest.raises(MarketDataError, match="No data fetched|did not return 'close'"):
            fetch_stock_data(["AAPL"], "2023-01-01", "2023-06-01", provider="tiingo")

    @patch("quantum_portfolio_optimizer.data.data_fetcher.obb")
    def test_empty_after_dropna_raises_error(self, mock_obb):
        """Empty DataFrame after dropna should raise MarketDataError."""
        # Return DataFrame with only NaN values
        mock_obb.equity.price.historical.return_value.to_df.return_value = pd.DataFrame({
            "close": [None, None, None]
        })
        with pytest.raises(MarketDataError, match="No valid data remaining"):
            fetch_stock_data(["AAPL"], "2023-01-01", "2023-06-01", provider="tiingo")

    @patch("quantum_portfolio_optimizer.data.data_fetcher.obb")
    def test_insufficient_data_points_raises_error(self, mock_obb):
        """Insufficient data points should raise InsufficientDataError."""
        dates = pd.date_range("2023-01-01", periods=5)
        tickers = [f"T{i}" for i in range(10)]
        
        def mock_hist(*args, **kwargs):
            m = MagicMock()
            m.to_df.return_value = pd.DataFrame({"close": [100.0] * 5}, index=dates)
            return m
        mock_obb.equity.price.historical.side_effect = mock_hist

        with pytest.raises(InsufficientDataError, match="Insufficient data"):
            fetch_stock_data(tickers, "2023-01-01", "2023-06-01", provider="tiingo")

    @patch("quantum_portfolio_optimizer.data.data_fetcher.obb")
    def test_successful_fetch_returns_dataframe(self, mock_obb):
        """Successful fetch should return proper DataFrame."""
        dates = pd.date_range("2023-01-01", periods=30)
        
        def mock_hist(*args, **kwargs):
            m = MagicMock()
            m.to_df.return_value = pd.DataFrame({"close": [100.0] * 30}, index=dates)
            return m
        mock_obb.equity.price.historical.side_effect = mock_hist

        result = fetch_stock_data(["AAPL", "MSFT", "GOOGL"], "2023-01-01", "2023-06-01", provider="tiingo")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30
        assert list(result.columns) == ["AAPL", "MSFT", "GOOGL"]
        assert not result.empty

    @patch("quantum_portfolio_optimizer.data.data_fetcher.obb")
    def test_validation_called_before_fetch(self, mock_obb):
        """Validation should catch errors before API call."""
        with pytest.raises(InvalidTickerError, match="At least one ticker"):
            fetch_stock_data([], "2023-01-01", "2023-06-01", provider="tiingo")

        mock_obb.equity.price.historical.assert_not_called()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_25_tickers_passes(self):
        """Exactly 25 tickers should pass validation."""
        tickers = [f"T{i:02d}" for i in range(25)]
        validate_inputs(tickers, "2023-01-01", "2023-12-31")

    def test_exactly_5_day_range_passes(self):
        """Exactly 5 day range should pass validation."""
        validate_inputs(["AAPL"], "2023-01-01", "2023-01-06")

    def test_single_ticker_passes(self):
        """Single ticker should pass validation."""
        validate_inputs(["AAPL"], "2023-01-01", "2023-12-31")

    def test_numeric_ticker_passes(self):
        """Numeric ticker (e.g., 0001.HK) should pass validation."""
        validate_inputs(["0001.HK"], "2023-01-01", "2023-12-31")


class TestExceptionAttributes:
    """Test that exceptions carry proper attributes."""

    def test_invalid_ticker_error_has_ticker(self):
        """InvalidTickerError should have ticker attribute."""
        try:
            validate_inputs(["INVALID!"], "2024-01-01", "2024-06-01")
        except InvalidTickerError as e:
            assert e.ticker == "INVALID!"
            assert "ticker" in e.details

    def test_invalid_date_range_error_has_dates(self):
        """InvalidDateRangeError should have date attributes."""
        try:
            validate_inputs(["AAPL"], "2024-06-01", "2024-01-01")
        except InvalidDateRangeError as e:
            assert e.start_date == "2024-06-01"
            assert e.end_date == "2024-01-01"
            assert "start_date" in e.details
            assert "end_date" in e.details

    @patch("quantum_portfolio_optimizer.data.data_fetcher.obb")
    def test_insufficient_data_error_has_details(self, mock_obb):
        """InsufficientDataError should have data point details."""
        dates = pd.date_range("2023-01-01", periods=5)
        tickers = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
        mock_obb.equity.price.historical.return_value.to_df.return_value = pd.DataFrame(
            {"close": [100.0] * 5},
            index=dates
        )

        try:
            fetch_stock_data(tickers, "2023-01-01", "2023-06-01")
        except InsufficientDataError as e:
            assert e.actual == 5
            assert e.required >= 10
            assert len(e.tickers) == 10

    def test_exception_to_dict(self):
        """Exceptions should convert to dict for API responses."""
        try:
            validate_inputs(["BAD!"], "2024-01-01", "2024-06-01")
        except InvalidTickerError as e:
            error_dict = e.to_dict()
            assert "error_type" in error_dict
            assert error_dict["error_type"] == "InvalidTickerError"
            assert "message" in error_dict
            assert "details" in error_dict
