# quantum_portfolio_optimizer/src/quantum_portfolio_optimizer/data/data_fetcher.py
"""Data fetcher module for retrieving stock market data."""

import re
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from quantum_portfolio_optimizer.exceptions import (
    DataError,
    InsufficientDataError,
    InvalidDateRangeError,
    InvalidTickerError,
    MarketDataError,
)


def validate_inputs(tickers: List[str], start_date: str, end_date: str) -> None:
    """Validate inputs before making API calls.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Raises:
        InvalidTickerError: If ticker validation fails
        InvalidDateRangeError: If date validation fails
    """
    # Ticker validation
    if not tickers:
        raise InvalidTickerError("", "At least one ticker is required")

    if len(tickers) > 25:
        raise InvalidTickerError(
            ", ".join(tickers[:3]) + "...",
            f"Maximum 25 tickers supported, got {len(tickers)}"
        )

    # Ticker format: 1-10 uppercase letters, numbers, dots, or hyphens
    ticker_pattern = re.compile(r'^[A-Z0-9.\-]{1,10}$')
    for ticker in tickers:
        if not ticker_pattern.match(ticker.upper()):
            raise InvalidTickerError(
                ticker,
                "Use 1-10 uppercase letters, numbers, dots, or hyphens (e.g., AAPL, BRK.B)"
            )

    # Check for duplicates
    seen = set()
    for ticker in tickers:
        upper = ticker.upper()
        if upper in seen:
            raise InvalidTickerError(
                ticker,
                "Duplicate tickers detected. Each ticker should appear only once."
            )
        seen.add(upper)

    # Date validation
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
    except ValueError:
        raise InvalidDateRangeError(
            start_date, end_date,
            f"Invalid start date format. Use YYYY-MM-DD format."
        )

    try:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise InvalidDateRangeError(
            start_date, end_date,
            f"Invalid end date format. Use YYYY-MM-DD format."
        )

    if start >= end:
        raise InvalidDateRangeError(
            start_date, end_date,
            "Start date must be before end date"
        )

    if end > datetime.now():
        raise InvalidDateRangeError(
            start_date, end_date,
            "End date cannot be in the future"
        )

    # Minimum date range for meaningful data
    if (end - start).days < 5:
        raise InvalidDateRangeError(
            start_date, end_date,
            "Date range must be at least 5 days for meaningful analysis"
        )


def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        tickers (List[str]): A list of stock tickers.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the adjusted close prices of the stocks,
                      with dates as the index and tickers as columns.

    Raises:
        InvalidTickerError: If ticker validation fails
        InvalidDateRangeError: If date validation fails
        MarketDataError: If data fetch from yfinance fails
        InsufficientDataError: If not enough data points are returned
    """
    # Validate inputs before making API call
    validate_inputs(tickers, start_date, end_date)

    try:
        # Set auto_adjust=False to get a consistent multi-level column output
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    except Exception as e:
        raise MarketDataError(
            f"Failed to fetch data: {str(e)}"
        )

    if data.empty:
        raise MarketDataError(
            "No data fetched. Check that ticker symbols are valid "
            "and the date range contains trading days."
        )

    # Select Adjusted Close prices and drop rows with any missing values
    adj_close = data['Adj Close'].dropna()

    if adj_close.empty:
        raise MarketDataError(
            "No valid data remaining after removing missing values."
        )

    # Check minimum data points for reliable covariance estimation
    min_points = max(10, len(tickers) + 1)
    if len(adj_close) < min_points:
        raise InsufficientDataError(
            required=min_points,
            actual=len(adj_close),
            tickers=tickers,
        )

    return adj_close
