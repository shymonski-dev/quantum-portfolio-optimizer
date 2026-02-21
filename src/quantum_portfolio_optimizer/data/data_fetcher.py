# quantum_portfolio_optimizer/src/quantum_portfolio_optimizer/data/data_fetcher.py
"""Data fetcher module for retrieving stock market data."""

import re
from datetime import datetime
from typing import List

import pandas as pd
from openbb import obb

from quantum_portfolio_optimizer.exceptions import (
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
            f"Maximum 25 tickers supported, got {len(tickers)}",
        )

    # Ticker format: 1-10 uppercase letters, numbers, dots, or hyphens
    ticker_pattern = re.compile(r"^[A-Z0-9.\-]{1,10}$")
    for ticker in tickers:
        if not ticker_pattern.match(ticker.upper()):
            raise InvalidTickerError(
                ticker,
                "Use 1-10 uppercase letters, numbers, dots, or hyphens (e.g., AAPL, BRK.B)",
            )

    # Check for duplicates
    seen = set()
    for ticker in tickers:
        upper = ticker.upper()
        if upper in seen:
            raise InvalidTickerError(
                ticker,
                "Duplicate tickers detected. Each ticker should appear only once.",
            )
        seen.add(upper)

    # Date validation
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise InvalidDateRangeError(
            start_date, end_date, "Invalid start date format. Use YYYY-MM-DD format."
        )

    try:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise InvalidDateRangeError(
            start_date, end_date, "Invalid end date format. Use YYYY-MM-DD format."
        )

    if start >= end:
        raise InvalidDateRangeError(
            start_date, end_date, "Start date must be before end date"
        )

    if end > datetime.now():
        raise InvalidDateRangeError(
            start_date, end_date, "End date cannot be in the future"
        )

    # Minimum date range for meaningful data
    if (end - start).days < 5:
        raise InvalidDateRangeError(
            start_date,
            end_date,
            "Date range must be at least 5 days for meaningful analysis",
        )


def fetch_stock_data(
    tickers: List[str], start_date: str, end_date: str, provider: str = "tiingo"
) -> pd.DataFrame:
    """
    Fetches historical stock data using OpenBB SDK.

    Args:
        tickers (List[str]): A list of stock tickers.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        provider (str): OpenBB data provider (default: 'tiingo').

    Returns:
        pd.DataFrame: A DataFrame containing the close prices of the stocks,
                      with dates as the index and tickers as columns.

    Raises:
        InvalidTickerError: If ticker validation fails
        InvalidDateRangeError: If date validation fails
        MarketDataError: If data fetch fails
        InsufficientDataError: If not enough data points are returned
    """
    # Validate inputs before making API call
    validate_inputs(tickers, start_date, end_date)

    try:
        # OpenBB Platform SDK call
        # We use a loop for multiple tickers to ensure per-ticker error handling
        # and consistent DataFrame construction across different providers.
        all_data = []
        for ticker in tickers:
            res = obb.equity.price.historical(
                symbol=ticker,
                start_date=start_date,
                end_date=end_date,
                provider=provider,
            ).to_df()

            # Select 'close' and rename to ticker
            if "close" in res.columns:
                series = res["close"]
                series.name = ticker
                all_data.append(series)
            else:
                raise MarketDataError(
                    f"Provider {provider} did not return 'close' price for {ticker}"
                )

        df_close = pd.concat(all_data, axis=1)

    except Exception as e:
        raise MarketDataError(
            f"Failed to fetch data from {provider}: {str(e)}", provider=provider
        )

    if df_close.empty:
        raise MarketDataError(
            f"No data fetched from {provider}. Check that ticker symbols are valid "
            "and the date range contains trading days."
        )

    # Cleanup missing values
    df_close = df_close.dropna()

    if df_close.empty:
        raise MarketDataError(
            "No valid data remaining after removing missing values (possibly due to date mismatch)."
        )

    # Check minimum data points for reliable covariance estimation
    min_points = max(10, len(tickers) + 1)
    if len(df_close) < min_points:
        raise InsufficientDataError(
            required=min_points,
            actual=len(df_close),
            tickers=tickers,
        )

    return df_close
