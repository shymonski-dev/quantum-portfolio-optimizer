# quantum_portfolio_optimizer/src/quantum_portfolio_optimizer/data/data_fetcher.py
import yfinance as yf
import pandas as pd
from typing import List

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
        ValueError: If no data can be fetched for the given tickers and dates.
    """
    # Set auto_adjust=False to get a consistent multi-level column output
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    if data.empty:
        raise ValueError("No data fetched. Check tickers and date range.")
    
    # Select Adjusted Close prices and drop rows with any missing values
    adj_close = data['Adj Close'].dropna()
    
    if adj_close.empty:
        raise ValueError("No valid data remaining after removing missing values.")

    return adj_close
