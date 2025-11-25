"""Calculate logarithmic returns as specified in the research paper."""

import warnings
import numpy as np
import pandas as pd
from typing import Union, Optional


def calculate_logarithmic_returns(
    prices: Union[np.ndarray, pd.DataFrame],
    time_window: int = 30
) -> np.ndarray:
    """
    Calculate logarithmic returns: μt,a = log(Pt+1,a/Pt,a)

    Parameters:
        prices: Asset price data (time x assets)
        time_window: Deprecated - this parameter has no effect.

    Returns:
        Logarithmic returns array
    """
    if time_window != 30:
        warnings.warn(
            "time_window parameter has no effect and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2
        )

    if isinstance(prices, pd.DataFrame):
        prices = prices.values

    # Ensure prices are positive
    if np.any(prices <= 0):
        raise ValueError("All prices must be positive for logarithmic returns")

    # Calculate log returns
    log_returns = np.log(prices[1:] / prices[:-1])

    return log_returns


def calculate_rolling_covariance(
    returns: Union[np.ndarray, pd.DataFrame],
    window: int = 30
) -> np.ndarray:
    """
    Calculate rolling covariance matrix over specified window.

    Parameters:
        returns: Returns data (time x assets)
        window: Rolling window size (default 30 days as per paper)

    Returns:
        Covariance matrix
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.values

    if len(returns) < window:
        # Use all available data if less than window
        return np.cov(returns.T)

    # Use last 'window' periods for covariance
    return np.cov(returns[-window:].T)