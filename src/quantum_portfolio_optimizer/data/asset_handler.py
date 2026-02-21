"""Utilities for loading and validating asset price data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class AssetDataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    volatility: pd.Series


class AssetDataLoader:
    """Load asset price histories from CSV or JSON sources."""

    def __init__(self, forward_fill: bool = True) -> None:
        self.forward_fill = forward_fill

    def load(self, path: str | Path, date_column: Optional[str] = None) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".json", ".jsonl"}:
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file type '{suffix}' for asset data.")

        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)

        return self._preprocess(df)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()
        if self.forward_fill:
            df = df.ffill()
        if df.isna().any().any():
            raise ValueError("Asset data contains NaN values after forward-fill.")
        if (df <= 0).any().any():
            raise ValueError("Asset prices must be strictly positive.")
        return df

    def compute_returns(
        self,
        prices: pd.DataFrame,
        method: Literal["log", "pct"] = "log",
    ) -> pd.DataFrame:
        if method == "log":
            returns = np.log(prices / prices.shift(1)).dropna(how="all")
        elif method == "pct":
            returns = prices.pct_change().dropna(how="all")
        else:
            raise ValueError(f"Unsupported returns method '{method}'.")
        return returns

    def compute_volatility(
        self, returns: pd.DataFrame, window: Optional[int] = None
    ) -> pd.Series:
        if window:
            vol = returns.rolling(window=window).std().dropna(how="all").iloc[-1]
        else:
            vol = returns.std()
        return vol

    def prepare_dataset(
        self,
        prices: pd.DataFrame,
        returns_method: Literal["log", "pct"] = "log",
        volatility_window: Optional[int] = None,
    ) -> AssetDataset:
        returns = self.compute_returns(prices, method=returns_method)
        volatility = self.compute_volatility(returns, window=volatility_window)
        return AssetDataset(prices=prices, returns=returns, volatility=volatility)
