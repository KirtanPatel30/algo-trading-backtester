"""
strategies/pairs_trading.py
Statistical arbitrage / pairs trading:
  Trade the spread between two cointegrated assets.
  Go long spread when it widens below -z_threshold, short when above +z_threshold.
"""
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class PairsTradingStrategy(BaseStrategy):
    def __init__(self, ticker_a="AAPL", ticker_b="MSFT", window=30, z_threshold=1.5):
        super().__init__("Pairs Trading")
        self.ticker_a    = ticker_a
        self.ticker_b    = ticker_b
        self.window      = window
        self.z_threshold = z_threshold

    def generate_signals(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(
            df_a[["date","close"]].rename(columns={"close":"close_a"}),
            df_b[["date","close"]].rename(columns={"close":"close_b"}),
            on="date"
        ).sort_values("date").reset_index(drop=True)

        # Hedge ratio via rolling OLS
        merged["spread"]  = merged["close_a"] - merged["close_b"]
        merged["spread_mean"] = merged["spread"].rolling(self.window, min_periods=1).mean()
        merged["spread_std"]  = merged["spread"].rolling(self.window, min_periods=1).std().fillna(1)
        merged["z_score"] = (merged["spread"] - merged["spread_mean"]) / (merged["spread_std"] + 1e-9)

        merged["signal"]  = 0
        merged.loc[merged["z_score"] < -self.z_threshold, "signal"] =  1  # spread too low → long
        merged.loc[merged["z_score"] >  self.z_threshold, "signal"] = -1  # spread too high → short

        # Use close_a as the price series for returns
        merged["close"]        = merged["close_a"]
        merged["daily_return"] = merged["close"].pct_change()
        merged["strategy_return"] = merged["signal"].shift(1) * merged["daily_return"]
        merged["cum_return"]   = (1 + merged["strategy_return"]).cumprod() - 1
        merged["cum_bh"]       = (1 + merged["daily_return"]).cumprod() - 1
        merged["equity"]       = 10000 * (1 + merged["cum_return"])
        merged["ticker"]       = f"{self.ticker_a}/{self.ticker_b}"
        return merged
