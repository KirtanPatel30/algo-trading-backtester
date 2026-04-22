"""
strategies/momentum.py
Momentum strategy: go long when short MA > long MA (golden cross),
short when short MA < long MA (death cross).
"""
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    def __init__(self, fast=20, slow=50):
        super().__init__("Momentum")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("date").reset_index(drop=True)
        df["ma_fast"] = df["close"].rolling(self.fast, min_periods=1).mean()
        df["ma_slow"] = df["close"].rolling(self.slow, min_periods=1).mean()
        df["rsi"]     = self._rsi(df["close"])
        df["signal"]  = 0
        # Long when fast > slow AND RSI not overbought
        df.loc[(df["ma_fast"] > df["ma_slow"]) & (df["rsi"] < 70), "signal"] = 1
        # Short when fast < slow AND RSI not oversold
        df.loc[(df["ma_fast"] < df["ma_slow"]) & (df["rsi"] > 30), "signal"] = -1
        return self.compute_returns(df)

    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
