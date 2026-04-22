"""
strategies/mean_reversion.py
Mean reversion using Bollinger Bands:
  Buy when price touches lower band, sell when touches upper band.
"""
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, window=20, num_std=2.0):
        super().__init__("Mean Reversion")
        self.window  = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("date").reset_index(drop=True)
        df["ma"]     = df["close"].rolling(self.window, min_periods=1).mean()
        df["std"]    = df["close"].rolling(self.window, min_periods=1).std().fillna(0)
        df["upper"]  = df["ma"] + self.num_std * df["std"]
        df["lower"]  = df["ma"] - self.num_std * df["std"]
        df["z_score"]= (df["close"] - df["ma"]) / (df["std"] + 1e-9)
        df["signal"] = 0
        df.loc[df["close"] < df["lower"], "signal"] =  1   # oversold → buy
        df.loc[df["close"] > df["upper"], "signal"] = -1   # overbought → sell
        # Hold signal between crosses
        df["signal"] = df["signal"].replace(0, method="ffill").fillna(0)
        return self.compute_returns(df)
