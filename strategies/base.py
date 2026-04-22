"""strategies/base.py — Base strategy class."""
import pandas as pd
import numpy as np


class BaseStrategy:
    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["daily_return"]   = df["close"].pct_change()
        df["strategy_return"]= df["signal"].shift(1) * df["daily_return"]
        df["cum_return"]     = (1 + df["strategy_return"]).cumprod() - 1
        df["cum_bh"]         = (1 + df["daily_return"]).cumprod() - 1
        df["equity"]         = 10000 * (1 + df["cum_return"])
        return df
