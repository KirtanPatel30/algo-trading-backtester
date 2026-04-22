"""tests/test_strategies.py — Unit tests for backtesting engine."""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_price_df(n=200, seed=42):
    np.random.seed(seed)
    dates   = pd.date_range("2023-01-01", periods=n, freq="B")
    closes  = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, n)))
    opens   = closes * np.random.uniform(0.998, 1.002, n)
    highs   = np.maximum(opens, closes) * np.random.uniform(1.001, 1.01, n)
    lows    = np.minimum(opens, closes) * np.random.uniform(0.99, 0.999, n)
    vols    = np.random.randint(1e7, 5e7, n)
    return pd.DataFrame({"date":dates,"open":opens,"high":highs,"low":lows,
                          "close":closes,"volume":vols,"ticker":"TEST"})


class TestMomentum:
    def test_signals_valid(self):
        from strategies.momentum import MomentumStrategy
        df = make_price_df()
        result = MomentumStrategy().generate_signals(df)
        assert "signal" in result.columns
        assert result["signal"].isin([-1, 0, 1]).all()

    def test_equity_starts_at_10k(self):
        from strategies.momentum import MomentumStrategy
        df = make_price_df()
        result = MomentumStrategy().generate_signals(df)
        assert abs(result["equity"].iloc[0] - 10000) < 100

    def test_returns_computed(self):
        from strategies.momentum import MomentumStrategy
        df = make_price_df()
        result = MomentumStrategy().generate_signals(df)
        assert "strategy_return" in result.columns
        assert "cum_return" in result.columns


class TestMeanReversion:
    def test_bollinger_bands(self):
        from strategies.mean_reversion import MeanReversionStrategy
        df = make_price_df()
        result = MeanReversionStrategy().generate_signals(df)
        assert "upper" in result.columns
        assert "lower" in result.columns
        assert (result["upper"] >= result["lower"]).all()

    def test_signals_valid(self):
        from strategies.mean_reversion import MeanReversionStrategy
        df = make_price_df()
        result = MeanReversionStrategy().generate_signals(df)
        assert result["signal"].isin([-1, 0, 1]).all()


class TestPairsTrading:
    def test_spread_computed(self):
        from strategies.pairs_trading import PairsTradingStrategy
        df_a = make_price_df(seed=1)
        df_b = make_price_df(seed=2)
        result = PairsTradingStrategy().generate_signals(df_a, df_b)
        assert "z_score" in result.columns
        assert "spread" in result.columns

    def test_signals_valid(self):
        from strategies.pairs_trading import PairsTradingStrategy
        df_a = make_price_df(seed=1)
        df_b = make_price_df(seed=2)
        result = PairsTradingStrategy().generate_signals(df_a, df_b)
        assert result["signal"].isin([-1, 0, 1]).all()


class TestMetrics:
    def test_sharpe_positive_trend(self):
        from backtest.engine import sharpe_ratio
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe  = sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_max_drawdown_negative(self):
        from backtest.engine import max_drawdown
        equity = pd.Series([10000, 11000, 9000, 9500, 10500])
        mdd    = max_drawdown(equity)
        assert mdd < 0

    def test_win_rate_range(self):
        from backtest.engine import win_rate
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        wr      = win_rate(returns)
        assert 0 <= wr <= 1


class TestMLStrategy:
    def test_feature_engineering(self):
        from strategies.ml_signal import MLSignalStrategy
        df    = make_price_df(n=300)
        strat = MLSignalStrategy()
        feat  = strat._engineer_features(df)
        assert "rsi" in feat.columns
        assert "return_1d" in feat.columns
        assert "target" in feat.columns

    def test_signals_after_fit(self):
        from strategies.ml_signal import MLSignalStrategy
        df    = make_price_df(n=300)
        strat = MLSignalStrategy()
        result= strat.generate_signals(df)
        assert "signal" in result.columns
        assert result["signal"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
