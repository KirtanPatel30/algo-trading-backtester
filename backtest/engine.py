"""
backtest/engine.py
Core backtesting engine — runs all strategies, computes risk metrics,
saves results to data/processed/
"""

import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Risk Metrics ──────────────────────────────────────────────────────────────

def sharpe_ratio(returns, risk_free=0.05/252):
    excess = returns - risk_free
    return np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)

def sortino_ratio(returns, risk_free=0.05/252):
    excess    = returns - risk_free
    downside  = excess[excess < 0].std()
    return np.sqrt(252) * excess.mean() / (downside + 1e-9)

def max_drawdown(equity):
    peak = equity.expanding().max()
    dd   = (equity - peak) / (peak + 1e-9)
    return dd.min()

def calmar_ratio(returns, equity):
    ann_return = (1 + returns.mean()) ** 252 - 1
    mdd        = abs(max_drawdown(equity))
    return ann_return / (mdd + 1e-9)

def win_rate(strategy_returns):
    wins = (strategy_returns > 0).sum()
    total= (strategy_returns != 0).sum()
    return wins / (total + 1e-9)

def compute_metrics(df, name, ticker):
    r  = df["strategy_return"].dropna()
    eq = df["equity"].dropna()
    ann_return = (1 + r.mean()) ** 252 - 1
    ann_vol    = r.std() * np.sqrt(252)
    return {
        "strategy":       name,
        "ticker":         ticker,
        "ann_return":     round(ann_return * 100, 2),
        "ann_volatility": round(ann_vol * 100, 2),
        "sharpe":         round(sharpe_ratio(r), 3),
        "sortino":        round(sortino_ratio(r), 3),
        "max_drawdown":   round(max_drawdown(eq) * 100, 2),
        "calmar":         round(calmar_ratio(r, eq), 3),
        "win_rate":       round(win_rate(r) * 100, 2),
        "total_trades":   int((df["signal"].diff() != 0).sum()),
        "final_equity":   round(float(eq.iloc[-1]), 2),
        "bh_return":      round(float(df["cum_bh"].iloc[-1] * 100), 2),
    }


# ── Run All Strategies ────────────────────────────────────────────────────────

def run_all():
    print("=" * 60)
    print("ALGO TRADING BACKTESTER — ENGINE")
    print("=" * 60)

    prices_path = RAW_DIR / "prices.csv"
    if not prices_path.exists():
        print("No data found. Fetching...")
        from data.fetch import fetch_all
        fetch_all()

    prices = pd.read_csv(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    from strategies.momentum       import MomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.pairs_trading  import PairsTradingStrategy
    from strategies.ml_signal      import MLSignalStrategy

    all_metrics   = []
    all_equity    = []
    all_trades    = []

    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY"]

    # ── Momentum ──
    print("\n[1/4] Running Momentum Strategy...")
    strat = MomentumStrategy(fast=20, slow=50)
    for t in tickers:
        df = prices[prices["ticker"] == t].copy()
        if len(df) < 60: continue
        result = strat.generate_signals(df)
        m = compute_metrics(result, "Momentum", t)
        all_metrics.append(m)
        result["strategy"] = "Momentum"
        all_equity.append(result[["date","ticker","strategy","equity","cum_return","cum_bh","signal","close"]])
        trades = result[result["signal"].diff() != 0][["date","ticker","signal","close"]]
        trades["strategy"] = "Momentum"
        all_trades.append(trades)
        print(f"  {t}: Sharpe={m['sharpe']:.2f} | Ann.Return={m['ann_return']:.1f}% | MaxDD={m['max_drawdown']:.1f}%")

    # ── Mean Reversion ──
    print("\n[2/4] Running Mean Reversion Strategy...")
    strat = MeanReversionStrategy(window=20, num_std=2.0)
    for t in tickers:
        df = prices[prices["ticker"] == t].copy()
        if len(df) < 60: continue
        result = strat.generate_signals(df)
        m = compute_metrics(result, "Mean Reversion", t)
        all_metrics.append(m)
        result["strategy"] = "Mean Reversion"
        all_equity.append(result[["date","ticker","strategy","equity","cum_return","cum_bh","signal","close"]])
        trades = result[result["signal"].diff() != 0][["date","ticker","signal","close"]]
        trades["strategy"] = "Mean Reversion"
        all_trades.append(trades)
        print(f"  {t}: Sharpe={m['sharpe']:.2f} | Ann.Return={m['ann_return']:.1f}% | MaxDD={m['max_drawdown']:.1f}%")

    # ── Pairs Trading ──
    print("\n[3/4] Running Pairs Trading Strategy...")
    pairs = [("AAPL","MSFT"), ("GOOGL","META"), ("JPM","GS")]
    strat = PairsTradingStrategy()
    for a, b in pairs:
        df_a = prices[prices["ticker"] == a].copy()
        df_b = prices[prices["ticker"] == b].copy()
        if len(df_a) < 60 or len(df_b) < 60: continue
        result = strat.generate_signals(df_a, df_b)
        result["ticker"] = f"{a}/{b}"
        m = compute_metrics(result, "Pairs Trading", f"{a}/{b}")
        all_metrics.append(m)
        result["strategy"] = "Pairs Trading"
        all_equity.append(result[["date","ticker","strategy","equity","cum_return","cum_bh","signal","close"]])
        print(f"  {a}/{b}: Sharpe={m['sharpe']:.2f} | Ann.Return={m['ann_return']:.1f}% | MaxDD={m['max_drawdown']:.1f}%")

    # ── ML Signal ──
    print("\n[4/4] Running ML Signal Strategy...")
    strat = MLSignalStrategy()
    for t in ["AAPL","MSFT","NVDA","SPY"]:
        df = prices[prices["ticker"] == t].copy()
        if len(df) < 100: continue
        strat.fitted = False
        result = strat.generate_signals(df)
        m = compute_metrics(result, "ML Signal", t)
        all_metrics.append(m)
        result["strategy"] = "ML Signal"
        all_equity.append(result[["date","ticker","strategy","equity","cum_return","cum_bh","signal","close"]])
        trades = result[result["signal"].diff() != 0][["date","ticker","signal","close"]]
        trades["strategy"] = "ML Signal"
        all_trades.append(trades)
        print(f"  {t}: Sharpe={m['sharpe']:.2f} | Ann.Return={m['ann_return']:.1f}% | MaxDD={m['max_drawdown']:.1f}%")

    # ── Save results ──
    metrics_df = pd.DataFrame(all_metrics)
    equity_df  = pd.concat(all_equity,  ignore_index=True)
    trades_df  = pd.concat(all_trades,  ignore_index=True) if all_trades else pd.DataFrame()

    metrics_df.to_csv(PROCESSED_DIR / "metrics.csv",  index=False)
    equity_df.to_csv( PROCESSED_DIR / "equity.csv",   index=False)
    trades_df.to_csv( PROCESSED_DIR / "trades.csv",   index=False)

    print(f"\n[ENGINE] Saved results to data/processed/")
    print(f"[ENGINE] Best Sharpe: {metrics_df.loc[metrics_df['sharpe'].idxmax(), 'strategy']} "
          f"on {metrics_df.loc[metrics_df['sharpe'].idxmax(), 'ticker']} "
          f"({metrics_df['sharpe'].max():.3f})")
    return metrics_df


if __name__ == "__main__":
    run_all()
