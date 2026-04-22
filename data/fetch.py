"""
data/fetch.py
Fetches real OHLCV data via yfinance. Falls back to synthetic GBM data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM", "GS", "SPY"]
PAIRS   = [("AAPL", "MSFT"), ("GOOGL", "META"), ("JPM", "GS")]

BASE_PRICES = {
    "AAPL": 185.0, "MSFT": 415.0, "GOOGL": 165.0, "TSLA": 245.0,
    "NVDA": 620.0, "AMZN": 178.0, "META": 490.0,
    "JPM": 198.0,  "GS": 385.0,   "SPY": 480.0,
}


def fetch_real(ticker, days=365):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)
        if df is None or len(df) < 50:
            return None
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = df[["date","open","high","low","close","volume"]].copy()
        df["ticker"] = ticker
        return df
    except Exception:
        return None


def generate_synthetic(ticker, days=365):
    np.random.seed(hash(ticker) % 2**31)
    base   = BASE_PRICES.get(ticker, 100.0)
    dates  = pd.date_range(end=datetime.now(), periods=days, freq="B")
    mu, sigma = 0.0003, 0.018
    returns   = np.random.normal(mu, sigma, days)
    closes    = base * np.exp(np.cumsum(returns))
    opens     = closes * np.random.uniform(0.995, 1.005, days)
    highs     = np.maximum(opens, closes) * np.random.uniform(1.001, 1.015, days)
    lows      = np.minimum(opens, closes) * np.random.uniform(0.985, 0.999, days)
    vols      = np.random.randint(15_000_000, 80_000_000, days)
    return pd.DataFrame({
        "date": dates, "open": opens.round(2), "high": highs.round(2),
        "low": lows.round(2), "close": closes.round(2),
        "volume": vols, "ticker": ticker,
    })


def fetch_all():
    print("[DATA] Fetching market data...")
    frames = []
    for t in TICKERS:
        df = fetch_real(t, 365)
        if df is None or len(df) < 50:
            print(f"  {t}: synthetic")
            df = generate_synthetic(t, 365)
        else:
            print(f"  {t}: {len(df)} real rows")
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    out = RAW_DIR / "prices.csv"
    result.to_csv(out, index=False)
    print(f"[DATA] Saved {len(result):,} rows to {out}")
    return result


if __name__ == "__main__":
    fetch_all()
