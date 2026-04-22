"""api/main.py — FastAPI REST API for backtesting results."""
import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

app = FastAPI(title="Algo Trading Backtester API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load(name):
    p = PROCESSED_DIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def get_metrics(strategy: Optional[str] = None, ticker: Optional[str] = None):
    df = load("metrics")
    if df is None: raise HTTPException(404, "Run backtest first.")
    if strategy: df = df[df["strategy"] == strategy]
    if ticker:   df = df[df["ticker"] == ticker]
    return df.to_dict(orient="records")


@app.get("/metrics/best")
def best_strategies():
    df = load("metrics")
    if df is None: raise HTTPException(404, "Run backtest first.")
    return {
        "best_sharpe":    df.loc[df["sharpe"].idxmax()].to_dict(),
        "best_return":    df.loc[df["ann_return"].idxmax()].to_dict(),
        "lowest_drawdown":df.loc[df["max_drawdown"].idxmax()].to_dict(),
    }


@app.get("/equity/{strategy}/{ticker}")
def get_equity(strategy: str, ticker: str):
    df = load("equity")
    if df is None: raise HTTPException(404, "Run backtest first.")
    filtered = df[(df["strategy"]==strategy) & (df["ticker"]==ticker)]
    if filtered.empty: raise HTTPException(404, f"No data for {strategy}/{ticker}")
    return filtered[["date","equity","cum_return","cum_bh","close"]].tail(252).to_dict(orient="records")


@app.get("/trades")
def get_trades(strategy: Optional[str] = None, ticker: Optional[str] = None):
    df = load("trades")
    if df is None: raise HTTPException(404, "Run backtest first.")
    if strategy: df = df[df["strategy"] == strategy]
    if ticker:   df = df[df["ticker"]   == ticker]
    return df.tail(100).to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
