# 📈 Algorithmic Trading Backtesting Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?style=flat-square&logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?style=flat-square&logo=fastapi)

> Full backtesting framework implementing 4 trading strategies on real market data — Sharpe ratio, max drawdown, win rate analytics, and live paper trading simulation.

## Strategies
- **Momentum** — Buy strength, ride the trend
- **Mean Reversion** — Bollinger Bands reversion to mean
- **Pairs Trading** — Statistical arbitrage on correlated assets
- **ML Signal** — XGBoost-based price movement predictor

## Quick Start
```bash
pip install -r requirements.txt
python run_all.py
streamlit run dashboard/app.py
```

## Resume Bullets
- Built algorithmic trading backtesting engine implementing 4 strategies (momentum, mean reversion, pairs trading, ML signal) on real OHLCV data
- Engineered risk metrics including Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, and win rate across all strategies
- Implemented walk-forward optimization to prevent overfitting; compared strategies against buy-and-hold benchmark
- Served backtest results via FastAPI REST API; visualized P&L curves, drawdown charts, and trade logs in Streamlit dashboard
