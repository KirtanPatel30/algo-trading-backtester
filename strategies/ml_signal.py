"""
strategies/ml_signal.py
ML-based strategy: XGBoost classifier predicts next-day direction.
Features: technical indicators + lagged returns.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from strategies.base import BaseStrategy

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class MLSignalStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("ML Signal")
        self.model  = None
        self.fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["return_1d"]  = df["close"].pct_change()
        df["return_3d"]  = df["close"].pct_change(3)
        df["return_5d"]  = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)
        df["ma_5"]       = df["close"].rolling(5,  min_periods=1).mean()
        df["ma_20"]      = df["close"].rolling(20, min_periods=1).mean()
        df["ma_50"]      = df["close"].rolling(50, min_periods=1).mean()
        df["vol_10"]     = df["return_1d"].rolling(10, min_periods=1).std()
        df["vol_20"]     = df["return_1d"].rolling(20, min_periods=1).std()
        df["price_vs_ma5"]  = df["close"] / (df["ma_5"]  + 1e-9) - 1
        df["price_vs_ma20"] = df["close"] / (df["ma_20"] + 1e-9) - 1
        # RSI
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
        # Volume features
        df["vol_change"] = df["volume"].pct_change()
        df["vol_ma20"]   = df["volume"].rolling(20, min_periods=1).mean()
        df["vol_ratio"]  = df["volume"] / (df["vol_ma20"] + 1e-9)
        df["target"]     = (df["close"].shift(-1) > df["close"]).astype(int)
        return df

    FEATURE_COLS = [
        "return_1d","return_3d","return_5d","return_10d",
        "price_vs_ma5","price_vs_ma20","vol_10","vol_20",
        "rsi","vol_change","vol_ratio",
    ]

    def fit(self, df: pd.DataFrame):
        df = self._engineer_features(df).dropna()
        X  = df[self.FEATURE_COLS].replace([np.inf,-np.inf], 0).fillna(0)
        y  = df["target"]
        tscv = TimeSeriesSplit(n_splits=5)
        self.model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        self.model.fit(X, y)
        self.fitted = True
        joblib.dump(self.model, MODEL_DIR / "ml_signal_model.pkl")
        preds = self.model.predict(X)
        print(f"  [ML] Train accuracy: {accuracy_score(y, preds):.4f}")
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            self.fit(df)
        df = self._engineer_features(df.copy())
        X  = df[self.FEATURE_COLS].replace([np.inf,-np.inf], 0).fillna(0)
        df["signal"] = self.model.predict(X) * 2 - 1  # map 0/1 → -1/+1
        df["signal"] = df["signal"].shift(1).fillna(0)
        return self.compute_returns(df)
