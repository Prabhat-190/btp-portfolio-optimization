"""Masuda (2024) §3.4: standardise OHLCV, then PCA on the training window only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_pipeline import MarketData


def ohlcv_feature_frame(market: MarketData, ticker: str) -> pd.DataFrame:
    close = market.prices[ticker]
    ohlcv = market.ohlcv or {}
    open_ = ohlcv.get("Open", pd.DataFrame()).get(ticker, close.shift(1))
    high = ohlcv.get("High", pd.DataFrame()).get(ticker, close)
    low = ohlcv.get("Low", pd.DataFrame()).get(ticker, close)
    volume = ohlcv.get("Volume", pd.DataFrame()).get(ticker, pd.Series(0.0, index=close.index))
    feat = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "ret": close.pct_change(),
            "hl_range": (high - low) / close.replace(0, np.nan),
        },
        index=close.index,
    )
    return feat.loc[market.returns.index].replace([np.inf, -np.inf], np.nan).ffill().bfill()


def fit_pca_window(frame: pd.DataFrame, train_idx, n_components: float = 0.99) -> tuple[np.ndarray, StandardScaler, PCA]:
    """Fit scaler/PCA on train rows only, then transform the full window (no leakage)."""
    train = frame.loc[frame.index.intersection(train_idx)]
    scaler = StandardScaler().fit(train.values)
    scaled_train = scaler.transform(train.values)
    pca = PCA(n_components=n_components, svd_solver="full").fit(scaled_train)
    scores = pca.transform(scaler.transform(frame.values))
    return scores, scaler, pca


def pca_sequences(scores: np.ndarray, target: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(scores)):
        xs.append(scores[i - seq_len : i])
        ys.append(target[i])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
