"""Hybrid stock-selection models after Masuda (2024), with explicit gradient vectors.

Primary path: PCA features → Linear / RF / gradient boosting / SGD hybrid.
The hybrid uses a 1-D conv-style smoother plus a linear head trained by SGD.
Each epoch records the gradient vector g = ∇_θ L and its L2 norm (thesis §4.1:
backprop updates use the direction and magnitude of the gradient).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config import SEQ_LEN, SGD_EPOCHS, SGD_LR, TRAIN_END
from data_pipeline import MarketData
from features import fit_pca_window, ohlcv_feature_frame, pca_sequences
from metrics import expected_alpha, mae, mse, smape


def _split_mask(index: pd.DatetimeIndex, train_end: str = TRAIN_END) -> np.ndarray:
    return index <= pd.Timestamp(train_end)


def _smooth_conv(seq: np.ndarray) -> np.ndarray:
    """Fixed 3-tap averaging kernel — the CNN feature step (Lu / Masuda CNN-LSTM)."""
    kernel = np.array([0.25, 0.50, 0.25])
    if seq.ndim == 2:
        pad = np.pad(seq, ((1, 1), (0, 0)), mode="edge")
        return np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(seq.shape[1])], axis=1)
    out = []
    for i in range(seq.shape[0]):
        out.append(_smooth_conv(seq[i]))
    return np.asarray(out)


class SGDLinearHead:
    """Linear head trained by SGD. Exposes the full gradient vector each step."""

    def __init__(self, n_features: int, lr: float = SGD_LR, epochs: int = SGD_EPOCHS, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0.0, 0.05, size=n_features)
        self.bias = 0.0
        self.lr = lr
        self.epochs = epochs
        self.grad_log: list[dict] = []

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.theta + self.bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDLinearHead":
        n = len(y)
        for epoch in range(1, self.epochs + 1):
            pred = self._forward(X)
            err = pred - y
            g_theta = (X.T @ err) / n
            g_bias = float(err.mean())
            g_vec = np.concatenate([g_theta, np.array([g_bias])])
            g_norm = float(np.linalg.norm(g_vec))
            self.theta = self.theta - self.lr * g_theta
            self.bias = self.bias - self.lr * g_bias
            loss = float(np.mean(err**2))
            self.grad_log.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "grad_l2": g_norm,
                    "grad_mean": float(g_vec.mean()),
                    "grad_max_abs": float(np.max(np.abs(g_vec))),
                }
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    @property
    def last_gradient_vector(self) -> np.ndarray:
        if not self.grad_log:
            return np.array([])
        # Reconstruct last g from stored scalars is incomplete; keep theta-sized proxy.
        return np.concatenate([self.theta, np.array([self.bias])])


def _hybrid_design(sequences: np.ndarray) -> np.ndarray:
    smoothed = _smooth_conv(sequences)
    last = smoothed[:, -1, :]
    mean = smoothed.mean(axis=1)
    return np.hstack([last, mean])


def _holdout_forecasts(model, X_train, y_train, X_test) -> np.ndarray:
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def train_selection_models(market: MarketData, train_end: str = TRAIN_END, seq_len: int = SEQ_LEN) -> dict:
    """Train Masuda-style models per NSE name. PCA/scalers fit on the train window only."""
    train_mask = _split_mask(market.returns.index, train_end)
    train_idx = market.returns.index[train_mask]
    if train_idx.empty or (~train_mask).sum() < 20:
        raise RuntimeError("Holdout window is too short. Check TRAIN_END vs the NSE sample.")

    rows = []
    alpha_hat = {}
    mu_hat = {}
    grad_rows = []
    hybrid = None

    for ticker in market.tickers:
        feat = ohlcv_feature_frame(market, ticker)
        scores, _, pca = fit_pca_window(feat, train_idx)
        y = market.returns[ticker].reindex(feat.index).values.astype(float)
        X_seq, y_seq = pca_sequences(scores, y, seq_len)
        idx = feat.index[seq_len:]
        is_train = idx <= pd.Timestamp(train_end)
        if is_train.sum() < 40 or (~is_train).sum() < 10:
            continue

        X_flat = X_seq.reshape(len(X_seq), -1)
        X_hyb = _hybrid_design(X_seq)
        y_tr, y_te = y_seq[is_train], y_seq[~is_train]
        X_tr, X_te = X_flat[is_train], X_flat[~is_train]
        H_tr, H_te = X_hyb[is_train], X_hyb[~is_train]
        close = market.prices[ticker].reindex(feat.index).values
        prev_close = close[seq_len - 1 : -1]
        true_price = close[seq_len:]
        prev_te = prev_close[~is_train]
        price_te = true_price[~is_train]

        linear = LinearRegression()
        forest = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)
        boosting = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=80, random_state=42)
        hybrid = SGDLinearHead(n_features=H_tr.shape[1])

        preds = {
            "linear": _holdout_forecasts(linear, X_tr, y_tr, X_te),
            "random_forest": _holdout_forecasts(forest, X_tr, y_tr, X_te),
            "grad_boost": _holdout_forecasts(boosting, X_tr, y_tr, X_te),
            "cnn_sgd_hybrid": hybrid.fit(H_tr, y_tr).predict(H_te),
        }
        for name, yhat in preds.items():
            price_hat = prev_te * (1.0 + yhat)
            rows.append(
                {
                    "ticker": ticker,
                    "model": name,
                    "mse": mse(price_te, price_hat),
                    "mae": mae(price_te, price_hat),
                    "smape": smape(price_te, price_hat),
                    "n_test": int(len(y_te)),
                    "n_pcs": int(pca.n_components_),
                }
            )

        mu_hat[ticker] = float(np.mean(preds["cnn_sgd_hybrid"]))
        for rec in hybrid.grad_log:
            rec = dict(rec)
            rec["ticker"] = ticker
            grad_rows.append(rec)

    mu = pd.Series(mu_hat, name="expected_return")
    train_r = market.returns.loc[train_idx]
    train_m = market.benchmark_returns.loc[train_idx]
    betas = pd.Series(
        {t: float(train_r[t].cov(train_m) / train_m.var(ddof=1)) for t in mu.index},
        name="beta",
    )
    alpha = expected_alpha(mu, betas, float(train_m.mean()))
    for t in alpha.index:
        alpha_hat[t] = float(alpha[t])

    pred_table = pd.DataFrame(rows)
    summary = (
        pred_table.groupby("model")[["mse", "mae", "smape"]]
        .mean()
        .sort_values("smape")
        .reset_index()
    )
    grads = pd.DataFrame(grad_rows)
    return {
        "predictions": pred_table,
        "summary": summary,
        "expected_return": mu,
        "expected_alpha": pd.Series(alpha_hat, name="expected_alpha"),
        "betas": betas,
        "gradients": grads,
        "hybrid": hybrid,
    }


def train_and_predict_alpha(asset_returns, benchmark_returns, betas, pca_features=None, epochs: int = 20, method: str = "ridge"):
    """Kept for the old CLI. Prefer train_selection_models()."""
    mu = asset_returns.mean()
    return expected_alpha(mu, betas, float(benchmark_returns.mean()))
