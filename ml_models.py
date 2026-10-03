"""Expected-return models with no in-sample leakage.

Default: historical mean (what MVSK actually needs).
Optional: Ridge on lagged returns, fit only on the training window.
Optional: CNN-LSTM if PyTorch is installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import TRADING_DAYS


def historical_expected_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean()


def _lag_matrix(series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(lags, len(series)):
        x.append(series[i - lags : i])
        y.append(series[i])
    return np.asarray(x), np.asarray(y)


def ridge_expected_returns(returns: pd.DataFrame, lags: int = 20) -> pd.Series:
    """Walk-forward 1-day forecasts averaged over the last `lags` days."""
    out = {}
    for col in returns.columns:
        y = returns[col].values.astype(float)
        X, target = _lag_matrix(y, lags)
        if len(X) < 60:
            out[col] = float(np.mean(y))
            continue
        split = int(0.8 * len(X))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[:split])
        model = Ridge(alpha=1.0)
        model.fit(X_train, target[:split])
        X_tail = scaler.transform(X[split:])
        preds = model.predict(X_tail)
        out[col] = float(np.mean(preds)) if preds.size else float(np.mean(y))
    return pd.Series(out, name="expected_return")


def train_and_predict_alpha(
    asset_returns,
    benchmark_returns,
    betas,
    pca_features=None,
    epochs: int = 20,
    method: str = "ridge",
):
    """Compatibility wrapper. Prefer historical/ridge over a 1-day CNN forecast."""
    if method == "historical":
        mu = historical_expected_returns(asset_returns)
    else:
        mu = ridge_expected_returns(asset_returns)

    market_mu = float(benchmark_returns.mean())
    beta_s = pd.Series(betas)
    alpha = mu - beta_s.reindex(mu.index) * market_mu
    alpha.name = "expected_alpha"
    print("Expected daily returns")
    print(mu.apply(lambda x: f"{x*TRADING_DAYS*100:.2f}% ann. proxy"))
    print("Expected daily alphas vs Nifty 50")
    print(alpha)
    return alpha


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class CNNLSTM(nn.Module):
        def __init__(self, input_dim, cnn_filters=16, lstm_hidden=32):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, cnn_filters, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(2)
            self.lstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True)
            self.fc = nn.Linear(lstm_hidden, 1)

        def forward(self, x):
            x = self.relu(self.conv1(x.permute(0, 2, 1)))
            x = self.pool(x).permute(0, 2, 1)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    def cnn_lstm_expected_returns(returns: pd.DataFrame, seq_len: int = 20, epochs: int = 15) -> pd.Series:
        """Optional DL forecast. PCA/scalers are fit on the train split only."""
        preds = {}
        for col in returns.columns:
            y = returns[col].values.astype(float).reshape(-1, 1)
            X, target = _lag_matrix(y.ravel(), seq_len)
            if len(X) < 80:
                preds[col] = float(np.mean(y))
                continue
            split = int(0.8 * len(X))
            scaler_x = StandardScaler().fit(X[:split])
            scaler_y = StandardScaler().fit(target[:split].reshape(-1, 1))
            X_train = scaler_x.transform(X[:split])
            y_train = scaler_y.transform(target[:split].reshape(-1, 1))
            loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
                    torch.tensor(y_train, dtype=torch.float32),
                ),
                batch_size=32,
                shuffle=True,
            )
            model = CNNLSTM(input_dim=1)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            model.train()
            for _ in range(epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                tail = scaler_x.transform(X[split:])
                yhat = model(torch.tensor(tail, dtype=torch.float32).unsqueeze(-1)).numpy()
                yhat = scaler_y.inverse_transform(yhat).ravel()
            preds[col] = float(np.mean(yhat)) if yhat.size else float(np.mean(y))
        return pd.Series(preds, name="expected_return")

except ImportError:
    def cnn_lstm_expected_returns(returns: pd.DataFrame, seq_len: int = 20, epochs: int = 15) -> pd.Series:
        print("PyTorch not installed; falling back to Ridge forecasts.")
        return ridge_expected_returns(returns, lags=seq_len)
