"""Portfolio moments, CVaR, and annualised performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from config import CVAR_ALPHA, RISK_FREE_ANNUAL, TRADING_DAYS


def daily_risk_free(rf_annual: float = RISK_FREE_ANNUAL) -> float:
    return rf_annual / TRADING_DAYS


def portfolio_returns(weights: np.ndarray, returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    r = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    w = np.asarray(weights, dtype=float)
    return r @ w


def pearson_kurtosis(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Pearson kurtosis (normal = 3). scipy default is excess kurtosis."""
    arr = np.asarray(x, dtype=float)
    return kurtosis(np.nan_to_num(arr, nan=0.0), axis=axis, fisher=False, bias=False)


def moments(weights: np.ndarray, returns: pd.DataFrame | np.ndarray) -> dict[str, float]:
    rp = portfolio_returns(weights, returns)
    return {
        "mean": float(np.mean(rp)),
        "variance": float(np.var(rp, ddof=1)),
        "vol": float(np.std(rp, ddof=1)),
        "skew": float(skew(rp, bias=False)),
        "kurtosis": float(pearson_kurtosis(rp)),
    }


def historical_cvar(returns: np.ndarray, alpha: float = CVAR_ALPHA) -> float:
    """Positive loss CVaR: average of the worst alpha tail."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float("nan")
    cutoff = np.nanquantile(r, alpha)
    tail = r[r <= cutoff]
    if tail.size == 0:
        return float(-cutoff)
    return float(-np.mean(tail))


def sharpe_ratio(mean: float, vol: float, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    if vol <= 0:
        return float("nan")
    return (mean - daily_risk_free(rf_annual)) / vol


def annualise_return(daily_mean: float) -> float:
    return (1.0 + daily_mean) ** TRADING_DAYS - 1.0


def annualise_vol(daily_vol: float) -> float:
    return daily_vol * np.sqrt(TRADING_DAYS)


def performance_summary(
    daily_returns: np.ndarray,
    rf_annual: float = RISK_FREE_ANNUAL,
    alpha: float = CVAR_ALPHA,
) -> dict[str, float]:
    r = np.asarray(daily_returns, dtype=float)
    r = r[~np.isnan(r)]
    mu = float(np.mean(r)) if r.size else float("nan")
    vol = float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
    equity = np.cumprod(1.0 + r) if r.size else np.array([1.0])
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0)) if equity.size else float("nan")
    return {
        "daily_mean": mu,
        "daily_vol": vol,
        "ann_return": annualise_return(mu) if r.size else float("nan"),
        "ann_vol": annualise_vol(vol) if r.size > 1 else float("nan"),
        "sharpe": sharpe_ratio(mu, vol, rf_annual) * np.sqrt(TRADING_DAYS) if r.size > 1 else float("nan"),
        "skew": float(skew(r, bias=False)) if r.size > 2 else float("nan"),
        "kurtosis": float(pearson_kurtosis(r)) if r.size > 3 else float("nan"),
        "cvar": historical_cvar(r, alpha),
        "max_drawdown": max_dd,
        "n_days": int(r.size),
    }
