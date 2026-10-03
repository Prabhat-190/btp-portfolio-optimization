"""Holdout simulation after Masuda (2024) §5.2, plus an optional MVSK extra."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MAX_WEIGHT, N_SELECT, TEST_DAYS, TRAIN_DAYS, TRAIN_END
from metrics import performance_summary, portfolio_returns
from portfolio_optimization import (
    allocate_by_alpha,
    equal_weight,
    ledoit_wolf_cov,
    markowitz_max_sharpe,
    markowitz_min_variance,
    optimize_nsga3,
    pareto_frame,
    project_to_capped_simplex,
    select_best_sharpe,
)


def _historical_mu(returns: pd.DataFrame) -> pd.Series:
    return returns.mean()


def holdout_paths(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    alpha: pd.Series,
    train_end: str = TRAIN_END,
    n_select: int = N_SELECT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze the train-set α book and mark it to the unseen NSE holdout."""
    hold = returns.index > pd.Timestamp(train_end)
    test = returns.loc[hold]
    bench = benchmark_returns.reindex(test.index)
    train = returns.loc[~hold]
    if test.empty:
        raise RuntimeError("No holdout rows after TRAIN_END.")

    cov = ledoit_wolf_cov(train)
    w_eq = equal_weight(train.shape[1])
    w_mv = markowitz_min_variance(train, cov=cov)
    w_ms = markowitz_max_sharpe(_historical_mu(train), train, cov=cov)
    w_alpha, names = allocate_by_alpha(alpha.reindex(train.columns).fillna(0.0), train, n_select=n_select)

    daily = pd.DataFrame(
        {
            "alpha_mvo": portfolio_returns(w_alpha, test),
            "max_sharpe": portfolio_returns(w_ms, test),
            "min_var": portfolio_returns(w_mv, test),
            "equal": portfolio_returns(w_eq, test),
            "nifty": bench.values,
        },
        index=test.index,
    )
    rows = []
    for col in daily.columns:
        row = performance_summary(daily[col].values)
        row["strategy"] = col
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("strategy")
    summary.attrs["selected"] = names
    return summary, daily


def walk_forward(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
    max_weight: float = MAX_WEIGHT,
    n_gen: int = 25,
):
    """Optional higher-order extra: rolling NSGA-III vs Markowitz vs Nifty."""
    aligned = returns.join(benchmark_returns.rename("NIFTY50"), how="inner")
    asset_r = aligned[returns.columns]
    bench_r = aligned["NIFTY50"]
    folds = []
    start = 0
    n = len(asset_r)
    while start + train_days + 1 < n:
        tr_end = start + train_days
        te_end = min(n, tr_end + test_days)
        folds.append((start, tr_end, te_end))
        start += test_days
        if te_end >= n:
            break
    if not folds:
        raise RuntimeError("Not enough NSE history for a walk-forward window.")

    oos = {k: [] for k in ("equal", "min_var", "max_sharpe", "nsga3", "nifty")}
    oos_index = []
    records = []
    for fold, (a, b, c) in enumerate(folds, start=1):
        train = asset_r.iloc[a:b]
        test = asset_r.iloc[b:c]
        bench_test = bench_r.iloc[b:c]
        mu = train.mean()
        cov = ledoit_wolf_cov(train)
        w_eq = equal_weight(train.shape[1])
        w_mv = markowitz_min_variance(train, cov=cov, max_weight=max_weight)
        w_ms = markowitz_max_sharpe(mu, train, cov=cov, max_weight=max_weight)
        w_nsga, _ = optimize_nsga3(mu, train, cov=cov, max_weight=max_weight, n_gen=n_gen)
        front = pareto_frame(w_nsga, train, mu)
        w_sel = project_to_capped_simplex(front.loc[select_best_sharpe(front).name, train.columns].values)
        paths = {
            "equal": portfolio_returns(w_eq, test),
            "min_var": portfolio_returns(w_mv, test),
            "max_sharpe": portfolio_returns(w_ms, test),
            "nsga3": portfolio_returns(w_sel, test),
            "nifty": bench_test.values,
        }
        for k, series in paths.items():
            oos[k].append(np.asarray(series, dtype=float))
        oos_index.extend(list(test.index))
        print(f"  fold {fold}/{len(folds)}  {train.index[0].date()} → {test.index[-1].date()}")
        rec = {
            "fold": fold,
            "train_start": train.index[0].date().isoformat(),
            "test_end": test.index[-1].date().isoformat(),
        }
        for name, series in paths.items():
            stats = performance_summary(series)
            rec[f"{name}_sharpe"] = stats["sharpe"]
        records.append(rec)

    oos_daily = pd.DataFrame({k: np.concatenate(v) for k, v in oos.items()}, index=pd.DatetimeIndex(oos_index))
    summary = pd.DataFrame(
        [{**performance_summary(oos_daily[c].values), "strategy": c} for c in oos_daily.columns]
    ).set_index("strategy")
    return summary, pd.DataFrame(records), oos_daily
