"""Quarterly walk-forward test on NSE returns vs Nifty 50."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MAX_WEIGHT, TEST_DAYS, TRAIN_DAYS
from metrics import performance_summary, portfolio_returns
from ml_models import historical_expected_returns
from portfolio_optimization import (
    equal_weight,
    ledoit_wolf_cov,
    markowitz_max_sharpe,
    markowitz_min_variance,
    optimize_nsga3,
    project_to_capped_simplex,
    select_best_sharpe,
    pareto_frame,
)


def _rebalance_dates(n: int, train_days: int, test_days: int) -> list[tuple[int, int, int]]:
    folds = []
    start = 0
    while start + train_days + 1 < n:
        tr_end = start + train_days
        te_end = min(n, tr_end + test_days)
        folds.append((start, tr_end, te_end))
        start += test_days
        if te_end >= n:
            break
    return folds


def walk_forward(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
    max_weight: float = MAX_WEIGHT,
    n_gen: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned = returns.join(benchmark_returns.rename("NIFTY50"), how="inner")
    asset_r = aligned[returns.columns]
    bench_r = aligned["NIFTY50"]
    folds = _rebalance_dates(len(asset_r), train_days, test_days)
    if not folds:
        raise RuntimeError("Not enough NSE history for a walk-forward window.")

    oos = {k: [] for k in ("equal", "min_var", "max_sharpe", "nsga3", "nifty")}
    oos_index = []
    records = []

    for fold, (a, b, c) in enumerate(folds, start=1):
        train = asset_r.iloc[a:b]
        test = asset_r.iloc[b:c]
        bench_test = bench_r.iloc[b:c]
        mu = historical_expected_returns(train)
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
            "train_end": train.index[-1].date().isoformat(),
            "test_start": test.index[0].date().isoformat(),
            "test_end": test.index[-1].date().isoformat(),
            "n_test_days": int(len(test)),
        }
        for name, series in paths.items():
            stats = performance_summary(series)
            rec[f"{name}_ann_return"] = stats["ann_return"]
            rec[f"{name}_sharpe"] = stats["sharpe"]
            rec[f"{name}_cvar"] = stats["cvar"]
        records.append(rec)

    oos_daily = pd.DataFrame({k: np.concatenate(v) for k, v in oos.items()}, index=pd.DatetimeIndex(oos_index))
    summary_rows = []
    for name, col in oos_daily.items():
        row = performance_summary(col.values)
        row["strategy"] = name
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).set_index("strategy")
    folds_df = pd.DataFrame(records)
    return summary, folds_df, oos_daily
