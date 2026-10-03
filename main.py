"""CLI for the IIT Kharagpur BTP: higher-order NSE portfolio optimisation."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

from backtest import walk_forward
from config import END_DATE, MAX_WEIGHT, OUTPUT_DIR, START_DATE
from data_pipeline import build_market_data, jarque_bera_table, load_market_cache, save_market_cache
from metrics import historical_cvar
from ml_models import historical_expected_returns, ridge_expected_returns
from portfolio_optimization import (
    equal_weight,
    evaluate_weights,
    ledoit_wolf_cov,
    markowitz_max_sharpe,
    markowitz_min_variance,
    optimize_nsga3,
    pareto_frame,
    select_best_sharpe,
)


def _save_baselines(rows: list[dict], returns: pd.DataFrame, weights_map: dict) -> pd.DataFrame:
    out = []
    for row, (name, w) in zip(rows, weights_map.items()):
        rec = dict(row)
        rec["cvar"] = historical_cvar(returns.values @ w)
        for t, wi in zip(returns.columns, w):
            rec[t] = float(wi)
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(OUTPUT_DIR / "baselines.csv", index=False)
    return df


def _plot_pareto(front: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sc = ax.scatter(
        front["vol"] * 100,
        front["mean"] * 252 * 100,
        c=front["skew"],
        cmap="viridis",
        s=55,
        edgecolor="k",
        linewidth=0.3,
    )
    fig.colorbar(sc, ax=ax, label="Skewness")
    ax.set_xlabel("Daily volatility (%)")
    ax.set_ylabel("Annualised mean return proxy (%)")
    ax.set_title("NSGA-III MVSK Pareto front — current NSE universe")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_oos(oos: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for col, label in {
        "nsga3": "NSGA-III (best IS Sharpe)",
        "min_var": "Min-variance",
        "max_sharpe": "Max-Sharpe",
        "equal": "Equal weight",
        "nifty": "Nifty 50",
    }.items():
        equity = (1 + oos[col]).cumprod()
        ax.plot(equity.index, equity.values, label=label, lw=1.6)
    ax.set_title("Walk-forward wealth — NSE names vs Nifty 50")
    ax.set_ylabel("Growth of INR 1")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(start: str, end: str | None, n_gen: int, skip_backtest: bool, refresh: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("==================================================")
    print(" IIT KGP BTP: MVSK Portfolio Optimisation")
    print(" Current NSE / Nifty 50 data")
    print("==================================================")

    market = None if refresh else load_market_cache()
    if market is None:
        market = build_market_data(start=start, end=end)
        save_market_cache(market)
    else:
        print(f"Using cached NSE panel {market.start} → {market.end}")
    jb = jarque_bera_table(market.returns, market.meta)
    jb.to_csv(OUTPUT_DIR / "jarque_bera.csv", index=False)
    print("\nUniverse")
    print(market.meta.to_string(index=False))
    print("\nJarque–Bera (reject normality at 5%)")
    print(jb[["ticker", "name", "p_value", "reject_normal_5pct"]].to_string(index=False))

    mu = historical_expected_returns(market.returns)
    mu_ridge = ridge_expected_returns(market.returns)
    cov = ledoit_wolf_cov(market.returns)

    w_eq = equal_weight(market.returns.shape[1])
    w_mv = markowitz_min_variance(market.returns, cov=cov)
    w_ms = markowitz_max_sharpe(mu, market.returns, cov=cov)
    weights, _ = optimize_nsga3(mu, market.returns, cov=cov, n_gen=n_gen)
    front = pareto_frame(weights, market.returns, mu)
    front.to_csv(OUTPUT_DIR / "pareto_front.csv", index=False)
    best = select_best_sharpe(front)
    w_nsga = best[market.returns.columns].to_numpy(dtype=float)

    baseline_rows = [
        evaluate_weights(w_eq, market.returns, "equal-weight"),
        evaluate_weights(w_mv, market.returns, "min-variance"),
        evaluate_weights(w_ms, market.returns, "max-sharpe"),
        evaluate_weights(w_nsga, market.returns, "nsga3-best-sharpe"),
    ]
    baselines = _save_baselines(
        baseline_rows,
        market.returns,
        {
            "equal-weight": w_eq,
            "min-variance": w_mv,
            "max-sharpe": w_ms,
            "nsga3-best-sharpe": w_nsga,
        },
    )
    print("\nIn-sample baselines")
    print(baselines[["portfolio", "mean", "vol", "skew", "kurtosis", "sharpe", "largest_weight"]].to_string(index=False))

    _plot_pareto(front, OUTPUT_DIR / "pareto_front.png")

    if not skip_backtest:
        print("\nWalk-forward out-of-sample (2y train / 1q test)...")
        summary, folds, oos = walk_forward(market.returns, market.benchmark_returns, n_gen=max(25, n_gen // 2))
        summary.to_csv(OUTPUT_DIR / "walkforward_summary.csv")
        folds.to_csv(OUTPUT_DIR / "walkforward_folds.csv", index=False)
        oos.to_csv(OUTPUT_DIR / "walkforward_daily.csv")
        _plot_oos(oos, OUTPUT_DIR / "walkforward_equity.png")
        print(summary[["ann_return", "ann_vol", "sharpe", "cvar", "max_drawdown"]].to_string())

    ridge_vs_hist = pd.DataFrame({"historical": mu, "ridge": mu_ridge})
    ridge_vs_hist.to_csv(OUTPUT_DIR / "expected_returns.csv")
    print(f"\nArtifacts written to {OUTPUT_DIR}")
    print("==================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE MVSK portfolio BTP pipeline")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--n-gen", type=int, default=80)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Re-download NSE prices")
    parser.add_argument("--max-weight", type=float, default=MAX_WEIGHT)
    args = parser.parse_args()
    if args.max_weight != MAX_WEIGHT:
        print(f"Note: concentration cap override {args.max_weight} is used only if you edit config.MAX_WEIGHT.")
    run(start=args.start, end=args.end, n_gen=args.n_gen, skip_backtest=args.skip_backtest, refresh=args.refresh)


if __name__ == "__main__":
    main()
