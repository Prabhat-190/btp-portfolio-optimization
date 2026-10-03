"""CLI: Masuda-style hybrid ML stock selection on current NSE data."""

from __future__ import annotations

import argparse

import pandas as pd

from backtest import holdout_paths
from config import END_DATE, MAX_WEIGHT, N_SELECT, OUTPUT_DIR, START_DATE, TRAIN_END
from plots import plot_gradients, plot_holdout
from data_pipeline import build_market_data, jarque_bera_table, load_market_cache, save_market_cache
from metrics import historical_cvar, pearson_kurtosis
from ml_models import train_selection_models
from portfolio_optimization import (
    allocate_by_alpha,
    equal_weight,
    evaluate_weights,
    ledoit_wolf_cov,
    markowitz_max_sharpe,
    markowitz_min_variance,
    optimize_nsga3,
    pareto_frame,
    select_best_sharpe,
)
from scipy.stats import skew


def run(start: str, end: str | None, refresh: bool, higher_order: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("==================================================")
    print(" IIT KGP BTP: Hybrid ML stock selection")
    print(" Masuda (2024) on current NSE / Nifty 50 data")
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

    print("\n[1] PCA + prediction models (train ≤", TRAIN_END, ")")
    fitted = train_selection_models(market)
    fitted["summary"].to_csv(OUTPUT_DIR / "prediction_smape.csv", index=False)
    fitted["predictions"].to_csv(OUTPUT_DIR / "prediction_by_ticker.csv", index=False)
    fitted["gradients"].to_csv(OUTPUT_DIR / "gradient_norms.csv", index=False)
    fitted["expected_alpha"].to_csv(OUTPUT_DIR / "expected_alpha.csv")
    print(fitted["summary"].to_string(index=False))
    print("\nExpected alpha vs Nifty 50 (Masuda Eq. 4.8)")
    print(fitted["expected_alpha"].sort_values(ascending=False).to_string())
    plot_gradients(fitted["gradients"], OUTPUT_DIR / "gradient_norms.png")

    print("\n[2] Mean-variance on top-N expected alpha")
    w_alpha, picked = allocate_by_alpha(fitted["expected_alpha"], market.returns, n_select=N_SELECT)
    print("Selected sleeve:", ", ".join(picked))
    train = market.returns.loc[market.returns.index <= pd.Timestamp(TRAIN_END)]
    cov = ledoit_wolf_cov(train)
    w_eq = equal_weight(train.shape[1])
    w_mv = markowitz_min_variance(train, cov=cov)
    w_ms = markowitz_max_sharpe(train.mean(), train, cov=cov)
    rows = []
    for label, w in [
        ("equal-weight", w_eq),
        ("min-variance", w_mv),
        ("hist-max-sharpe", w_ms),
        ("alpha-mvo", w_alpha),
    ]:
        rec = evaluate_weights(w, train, label)
        rec["cvar"] = historical_cvar(train.values @ w)
        rows.append(rec)
    baselines = pd.DataFrame(rows)
    baselines.to_csv(OUTPUT_DIR / "baselines.csv", index=False)
    print(baselines[["portfolio", "mean", "vol", "skew", "kurtosis", "sharpe"]].to_string(index=False))

    print("\n[3] Holdout mark-to-market after", TRAIN_END)
    summary, daily = holdout_paths(
        market.returns, market.benchmark_returns, fitted["expected_alpha"]
    )
    summary.to_csv(OUTPUT_DIR / "holdout_summary.csv")
    daily.to_csv(OUTPUT_DIR / "holdout_daily.csv")
    plot_holdout(daily, OUTPUT_DIR / "holdout_equity.png")
    print(summary[["ann_return", "ann_vol", "sharpe", "cvar", "max_drawdown"]].to_string())

    hold = daily["alpha_mvo"]
    print("\n[4] Higher-order check (diagnostic only)")
    print(
        f"Alpha-MVO holdout skew={skew(hold, bias=False):.3f}  "
        f"Pearson kurtosis={float(pearson_kurtosis(hold.values)):.2f}"
    )
    print(f"Names rejecting normality (5%): {int(jb['reject_normal_5pct'].sum())}/{len(jb)}")

    if higher_order:
        print("\n[extra] Light NSGA-III MVSK on the train window...")
        weights, _ = optimize_nsga3(train.mean(), train, cov=cov, n_gen=40)
        front = pareto_frame(weights, train, train.mean())
        front.to_csv(OUTPUT_DIR / "pareto_front.csv", index=False)
        best = select_best_sharpe(front)
        print(best[["mean", "vol", "skew", "kurtosis", "sharpe"]].to_string())

    print(f"\nArtifacts written to {OUTPUT_DIR}")
    print("==================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Masuda-style NSE hybrid ML BTP")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--higher-order", action="store_true", help="Optional MVSK extra")
    parser.add_argument("--max-weight", type=float, default=MAX_WEIGHT)
    args = parser.parse_args()
    run(start=args.start, end=args.end, refresh=args.refresh, higher_order=args.higher_order)


if __name__ == "__main__":
    main()
