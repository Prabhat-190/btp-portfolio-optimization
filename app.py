"""Streamlit dashboard for the NSE MVSK BTP."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
import streamlit as st

from config import END_DATE, OUTPUT_DIR, START_DATE
from data_pipeline import build_market_data, jarque_bera_table, save_market_cache
from metrics import historical_cvar
from ml_models import historical_expected_returns
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
from backtest import walk_forward

st.set_page_config(page_title="NSE MVSK Portfolio BTP", layout="wide")
st.title("Higher-order portfolio optimisation — current NSE data")
st.caption("BTP · Prabhat Kumar · 23MA10046 · IIT Kharagpur · Nifty 50 names, live Yahoo Finance pull")


def _read_csv(name: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


with st.sidebar:
    st.header("Sample")
    start = st.date_input("Start", value=pd.to_datetime(START_DATE).date())
    end = st.date_input("End", value=pd.to_datetime(END_DATE).date())
    n_gen = st.slider("NSGA-III generations", min_value=20, max_value=120, value=60, step=10)
    run_backtest = st.checkbox("Run walk-forward (slower)", value=False)
    run = st.button("Fetch current NSE data and optimise", type="primary")

universe_path = Path(__file__).with_name("data") / "universe.csv"
universe = pd.read_csv(universe_path) if universe_path.exists() else None
jb = _read_csv("jarque_bera.csv")
baselines = _read_csv("baselines.csv")
front = _read_csv("pareto_front.csv")
wf = _read_csv("walkforward_summary.csv")

if run:
    with st.spinner("Downloading current NSE / Nifty 50 prices..."):
        market = build_market_data(start=start.isoformat(), end=end.isoformat())
        save_market_cache(market)
        jb = jarque_bera_table(market.returns, market.meta)
        jb.to_csv(OUTPUT_DIR / "jarque_bera.csv", index=False)
        universe = market.meta
    with st.spinner("Solving Markowitz baselines and NSGA-III MVSK front..."):
        mu = historical_expected_returns(market.returns)
        cov = ledoit_wolf_cov(market.returns)
        w_eq = equal_weight(market.returns.shape[1])
        w_mv = markowitz_min_variance(market.returns, cov=cov)
        w_ms = markowitz_max_sharpe(mu, market.returns, cov=cov)
        weights, _ = optimize_nsga3(mu, market.returns, cov=cov, n_gen=n_gen)
        front = pareto_frame(weights, market.returns, mu)
        front.to_csv(OUTPUT_DIR / "pareto_front.csv", index=False)
        best = select_best_sharpe(front)
        w_nsga = best[market.returns.columns].to_numpy(dtype=float)
        rows = []
        for label, w in [
            ("equal-weight", w_eq),
            ("min-variance", w_mv),
            ("max-sharpe", w_ms),
            ("nsga3-best-sharpe", w_nsga),
        ]:
            rec = evaluate_weights(w, market.returns, label)
            rec["cvar"] = historical_cvar(market.returns.values @ w)
            for t, wi in zip(market.returns.columns, w):
                rec[t] = float(wi)
            rows.append(rec)
        baselines = pd.DataFrame(rows)
        baselines.to_csv(OUTPUT_DIR / "baselines.csv", index=False)
    if run_backtest:
        with st.spinner("Walk-forward out-of-sample vs Nifty 50..."):
            summary, folds, oos = walk_forward(market.returns, market.benchmark_returns, n_gen=max(25, n_gen // 2))
            summary.to_csv(OUTPUT_DIR / "walkforward_summary.csv")
            folds.to_csv(OUTPUT_DIR / "walkforward_folds.csv", index=False)
            oos.to_csv(OUTPUT_DIR / "walkforward_daily.csv")
            wf = summary.reset_index()
    st.success(f"Loaded NSE closes {market.start} → {market.end}.")

tabs = st.tabs(["Universe", "Normality", "Baselines", "Pareto front", "Walk-forward"])

with tabs[0]:
    st.subheader("Current Indian equity book")
    st.write(
        "Twelve liquid Nifty 50 names, sector-balanced, with history from 2020 through the latest NSE close. "
        "LIC was removed because the 2022 IPO leaves a hole in the early sample."
    )
    if universe is not None:
        view = universe.copy()
        if "ann_return" in view.columns:
            view["ann_return"] = view["ann_return"].map(lambda x: f"{x*100:.1f}%")
            view["ann_vol"] = view["ann_vol"].map(lambda x: f"{x*100:.1f}%")
            view["last_price"] = view["last_price"].map(lambda x: f"₹{x:,.1f}")
        if "beta" in view.columns:
            view["beta"] = view["beta"].map(lambda x: f"{float(x):.2f}")
        st.dataframe(view, width="stretch", hide_index=True)
    else:
        st.info("Run the sidebar job to pull the latest NSE prices.")

with tabs[1]:
    st.subheader("Jarque–Bera test on daily NSE returns")
    if jb is not None:
        rejected = int(jb["reject_normal_5pct"].sum()) if "reject_normal_5pct" in jb.columns else 0
        st.metric("Names rejecting normality (5%)", f"{rejected} / {len(jb)}")
        st.dataframe(jb, width="stretch", hide_index=True)
        st.caption("Fat tails and skew are why a two-moment Markowitz book is incomplete — the BTP adds skewness and kurtosis.")
    else:
        st.info("No Jarque–Bera table yet.")

with tabs[2]:
    st.subheader("In-sample baselines")
    if baselines is not None:
        show_cols = [c for c in ["portfolio", "mean", "vol", "skew", "kurtosis", "sharpe", "cvar", "largest_weight"] if c in baselines.columns]
        st.dataframe(baselines[show_cols], width="stretch", hide_index=True)
        weight_cols = [c for c in baselines.columns if c.endswith(".NS")]
        if weight_cols:
            pick = st.selectbox("Allocation", baselines["portfolio"].tolist())
            w = baselines.loc[baselines["portfolio"] == pick, weight_cols].iloc[0]
            w = w[w > 0.01]
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.pie(w.values, labels=w.index.str.replace(".NS", ""), autopct="%1.1f%%")
            ax.set_title(pick)
            st.pyplot(fig)
    else:
        st.info("No baselines yet.")

with tabs[3]:
    st.subheader("NSGA-III Mean–Variance–Skewness–Kurtosis front")
    if front is not None and {"vol", "mean", "skew"}.issubset(front.columns):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sc = ax.scatter(front["vol"] * 100, front["mean"] * 252 * 100, c=front["skew"], cmap="viridis", s=50)
        fig.colorbar(sc, ax=ax, label="Skewness")
        ax.set_xlabel("Daily volatility (%)")
        ax.set_ylabel("Annualised mean proxy (%)")
        st.pyplot(fig)
        st.dataframe(front, width="stretch", hide_index=True)
    else:
        st.info("No Pareto front yet.")

with tabs[4]:
    st.subheader("Out-of-sample vs Nifty 50")
    if wf is not None:
        st.dataframe(wf, width="stretch", hide_index=True)
        oos_path = OUTPUT_DIR / "walkforward_daily.csv"
        if oos_path.exists():
            oos = pd.read_csv(oos_path, index_col=0, parse_dates=True)
            fig, ax = plt.subplots(figsize=(8.5, 4.6))
            for col in oos.columns:
                ax.plot((1 + oos[col]).cumprod(), label=col)
            ax.legend(frameon=False)
            ax.set_ylabel("Growth of INR 1")
            st.pyplot(fig)
    else:
        st.info("Enable walk-forward in the sidebar, then run the job.")
