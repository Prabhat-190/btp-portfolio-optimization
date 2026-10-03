"""Streamlit dashboard: Masuda-style hybrid ML selection on current NSE data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

from config import END_DATE, OUTPUT_DIR, START_DATE, TRAIN_END
from data_pipeline import build_market_data, jarque_bera_table, save_market_cache
from plots import plot_gradients, plot_holdout
from ml_models import train_selection_models
from backtest import holdout_paths
from portfolio_optimization import allocate_by_alpha

st.set_page_config(page_title="NSE Hybrid ML BTP", layout="wide")
st.title("Hybrid ML stock selection — current NSE data")
st.caption(
    "BTP · Prabhat Kumar · 23MA10046 · IIT Kharagpur · "
    "Masuda (MIT, 2024) pipeline on Nifty 50 names: PCA → models → α → mean–variance. "
    "Higher-order moments are a diagnostic only."
)


def _read_csv(name: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else None


with st.sidebar:
    st.header("Sample")
    start = st.date_input("Start", value=pd.to_datetime(START_DATE).date())
    end = st.date_input("End", value=pd.to_datetime(END_DATE).date())
    run = st.button("Fetch NSE data and train models", type="primary")

uni_path = Path(__file__).with_name("data") / "universe.csv"
universe = pd.read_csv(uni_path) if uni_path.exists() else None
smape = _read_csv("prediction_smape.csv")
jb = _read_csv("jarque_bera.csv")
alpha = _read_csv("expected_alpha.csv")
hold = _read_csv("holdout_summary.csv")
grads = _read_csv("gradient_norms.csv")

if run:
    with st.spinner("Downloading current NSE / Nifty 50 prices..."):
        market = build_market_data(start=start.isoformat(), end=end.isoformat())
        save_market_cache(market)
        jb = jarque_bera_table(market.returns, market.meta)
        jb.to_csv(OUTPUT_DIR / "jarque_bera.csv", index=False)
        universe = market.meta
    with st.spinner("PCA + Linear / RF / gradient boosting / CNN-SGD hybrid..."):
        fitted = train_selection_models(market)
        fitted["summary"].to_csv(OUTPUT_DIR / "prediction_smape.csv", index=False)
        fitted["gradients"].to_csv(OUTPUT_DIR / "gradient_norms.csv", index=False)
        fitted["expected_alpha"].to_csv(OUTPUT_DIR / "expected_alpha.csv")
        smape = fitted["summary"]
        alpha = fitted["expected_alpha"].reset_index()
        alpha.columns = ["ticker", "expected_alpha"]
        grads = fitted["gradients"]
        plot_gradients(grads, OUTPUT_DIR / "gradient_norms.png")
        w, picked = allocate_by_alpha(fitted["expected_alpha"], market.returns)
        summary, daily = holdout_paths(market.returns, market.benchmark_returns, fitted["expected_alpha"])
        summary.to_csv(OUTPUT_DIR / "holdout_summary.csv")
        daily.to_csv(OUTPUT_DIR / "holdout_daily.csv")
        plot_holdout(daily, OUTPUT_DIR / "holdout_equity.png")
        hold = summary.reset_index()
        st.success(f"Holdout after {TRAIN_END}. Sleeve: {', '.join(picked)}")

tabs = st.tabs(["Universe", "Prediction", "Gradient vector", "Alpha + MVO", "Higher-order"])

with tabs[0]:
    st.subheader("Current Indian equity book")
    if universe is not None:
        st.dataframe(universe, width="stretch", hide_index=True)
    else:
        st.info("Run the sidebar job.")

with tabs[1]:
    st.subheader("Holdout SMAPE (Masuda Eq. 5.3)")
    if smape is not None:
        st.dataframe(smape, width="stretch", hide_index=True)
        st.caption("Hybrid and gradient-boosted models should beat plain linear regression on NSE returns.")
    else:
        st.info("No prediction table yet.")

with tabs[2]:
    st.subheader("SGD gradient vector")
    st.write(
        "The hybrid head is trained by SGD. Each epoch stores the gradient vector "
        r"$g=\nabla_\theta L$ and plots the mean $||g||_2$. LSTM/CNN hybrids are used "
        "because plain RNNs suffer vanishing gradients (Masuda §4.1)."
    )
    gpath = OUTPUT_DIR / "gradient_norms.png"
    if gpath.exists():
        st.image(str(gpath), width="stretch")
    if grads is not None and not grads.empty:
        last = grads.groupby("ticker").tail(1)[["ticker", "loss", "grad_l2", "grad_max_abs"]]
        st.dataframe(last, width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Expected alpha and holdout vs Nifty 50")
    if alpha is not None:
        st.dataframe(alpha, width="stretch", hide_index=True)
    if hold is not None:
        st.dataframe(hold, width="stretch", hide_index=True)
    epath = OUTPUT_DIR / "holdout_equity.png"
    if epath.exists():
        st.image(str(epath), width="stretch")

with tabs[4]:
    st.subheader("Why two-moment MVO is incomplete")
    if jb is not None:
        st.metric("Names rejecting normality (5%)", f"{int(jb['reject_normal_5pct'].sum())} / {len(jb)}")
        st.dataframe(jb, width="stretch", hide_index=True)
        st.caption("Skew and kurtosis are reported as a check. The allocator itself stays mean–variance + alpha, as in Masuda.")
    else:
        st.info("No Jarque–Bera table yet.")
