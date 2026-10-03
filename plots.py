"""Shared matplotlib helpers for the CLI and Streamlit app."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def plot_gradients(grads: pd.DataFrame, path) -> None:
    if grads is None or grads.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    mean_g = grads.groupby("epoch")["grad_l2"].mean()
    ax.plot(mean_g.index, mean_g.values, lw=1.8, color="#0ea5e9")
    ax.set_xlabel("SGD epoch")
    ax.set_ylabel("Mean ||g||_2")
    ax.set_title("Gradient-vector L2 norm — CNN-SGD hybrid")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_holdout(daily: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    labels = {
        "alpha_mvo": "Alpha + mean-variance",
        "max_sharpe": "Historical max-Sharpe",
        "min_var": "Min-variance",
        "equal": "Equal weight",
        "nifty": "Nifty 50",
    }
    for col, label in labels.items():
        if col in daily.columns:
            ax.plot((1 + daily[col]).cumprod(), label=label, lw=1.6)
    ax.set_title("Holdout wealth — NSE names vs Nifty 50")
    ax.set_ylabel("Growth of INR 1")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
