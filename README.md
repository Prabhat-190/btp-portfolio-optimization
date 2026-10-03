# Hybrid ML stock selection on NSE (BTP)

Bachelor's Thesis Project, Department of Mathematics, IIT Kharagpur.

**Student:** Prabhat Kumar (23MA10046)  
**Supervisor:** Prof. Geetanjli Panda  
**Primary reference:** Joshua S. Masuda, *Portfolio Optimization Using a Hybrid Machine Learning Stock Selection Model*, M.Eng. thesis, MIT EECS, 2024.

The allocator is the Masuda pipeline, run on **current Indian data** (Nifty 50 names vs `^NSEI`). Higher-order moments are only a diagnostic: NSE returns fail normality, so a two-moment book is incomplete, but the optimiser itself stays **expected alpha + mean–variance**.

## What the thesis does (and what we copy)

1. Clean OHLCV, standardise, **PCA** on the **training window only** (Masuda §3.4–3.5).
2. Predict the next return with Linear, Random Forest, **gradient boosting**, and a **CNN-style + SGD** hybrid.
3. Convert forecasts to **alpha** against the market: \(\alpha_i=\hat R_i-\beta_i\hat R_m\) (Eq. 4.8). \(\beta\) is estimated on the train set vs Nifty 50, not the S&P 500.
4. Keep the **top 5** names by \(\alpha\), then **mean–variance** (max-Sharpe) on that sleeve (§5.2.1).
5. Score forecasts with **SMAPE / MAE / MSE** (Eq. 5.1–5.3) and the book with **alpha and Sharpe**.

## Gradient vector

Masuda §4.1: backprop updates weights using the **direction and magnitude of the gradient**; RNNs fail when that vector vanishes; LightGBM keeps large-gradient instances (GOSS).

This repo makes that explicit. The hybrid head is a linear layer on CNN-smoothed PCA windows, trained by SGD. Each epoch stores

\[
g=\nabla_\theta L,\qquad \|g\|_2,\quad \max_i |g_i|.
\]

Plots and `output/gradient_norms.csv` are part of the BTP record.

## Universe

Twelve liquid Nifty 50 names, 2020-01-01 → latest NSE close. LIC is out (IPO 2022). Train / holdout cut: **31 Dec 2024** (the Indian analogue of Masuda’s 2019–22 / 2023 split). Risk-free rate: 6.5% India 10Y.

## Higher-order (light touch)

Jarque–Bera is reported because every name rejects normality. Skew and Pearson kurtosis of the alpha-MVO book are printed as a check. `python main.py --higher-order` runs a small NSGA-III MVSK front — it is **not** the primary allocator.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --refresh
streamlit run app.py
```

Main artifacts: `prediction_smape.csv`, `expected_alpha.csv`, `gradient_norms.csv`, `holdout_summary.csv`, `jarque_bera.csv`.
