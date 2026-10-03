# Higher-order NSE portfolio optimisation (BTP)

Bachelor's Thesis Project, Department of Mathematics, IIT Kharagpur.

**Student:** Prabhat Kumar (23MA10046)  
**Supervisor:** Prof. Geetanjli Panda  
**Topic:** Mean–Variance–Skewness–Kurtosis (MVSK) allocation on current Indian equity data

Classical Markowitz uses only mean and variance. NSE daily returns are skewed and fat-tailed, so a two-moment book understates tail risk. This repository treats long-only allocation as a four-objective problem and approximates the Pareto set with NSGA-III.

## What was corrected

The first public version mixed a **synthetic 10×756 matrix**, a **US/MIT-thesis CNN-LSTM story**, and a **stale NSE pull that stopped on 2024-06-30**. It also kept `LICI.NS` (listed May 2022), read `Adj Close` after yfinance started auto-adjusting, fitted PCA on the full sample (leakage), and used excess kurtosis while the write-up reported Pearson kurtosis.

This revision freezes a **12-name Nifty 50 book**, downloads **prices through the latest NSE close**, and evaluates portfolios **out of sample**.

## Universe

| Ticker | Name | Sector |
|---|---|---|
| RELIANCE.NS | Reliance Industries | Energy |
| TCS.NS | Tata Consultancy Services | IT |
| HDFCBANK.NS | HDFC Bank | Banking |
| ICICIBANK.NS | ICICI Bank | Banking |
| SBIN.NS | State Bank of India | Banking |
| BHARTIARTL.NS | Bharti Airtel | Telecom |
| INFY.NS | Infosys | IT |
| ITC.NS | ITC | FMCG |
| HINDUNILVR.NS | Hindustan Unilever | FMCG |
| LT.NS | Larsen & Toubro | Capital Goods |
| SUNPHARMA.NS | Sun Pharmaceutical | Pharma |
| MARUTI.NS | Maruti Suzuki | Auto |

Benchmark: **Nifty 50** (`^NSEI`). Sample: **2020-01-01 → latest close**. Risk-free rate: **6.5%** annual (India 10Y / policy corridor), converted as \(r_f/252\).

## Method

1. Download auto-adjusted NSE closes (Yahoo Finance) and align on common trading days.
2. Jarque–Bera test per name.
3. Expected returns = historical mean on the estimation window (Ridge forecast is optional; a one-day CNN-LSTM print is not treated as a usable \(\mu\)).
4. Covariance = Ledoit–Wolf shrinkage.
5. Baselines: equal weight, min-variance, max-Sharpe (SLSQP, long-only, 30% cap).
6. NSGA-III on \((\max \mu, \min \sigma^2, \max \text{skew}, \min \text{kurtosis})\) with simplex repair.
7. Walk-forward: 504-day train, 63-day test, rolled quarterly; report Sharpe, CVaR, drawdown versus Nifty 50.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py
python main.py --skip-backtest          # faster in-sample only
streamlit run app.py
```

Artifacts land in `output/`:

- `jarque_bera.csv`
- `baselines.csv`
- `pareto_front.csv` / `pareto_front.png`
- `walkforward_summary.csv` / `walkforward_equity.png`

Cached NSE panels are written to `data/`.

## Latest run (2020-01-02 → 2026-10-01, 1,670 NSE days)

All 12 names reject normality (Jarque–Bera p-value ≈ 0). Pearson kurtosis is 8–16, so a variance-only book is incomplete.

Walk-forward (2y train / 1q test, 19 folds, vs Nifty 50):

| Strategy | Ann. return | Sharpe | CVaR 5% | Max DD |
|---|---:|---:|---:|---:|
| Max-Sharpe | 12.5% | 0.36 | 2.01% | −17.3% |
| NSGA-III (best IS Sharpe) | 10.7% | 0.27 | 1.85% | −18.1% |
| Equal weight | 8.8% | 0.15 | 1.77% | −19.3% |
| Min-variance | 7.4% | 0.05 | 1.68% | −22.9% |
| Nifty 50 | 5.8% | −0.06 | 1.98% | −16.5% |

Max-Sharpe wins raw Sharpe. NSGA-III sits between that corner and equal weight: lower CVaR than max-Sharpe, higher return than the index. That is the MVSK point — you trade a bit of Sharpe for a less concentrated, less tail-heavy book.

## Notes

- `max_weight = 0.30` replaces the old 57% cap that came from a synthetic corner solution.
- Pearson kurtosis is used (`fisher=False`); a normal series has kurtosis 3.
- CNN-LSTM remains an optional extra if PyTorch is installed. It is not required for the BTP results.
