import numpy as np
import pandas as pd
from data_pipeline import get_data
from ml_models import train_and_predict_alpha
from portfolio_optimization import optimize_nsga3, markowitz_max_sharpe

def main():
    print("==================================================")
    print(" IIT KGP BTP: Hybrid ML/DL Portfolio Optimization")
    print(" (Based on MIT Thesis: PCA + Alpha Optimization)")
    print("==================================================")
    
    # 1. Data Pipeline (OHLCV, PCA, Beta, Jarque-Bera)
    asset_returns, benchmark_returns, betas, pca_features, jb_results = get_data()
    
    # 2. ML/DL Predictive Signals (CNN-LSTM on PCA -> Expected Alpha)
    # This generates predicted expected alphas based on historical data & PCA components
    expected_alphas_series = train_and_predict_alpha(asset_returns, benchmark_returns, betas, pca_features, epochs=20)
    expected_alphas = expected_alphas_series.values
    
    # Calculate covariance matrix using historical data for risk models
    cov_matrix = asset_returns.cov().values
    
    # 3. Markowitz Baseline (Optimizing for Alpha)
    print("\n[Step 3] Running Markowitz Baseline...")
    w_markowitz, alpha_markowitz, var_markowitz = markowitz_max_sharpe(expected_alphas, cov_matrix, max_weight=0.57)
    vol_markowitz = np.sqrt(var_markowitz)
    print(f"Markowitz Max-Sharpe Portfolio:")
    print(f"  Expected Alpha: {alpha_markowitz*100:.4f}%")
    print(f"  Volatility:     {vol_markowitz*100:.4f}%")
    print(f"  Sharpe Ratio:   {alpha_markowitz/vol_markowitz:.4f}")
    
    # 4. NSGA-III Optimization (MVSK based on Alpha)
    print("\n[Step 4] Running NSGA-III Optimization...")
    weights_nsga3, obj_nsga3 = optimize_nsga3(expected_alphas, cov_matrix, asset_returns.values, max_weight=0.57, n_portfolios=24)
    
    # Analyze the 24 portfolios
    # Find the portfolio in the Pareto front that minimizes variance
    min_var_idx = np.argmin(obj_nsga3[:, 1])
    w_min_var = weights_nsga3[min_var_idx]
    
    # Calculate alpha and variance for this portfolio
    port_alpha = np.dot(w_min_var, expected_alphas)
    port_var = np.dot(w_min_var.T, np.dot(cov_matrix, w_min_var))
    vol_nsga3 = np.sqrt(port_var)
    
    print(f"\nNSGA-III Selected Portfolio (Min Variance from Pareto front):")
    print(f"  Expected Alpha: {port_alpha*100:.4f}%")
    print(f"  Volatility:     {vol_nsga3*100:.4f}%")
    print(f"  Sharpe Ratio:   {port_alpha/vol_nsga3:.4f}")
    
    # Calculate volatility reduction
    vol_reduction = (vol_markowitz - vol_nsga3) / vol_markowitz * 100
    print(f"\nVolatility Reduction vs Markowitz: {vol_reduction:.2f}%")
    
    # Save the 24 portfolios to CSV
    portfolios_df = pd.DataFrame(weights_nsga3, columns=asset_returns.columns)
    portfolios_df['Exp_Alpha'] = -obj_nsga3[:, 0]
    portfolios_df['Variance'] = obj_nsga3[:, 1]
    portfolios_df['Skewness'] = -obj_nsga3[:, 2]
    portfolios_df['Kurtosis'] = obj_nsga3[:, 3]
    portfolios_df.to_csv('nsga3_24_portfolios_alpha.csv', index=False)
    
    print("\nResults saved to nsga3_24_portfolios_alpha.csv")
    print("==================================================")
    print(" Pipeline Execution Complete.")
    print("==================================================")

if __name__ == '__main__':
    main()
