import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def fetch_and_preprocess_data(tickers, benchmark_ticker, start_date, end_date):
    """
    Fetches OHLCV data, calculates daily returns and beta, and applies PCA.
    """
    print(f"Fetching data for {len(tickers)} assets and benchmark {benchmark_ticker}...")
    
    all_tickers = tickers + [benchmark_ticker]
    data = yf.download(all_tickers, start=start_date, end=end_date)
    
    # Forward fill then backward fill to handle missing data
    data = data.ffill().bfill()
    
    # Calculate daily returns based on Adj Close
    returns = data['Adj Close'].pct_change().dropna()
    
    benchmark_returns = returns[benchmark_ticker]
    asset_returns = returns[tickers]
    
    # Calculate Beta for each asset relative to the benchmark
    print("Calculating Beta for each asset...")
    betas = {}
    benchmark_var = benchmark_returns.var()
    for ticker in tickers:
        cov = asset_returns[ticker].cov(benchmark_returns)
        beta = cov / benchmark_var
        betas[ticker] = beta
        
    print(pd.Series(betas, name="Beta"))
    
    # PCA Dimensionality Reduction on OHLCV features (aligning with MIT thesis)
    print("\nApplying PCA Dimensionality Reduction to OHLCV features...")
    pca_features_dict = {}
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    for ticker in tickers:
        # Extract features for this ticker
        ticker_data = data.loc[:, (features, ticker)]
        ticker_data.columns = ticker_data.columns.droplevel(1) # Drop ticker name from columns
        # Align with returns index (drops first row since returns are pct_change)
        ticker_data = ticker_data.loc[asset_returns.index] 
        
        # Standardize features (Mean=0, Var=1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(ticker_data)
        
        # Apply PCA to retain 99% of variance
        pca = PCA(n_components=0.99) 
        pca_result = pca.fit_transform(scaled_data)
        pca_features_dict[ticker] = pca_result
        
    avg_components = np.mean([v.shape[1] for v in pca_features_dict.values()])
    print(f"PCA reduced OHLCV features to an average of {avg_components:.1f} principal components per asset.")
    
    return asset_returns, benchmark_returns, betas, pca_features_dict

def perform_jarque_bera_tests(returns):
    """
    Performs the Jarque-Bera test for non-normality on each asset's returns.
    """
    print("\n--- Jarque-Bera Testing ---")
    results = []
    for col in returns.columns:
        stat, p_value = jarque_bera(returns[col])
        is_normal = p_value > 0.05
        results.append({
            'Asset': col,
            'JB_Statistic': stat,
            'p_value': p_value,
            'Normal': is_normal
        })
        
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df

def get_data():
    tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
        'BHARTIARTL.NS', 'SBIN.NS', 'INFY.NS', 'LICI.NS', 
        'ITC.NS', 'HINDUNILVR.NS'
    ]
    benchmark_ticker = '^NSEI' # Nifty 50 index
    start_date = '2021-01-01'
    end_date = '2024-06-30'
    
    asset_returns, benchmark_returns, betas, pca_features = fetch_and_preprocess_data(tickers, benchmark_ticker, start_date, end_date)
    jb_results = perform_jarque_bera_tests(asset_returns)
    
    return asset_returns, benchmark_returns, betas, pca_features, jb_results

if __name__ == '__main__':
    get_data()
    print("\nData pipeline completed successfully.")
