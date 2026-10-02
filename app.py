import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from data_pipeline import get_data
from ml_models import train_and_predict_alpha
from portfolio_optimization import optimize_nsga3, markowitz_max_sharpe

st.set_page_config(page_title="Portfolio Optimization BTP", layout="wide")

st.title("📈 Hybrid ML/DL Portfolio Optimization")
st.markdown("### BTP Project by Prabhat Kumar (IIT KGP)")

st.write("This application implements a quantitative pipeline based on the MIT Thesis using PCA, CNN-LSTM for Expected Alpha prediction, and NSGA-III for multi-objective optimization (MVSK).")

@st.cache_data
def load_existing_results():
    if os.path.exists("nsga3_24_portfolios_alpha.csv"):
        return pd.read_csv("nsga3_24_portfolios_alpha.csv")
    return None

results_df = load_existing_results()

if results_df is not None:
    st.success("Loaded previously generated Pareto-optimal portfolios.")
    
    st.subheader("24 Non-Dominated Portfolios (Pareto Front)")
    st.dataframe(results_df.style.highlight_max(axis=0))
    
    # Identify Min Variance Portfolio
    min_var_idx = results_df['Variance'].idxmin()
    best_port = results_df.iloc[min_var_idx]
    
    st.subheader("Selected Portfolio (Minimum Variance)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Alpha", f"{best_port['Exp_Alpha']*100:.4f}%")
    col2.metric("Volatility", f"{np.sqrt(best_port['Variance'])*100:.4f}%")
    col3.metric("Sharpe Ratio", f"{best_port['Exp_Alpha']/np.sqrt(best_port['Variance']):.4f}")
    
    st.subheader("Asset Allocation Weights")
    weights = best_port.drop(['Exp_Alpha', 'Variance', 'Skewness', 'Kurtosis'])
    fig, ax = plt.subplots(figsize=(8,8))
    # Filter out very small weights for cleaner pie chart
    filtered_weights = weights[weights > 0.01]
    ax.pie(filtered_weights, labels=filtered_weights.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

st.divider()

st.subheader("Run Full Pipeline (Warning: Takes time)")
if st.button("Execute Pipeline"):
    with st.spinner("Fetching Data and running Jarque-Bera Tests..."):
        asset_returns, benchmark_returns, betas, pca_features, jb_results = get_data()
        st.success("Data Pipeline Complete.")
        
    with st.spinner("Training CNN-LSTM Models with Stochastic Gradient Descent..."):
        expected_alphas_series = train_and_predict_alpha(asset_returns, benchmark_returns, betas, pca_features, epochs=20)
        expected_alphas = expected_alphas_series.values
        cov_matrix = asset_returns.cov().values
        st.success("ML/DL Predictive Signals Generated.")
        
    with st.spinner("Running NSGA-III 4-Objective Optimization..."):
        weights_nsga3, obj_nsga3 = optimize_nsga3(expected_alphas, cov_matrix, asset_returns.values, max_weight=0.57, n_portfolios=24)
        
        portfolios_df = pd.DataFrame(weights_nsga3, columns=asset_returns.columns)
        portfolios_df['Exp_Alpha'] = -obj_nsga3[:, 0]
        portfolios_df['Variance'] = obj_nsga3[:, 1]
        portfolios_df['Skewness'] = -obj_nsga3[:, 2]
        portfolios_df['Kurtosis'] = obj_nsga3[:, 3]
        portfolios_df.to_csv('nsga3_24_portfolios_alpha.csv', index=False)
        st.success("Optimization Complete! Refresh the page to see the new results.")
        st.rerun()
