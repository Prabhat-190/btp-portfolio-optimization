import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize as scipy_minimize

class PortfolioOptimizationProblem(Problem):
    def __init__(self, expected_alphas, cov_matrix, returns_data, max_weight=0.57):
        """
        4-objective Portfolio Optimization Problem based on MIT Thesis parameters.
        Objectives: Maximize Expected Alpha, Minimize Variance, Maximize Skewness, Minimize Kurtosis
        """
        self.n_assets = len(expected_alphas)
        self.expected_alphas = expected_alphas
        self.cov_matrix = cov_matrix
        self.returns_data = returns_data # Shape: (T, n_assets)
        self.max_weight = max_weight
        
        super().__init__(n_var=self.n_assets,
                         n_obj=4,
                         n_ieq_constr=2, 
                         xl=np.zeros(self.n_assets),
                         xu=np.ones(self.n_assets) * self.max_weight)

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        
        # 1. Inequality Constraints for sum(w) == 1
        sum_w = np.sum(x, axis=1)
        g1 = sum_w - 1.0 - 1e-4  
        g2 = 1.0 - sum_w - 1e-4  
        
        # 2. Vectorised moment evaluation
        # Expected Alpha (MIT Thesis objective)
        f1 = -np.dot(x, self.expected_alphas) 
        
        # Portfolio Variance
        f2 = np.sum(x * np.dot(x, self.cov_matrix), axis=1)
        
        # Skewness and Kurtosis
        port_returns_ts = np.dot(self.returns_data, x.T)
        
        port_skew = skew(port_returns_ts, axis=0)
        port_kurt = kurtosis(port_returns_ts, axis=0)
        
        f3 = -port_skew 
        f4 = port_kurt  
        
        out["F"] = np.column_stack([f1, f2, f3, f4])
        out["G"] = np.column_stack([g1, g2])

def optimize_nsga3(expected_alphas, cov_matrix, returns_data, max_weight=0.57, n_portfolios=24):
    """
    Runs the NSGA-III algorithm to find non-dominated portfolios optimizing Alpha.
    """
    print("\n--- Running 4-Objective NSGA-III Optimization (Target: Alpha) ---")
    problem = PortfolioOptimizationProblem(expected_alphas, cov_matrix, returns_data, max_weight)
    
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)
    algorithm = NSGA3(pop_size=200, ref_dirs=ref_dirs)
    
    res = minimize(problem,
                   algorithm,
                   seed=42,
                   termination=('n_gen', 300),
                   verbose=False)
    
    weights = res.X
    objectives = res.F
    
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    if len(weights) > n_portfolios:
        indices = np.linspace(0, len(weights) - 1, n_portfolios, dtype=int)
        weights = weights[indices]
        objectives = objectives[indices]
        
    print(f"Generated {len(weights)} non-dominated portfolios.")
    return weights, objectives

def markowitz_max_sharpe(expected_alphas, cov_matrix, max_weight=0.57):
    """
    Baseline: Markowitz Mean-Variance Max Sharpe Ratio optimization 
    (using Expected Alpha instead of Expected Return for direct comparison).
    """
    print("\n--- Running Baseline Markowitz Max-Sharpe (Using Alpha) ---")
    n = len(expected_alphas)
    
    def negative_sharpe(w):
        port_alpha = np.dot(w, expected_alphas)
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        return -port_alpha / np.sqrt(port_var)
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, max_weight) for _ in range(n))
    init_guess = np.ones(n) / n
    
    res = scipy_minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    w_opt = res.x
    port_alpha = np.dot(w_opt, expected_alphas)
    port_var = np.dot(w_opt.T, np.dot(cov_matrix, w_opt))
    
    return w_opt, port_alpha, port_var
