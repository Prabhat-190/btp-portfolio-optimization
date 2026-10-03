"""Long-only MVSK optimisation: Markowitz baselines + NSGA-III Pareto front."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from scipy.optimize import minimize as scipy_minimize
from sklearn.covariance import LedoitWolf

from config import MAX_WEIGHT, NSGA_GENERATIONS, NSGA_PARTITIONS, NSGA_SEED, N_PARETO_KEEP
from metrics import moments, sharpe_ratio


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_


def _as_mu_cov(
    expected_returns: pd.Series | np.ndarray,
    returns: pd.DataFrame,
    cov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = list(returns.columns)
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    if mu.shape[0] != len(names):
        raise ValueError("expected_returns length must match the return matrix.")
    sigma = cov if cov is not None else ledoit_wolf_cov(returns)
    return mu, sigma, names


def project_to_capped_simplex(weights: np.ndarray, max_weight: float = MAX_WEIGHT) -> np.ndarray:
    """Project each row onto {w >= 0, sum w = 1, w_i <= max_weight}."""
    w = np.asarray(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    if w.ndim == 1:
        w = w.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    n = w.shape[1]
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError("max_weight is too small to allow a fully invested book.")

    for i in range(w.shape[0]):
        x = w[i]
        if x.sum() <= 0:
            x = np.ones(n)
        for _ in range(32):
            x = np.clip(x, 0.0, None)
            total = x.sum()
            if total <= 0:
                x = np.ones(n) / n
                continue
            x = x / total
            overflow = x > max_weight + 1e-12
            if not np.any(overflow):
                break
            free = ~overflow
            excess = float((x[overflow] - max_weight).sum())
            x[overflow] = max_weight
            if np.any(free) and x[free].sum() > 0:
                x[free] += excess * x[free] / x[free].sum()
            else:
                x = np.clip(x, 0.0, max_weight)
                x = x / x.sum()
                break
        w[i] = x / x.sum()
    return w[0] if squeeze else w


class SimplexRepair(Repair):
    def __init__(self, max_weight: float = MAX_WEIGHT):
        super().__init__()
        self.max_weight = max_weight

    def _do(self, problem, X, **kwargs):
        return project_to_capped_simplex(X, self.max_weight)


class MVSKProblem(Problem):
    """max mean, min variance, max skew, min Pearson kurtosis."""

    def __init__(self, mu: np.ndarray, cov: np.ndarray, returns: np.ndarray, max_weight: float):
        self.mu = mu
        self.cov = cov
        self.returns = returns
        self.max_weight = max_weight
        super().__init__(
            n_var=len(mu),
            n_obj=4,
            n_ieq_constr=0,
            xl=np.zeros(len(mu)),
            xu=np.full(len(mu), max_weight),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        w = project_to_capped_simplex(x, self.max_weight)
        mean = w @ self.mu
        var = np.einsum("ij,jk,ik->i", w, self.cov, w)
        port_ts = self.returns @ w.T
        from metrics import pearson_kurtosis
        from scipy.stats import skew

        sk = skew(np.nan_to_num(port_ts, nan=0.0), axis=0, bias=False)
        kt = pearson_kurtosis(port_ts, axis=0)
        F = np.column_stack([-mean, var, -sk, kt])
        bad = ~np.isfinite(F).all(axis=1)
        if np.any(bad):
            F[bad] = np.array([0.0, 1.0, 10.0, 50.0])
        out["F"] = F


def optimize_nsga3(
    expected_returns: pd.Series | np.ndarray,
    returns: pd.DataFrame,
    cov: np.ndarray | None = None,
    max_weight: float = MAX_WEIGHT,
    n_portfolios: int = N_PARETO_KEEP,
    n_partitions: int = NSGA_PARTITIONS,
    n_gen: int = NSGA_GENERATIONS,
    seed: int = NSGA_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma, _ = _as_mu_cov(expected_returns, returns, cov)
    problem = MVSKProblem(mu, sigma, returns.values, max_weight)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
    algorithm = NSGA3(
        pop_size=max(len(ref_dirs), 60),
        ref_dirs=ref_dirs,
        repair=SimplexRepair(max_weight),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        res = minimize(
            problem,
            algorithm,
            termination=("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
    if res.X is None:
        raise RuntimeError("NSGA-III returned no feasible portfolios.")

    weights = project_to_capped_simplex(np.atleast_2d(res.X), max_weight)
    objectives = np.atleast_2d(res.F)
    finite = np.isfinite(objectives).all(axis=1) & np.isfinite(weights).all(axis=1)
    if not np.any(finite):
        raise RuntimeError("NSGA-III returned only invalid portfolios. Try more generations.")
    weights, objectives = weights[finite], objectives[finite]
    if len(weights) > n_portfolios:
        idx = np.linspace(0, len(weights) - 1, n_portfolios, dtype=int)
        weights, objectives = weights[idx], objectives[idx]
    return weights, objectives


def _slsqp(n: int, objective, jac, max_weight: float) -> np.ndarray:
    bounds = tuple((0.0, max_weight) for _ in range(n))
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.ones(n) / n
    res = scipy_minimize(
        objective,
        w0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 400, "ftol": 1e-14},
    )
    return project_to_capped_simplex(np.array(res.x, copy=True), max_weight)


def markowitz_min_variance(returns: pd.DataFrame, cov: np.ndarray | None = None, max_weight: float = MAX_WEIGHT) -> np.ndarray:
    sigma = cov if cov is not None else ledoit_wolf_cov(returns)

    def obj(w):
        return float(w @ sigma @ w) * 1e4

    def jac(w):
        return 2.0 * (sigma @ w) * 1e4

    return _slsqp(sigma.shape[0], obj, jac, max_weight)


def markowitz_max_sharpe(
    expected_returns: pd.Series | np.ndarray,
    returns: pd.DataFrame,
    cov: np.ndarray | None = None,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    mu, sigma, _ = _as_mu_cov(expected_returns, returns, cov)

    def neg_sharpe(w):
        vol = np.sqrt(float(w @ sigma @ w))
        return -sharpe_ratio(float(w @ mu), vol)

    return _slsqp(len(mu), neg_sharpe, "2-point", max_weight)


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def evaluate_weights(weights: np.ndarray, returns: pd.DataFrame, label: str) -> dict:
    stats = moments(weights, returns)
    stats["sharpe"] = sharpe_ratio(stats["mean"], stats["vol"])
    stats["largest_weight"] = float(np.max(weights))
    stats["n_names"] = int(np.sum(weights > 0.01))
    stats["portfolio"] = label
    return stats


def pareto_frame(
    weights: np.ndarray,
    returns: pd.DataFrame,
    expected_returns: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    mu = None
    if expected_returns is not None:
        mu = np.asarray(expected_returns, dtype=float)
    for i, w in enumerate(weights):
        row = evaluate_weights(w, returns, f"nsga3_{i+1:02d}")
        if mu is not None:
            row["exp_return"] = float(w @ mu)
        for t, wi in zip(returns.columns, w):
            row[t] = float(wi)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_sharpe(pareto: pd.DataFrame) -> pd.Series:
    valid = pareto["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        raise RuntimeError("No finite Sharpe ratios on the Pareto front.")
    return pareto.loc[valid.idxmax()]
