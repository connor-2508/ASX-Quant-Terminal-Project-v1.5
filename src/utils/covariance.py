"""
covariance.py
=============
Covariance estimators shared by the optimisers and risk models.

ledoit_wolf_cov : Ledoit & Wolf (2004) shrinkage towards a scaled identity
ewma_cov        : RiskMetrics exponentially-weighted covariance
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf_cov(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf (2004) "well-conditioned estimator for large-dimensional
    covariance matrices": shrink the sample covariance S towards
    F = mu * I, where mu = trace(S) / p.

        Sigma_hat = delta * F + (1 - delta) * S
        delta     = min(b^2, d^2) / d^2
        d^2       = ||S - mu I||_F^2 / p
        b^2       = (1/n^2) * sum_k ||x_k x_k' - S||_F^2 / p

    with x_k the *demeaned* observations and S the MLE (1/n) covariance.
    Matches ``sklearn.covariance.LedoitWolf`` (tested).

    Returns a (p x p) array in the same (per-period) units as the input.
    """
    X = np.asarray(returns, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    n, p = X.shape
    if n < 2:
        return np.eye(p) * (np.nanvar(X) if X.size else 1.0)
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    if d2 <= 0:
        return S
    # sum_k ||x_k x_k' - S||^2 computed without an explicit loop
    x2 = Xc ** 2
    b2_bar = (np.sum(x2.T @ x2) / n - np.sum(S ** 2)) / (n * p)
    b2 = min(b2_bar, d2)
    shrink = b2 / d2
    return shrink * mu * np.eye(p) + (1 - shrink) * S


def ewma_cov(returns: pd.DataFrame | np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    RiskMetrics EWMA covariance:  S_t = lam * S_{t-1} + (1 - lam) * r_t r_t'
    (zero-mean convention, seeded with the sample covariance of the first
    min(20, n) observations).
    """
    X = np.asarray(returns, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    n, p = X.shape
    seed_n = min(20, n)
    S = np.cov(X[:seed_n].T) if seed_n > 1 else np.outer(X[0], X[0])
    S = np.atleast_2d(S)
    for r in X[seed_n:]:
        S = lam * S + (1 - lam) * np.outer(r, r)
    return S
