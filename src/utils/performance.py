"""
performance.py
==============
Return-series statistics shared by the backtest tearsheet and the
portfolio analytics module, so both report identical numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def downside_deviation(excess: pd.Series) -> float:
    """
    Target downside deviation (Sortino & Price 1994):
        DD = sqrt( mean( min(excess_t, 0)^2 ) )   averaged over ALL periods.
    """
    x = np.asarray(excess, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)))


def sortino_ratio(returns: pd.Series, rfr_daily: float = 0.0, periods: int = 252) -> float:
    excess = returns - rfr_daily
    dd = downside_deviation(excess)
    return float(excess.mean() / dd * np.sqrt(periods)) if dd > 0 else 0.0


def jensens_alpha_beta(
    returns: pd.Series, benchmark: pd.Series, rfr_daily: float = 0.0, periods: int = 252,
) -> tuple[float, float, float]:
    """
    OLS of excess portfolio returns on excess benchmark returns:
        (R_p - R_f) = alpha + beta (R_m - R_f) + e
    Returns (annualised alpha, beta, R^2).
    """
    common = returns.index.intersection(benchmark.dropna().index)
    if len(common) < 10:
        return 0.0, 1.0, 0.0
    y = returns.loc[common].values - rfr_daily
    x = benchmark.loc[common].values - rfr_daily
    vx = np.var(x)
    if vx == 0:
        return 0.0, 0.0, 0.0
    beta = np.cov(x, y, bias=True)[0, 1] / vx
    alpha = y.mean() - beta * x.mean()
    resid = y - alpha - beta * x
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
    return float(alpha * periods), float(beta), float(r2)


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """
    Year x month table of compounded monthly returns plus an 'Annual' column.
    Robust to histories shorter than 12 months.
    """
    if returns.empty:
        return pd.DataFrame(columns=MONTH_NAMES + ["Annual"])
    monthly = (1 + returns).groupby(
        [returns.index.year, returns.index.month]
    ).prod() - 1
    table = monthly.unstack(level=1)
    table = table.reindex(columns=range(1, 13))
    table.columns = MONTH_NAMES
    table.index.name = "Year"
    table["Annual"] = (1 + table[MONTH_NAMES].fillna(0)).prod(axis=1) - 1
    return table
