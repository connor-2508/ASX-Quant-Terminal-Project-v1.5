"""
Shared pytest fixtures.

All tests run offline on synthetic data with a known data-generating
process (src/data/synthetic_data.py), so results are deterministic and
model outputs can be checked against ground truth (planted betas,
planted regime switches, no true alpha).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.synthetic_data import generate_market_data  # noqa: E402

TICKERS = ["BHP.AX", "CBA.AX", "CSL.AX", "WES.AX", "MQG.AX", "RIO.AX", "WOW.AX", "ANZ.AX"]


@pytest.fixture(scope="session")
def market():
    return generate_market_data(TICKERS, n_days=1500, seed=7)


@pytest.fixture(scope="session")
def prices(market):
    return market["close"]


@pytest.fixture(scope="session")
def benchmark(market):
    return market["benchmark_returns"]


@pytest.fixture(scope="session")
def returns(prices):
    return prices.pct_change().dropna()


def perturb_after(prices: pd.DataFrame, cutoff_idx: int, seed: int = 0) -> pd.DataFrame:
    """Replace every price AFTER row ``cutoff_idx`` with a different random path."""
    rng = np.random.default_rng(seed)
    out = prices.copy()
    n_after = len(prices) - cutoff_idx - 1
    shock = np.exp(np.cumsum(rng.normal(0, 0.03, (n_after, prices.shape[1])), axis=0))
    out.iloc[cutoff_idx + 1:] = prices.iloc[cutoff_idx + 1:].values * shock
    return out
