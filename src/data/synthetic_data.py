"""
synthetic_data.py
=================
Synthetic OHLCV market data with a known data-generating process.

Used for:
    - the offline "Synthetic demo data" mode in the dashboard (useful when
      Yahoo Finance is rate-limiting or there is no network), and
    - the test-suite, where models are checked against planted ground truth
      (known betas, known regime switches, no true alpha).

The data are NOT real prices. The dashboard labels them clearly.

Model
-----
    r_m,t      market return with a two-state volatility regime
               (calm: 0.8% daily vol, stressed: 2.2% with a negative drift)
    r_i,t    = beta_i * r_m,t + gamma_s(i) * f_s,t + e_i,t
               with a sector factor f_s and idiosyncratic noise e_i.
    OHLC       built around the close with realistic intraday ranges;
    Volume     lognormal, higher for larger names and on high-vol days.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def generate_market_data(
    tickers: Optional[list[str]] = None,
    start: str = "2019-01-01",
    end: Optional[str] = None,
    n_days: Optional[int] = None,
    seed: int = 7,
    regimes: bool = True,
) -> dict:
    """
    Returns
    -------
    dict with keys
        close, open, high, low, volume : pd.DataFrame (dates x tickers)
        benchmark_returns              : pd.Series
        true_betas                     : pd.Series
        market_regime                  : pd.Series (0 = calm, 1 = stressed)
    """
    tickers = tickers or ["BHP.AX", "CBA.AX", "CSL.AX", "WES.AX",
                          "MQG.AX", "RIO.AX", "WOW.AX", "ANZ.AX"]
    rng = np.random.default_rng(seed)

    if n_days is not None:
        dates = pd.bdate_range(start, periods=n_days)
    else:
        dates = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    n = len(dates)
    k = len(tickers)

    # Market with a persistent two-state volatility regime
    regime = np.zeros(n, dtype=int)
    if regimes:
        p_stay = {0: 0.995, 1: 0.97}
        for t in range(1, n):
            stay = rng.random() < p_stay[regime[t - 1]]
            regime[t] = regime[t - 1] if stay else 1 - regime[t - 1]
    mu = np.where(regime == 0, 0.0006, -0.0015)
    sig = np.where(regime == 0, 0.008, 0.022)
    r_m = mu + sig * rng.standard_normal(n)

    betas = np.round(np.linspace(0.6, 1.4, k), 3)
    rng.shuffle(betas)
    n_sectors = max(1, min(4, k // 2))
    sector = np.arange(k) % n_sectors
    f_s = rng.normal(0, 0.006, (n, n_sectors))
    idio_vol = rng.uniform(0.008, 0.016, k)
    eps = rng.standard_normal((n, k)) * idio_vol
    rets = r_m[:, None] * betas[None, :] + f_s[:, sector] + eps
    rets = np.clip(rets, -0.5, 0.5)

    start_px = rng.uniform(5, 150, k)
    close = pd.DataFrame(
        start_px * np.exp(np.cumsum(np.log1p(rets), axis=0)),
        index=dates, columns=tickers,
    )

    gap = rng.normal(0, 0.003, (n, k))
    prev = close.shift(1).bfill().values
    opn = prev * (1 + gap)
    cl = close.values
    rng_hi = np.abs(rng.normal(0, 0.006, (n, k))) * (1 + 2 * regime[:, None])
    rng_lo = np.abs(rng.normal(0, 0.006, (n, k))) * (1 + 2 * regime[:, None])
    high = np.maximum(opn, cl) * (1 + rng_hi)
    low = np.minimum(opn, cl) * (1 - rng_lo)

    size = rng.uniform(13, 16, k)                 # log average share volume
    vol = np.exp(size[None, :] + 0.3 * rng.standard_normal((n, k)) + 0.5 * regime[:, None])

    frame = lambda a: pd.DataFrame(a, index=dates, columns=tickers)  # noqa: E731
    return {
        "close": close,
        "open": frame(opn),
        "high": frame(high),
        "low": frame(low),
        "volume": frame(np.round(vol)),
        "benchmark_returns": pd.Series(r_m, index=dates, name="Synthetic Market"),
        "true_betas": pd.Series(betas, index=tickers, name="beta"),
        "market_regime": pd.Series(regime, index=dates, name="regime"),
    }
