"""
calendar.py
===========
Rebalance-date helpers.

Why this exists
---------------
``df.resample("ME").last().index`` returns *calendar* period-end labels
(e.g. 2023-12-31, a Sunday). Matching those labels against a trading-day
index silently skips every month whose last calendar day is a weekend or
holiday, and ``resample("W")`` labels (Sundays) never match at all.
These helpers return the *last actual trading day* in each period instead.
"""

from __future__ import annotations

import pandas as pd

_FREQ_ALIASES = {
    "daily": None,
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "annual": "Y",
    "yearly": "Y",
    # pandas-style aliases are accepted too
    "W": "W", "ME": "M", "M": "M", "QE": "Q", "Q": "Q", "YE": "Y", "Y": "Y",
}


def period_end_mask(index: pd.DatetimeIndex, freq: str = "monthly") -> pd.Series:
    """
    Boolean Series aligned to ``index``: True on the last trading day of
    each period (week / month / quarter / year). ``freq='daily'`` → all True.
    """
    idx = pd.DatetimeIndex(index)
    if freq not in _FREQ_ALIASES:
        raise ValueError(f"Unknown rebalance frequency: {freq!r}")
    period = _FREQ_ALIASES[freq]
    if period is None or len(idx) == 0:
        return pd.Series(True, index=idx)
    p = idx.to_period(period)
    # last observation of each period: next observation is in a different period
    is_last = p != pd.Series(p).shift(-1).values
    return pd.Series(is_last, index=idx)


def period_end_dates(index: pd.DatetimeIndex, freq: str = "monthly") -> set:
    """Set of trading dates that are the last trading day of each period."""
    mask = period_end_mask(index, freq)
    return set(mask.index[mask.values])


def contiguous_segments(labels: pd.Series) -> list[tuple]:
    """
    Split a label series into runs of identical consecutive values.
    Returns [(start_date, end_date, label), ...] - used to shade regime
    charts correctly (one rectangle per run, not one per label).
    """
    s = labels.dropna()
    if s.empty:
        return []
    change = s.ne(s.shift()).cumsum()
    out = []
    for _, run in s.groupby(change):
        out.append((run.index[0], run.index[-1], run.iloc[0]))
    return out
