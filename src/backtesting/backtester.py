"""
backtester.py
=============
Professional vectorised backtesting engine for ASX equity strategies.

Design Philosophy
-----------------
    Vectorised first  : All operations use NumPy/Pandas broadcasting.
                        No Python loops over time steps.
    Realistic costs   : Slippage, commission, and market impact modelled.
    No look-ahead     : Strict signal-to-execution lag enforcement.
    Reproducible      : Fixed random seeds, deterministic outputs.
    Extensible        : Clean base class for custom strategies.

Architecture
------------
    BacktestConfig    : All parameters in one place
    Portfolio         : Tracks positions, NAV, cash, turnover
    Backtester        : Orchestrates signal → portfolio → metrics
    BacktestResult    : Clean result container

Usage
-----
    from src.backtesting.backtester import Backtester, BacktestConfig

    config = BacktestConfig(
        initial_capital=1_000_000,
        commission=0.001,
        slippage=0.0005,
    )
    bt = Backtester(prices, signals, config)
    result = bt.run()
    print(result.metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd



# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """
    All backtesting parameters in one place.

    Parameters
    ----------
    initial_capital   : float   Starting portfolio value in AUD.
    commission        : float   One-way commission rate (e.g. 0.001 = 0.1%).
    slippage          : float   One-way slippage per trade (e.g. 0.0005 = 0.05%).
    market_impact     : float   Square-root market impact coefficient.
    signal_lag        : int     Trading days between the close at which a signal is
                                observed and the close at which it is traded.
                                1 = signal from close t-1 is traded at close t and
                                first earns the return from t to t+1 (no look-ahead).
    rebalance_freq    : str     'signal'  : trade only when target weights change
                                            (positions drift with prices in between)
                                'daily'   : trade back to target every day
                                'weekly'|'monthly'|'quarterly' : trade on the last
                                            trading day of each period only.
    max_position_size : float   Maximum weight per asset (e.g. 0.20 = 20%).
    min_position_size : float   Minimum weight to hold (below this = close).
    max_leverage      : float   Maximum gross leverage (1.0 = long only).
    risk_free_rate    : float   Annualised risk-free rate for Sharpe calculation.
    benchmark_ticker  : str     Benchmark label for information ratio.
    allow_short       : bool    Whether short positions are permitted.
    """
    initial_capital: float = 1_000_000.0
    commission: float = 0.001          # 0.1% one-way
    slippage: float = 0.0005           # 0.05% one-way
    market_impact: float = 0.0         # disabled by default
    signal_lag: int = 1                # execute next day
    rebalance_freq: str = "signal"
    max_position_size: float = 1.0
    min_position_size: float = 0.0
    max_leverage: float = 1.0
    risk_free_rate: float = 0.0435
    benchmark_ticker: str = "STW.AX"   # total-return ASX 200 proxy
    allow_short: bool = False


# ─────────────────────────────────────────────
#  Portfolio state tracker
# ─────────────────────────────────────────────

class Portfolio:
    """
    Tracks portfolio state through the backtest.
    Handles cash, positions, NAV, and trade records.

    Parameters
    ----------
    initial_capital : float
    tickers         : list of str
    dates           : pd.DatetimeIndex
    """

    def __init__(
        self,
        initial_capital: float,
        tickers: list[str],
        dates: pd.DatetimeIndex,
    ) -> None:
        self.n = len(dates)
        self.tickers = tickers
        self.dates = dates
        self.n_assets = len(tickers)

        # NAV history
        self.nav = np.full(self.n, initial_capital)
        self.cash = np.full(self.n, initial_capital)

        # Position matrix: rows=dates, cols=assets (in $ value)
        self.positions = np.zeros((self.n, self.n_assets))

        # Weight matrix
        self.weights = np.zeros((self.n, self.n_assets))

        # Trade records
        self.trades: list[dict] = []

        # Turnover per day
        self.turnover = np.zeros(self.n)

        # Transaction costs per day
        self.costs = np.zeros(self.n)

    def to_dataframe(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Export all portfolio state as DataFrames."""
        return {
            "nav": pd.Series(self.nav, index=self.dates, name="NAV"),
            "cash": pd.Series(self.cash, index=self.dates, name="Cash"),
            "positions": pd.DataFrame(
                self.positions, index=self.dates, columns=self.tickers
            ),
            "weights": pd.DataFrame(
                self.weights, index=self.dates, columns=self.tickers
            ),
            "turnover": pd.Series(self.turnover, index=self.dates, name="Turnover"),
            "costs": pd.Series(self.costs, index=self.dates, name="Costs"),
        }


# ─────────────────────────────────────────────
#  Backtest Result
# ─────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    nav: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.Series = field(default_factory=pd.Series)
    costs: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: Optional[pd.Series] = None
    config: Optional[BacktestConfig] = None
    strategy_name: str = "Strategy"


# ─────────────────────────────────────────────
#  Core Backtester
# ─────────────────────────────────────────────

class Backtester:
    """
    Vectorised backtesting engine.

    Parameters
    ----------
    prices          : pd.DataFrame
        Adjusted close prices. Rows=dates, cols=tickers.
    signals         : pd.DataFrame or pd.Series
        Trading signals. Same index as prices.
        Values: +1 (long), 0 (flat), -1 (short) OR continuous weights.
        If DataFrame: columns = tickers (individual asset signals).
        If Series: single signal applied to equal-weight portfolio.
    config          : BacktestConfig
    benchmark       : pd.Series, optional
        Benchmark return series for relative metrics.
    strategy_name   : str
        Label for display.
    weight_func     : Callable, optional
        Custom function to convert signals → weights.
        Signature: weight_func(signals, prices, config) → pd.DataFrame
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame | pd.Series,
        config: Optional[BacktestConfig] = None,
        benchmark: Optional[pd.Series] = None,
        strategy_name: str = "Strategy",
        weight_func: Optional[Callable] = None,
    ) -> None:
        self.prices = prices.copy()
        self.config = config or BacktestConfig()
        self.benchmark = benchmark
        self.strategy_name = strategy_name
        self.weight_func = weight_func or self._default_weight_func

        # Align signals to prices
        if isinstance(signals, pd.Series):
            signals = signals.to_frame()
            # Broadcast single signal to all tickers
            if signals.columns[0] not in prices.columns:
                for t in prices.columns:
                    signals[t] = signals.iloc[:, 0]
                signals = signals.drop(columns=[signals.columns[0]])

        self.signals = signals.reindex(prices.index).fillna(0)

        # Enforce signal lag (no look-ahead)
        self.signals = self.signals.shift(self.config.signal_lag).fillna(0)

    # ──────────────────────────────────────
    #  Weight generation
    # ──────────────────────────────────────

    def _default_weight_func(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: BacktestConfig,
    ) -> pd.DataFrame:
        """
        Convert signals to portfolio weights.
        +1 → equal positive weight, -1 → equal negative weight, 0 → flat.
        Normalises so long weights sum to 1 (long-only) or ±1 (long/short).
        """
        if not config.allow_short:
            raw = signals.clip(lower=0)
        else:
            raw = signals.copy()

        row_sum = raw.abs().sum(axis=1).replace(0, np.nan)
        weights = raw.div(row_sum, axis=0).fillna(0)

        # Apply position size limits
        weights = weights.clip(
            lower=-config.max_position_size,
            upper=config.max_position_size,
        )

        # Apply minimum position size
        if config.min_position_size > 0:
            too_small = weights.abs() < config.min_position_size
            weights[too_small] = 0

        return weights

    def _rebalance_mask(self, target: pd.DataFrame) -> np.ndarray:
        """
        Boolean array: True on days the portfolio is traded back to target.
        Between rebalances positions drift with prices (no implicit,
        cost-free daily rebalancing).
        """
        freq = self.config.rebalance_freq
        n = len(target)
        if freq == "daily":
            return np.ones(n, dtype=bool)
        if freq == "signal":
            arr = target.values
            changed = np.zeros(n, dtype=bool)
            changed[0] = np.abs(arr[0]).sum() > 0
            if n > 1:
                changed[1:] = (np.abs(np.diff(arr, axis=0)) > 1e-12).any(axis=1)
            return changed
        from src.utils.calendar import period_end_mask
        return period_end_mask(target.index, freq).values

    # ──────────────────────────────────────
    #  Transaction cost model
    # ──────────────────────────────────────

    def _compute_transaction_costs(
        self,
        prev_weights: np.ndarray,
        new_weights: np.ndarray,
        nav: float,
        prices_row: np.ndarray,
        adv: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """
        Compute total transaction costs for a rebalancing event.

        Cost components:
            1. Commission: fixed % of traded value
            2. Bid-ask spread (slippage): fixed % of traded value
            3. Market impact: sqrt model (optional)

        Returns
        -------
        (total_cost, turnover)
            total_cost : AUD cost of the rebalancing
            turnover   : fraction of NAV traded
        """
        weight_change = np.abs(new_weights - prev_weights)
        turnover = float(weight_change.sum() / 2)

        traded_value = nav * weight_change
        commission_cost = (traded_value * self.config.commission).sum()
        slippage_cost = (traded_value * self.config.slippage).sum()

        # Square-root market impact (optional)
        impact_cost = 0.0
        if self.config.market_impact > 0 and adv is not None:
            adv_safe = np.where(adv > 0, adv, 1e6)
            participation = traded_value / adv_safe
            impact = self.config.market_impact * np.sqrt(participation) * traded_value
            impact_cost = float(impact.sum())

        total_cost = float(commission_cost + slippage_cost + impact_cost)
        return total_cost, turnover

    # ──────────────────────────────────────
    #  Main run loop
    # ──────────────────────────────────────

    def run(self) -> BacktestResult:
        """
        Execute the backtest simulation.

        Timing convention (daily close-to-close):
            target_w[t]  is decided at the close of day t using signals that
                         were already lagged by ``signal_lag`` days;
            the trade is executed at the close of t (costs charged then);
            those weights earn the asset returns from t to t+1.
        Between rebalances the weights drift with relative price moves.

        Returns
        -------
        BacktestResult  containing NAV, returns, positions, metrics.
        The reported period starts on the first day capital is deployed, so
        the signal warm-up period (e.g. a 12-month lookback) does not dilute
        CAGR / Sharpe with zero-return days.
        """
        common = self.prices.index.intersection(self.signals.index)
        prices = self.prices.loc[common].copy()
        signals = self.signals.loc[common].copy()

        tickers = [t for t in signals.columns if t in prices.columns]
        if not tickers:
            raise ValueError("No overlapping tickers between signals and prices.")

        prices = prices[tickers]
        signals = signals[tickers]
        dates = prices.index
        n = len(dates)
        n_assets = len(tickers)

        target = self.weight_func(signals, prices, self.config)[tickers].fillna(0.0)
        rebalance = self._rebalance_mask(target)
        target_arr = target.values

        # Returns are only defined between two valid prices; a missing price
        # contributes a 0 return (position is held flat over the gap).
        daily_returns = prices.pct_change(fill_method=None).fillna(0.0).values

        portfolio = Portfolio(self.config.initial_capital, tickers, dates)

        nav = float(self.config.initial_capital)
        w = np.zeros(n_assets)          # weights held over (t-1, t]
        trades = []

        for t in range(n):
            # 1) P&L on positions held from the previous close
            if t > 0:
                r = daily_returns[t]
                port_ret = float(w @ r)
                nav *= (1.0 + port_ret)
                denom = 1.0 + port_ret
                w = w * (1.0 + r) / denom if abs(denom) > 1e-12 else np.zeros(n_assets)

            # 2) Trade at today's close if this is a rebalance day
            if rebalance[t]:
                new_w = target_arr[t]
                cost, turnover = self._compute_transaction_costs(
                    w, new_w, nav, prices.values[t],
                )
                if turnover > 0:
                    nav -= cost
                    trades.append({
                        "date": dates[t],
                        "turnover": turnover,
                        "cost": cost,
                        "n_positions": int((np.abs(new_w) > 1e-9).sum()),
                    })
                portfolio.turnover[t] = turnover
                portfolio.costs[t] = cost
                w = new_w.copy()

            portfolio.nav[t] = nav
            portfolio.weights[t] = w
            portfolio.positions[t] = nav * w
            portfolio.cash[t] = nav * (1.0 - w.sum())

        state = portfolio.to_dataframe()

        # Trim the warm-up period before the first trade
        invested = np.abs(portfolio.weights).sum(axis=1) > 0
        first = int(np.argmax(invested)) if invested.any() else n - 1
        sl = slice(first, n)
        nav_series = state["nav"].iloc[sl]
        returns = nav_series.pct_change().dropna()

        result = BacktestResult(
            nav=nav_series,
            returns=returns,
            positions=state["positions"].iloc[sl],
            weights=state["weights"].iloc[sl],
            turnover=state["turnover"].iloc[sl],
            costs=state["costs"].iloc[sl],
            trades=pd.DataFrame(trades),
            benchmark_returns=self.benchmark,
            config=self.config,
            strategy_name=self.strategy_name,
        )

        if len(returns) < 2:
            # Nothing was ever traded – return an empty, clearly-flagged result
            result.metrics = pd.DataFrame(columns=["Metric", "Value"])
            return result

        from src.backtesting.performance_metrics import PerformanceMetrics
        pm = PerformanceMetrics(result)
        result.metrics = pm.compute_all()
        result.monthly_returns = pm.monthly_returns()
        result.drawdown = pm.drawdown_series()

        return result

    # ──────────────────────────────────────
    #  Walk-forward backtesting
    # ──────────────────────────────────────

    def run_walk_forward(
        self,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int = 21,
    ) -> list[BacktestResult]:
        """
        Rolling out-of-sample windows over a *pre-computed* signal.

        Nothing is re-fitted here: this splits the history into consecutive
        test windows (skipping ``train_days`` at the start) so you can check
        performance stability through time. For genuine walk-forward
        parameter re-optimisation see
        ``src.optimisation.parameter_optimisation.WalkForwardOptimisation``.

        Parameters
        ----------
        train_days : int   In-sample window.
        test_days  : int   Out-of-sample window.
        step_days  : int   How far to step forward each iteration.

        Returns
        -------
        list of BacktestResult  (one per test window)
        """
        common = self.prices.index.intersection(self.signals.index)
        n = len(common)
        results = []
        start = 0

        while start + train_days + test_days <= n:
            test_start = start + train_days
            test_end = min(test_start + test_days, n)

            test_prices = self.prices.iloc[test_start:test_end]
            test_signals = self.signals.iloc[test_start:test_end]

            # self.signals are already lagged – do not lag them twice
            from dataclasses import replace
            bt = Backtester(
                prices=test_prices,
                signals=test_signals,
                config=replace(self.config, signal_lag=0),
                benchmark=self.benchmark,
                strategy_name=f"{self.strategy_name}_WF_{start}",
                weight_func=self.weight_func,
            )
            try:
                r = bt.run()
                results.append(r)
            except Exception as e:
                print(f"  [WF] Window {start} failed: {e}")

            start += step_days

        return results

    def stitch_walk_forward(
        self, results: list[BacktestResult]
    ) -> BacktestResult:
        """
        Combine walk-forward windows into a single continuous equity curve.
        Each window is chain-linked from the previous ending NAV.

        Parameters
        ----------
        results : list   From run_walk_forward().

        Returns
        -------
        BacktestResult  Combined out-of-sample backtest.
        """
        if not results:
            raise ValueError("No walk-forward results to stitch.")

        nav_parts = []
        base_nav = self.config.initial_capital

        for r in results:
            scaled = r.nav / r.nav.iloc[0] * base_nav
            nav_parts.append(scaled)
            base_nav = float(scaled.iloc[-1])

        combined_nav = pd.concat(nav_parts)
        combined_nav = combined_nav[~combined_nav.index.duplicated(keep="last")]
        combined_returns = combined_nav.pct_change().dropna()

        combined = BacktestResult(
            nav=combined_nav,
            returns=combined_returns,
            benchmark_returns=self.benchmark,
            config=self.config,
            strategy_name=f"{self.strategy_name}_WalkForward",
        )

        from src.backtesting.performance_metrics import PerformanceMetrics
        pm = PerformanceMetrics(combined)
        combined.metrics = pm.compute_all()
        combined.monthly_returns = pm.monthly_returns()
        combined.drawdown = pm.drawdown_series()

        return combined
