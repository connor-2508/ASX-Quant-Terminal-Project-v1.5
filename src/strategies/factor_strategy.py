"""
factor_strategy.py  /  regime_adaptive_strategy.py
===================================================
Factor-based and regime-adaptive strategies for ASX equities.

Strategies Implemented
----------------------
FactorStrategy          : Rank and trade on combined factor scores
RegimeAdaptiveStrategy  : Switch strategy based on detected market regime
FactorTimingStrategy    : Time factor exposure based on macro conditions

Usage
-----
    from src.strategies.factor_strategy import FactorStrategy
    from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.calendar import period_end_dates



# ═══════════════════════════════════════════════
#  FACTOR STRATEGY
# ═══════════════════════════════════════════════

class FactorStrategy:
    """
    Multi-factor long-only or long/short strategy.

    Combines momentum, value, and quality factor scores into
    a composite ranking. Long top-ranked stocks, optionally
    short bottom-ranked stocks.

    Parameters
    ----------
    universe_prices  : pd.DataFrame   Close prices.
    factor_weights   : dict           {'momentum': 0.4, 'value': 0.3, 'quality': 0.3}
    n_long           : int            Number of long positions.
    n_short          : int            Number of short positions.
    rebalance_freq   : str            'monthly' | 'quarterly'.
    min_score        : float          Minimum composite score to enter long.
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        factor_weights: Optional[dict] = None,
        n_long: int = 10,
        n_short: int = 0,
        rebalance_freq: str = "monthly",
        min_score: float = 0.0,
    ) -> None:
        self.prices = universe_prices.copy()
        self.factor_weights = factor_weights or {
            "momentum": 0.40,
            "value": 0.30,
            "quality": 0.30,
        }
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_freq = rebalance_freq
        self.min_score = min_score

    def _compute_composite_score(self) -> pd.DataFrame:
        """Compute weighted composite factor score."""
        from src.factor_models.momentum_factor import (
            MomentumFactor, ValueFactor, QualityFactor
        )

        factors = {}
        if "momentum" in self.factor_weights:
            mom = MomentumFactor(self.prices)
            factors["momentum"] = mom.compute_signals(zscore=True)

        if "value" in self.factor_weights:
            val = ValueFactor(self.prices)
            factors["value"] = val.compute_signals(zscore=True)

        if "quality" in self.factor_weights:
            qual = QualityFactor(self.prices)
            factors["quality"] = qual.compute_signals(zscore=True)

        if not factors:
            return pd.DataFrame()

        composite = None
        for fname, fdf in factors.items():
            w = self.factor_weights.get(fname, 0)
            if composite is None:
                composite = fdf * w
            else:
                common = composite.index.intersection(fdf.index)
                composite = composite.loc[common] + fdf.loc[common] * w

        return composite

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate factor strategy signals.

        Returns
        -------
        pd.DataFrame  Position weights.
        """
        score = self._compute_composite_score()
        if score is None or score.empty:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=score.index, columns=score.columns)

        rebalance_dates = period_end_dates(score.index, self.rebalance_freq)

        last_signal = pd.Series(0.0, index=score.columns)

        for date in score.index:
            if date in rebalance_dates:
                row = score.loc[date].dropna()
                n_long = min(self.n_long, len(row) - self.n_short)
                if n_long < 1:
                    last_signal = pd.Series(0.0, index=score.columns)
                    signals.loc[date] = last_signal
                    continue

                new_signal = pd.Series(0.0, index=score.columns)

                # Long: top N by composite score above min_score
                longs = row.nlargest(n_long)
                longs = longs[longs >= self.min_score]
                if not longs.empty:
                    new_signal[longs.index] = 1.0 / len(longs)

                # Short: bottom N by composite score
                if self.n_short > 0:
                    shorts = row.nsmallest(self.n_short)
                    if not shorts.empty:
                        new_signal[shorts.index] = -1.0 / len(shorts)

                last_signal = new_signal

            signals.loc[date] = last_signal

        return signals

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run factor strategy backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            allow_short=self.n_short > 0,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="Factor_Strategy",
        )
        return bt.run()


# ═══════════════════════════════════════════════
#  REGIME ADAPTIVE STRATEGY
# ═══════════════════════════════════════════════

class RegimeAdaptiveStrategy:
    """
    Regime-adaptive strategy that switches between sub-strategies
    based on detected market regime.

    Concept
    -------
    Different strategies work in different regimes:
        Bull markets   → Momentum performs best
        Bear markets   → Move to defensive / cash / low-vol
        Sideways       → Mean reversion / value works

    Parameters
    ----------
    universe_prices : pd.DataFrame   Close prices.
    market_prices   : pd.Series      Market index for regime detection.
    regime_map      : dict           {regime_label: strategy_class_or_signals}
    n_positions     : int            Number of positions per regime.
    min_regime_days : int            A new regime must persist this many consecutive
                                     trading days before the strategy switches to it.
                                     The raw MA-crossover regime flickers around
                                     crossovers; each flip replaces the whole portfolio,
                                     so confirmation cuts turnover and costs. 1 = no filter.
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        market_prices: pd.Series,
        n_positions: int = 10,
        use_hmm: bool = False,
        min_regime_days: int = 5,
    ) -> None:
        self.prices = universe_prices.copy()
        self.market = market_prices.dropna()
        self.n_positions = n_positions
        self.use_hmm = use_hmm
        self.min_regime_days = max(1, int(min_regime_days))

    @staticmethod
    def confirm_regimes(raw: pd.Series, min_days: int) -> pd.Series:
        """
        Persistence filter: switch to a new regime only after it has been
        observed for ``min_days`` consecutive days. Uses past values only,
        so it adds a confirmation lag but no look-ahead.
        """
        if min_days <= 1 or raw.empty:
            return raw
        vals = raw.astype(str).values
        out = np.empty(len(vals), dtype=object)
        current = vals[0]
        run_label, run_len = vals[0], 0
        for i, v in enumerate(vals):
            if v == run_label:
                run_len += 1
            else:
                run_label, run_len = v, 1
            if run_label != current and run_len >= min_days:
                current = run_label
            out[i] = current
        return pd.Series(out, index=raw.index, name=raw.name)

    def _detect_regimes(self) -> pd.Series:
        """Detect market regimes using TrendRegimeDetector."""
        from src.regime_detection.volatility_regimes import TrendRegimeDetector

        trd = TrendRegimeDetector(
            prices=self.market,
            returns=self.market.pct_change().dropna(),
        )
        regimes = trd.moving_average_regime()
        regimes = regimes.reindex(self.prices.index).ffill().fillna("Sideways")
        return self.confirm_regimes(regimes, self.min_regime_days)

    def _bull_signals(self) -> pd.DataFrame:
        """Momentum signals for bull regime."""
        from src.strategies.momentum_strategy import CrossSectionalMomentum
        strategy = CrossSectionalMomentum(
            self.prices, n_long=self.n_positions, n_short=0,
            rebalance_freq="monthly",
        )
        return strategy.generate_signals()

    def _bear_signals(self) -> pd.DataFrame:
        """Defensive signals for bear regime — move to lowest vol stocks."""
        returns = self.prices.pct_change()
        vol = returns.rolling(63).std() * np.sqrt(252)
        low_vol = -vol   # Invert: low vol = high score

        signals = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        rebalance_dates = period_end_dates(signals.index, "monthly")
        last_signal = pd.Series(0.0, index=self.prices.columns)

        for date in self.prices.index:
            if date in rebalance_dates:
                if date in low_vol.index:
                    row = low_vol.loc[date].dropna()
                    k = min(self.n_positions, len(row))
                    if k >= 1:
                        new_sig = pd.Series(0.0, index=self.prices.columns)
                        longs = row.nlargest(k).index
                        new_sig[longs] = 1.0 / k
                        last_signal = new_sig
            signals.loc[date] = last_signal

        return signals

    def _sideways_signals(self) -> pd.DataFrame:
        """Value signals for sideways regime."""
        from src.factor_models.momentum_factor import ValueFactor
        val = ValueFactor(self.prices)
        scores = val.compute_signals(zscore=True)

        signals = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        rebalance_dates = period_end_dates(signals.index, "monthly")
        last_signal = pd.Series(0.0, index=self.prices.columns)

        for date in self.prices.index:
            if date in rebalance_dates:
                if date in scores.index:
                    row = scores.loc[date].dropna()
                    k = min(self.n_positions, len(row))
                    if k >= 1:
                        new_sig = pd.Series(0.0, index=self.prices.columns)
                        longs = row.nlargest(k).index
                        new_sig[longs] = 1.0 / k
                        last_signal = new_sig
            signals.loc[date] = last_signal

        return signals

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate regime-adaptive signals by blending sub-strategy signals.

        Returns
        -------
        pd.DataFrame  Combined regime-aware position weights.
        """
        regimes = self._detect_regimes()
        bull_sig = self._bull_signals()
        bear_sig = self._bear_signals()
        sideways_sig = self._sideways_signals()

        # Align all signals to same index
        common = (
            self.prices.index
            .intersection(bull_sig.index)
            .intersection(bear_sig.index)
            .intersection(sideways_sig.index)
            .intersection(regimes.index)
        )

        cols = self.prices.columns
        reg = regimes.reindex(common).astype(str)
        bull = bull_sig.reindex(index=common, columns=cols).fillna(0.0)
        bear = bear_sig.reindex(index=common, columns=cols).fillna(0.0)
        side = sideways_sig.reindex(index=common, columns=cols).fillna(0.0)
        combined = side.copy()
        combined[(reg == "Bull").values] = bull[(reg == "Bull").values]
        combined[(reg == "Bear").values] = bear[(reg == "Bear").values]

        return combined

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run regime adaptive backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="Regime_Adaptive",
        )
        return bt.run()


# ═══════════════════════════════════════════════
#  FACTOR TIMING STRATEGY
# ═══════════════════════════════════════════════

class FactorTimingStrategy:
    """
    Dynamically tilt factor exposures based on macro conditions.

    In risk-on environments  → overweight momentum and quality.
    In risk-off environments → overweight value and low-vol.
    Reads risk sentiment from macro data.

    Parameters
    ----------
    universe_prices : pd.DataFrame
    macro_data      : pd.DataFrame   From MacroDataLoader.fetch_all().
    n_positions     : int
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
        n_positions: int = 10,
    ) -> None:
        self.prices = universe_prices.copy()
        self.macro = macro_data
        self.n_positions = n_positions

    def _get_factor_weights(self, date: pd.Timestamp) -> dict:
        """
        Determine factor weights based on macro environment.
        Returns weights for momentum, value, quality.
        """
        if self.macro is None or date not in self.macro.index:
            return {"momentum": 0.4, "value": 0.3, "quality": 0.3}

        row = self.macro.loc[date]
        risk_score = float(row.get("risk_on_score", 0))
        vix = float(row.get("VIX_level", 20))

        # Risk-on: tilt to momentum
        if risk_score > 0.3 and vix < 20:
            return {"momentum": 0.60, "value": 0.20, "quality": 0.20}
        # Risk-off: tilt to value and quality
        elif risk_score < -0.3 or vix > 30:
            return {"momentum": 0.10, "value": 0.45, "quality": 0.45}
        # Neutral
        else:
            return {"momentum": 0.40, "value": 0.30, "quality": 0.30}

    def generate_signals(self) -> pd.DataFrame:
        """Generate factor-timed signals."""
        from src.factor_models.momentum_factor import (
            MomentumFactor, ValueFactor, QualityFactor
        )

        mom_scores = MomentumFactor(self.prices).compute_signals(zscore=True)
        val_scores = ValueFactor(self.prices).compute_signals(zscore=True)
        qual_scores = QualityFactor(self.prices).compute_signals(zscore=True)

        common = (
            mom_scores.index
            .intersection(val_scores.index)
            .intersection(qual_scores.index)
        )

        signals = pd.DataFrame(0.0, index=common, columns=self.prices.columns)
        rebalance_dates = period_end_dates(common, "monthly")
        last_signal = pd.Series(0.0, index=self.prices.columns)

        for date in common:
            if date in rebalance_dates:
                fw = self._get_factor_weights(date)

                composite = pd.Series(0.0, index=self.prices.columns)
                if date in mom_scores.index:
                    composite += mom_scores.loc[date].reindex(self.prices.columns).fillna(0) * fw["momentum"]
                if date in val_scores.index:
                    composite += val_scores.loc[date].reindex(self.prices.columns).fillna(0) * fw["value"]
                if date in qual_scores.index:
                    composite += qual_scores.loc[date].reindex(self.prices.columns).fillna(0) * fw["quality"]

                valid = composite.dropna()
                k = min(self.n_positions, len(valid))
                if k >= 1:
                    new_sig = pd.Series(0.0, index=self.prices.columns)
                    longs = valid.nlargest(k).index
                    new_sig[longs] = 1.0 / k
                    last_signal = new_sig

            signals.loc[date] = last_signal

        return signals

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run factor timing backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="Factor_Timing",
        )
        return bt.run()
