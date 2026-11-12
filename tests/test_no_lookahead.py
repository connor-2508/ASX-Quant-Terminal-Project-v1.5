"""
Look-ahead tests.

Generic check: change every price AFTER a cutoff date and assert that no
signal / feature / factor value dated ON OR BEFORE the cutoff changes.
Anything that fails this uses future information.
"""

from __future__ import annotations


import pandas as pd
import pytest

from conftest import perturb_after

CUT = 1000


def _assert_same_up_to(a: pd.DataFrame | pd.Series, b, cutoff_date):
    a = a.loc[:cutoff_date]
    b = b.loc[:cutoff_date]
    pd.testing.assert_frame_equal(
        pd.DataFrame(a), pd.DataFrame(b), check_exact=False, rtol=1e-10, atol=1e-12,
    )


@pytest.fixture(scope="module")
def pair(prices):
    return prices, perturb_after(prices, CUT), prices.index[CUT]


@pytest.mark.parametrize("name", [
    "csm", "tsm", "ram", "bb", "rsi", "factor", "regime", "pairs_kalman", "pairs_ols",
])
def test_strategy_signals_have_no_lookahead(pair, name):
    from src.strategies.momentum_strategy import (
        CrossSectionalMomentum, TimeSeriesMomentum, RiskAdjustedMomentum,
    )
    from src.strategies.mean_reversion_strategy import (
        BollingerBandReversion, RSIMeanReversion, PairsTrading,
    )
    from src.strategies.factor_strategy import FactorStrategy, RegimeAdaptiveStrategy

    p0, p1, cut = pair
    build = {
        "csm": lambda p: CrossSectionalMomentum(p, n_long=3).generate_signals(),
        "tsm": lambda p: TimeSeriesMomentum(p).generate_signals(),
        "ram": lambda p: RiskAdjustedMomentum(p, n_long=3).generate_signals(),
        "bb": lambda p: BollingerBandReversion(p).generate_signals(),
        "rsi": lambda p: RSIMeanReversion(p).generate_signals(),
        "factor": lambda p: FactorStrategy(p, n_long=3).generate_signals(),
        "regime": lambda p: RegimeAdaptiveStrategy(p, p.mean(axis=1), n_positions=3).generate_signals(),
        "pairs_kalman": lambda p: PairsTrading(p, "BHP.AX", "RIO.AX", hedge_method="kalman").generate_signals(),
        "pairs_ols": lambda p: PairsTrading(p, "BHP.AX", "RIO.AX", hedge_method="ols").generate_signals(),
    }[name]
    _assert_same_up_to(build(p0), build(p1), cut)


def test_technical_indicators_have_no_lookahead(pair, market):
    from src.analytics.technical_indicators import TechnicalIndicators
    p0, p1, cut = pair
    t = "CBA.AX"
    base = pd.DataFrame({k.capitalize(): market[k][t] for k in ["open", "high", "low", "close", "volume"]})
    pert = base.copy()
    scale = p1[t] / p0[t]
    for c in ["Open", "High", "Low", "Close"]:
        pert[c] = base[c] * scale
    a = TechnicalIndicators(base).compute_all()
    b = TechnicalIndicators(pert).compute_all()
    assert "Chikou_Span" not in a.columns          # future close - excluded
    _assert_same_up_to(a, b, cut)


def test_factor_signals_have_no_lookahead(pair):
    from src.factor_models.momentum_factor import MomentumFactor, ValueFactor, QualityFactor
    p0, p1, cut = pair
    for cls in (MomentumFactor, ValueFactor, QualityFactor):
        _assert_same_up_to(cls(p0).compute_signals(), cls(p1).compute_signals(), cut)


def test_fama_french_factors_have_no_lookahead(pair, market, benchmark):
    from src.factor_models.fama_french import FamaFrench
    p0, p1, cut = pair
    size = market["close"] * market["volume"]
    f0 = FamaFrench(p0.pct_change(), benchmark, universe_prices=p0, size_data=size).get_factor_returns()
    f1 = FamaFrench(p1.pct_change(), benchmark, universe_prices=p1,
                    size_data=size * (p1 / p0)).get_factor_returns()
    _assert_same_up_to(f0, f1, cut)


def test_ml_features_have_no_lookahead(pair):
    from src.machine_learning.feature_engineering import FeatureEngineer
    p0, p1, cut = pair
    a = FeatureEngineer(p0).build_features("CSL.AX", include_target=False).drop(columns="ticker")
    b = FeatureEngineer(p1).build_features("CSL.AX", include_target=False).drop(columns="ticker")
    _assert_same_up_to(a, b, cut)


def test_ml_walk_forward_predictions_are_out_of_sample(prices):
    """
    Predictions dated <= cutoff may only depend on prices <= cutoff
    (features) and targets realised by then. Perturb the future and the
    walk-forward predictions up to the cutoff must not move.
    """
    from src.machine_learning.alpha_models import AlphaModelPipeline
    p = prices.iloc[:900]
    cut = 700
    p1 = perturb_after(p, cut)
    kw = dict(model_type="rf", horizon=21, model_params={"n_estimators": 20, "random_state": 0})
    a = AlphaModelPipeline(p, **kw).walk_forward_predict(min_train_days=300, retrain_every=100)
    b = AlphaModelPipeline(p1, **kw).walk_forward_predict(min_train_days=300, retrain_every=100)
    assert a.index[0] >= p.index[300]               # nothing before the first fit
    _assert_same_up_to(a, b, p.index[cut])


def test_backtester_lag_blocks_oracle_signal(prices):
    """A perfect-foresight signal must not make money once it is lagged."""
    from src.backtesting.backtester import Backtester, BacktestConfig
    fwd = prices.pct_change().shift(-1)                  # tomorrow's return (future!)
    oracle = (fwd.rank(axis=1) > 6).astype(float)        # best 2 stocks tomorrow
    cfg0 = BacktestConfig(signal_lag=0, commission=0, slippage=0)
    cfg1 = BacktestConfig(signal_lag=1, commission=0, slippage=0)
    r0 = Backtester(prices, oracle, cfg0).run()
    r1 = Backtester(prices, oracle, cfg1).run()
    ann0 = r0.returns.mean() * 252
    ann1 = r1.returns.mean() * 252
    assert ann0 > 1.0             # sanity: the oracle is wildly profitable unlagged...
    assert abs(ann1) < 0.25       # ...and worthless once lagged by a day


def test_kupiec_var_uses_only_past_data(returns):
    from src.simulation.monte_carlo_paths import StressTester
    st = StressTester(returns, {t: 1 / returns.shape[1] for t in returns.columns})
    port = st.port_returns
    var = port.rolling(250).quantile(0.05).shift(1) * -1
    out = st.var_backtest(confidence=0.95, window=250).set_index("Metric")["Value"]
    expected_exc = int((port < -var).loc[var.dropna().index].sum())
    assert int(out["Exceedances"]) == expected_exc
