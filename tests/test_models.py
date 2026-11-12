"""
Model correctness tests against closed forms, reference libraries and
planted ground truth.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ─────────────────────────── utilities ───────────────────────────

def test_ledoit_wolf_matches_sklearn(returns):
    from sklearn.covariance import LedoitWolf
    from src.utils.covariance import ledoit_wolf_cov
    np.testing.assert_allclose(ledoit_wolf_cov(returns), LedoitWolf().fit(returns.values).covariance_,
                               rtol=1e-10, atol=1e-14)


def test_rebalance_calendar_uses_trading_days():
    from src.utils.calendar import period_end_dates
    idx = pd.bdate_range("2023-01-01", "2023-12-31")
    m = sorted(period_end_dates(idx, "monthly"))
    assert len(m) == 12
    assert pd.Timestamp("2023-12-29") in m          # Dec 31 2023 was a Sunday
    assert len(period_end_dates(idx, "weekly")) == 52
    assert len(period_end_dates(idx, "quarterly")) == 4


def test_contiguous_segments():
    from src.utils.calendar import contiguous_segments
    s = pd.Series(list("AABBBA"), index=pd.bdate_range("2024-01-01", periods=6))
    segs = contiguous_segments(s)
    assert [x[2] for x in segs] == ["A", "B", "A"]


def test_data_package_imports_and_universe():
    import src.data  # noqa: F401  (used to fail with NameError: Path)
    from src.data.asx_data_loader import ASX_200_UNIVERSE
    assert len(ASX_200_UNIVERSE) == 60 == len(set(ASX_200_UNIVERSE))


def test_universe_cache_key_depends_on_tickers(tmp_path, monkeypatch):
    from src.data import asx_data_loader as m
    calls = []

    def fake_download(tickers, **kw):
        calls.append(tuple(tickers))
        idx = pd.bdate_range("2020-01-01", periods=300)
        cols = pd.MultiIndex.from_product([["Close"], tickers])
        return pd.DataFrame(1.0, index=idx, columns=cols)

    monkeypatch.setattr(m.yf, "download", fake_download)
    loader = m.ASXDataLoader(cache_dir=str(tmp_path))
    a = loader.fetch_universe_data(["BHP.AX", "CBA.AX"], "2020-01-01", "2021-01-01")
    b = loader.fetch_universe_data(["CSL.AX", "WES.AX"], "2020-01-01", "2021-01-01")
    assert list(a.columns) != list(b.columns) and len(calls) == 2


# ─────────────────────────── backtester ───────────────────────────

def test_buy_and_hold_tracks_price_with_drift(prices):
    """100% in one stock, never rebalanced: NAV must equal the price path less entry cost."""
    from src.backtesting.backtester import Backtester, BacktestConfig
    sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    sig["CSL.AX"] = 1.0
    cfg = BacktestConfig(commission=0.001, slippage=0.0005, signal_lag=1)
    r = Backtester(prices, sig, cfg).run()
    px = prices["CSL.AX"].loc[r.nav.index]
    expected = px / px.iloc[0] * cfg.initial_capital * (1 - 0.0015)
    np.testing.assert_allclose(r.nav.values, expected.values, rtol=1e-10)
    assert len(r.trades) == 1                          # one entry, no drift rebalancing


def test_equal_weight_monthly_rebalance_counts(prices):
    from src.backtesting.backtester import Backtester, BacktestConfig
    sig = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    r = Backtester(prices, sig, BacktestConfig(rebalance_freq="monthly")).run()
    n_months = prices.index.to_period("M").nunique()
    # first day of investing is a month-end; every later month-end trades the drift away
    assert n_months - 2 <= len(r.trades) <= n_months
    w = r.weights.sum(axis=1)
    assert np.allclose(w[w > 0], 1.0)


def test_warmup_is_trimmed(prices):
    from src.strategies.momentum_strategy import CrossSectionalMomentum
    from src.backtesting.backtester import Backtester
    sig = CrossSectionalMomentum(prices, lookback=252, n_long=3).generate_signals()
    r = Backtester(prices, sig).run()
    assert r.returns.index[0] > prices.index[252]
    assert (r.weights.abs().sum(axis=1).iloc[0]) > 0


def test_cross_sectional_momentum_trades_every_month_end(prices):
    from src.strategies.momentum_strategy import CrossSectionalMomentum
    sig = CrossSectionalMomentum(prices, n_long=3, rebalance_freq="monthly").generate_signals()
    active = sig.loc[sig.abs().sum(axis=1) > 0]
    first = active.index[0]
    months_after = prices.loc[first:].index.to_period("M").nunique()
    held = (sig.loc[first:] > 0).sum(axis=1)
    assert (held == 3).all() and months_after > 40
    # n_long larger than the universe falls back to holding everything
    big = CrossSectionalMomentum(prices, n_long=50).generate_signals()
    assert (big.abs().sum(axis=1) > 0).any()


def test_sortino_uses_full_sample_downside_deviation():
    from src.utils.performance import sortino_ratio
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert sortino_ratio(r, 0.0, 252) == pytest.approx(r.mean() / dd * np.sqrt(252))


def test_monthly_table_short_history():
    from src.utils.performance import monthly_return_table
    r = pd.Series(0.001, index=pd.bdate_range("2024-03-01", "2024-05-31"))
    t = monthly_return_table(r)
    assert list(t.columns[:12]) == ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    assert t.loc[2024, "Jan"] != t.loc[2024, "Jan"]   # NaN
    assert t.loc[2024, "Annual"] == pytest.approx((1.001) ** len(r) - 1)


# ─────────────────────────── factor models ───────────────────────────

def test_capm_recovers_planted_betas(market):
    from src.factor_models.capm import CAPM
    rets = market["close"].pct_change().dropna()
    capm = CAPM(rets, market["benchmark_returns"], risk_free_rate=0.03)
    df = capm.fit_to_dataframe()
    err = (df["Beta"] - market["true_betas"]).abs()
    assert err.max() < 0.08
    # SML consistency: abnormal return == Jensen's alpha (OLS identity)
    np.testing.assert_allclose(df["Abnormal Return"], df["Alpha (ann.)"], atol=1e-10)


def test_fama_french_factors_are_unbiased_on_noise():
    """Sorting on lagged characteristics in a world with no size/value premium -> ~0 Sharpe."""
    from src.data.synthetic_data import generate_market_data
    from src.factor_models.fama_french import FamaFrench
    sharpes = []
    for seed in range(6):
        d = generate_market_data(n_days=1300, seed=100 + seed, regimes=False)
        ff = FamaFrench(d["close"].pct_change(), d["benchmark_returns"],
                        universe_prices=d["close"], risk_free_rate=0.0,
                        size_data=d["close"] * d["volume"])
        fs = ff.factor_summary()
        sharpes.append(fs.loc[["SMB", "HML"], "Sharpe"].values)
    mean_sharpe = np.mean(sharpes, axis=0)
    assert np.all(np.abs(mean_sharpe) < 0.6), mean_sharpe


def test_fama_french_requires_benchmark(returns, prices):
    from src.factor_models.fama_french import FamaFrench
    with pytest.raises(ValueError):
        FamaFrench(returns, None, universe_prices=prices)


# ─────────────────────────── analytics ───────────────────────────

def test_statistical_suite_uses_statsmodels():
    from src.analytics import statistical_tests as m
    assert m.HAS_STATSMODELS, "statsmodels-backed tests are silently disabled"


def test_stationarity_and_cointegration():
    from src.analytics.statistical_tests import StatisticalTests
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0, 0.01, 1500))
    walk = noise.cumsum() + 100
    assert StatisticalTests(noise).adf_test().reject_null
    assert not StatisticalTests(walk).adf_test().reject_null
    x = pd.Series(np.cumsum(rng.normal(0, 1, 1500)) + 100)
    y = 0.7 * x + rng.normal(0, 1, 1500)
    assert StatisticalTests(y).engle_granger_cointegration(x).reject_null


def test_anderson_darling_pvalue_bound():
    from src.analytics.statistical_tests import StatisticalTests
    rng = np.random.default_rng(1)
    assert StatisticalTests(pd.Series(rng.standard_t(3, 3000))).anderson_darling_test().p_value <= 0.01
    assert StatisticalTests(pd.Series(rng.normal(size=500))).anderson_darling_test().p_value >= 0.05


def test_atr_is_wilder(market):
    from src.analytics.technical_indicators import TechnicalIndicators
    t = "BHP.AX"
    df = pd.DataFrame({k.capitalize(): market[k][t] for k in ["open", "high", "low", "close", "volume"]})
    atr = TechnicalIndicators(df).atr(14)
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    manual = tr.ewm(alpha=1 / 14, adjust=False).mean()
    np.testing.assert_allclose(atr.values, manual.values)


def test_garch_and_har(returns):
    from src.analytics.volatility_models import VolatilityModels
    vm = VolatilityModels(returns["CBA.AX"])
    g = vm.garch_volatility(forecast_horizon=10)
    assert 0 < g["params"]["alpha[1]"] + g["params"]["beta[1]"] < 1
    har = vm.har_rv_model(forecast_horizon=21)
    assert len(har["forecast"]) == 21 and har["forecast"].notna().all()


# ─────────────────────────── regimes ───────────────────────────

@pytest.fixture(scope="module")
def planted_regimes():
    rng = np.random.default_rng(1)
    r = np.r_[rng.normal(0.001, 0.008, 600), rng.normal(-0.002, 0.025, 300), rng.normal(0.0008, 0.009, 600)]
    y = pd.Series(r, index=pd.bdate_range("2019-01-01", periods=len(r)))
    truth = np.r_[np.ones(600), np.zeros(300), np.ones(600)]
    return y, truth


def test_regime_confirmation_filter(prices):
    """Persistence filter: short blips are ignored, real switches are delayed
    by exactly min_days-1 bars, it is causal, and it cuts regime changes."""
    from src.strategies.factor_strategy import RegimeAdaptiveStrategy as R
    idx = pd.bdate_range("2020-01-01", periods=12)
    raw = pd.Series(list("BBBSSBBBBSSS"), index=idx)
    out = R.confirm_regimes(raw, 3)
    assert list(out) == list("BBBBBBBBBBBS")          # 2-day blip ignored; switch confirmed on day 3
    assert R.confirm_regimes(raw, 1).equals(raw)      # 1 = no filter
    for k in range(1, len(raw)):                      # causal: prefix result never changes
        assert list(R.confirm_regimes(raw.iloc[:k], 3)) == list(out.iloc[:k])

    mkt = prices.mean(axis=1)
    fast = R(prices, mkt, n_positions=3, min_regime_days=1)._detect_regimes()
    slow = R(prices, mkt, n_positions=3, min_regime_days=10)._detect_regimes()
    switches = lambda r: int((r != r.shift()).sum())
    assert switches(slow) <= switches(fast)


def test_hmm_recovers_planted_regimes(planted_regimes):
    from src.regime_detection.hidden_markov_models import HiddenMarkovModel
    y, truth = planted_regimes
    res = HiddenMarkovModel(y, n_states=2).fit()
    assert (res.regime_sequence.values == truth).mean() > 0.97
    assert res.state_stds[0] > res.state_stds[1]      # Bear state is the volatile one


def test_markov_switching_recovers_planted_regimes(planted_regimes):
    from src.regime_detection.markov_regime_switching import MarkovRegimeSwitching
    y, truth = planted_regimes
    mrs = MarkovRegimeSwitching(y, n_regimes=2)
    res = mrs.fit()
    assert (res.regime_sequence.values == truth).mean() > 0.97
    np.testing.assert_allclose(res.transition_matrix.sum(axis=1), 1.0)
    assert list(mrs.state_probabilities().columns) == ["Bear", "Bull"]


# ─────────────────────────── optimisation ───────────────────────────

def test_markowitz_frontier_and_risk_parity(returns):
    from src.optimisation.markowitz_optimizer import MarkowitzOptimizer
    opt = MarkowitzOptimizer(returns, risk_free_rate=0.02)
    mv = opt.min_volatility()
    ms = opt.max_sharpe()
    fr = opt.efficient_frontier(n_points=15)
    assert mv.weights.sum() == pytest.approx(1) and ms.weights.sum() == pytest.approx(1)
    assert fr["Volatility"].min() >= mv.expected_volatility - 1e-6
    assert fr["Return"].is_monotonic_increasing
    assert ms.sharpe_ratio >= fr["Sharpe"].max() - 1e-3
    rp = opt.risk_parity()
    w = rp.weights.values
    rc = w * (opt.Sigma @ w)
    assert np.allclose(rc / rc.sum(), 1 / len(w), atol=1e-3)


def test_black_litterman_mechanics(returns):
    from src.optimisation.black_litterman import BlackLitterman
    rf = 0.04
    bl = BlackLitterman(returns, risk_free_rate=rf)
    eq = bl.equilibrium_vs_posterior()
    np.testing.assert_allclose(eq["Equilibrium Return"], eq["Posterior Return"])  # no views

    t = returns.columns[0]
    prior = eq.loc[t, "Equilibrium Return"]
    target = prior + 0.10
    post = {}
    for c in (0.1, 0.5, 0.95):
        bl.clear_views()
        bl.add_absolute_view(t, target, confidence=c)
        post[c] = bl.equilibrium_vs_posterior().loc[t, "Posterior Return"]
    assert prior < post[0.1] < post[0.5] < post[0.95] < target + 1e-9
    # c = 0.5 -> view precision equals prior precision -> halfway on that asset
    assert post[0.5] == pytest.approx((prior + target) / 2, rel=1e-6)

    bl.clear_views()
    bl.add_absolute_view(t, target, confidence=0.7)
    bl.confidence_sensitivity(n_levels=3)
    assert bl._views[0].confidence == 0.7                # restored
    with pytest.raises(ValueError):
        bl.add_relative_view(t, t, 0.02)


# ─────────────────────────── Monte Carlo ───────────────────────────

def test_monte_carlo_moments(returns):
    from src.simulation.monte_carlo_paths import MonteCarloSimulator
    r = returns["WES.AX"]
    mc = MonteCarloSimulator(r, n_simulations=40_000, horizon=252)
    gbm = mc.simulate_gbm(S0=100)
    assert gbm.final_prices.mean() / 100 == pytest.approx(np.exp(mc.mu), rel=0.01)
    jd = mc.simulate_jump_diffusion(S0=100)
    assert jd.final_prices.mean() / 100 == pytest.approx(np.exp(mc.mu), rel=0.02)
    lr = np.log(jd.paths[:, 1:] / jd.paths[:, :-1])
    assert lr.std() * np.sqrt(252) == pytest.approx(mc.sigma, rel=0.05)   # variance matched
    assert stats.skew(lr.ravel()) < 0                                      # negative jumps
    bs = mc.simulate_bootstrap(S0=100, block_size=5)
    # iid resampling: E[S_T/S_0] = (1 + mean r)^252, median ~ historical CAGR
    assert bs.final_prices.mean() / 100 == pytest.approx((1 + r.mean()) ** 252, rel=0.03)
    assert np.median(bs.final_prices) / 100 == pytest.approx((1 + r).prod() ** (252 / len(r)), rel=0.05)


def test_tail_risk_expected_return_not_degenerate(returns):
    from src.simulation.monte_carlo_paths import StressTester
    st = StressTester(returns, {t: 1 for t in returns.columns})
    tr = st.tail_risk_analysis(n_simulations=5000).set_index("Metric")["Value"]
    assert abs(tr["Expected Return"]) > 1e-4             # was identically 0


def test_kupiec_correct_model_not_rejected():
    from src.simulation.monte_carlo_paths import StressTester
    rng = np.random.default_rng(5)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, 3000)}, index=pd.bdate_range("2010-01-01", periods=3000))
    out = StressTester(r, {"A": 1}).var_backtest(0.95, 500).set_index("Metric")["Value"]
    assert float(out["P-Value"]) > 0.05


# ─────────────────────────── options ───────────────────────────

BS_ARGS = dict(S=100.0, K=95.0, T=0.75, r=0.04, sigma=0.25, q=0.02)


@pytest.mark.parametrize("ot", ["call", "put"])
def test_greeks_match_finite_differences(ot):
    from src.options.black_scholes import BlackScholes as BS
    a = BS_ARGS
    h = 1e-4
    p = lambda **kw: BS(**{**a, **kw}).price(ot)          # noqa: E731
    d = lambda **kw: BS(**{**a, **kw}).delta(ot)          # noqa: E731
    g = BS(**a).greeks(ot)
    assert g.delta == pytest.approx((p(S=a["S"] + h) - p(S=a["S"] - h)) / (2 * h), rel=1e-5)
    assert g.gamma == pytest.approx((p(S=a["S"] + h) - 2 * p() + p(S=a["S"] - h)) / h ** 2, rel=1e-3)
    assert g.vega == pytest.approx((p(sigma=a["sigma"] + h) - p(sigma=a["sigma"] - h)) / (2 * h) / 100, rel=1e-5)
    assert g.rho == pytest.approx((p(r=a["r"] + h) - p(r=a["r"] - h)) / (2 * h) / 100, rel=1e-5)
    assert g.theta == pytest.approx(-(p(T=a["T"] + h) - p(T=a["T"] - h)) / (2 * h) / 365, rel=1e-5)
    assert g.vanna == pytest.approx((d(sigma=a["sigma"] + h) - d(sigma=a["sigma"] - h)) / (2 * h), rel=1e-4)
    vega_raw = lambda s: BS(**{**a, "sigma": s}).vega() * 100   # noqa: E731
    assert g.volga == pytest.approx((vega_raw(a["sigma"] + h) - vega_raw(a["sigma"] - h)) / (2 * h), rel=1e-4)
    assert g.charm == pytest.approx(-(d(T=a["T"] + h) - d(T=a["T"] - h)) / (2 * h), rel=1e-4)


def test_put_call_parity_and_implied_vol():
    from src.options.black_scholes import BlackScholes, ImpliedVolatility
    bs = BlackScholes(**BS_ARGS)
    assert bs.put_call_parity_check()["passes"]
    a = BS_ARGS
    for ot in ("call", "put"):
        iv = ImpliedVolatility(bs.price(ot), a["S"], a["K"], a["T"], a["r"], q=a["q"], option_type=ot).calculate()
        assert iv == pytest.approx(a["sigma"], abs=1e-6)
    # below intrinsic -> no implied vol (previously returned a fake 0.1%)
    assert np.isnan(ImpliedVolatility(0.01, 100, 50, 0.5, 0.04).calculate())


def test_monte_carlo_option_pricer_converges():
    from src.options.black_scholes import MonteCarloOptionPricer
    mc = MonteCarloOptionPricer(**BS_ARGS, n_simulations=100_000, n_steps=50)
    eu = mc.european_call()
    assert abs(eu["mc_price"] - eu["bs_price"]) < 3 * eu["std_error"]
    dg = mc.digital_call()
    assert dg["mc_price"] == pytest.approx(dg["bs_price"], abs=0.01)
    assert mc.asian_call()["mc_price"] < eu["bs_price"]
    din = mc.barrier_call(85, "down-and-in")["mc_price"]
    dout = mc.barrier_call(85, "down-and-out")["mc_price"]
    assert din + dout == pytest.approx(eu["mc_price"], rel=1e-10)   # in-out parity


# ─────────────────────────── portfolio analytics ───────────────────────────

def test_risk_decomposition_euler(returns):
    from src.portfolio_analytics.risk_decomposition import RiskDecomposer
    w = {t: v for t, v in zip(returns.columns, np.linspace(1, 2, returns.shape[1]))}
    rd = RiskDecomposer(returns, w)
    assert rd.component_risk_contribution().sum() == pytest.approx(rd.portfolio_volatility())
    assert rd.pct_risk_contribution().sum() == pytest.approx(1.0)
    w_missing = {**w, "NOT.AX": 5.0}                       # ticker without data
    assert RiskDecomposer(returns, w_missing).w.sum() == pytest.approx(1.0)


def test_concentration_and_synthetic_factors(prices, benchmark):
    from src.portfolio_analytics.exposure_analysis import ExposureAnalyser
    ea = ExposureAnalyser(prices, {"BHP.AX": 0.9, "CBA.AX": 0.1}, benchmark=benchmark)
    assert ea.concentration_metrics()["gini_coefficient"] == pytest.approx(0.4)
    full = ExposureAnalyser(prices, {t: 1 for t in prices.columns}, benchmark=benchmark)
    f = full._build_synthetic_factors()
    assert f is not None and (f.std() > 1e-4).all()       # was identically zero
    assert not full.factor_exposure().empty


def test_bhb_attribution_identity_including_unknown_sector(returns):
    from src.portfolio_analytics.performance_attribution import PerformanceAttributor
    tick = list(returns.columns)
    pw = {t: v for t, v in zip(tick, np.linspace(1, 3, len(tick)))}
    bw = {t: 1 for t in tick}
    smap = {t: ("Materials" if i % 2 else "Financials") for i, t in enumerate(tick[:-2])}  # last 2 unmapped
    res = PerformanceAttributor(pw, bw, returns, sector_map=smap).sector_attribution()
    period = (1 + returns).prod() - 1
    rp = sum(pw[t] / sum(pw.values()) * period[t] for t in tick)
    rb = sum(bw[t] / sum(bw.values()) * period[t] for t in tick)
    assert res.total_active_return == pytest.approx(rp - rb)
    assert res.allocation_effect + res.selection_effect + res.interaction_effect == pytest.approx(rp - rb)


def test_portfolio_returns_drift_and_scorecard(prices):
    from src.portfolio_analytics.portfolio_metrics import calculate_portfolio_returns, PortfolioMetrics
    w = {"BHP.AX": 0.5, "CBA.AX": 0.5}
    bh = calculate_portfolio_returns(prices, w, rebalance="never")
    px = prices[["BHP.AX", "CBA.AX"]]
    expected = (0.5 * px["BHP.AX"] / px["BHP.AX"].iloc[0] + 0.5 * px["CBA.AX"] / px["CBA.AX"].iloc[0])
    np.testing.assert_allclose((1 + bh).cumprod().values, expected.iloc[1:].values, rtol=1e-10)
    daily = calculate_portfolio_returns(prices, w, rebalance="daily")
    assert not np.allclose(bh.values, daily.values)
    pm = PortfolioMetrics(pd.Series([0.01, -0.05, 0.02] * 50, index=pd.bdate_range("2020-01-01", periods=150)), risk_free_rate=0.0)
    s = pm.compute_all()
    assert s.score_card["Max Drawdown"][1] != "Unacceptable" or s.max_drawdown < -0.3


# ─────────────────────────── strategies / ML / WFO ───────────────────────────

def test_pairs_trading_profits_on_cointegrated_pair():
    from src.strategies.mean_reversion_strategy import PairsTrading
    from src.backtesting.backtester import Backtester, BacktestConfig
    rng = np.random.default_rng(4)
    n = 1500
    a = 50 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = 0.9 * ou[t - 1] + rng.normal(0, 0.4)
    px = pd.DataFrame({"A": a, "B": 0.8 * a + 5 + ou}, index=pd.bdate_range("2019-01-01", periods=n))
    pt = PairsTrading(px, "B", "A", hedge_method="kalman")
    assert pt.cointegration_test().reject_null
    r = Backtester(px, pt.generate_signals(),
                   BacktestConfig(allow_short=True, commission=0, slippage=0)).run()
    assert r.nav.iloc[-1] > r.nav.iloc[0]
    assert abs(pt.spread_analysis()["hedge_ratio"].iloc[-250:].mean() - 0.8) < 0.1


def test_xgboost_alpha_fits():
    from src.machine_learning.alpha_models import XGBoostAlpha
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(800, 4)))
    y = pd.Series(X[0] * 0.5 + rng.normal(size=800))
    res = XGBoostAlpha(n_estimators=100).fit(X.iloc[:600], y.iloc[:600], X.iloc[600:], y.iloc[600:])
    assert res.test_ic > 0.2


def test_walk_forward_optimisation_produces_oos_returns(prices):
    from src.optimisation.parameter_optimisation import WalkForwardOptimisation
    from src.strategies.momentum_strategy import CrossSectionalMomentum
    wfo = WalkForwardOptimisation(
        CrossSectionalMomentum, prices, {"lookback": [126, 252], "n_long": [2, 3]},
        train_days=504, test_days=126, step_days=252, optimiser="grid", verbose=False,
    ).run()
    assert wfo["n_windows"] >= 2
    assert (wfo["oos_returns"].abs() > 0).mean() > 0.9     # OOS windows actually trade


def test_remaining_strategies_and_utilities(prices, benchmark):
    """Smoke-test components that are library-only (not wired into the UI)."""
    from src.strategies.momentum_strategy import DualMomentumStrategy
    from src.strategies.factor_strategy import FactorTimingStrategy
    from src.analytics.statistical_tests import StatisticalTests
    from src.data.data_cleaning import DataCleaner
    from src.data.macro_data_loader import MacroDataLoader
    from src.backtesting.transaction_cost_model import TransactionCostModel

    safe = pd.Series(100 * 1.0002 ** np.arange(len(prices)), index=prices.index)
    dm = DualMomentumStrategy(prices.iloc[:, :3], safe).backtest(benchmark=benchmark)
    assert not dm.returns.empty
    ft = FactorTimingStrategy(prices, n_positions=3).generate_signals()
    assert np.allclose(ft.sum(axis=1)[ft.sum(axis=1) > 0], 1.0)

    jo = StatisticalTests(prices.iloc[:, 0]).johansen_cointegration([prices.iloc[:, 0], prices.iloc[:, 1]])
    assert "table" in jo

    out = DataCleaner().run_pipeline(prices, verbose=False)
    assert out["returns"].shape[1] == prices.shape[1]

    macro = pd.DataFrame({"VIX": 15 + np.abs(np.random.default_rng(0).normal(0, 5, len(prices))),
                          "AUDUSD": 0.7 * prices.iloc[:, 0] / prices.iloc[0, 0],
                          "ASX200": prices.mean(axis=1)}, index=prices.index)
    feats = MacroDataLoader.__new__(MacroDataLoader).compute_macro_features(macro)
    assert {"VIX_zscore", "risk_on_score"} <= set(feats.columns)

    tcm = TransactionCostModel(commission=0.001, spread=0.0005, slippage=0.0005)
    assert tcm.compute_cost(100_000).cost_bps == pytest.approx(20.0)
