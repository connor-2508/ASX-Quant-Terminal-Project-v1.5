"""
terminal_app.py
===============
ASX Quantitative Research Terminal
===================================
Main Streamlit application integrating all platform modules.

Run with:
    streamlit run app/terminal_app.py

Tabs
----
    Market        : price performance, return statistics, correlations
    Technical     : OHLC indicators, volatility models (GARCH family,
                    EWMA, range-based), statistical test battery
    Factors       : CAPM + Security Market Line, Fama-French 3F, factor scores
    Regimes       : Gaussian HMM, Hamilton Markov switching, rule-based consensus
    Backtest      : 9 strategies incl. Kalman pairs and walk-forward ML
    Optimisation  : Markowitz (max Sharpe, min vol, risk parity), Black-Litterman
    Monte Carlo   : GBM, Merton jump-diffusion, block bootstrap, Kupiec VaR test
    Options       : Black-Scholes-Merton, Greeks, implied vol, MC exotics
    Portfolio     : embedded portfolio analytics dashboard
    Guide         : plain-English explanation of every module
"""

from __future__ import annotations

import os
import sys
import warnings

# ── Path setup ──
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_APP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Keep the UI free of library noise (convergence chatter, pandas deprecations).
# The library code itself does not suppress warnings.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TRADING_DAYS = 252

# ─────────────────────────────────────────────
#  Page configuration — MUST be first st call
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ASX Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4FF, #7B2FBE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #90A4AE;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar — global configuration
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render the global sidebar and return configuration dict."""
    with st.sidebar:
        st.markdown("## ⚙️ Terminal Config")
        st.divider()

        source = st.radio(
            "Data source",
            ["Yahoo Finance", "Synthetic demo (offline)"],
            index=0,
            help=(
                "Synthetic demo data has a known data-generating process and "
                "needs no network - use it if Yahoo Finance is rate-limiting."
            ),
        )

        st.markdown("**ASX Universe**")
        default_tickers = "BHP.AX\nCBA.AX\nCSL.AX\nWES.AX\nMQG.AX\nRIO.AX\nWOW.AX\nANZ.AX"
        tickers_raw = st.text_area(
            "Tickers (one per line)",
            value=default_tickers,
            height=160,
            label_visibility="collapsed",
        )
        tickers = list(dict.fromkeys(
            t.strip().upper() for t in tickers_raw.split("\n") if t.strip()
        ))

        st.markdown("**Date Range**")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=pd.Timestamp("2019-01-01"))
        with col2:
            end_date = st.date_input("End", value=pd.Timestamp.today())

        from src.portfolio_analytics.portfolio_dashboard import BENCHMARK_HELP
        benchmark = st.selectbox(
            "Benchmark", ["STW.AX", "IOZ.AX", "^AXJO", "^AXKO", "^AORD", "^GSPC"], index=0,
            help=BENCHMARK_HELP,
        )
        if benchmark.startswith("^A"):
            st.caption("⚠️ Price index (no dividends): alphas will be overstated by "
                       "roughly the dividend yield.")

        rfr = st.number_input(
            "Risk-Free Rate (p.a.)",
            min_value=0.0, max_value=0.15,
            value=0.0435, step=0.0025, format="%.4f",
        )

        st.divider()
        load_btn = st.button("📥 Load Data", type="primary", width="stretch")

    return {
        "source": source,
        "tickers": tickers,
        "start": str(start_date),
        "end": str(end_date),
        "benchmark": benchmark,
        "rfr": rfr,
        "load": load_btn,
    }


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers: tuple, start: str, end: str, benchmark: str) -> dict:
    """Download OHLCV for the universe + benchmark from Yahoo Finance."""
    import yfinance as yf

    symbols = list(tickers) + [benchmark]
    raw = yf.download(
        symbols, start=start, end=end, auto_adjust=True,
        progress=False, group_by="column", threads=True,
    )
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError(
            "Yahoo Finance returned no data (it may be rate-limiting requests). "
            "Wait a minute and retry, or switch to the synthetic demo data source."
        )

    fields = {}
    for f in ["Open", "High", "Low", "Close", "Volume"]:
        df = raw[f].copy()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        fields[f.lower()] = df

    close_all = fields["close"]
    bm = None
    if benchmark in close_all.columns:
        bm_px = close_all[benchmark].dropna()
        bm = bm_px.pct_change().dropna()
        bm.name = benchmark

    avail = [t for t in tickers if t in close_all.columns and close_all[t].notna().sum() > 1]
    out = {k: v.reindex(columns=avail) for k, v in fields.items()}
    # drop dates on which none of the stocks traded (e.g. benchmark-only days)
    valid = out["close"].notna().any(axis=1)
    out = {k: v.loc[valid] for k, v in out.items()}
    if bm is not None:
        bm = bm.reindex(out["close"].index).dropna()
        if bm.empty:
            bm = None
    out.update(benchmark=bm, available=avail, source="Yahoo Finance")
    return out


@st.cache_data(show_spinner=False)
def load_synthetic_data(tickers: tuple, start: str, end: str) -> dict:
    """Offline synthetic OHLCV with a known DGP (clearly labelled in the UI)."""
    from src.data.synthetic_data import generate_market_data
    d = generate_market_data(list(tickers), start=start, end=end, seed=7)
    return {
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"],
        "benchmark": d["benchmark_returns"].rename("Synthetic Market"),
        "available": list(tickers), "source": "Synthetic demo",
    }


def market_proxy(data: dict) -> tuple[pd.Series, str]:
    """
    Market return series for regime work: the benchmark if loaded, else an
    equal-weighted index of the universe. (Averaging *prices* would be a
    price-weighted index dominated by high-priced stocks.)
    """
    bm = data.get("benchmark")
    if bm is not None and not bm.dropna().empty:
        return bm.dropna(), str(bm.name)
    ret = data["close"].pct_change(fill_method=None).mean(axis=1).dropna()
    return ret, "Equal-weight universe"


def regime_shapes(fig: go.Figure, labels: pd.Series, colours: dict) -> None:
    """Shade each contiguous regime period (one rectangle per run)."""
    from src.utils.calendar import contiguous_segments
    for x0, x1, lab in contiguous_segments(labels):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=colours.get(str(lab), "#90A4AE"),
            opacity=0.13, layer="below", line_width=0,
        )


# ── cached heavy computations (Streamlit re-runs every tab on each click) ──

@st.cache_data(show_spinner=False)
def cached_capm(returns: pd.DataFrame, bm: pd.Series, rfr: float):
    from src.factor_models.capm import CAPM
    capm = CAPM(returns, bm, risk_free_rate=rfr)
    return capm.fit_to_dataframe(), capm.security_market_line(), capm.market_premium(), capm.rfr_daily * TRADING_DAYS


@st.cache_data(show_spinner=False)
def cached_ff(returns: pd.DataFrame, bm: pd.Series, prices: pd.DataFrame,
              size: pd.DataFrame, rfr: float):
    from src.factor_models.fama_french import FamaFrench
    ff = FamaFrench(returns, bm, universe_prices=prices, risk_free_rate=rfr, size_data=size)
    return ff.fit_to_dataframe(), ff.factor_summary(), ff.factor_correlation()


@st.cache_data(show_spinner=False)
def cached_factor_scores(prices: pd.DataFrame):
    from src.factor_models.momentum_factor import FactorLibrary
    return FactorLibrary(prices).get_latest_factor_scores()


@st.cache_data(show_spinner=False)
def cached_hmm(returns: pd.Series, n_states: int):
    from src.regime_detection.hidden_markov_models import HiddenMarkovModel
    hmm = HiddenMarkovModel(returns, n_states=n_states)
    res = hmm.fit()
    return res, hmm.state_probabilities()


@st.cache_data(show_spinner=False)
def cached_mrs(returns: pd.Series, n_regimes: int):
    from src.regime_detection.markov_regime_switching import MarkovRegimeSwitching
    mrs = MarkovRegimeSwitching(returns, n_regimes=n_regimes)
    res = mrs.fit()
    return res, mrs.state_probabilities(True), mrs.state_probabilities(False)


@st.cache_data(show_spinner=False)
def cached_rule_regimes(returns: pd.Series, prices: pd.Series):
    from src.regime_detection.volatility_regimes import RegimeEngine
    engine = RegimeEngine(returns, prices=prices)
    return engine.current_regime(), engine.regime_history(use_hmm=False), engine.regime_statistics()


@st.cache_data(show_spinner=False)
def cached_vol_models(ohlc: pd.DataFrame, ticker: str):
    from src.analytics.volatility_models import VolatilityModels
    ret = ohlc["Close"].pct_change().dropna()
    vm = VolatilityModels(ret, prices=ohlc, name=ticker)
    summary = vm.volatility_summary()
    fits = {}
    for name, fn in [("GARCH(1,1)", vm.garch_volatility),
                     ("EGARCH(1,1,1)", vm.egarch_volatility),
                     ("GJR-GARCH(1,1)", vm.gjr_garch_volatility)]:
        fits[name] = fn()
    cols = {
        "EWMA (λ=0.94)": vm.ewma_volatility(),
        "Yang-Zhang 21D": vm.yang_zhang_volatility(21),
        "GARCH(1,1)": fits["GARCH(1,1)"].get("conditional_vol"),
        "GJR-GARCH": fits["GJR-GARCH(1,1)"].get("conditional_vol"),
    }
    series = pd.DataFrame({k: v for k, v in cols.items() if v is not None})
    forecast = vm.forecast_volatility(horizon=21)
    return summary, fits, series, forecast


@st.cache_data(show_spinner=False)
def cached_indicators(ohlc: pd.DataFrame, ticker: str) -> pd.DataFrame:
    from src.analytics.technical_indicators import TechnicalIndicators
    return TechnicalIndicators(ohlc, ticker=ticker).compute_all()


@st.cache_data(show_spinner=False)
def cached_markowitz(returns: pd.DataFrame, rfr: float, cov_method: str, max_w: float):
    from src.optimisation.markowitz_optimizer import MarkowitzOptimizer
    opt = MarkowitzOptimizer(returns, risk_free_rate=rfr, cov_method=cov_method)
    return (
        opt.max_sharpe(max_weight=max_w),
        opt.min_volatility(max_weight=max_w),
        opt.compare_portfolios(),
        opt.efficient_frontier(n_points=40, max_weight=max_w),
        opt.monte_carlo_portfolios(n_portfolios=3000),
    )


@st.cache_data(show_spinner=False)
def cached_stat_tests(returns: pd.Series, ticker: str):
    from src.analytics.statistical_tests import StatisticalTests
    stt = StatisticalTests(returns, name=ticker)
    battery = stt.run_full_battery()
    return battery.summary, stt.tail_index_estimation(), stt.descriptive_statistics()


# ─────────────────────────────────────────────
#  Page renderers
# ─────────────────────────────────────────────

def page_market_overview(data: dict, rfr: float):
    """Market overview: price charts, returns, correlations."""
    st.subheader("📊 Market Overview")
    prices = data["close"]
    benchmark = data["benchmark"]
    returns = prices.pct_change(fill_method=None)

    st.markdown("#### Normalised Price Performance")
    norm = prices / prices.bfill().iloc[0] * 100
    fig = go.Figure()
    colours = px.colors.qualitative.Set2
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col],
            name=col, line=dict(width=1.5, color=colours[i % len(colours)]),
        ))
    if benchmark is not None:
        bm_nav = (1 + benchmark.reindex(norm.index).fillna(0)).cumprod() * 100
        fig.add_trace(go.Scatter(
            x=bm_nav.index, y=bm_nav.values,
            name=f"Benchmark ({benchmark.name})",
            line=dict(dash="dot", color="#FF6B35", width=2),
        ))
    fig.update_layout(template="plotly_dark", height=420, hovermode="x unified",
                      title="Cumulative Performance (base=100)")
    st.plotly_chart(fig, use_container_width=True, key="market_overview_price")

    rf_d = (1 + rfr) ** (1 / TRADING_DAYS) - 1
    rows = {}
    for t in returns.columns:
        r = returns[t].dropna()
        if len(r) < 2:
            continue
        nav = (1 + r).cumprod()
        ex = r - rf_d
        rows[t] = {
            "CAGR": nav.iloc[-1] ** (TRADING_DAYS / len(r)) - 1,
            "Ann. Vol": r.std() * np.sqrt(TRADING_DAYS),
            "Sharpe": ex.mean() / ex.std() * np.sqrt(TRADING_DAYS) if ex.std() > 0 else np.nan,
            "Max DD": (nav / nav.cummax().clip(lower=1.0) - 1).min(),
        }
    stats = pd.DataFrame(rows).T

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Returns Summary")
        st.dataframe(
            stats.style.format({"CAGR": "{:.2%}", "Ann. Vol": "{:.2%}",
                                "Sharpe": "{:.2f}", "Max DD": "{:.2%}"}),
            width="stretch",
        )
        st.caption(f"Sharpe uses excess returns over the {rfr:.2%} risk-free rate.")

    with col2:
        st.markdown("#### Correlation Matrix")
        corr = returns.corr()
        fig_c = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, text_auto=".2f",
            template="plotly_dark",
        )
        fig_c.update_layout(height=320)
        st.plotly_chart(fig_c, use_container_width=True, key="market_corr")


def page_technical_analysis(data: dict):
    """Technical indicators, volatility models and statistical tests for one ticker."""
    st.subheader("📉 Technical Analysis")
    prices = data["close"]
    tickers = list(prices.columns)

    ticker = st.selectbox("Select Ticker", tickers, key="ta_ticker")

    ohlc = pd.DataFrame({
        "Open": data["open"][ticker], "High": data["high"][ticker],
        "Low": data["low"][ticker], "Close": data["close"][ticker],
        "Volume": data["volume"][ticker],
    }).dropna(subset=["Close"])
    indicators = cached_indicators(ohlc, ticker)
    st.caption(f"{indicators.shape[1]} indicator series computed from daily OHLCV.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ohlc.index, y=ohlc["Close"], name=ticker,
        line=dict(color="#00D4FF", width=1.5),
    ))
    for ma, color in [(20, "#FFD600"), (50, "#FF6B35"), (200, "#EF5350")]:
        col = f"SMA_{ma}"
        if col in indicators.columns:
            fig.add_trace(go.Scatter(
                x=indicators.index, y=indicators[col],
                name=f"SMA {ma}", line=dict(width=1, dash="dot", color=color),
            ))
    fig.update_layout(template="plotly_dark", height=380,
                      title=f"{ticker} Price + Moving Averages", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="ta_price")

    col1, col2 = st.columns(2)
    with col1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=indicators.index, y=indicators["RSI_14"],
            name="RSI 14", line=dict(color="#00D4FF"),
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#FF1744")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00C853")
        fig_rsi.update_layout(template="plotly_dark", height=260,
                              title="RSI (14)", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True, key="ta_rsi")

    with col2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=indicators.index, y=indicators["MACD"],
            name="MACD", line=dict(color="#00D4FF"),
        ))
        fig_macd.add_trace(go.Scatter(
            x=indicators.index, y=indicators["MACD_Signal"],
            name="Signal", line=dict(color="#FF6B35", dash="dot"),
        ))
        hist = indicators["MACD_Hist"]
        fig_macd.add_trace(go.Bar(
            x=indicators.index, y=hist, name="Histogram",
            marker_color=np.where(hist >= 0, "#00C853", "#FF1744"),
        ))
        fig_macd.update_layout(template="plotly_dark", height=260, title="MACD (12, 26, 9)")
        st.plotly_chart(fig_macd, use_container_width=True, key="ta_macd")

    col1, col2 = st.columns(2)
    with col1:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=ohlc.index, y=ohlc["Close"], name=ticker,
                                    line=dict(color="#00D4FF")))
        fig_bb.add_trace(go.Scatter(x=indicators.index, y=indicators["BB_Upper"],
                                    name="Upper", line=dict(color="#FFD600", dash="dot")))
        fig_bb.add_trace(go.Scatter(x=indicators.index, y=indicators["BB_Lower"],
                                    name="Lower", line=dict(color="#FFD600", dash="dot"),
                                    fill="tonexty", fillcolor="rgba(255, 214, 0, 0.05)"))
        fig_bb.update_layout(template="plotly_dark", height=300,
                             title="Bollinger Bands (20, 2σ)", hovermode="x unified")
        st.plotly_chart(fig_bb, use_container_width=True, key="ta_bb")
    with col2:
        fig_adx = go.Figure()
        adx_view = indicators.iloc[28:]      # skip the 2x14-day Wilder warm-up spike
        for c, colr in [("ADX", "#FFFFFF"), ("DI_Plus", "#00C853"), ("DI_Minus", "#FF1744")]:
            fig_adx.add_trace(go.Scatter(x=adx_view.index, y=adx_view[c], name=c,
                                         line=dict(color=colr, width=1.2)))
        fig_adx.add_hline(y=25, line_dash="dash", line_color="#90A4AE")
        fig_adx.update_layout(template="plotly_dark", height=300, title="ADX / DI (14)")
        st.plotly_chart(fig_adx, use_container_width=True, key="ta_adx")

    with st.expander("📋 Latest indicator values"):
        st.dataframe(indicators.iloc[-1].rename("Latest").to_frame(), width="stretch")

    # ── Volatility models ──
    st.markdown("#### Volatility Models")
    if st.toggle("Fit volatility models (GARCH, EGARCH, GJR-GARCH, EWMA, range-based)",
                 key="ta_vol_toggle"):
        with st.spinner("Fitting volatility models…"):
            summary, fits, series, forecast = cached_vol_models(ohlc, ticker)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(summary.style.format({"Latest Ann. Volatility": "{:.2%}"}),
                         width="stretch", hide_index=True)
        with c2:
            fig_v = go.Figure()
            for col in series.columns:
                fig_v.add_trace(go.Scatter(x=series.index, y=series[col] * 100,
                                           name=col, line=dict(width=1.2)))
            fig_v.update_layout(template="plotly_dark", height=330, yaxis_ticksuffix="%",
                                title="Annualised conditional / realised volatility")
            st.plotly_chart(fig_v, use_container_width=True, key="ta_vol_chart")
        rows = []
        for name, f in fits.items():
            if f:
                rows.append({"Model": name, "AIC": f.get("aic"), "BIC": f.get("bic"),
                             **{k: v for k, v in f.get("params", {}).items()}})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")
            st.caption("Parameters are estimated on percentage returns (arch library). "
                       "GJR 'gamma' > 0 / EGARCH 'gamma' < 0 indicate a leverage effect.")
        if not forecast.empty:
            st.line_chart(forecast * 100, height=220)
            st.caption("21-day ahead annualised volatility forecasts (%).")

    # ── Statistical tests ──
    st.markdown("#### Statistical Tests (daily returns)")
    if st.toggle("Run statistical test battery", key="ta_stats_toggle"):
        ret = ohlc["Close"].pct_change().dropna()
        with st.spinner("Running tests…"):
            battery, tails, desc = cached_stat_tests(ret, ticker)
        st.dataframe(battery, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hill tail-index estimates**")
            st.write({
                "Left tail α": tails.get("negative_tail_index"),
                "Left tail": tails.get("negative_tail_interpretation"),
                "Right tail α": tails.get("positive_tail_index"),
                "Right tail": tails.get("positive_tail_interpretation"),
            })
        with c2:
            st.dataframe(desc.set_index("Statistic"), width="stretch", height=280)


def page_factor_analysis(data: dict, rfr: float):
    """CAPM and Fama-French factor analysis."""
    st.subheader("🔬 Factor Analysis")
    prices = data["close"]
    benchmark = data["benchmark"]
    returns = prices.pct_change(fill_method=None)

    tab1, tab2, tab3 = st.tabs(["CAPM", "Fama-French 3F", "Factor Scores"])

    with tab1:
        st.markdown("#### CAPM Regression")
        if benchmark is None:
            st.warning("Benchmark required for CAPM analysis.")
        else:
            with st.spinner("Running CAPM…"):
                capm_df, sml, mkt_prem, rf_ann = cached_capm(returns, benchmark, rfr)
            st.dataframe(
                capm_df.style.format({
                    "Alpha (ann.)": "{:.2%}", "Beta": "{:.3f}", "R-Squared": "{:.3f}",
                    "Alpha t-stat": "{:.2f}", "Expected Return": "{:.2%}",
                    "Actual Return": "{:.2%}", "Abnormal Return": "{:.2%}", "CAGR": "{:.2%}",
                    "Treynor Ratio": "{:.3f}", "Info Ratio": "{:.3f}",
                    "Systematic Risk %": "{:.1%}", "Idiosyncratic Risk %": "{:.1%}",
                }),
                width="stretch",
            )
            st.caption("OLS on excess returns with Newey-West (HAC, 5 lags) t-stats. "
                       "Expected/Actual returns are arithmetic annualised means, so points "
                       "above the SML are exactly the stocks with positive alpha.")

            fig = px.scatter(
                sml, x="Beta", y="Actual Return", text="Ticker", color="Alpha",
                color_continuous_scale="RdYlGn", template="plotly_dark",
                title="Security Market Line",
            )
            betas = np.linspace(min(0.0, sml["Beta"].min()), sml["Beta"].max() * 1.2, 100)
            fig.add_trace(go.Scatter(
                x=betas, y=rf_ann + betas * mkt_prem, name="SML",
                line=dict(dash="dash", color="#90A4AE"),
            ))
            fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
            fig.update_layout(height=480, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True, key="factor_sml")

    with tab2:
        st.markdown("#### Fama-French 3-Factor Model")
        if benchmark is None:
            st.warning("Benchmark required for the market factor.")
        elif prices.shape[1] < 4:
            st.warning("Load at least 4 stocks to build SMB / HML long-short factors.")
        else:
            size = data["close"] * data["volume"]
            try:
                with st.spinner("Building factors and running regressions…"):
                    ff_df, factor_stats, factor_corr = cached_ff(returns, benchmark, prices, size, rfr)
                st.dataframe(
                    ff_df.style.format({
                        "Alpha (ann.)": "{:.2%}", "β MKT": "{:.3f}", "β SMB": "{:.3f}",
                        "β HML": "{:.3f}", "R²": "{:.3f}", "Adj R²": "{:.3f}",
                        "Alpha t-stat": "{:.2f}", "MKT Contrib": "{:.2%}",
                        "SMB Contrib": "{:.2%}", "HML Contrib": "{:.2%}",
                    }, na_rep="–"),
                    width="stretch",
                )
                st.markdown("#### Factor Return Summary")
                st.dataframe(
                    factor_stats.style.format({
                        "Ann. Return": "{:.2%}", "Ann. Volatility": "{:.2%}",
                        "Sharpe": "{:.3f}", "Skewness": "{:.2f}", "Kurtosis": "{:.2f}",
                    }),
                    width="stretch",
                )
                st.info(
                    "SMB and HML are **proxies built from the loaded universe**: portfolios "
                    "are formed each month-end on information available at that close "
                    "(SMB: 63-day average traded value; HML: price-based value score) and "
                    "held over the next month. With few stocks each leg holds only a "
                    "handful of names, so loadings are indicative."
                )
            except Exception as e:
                st.error(f"Fama-French failed: {e}")

    with tab3:
        st.markdown("#### Latest Factor Scores (cross-sectional z-scores)")
        with st.spinner("Computing factor scores…"):
            scores = cached_factor_scores(prices)
        st.dataframe(scores.style.format("{:.3f}", na_rep="–"), width="stretch")
        st.caption("Momentum = 12-1M, 6-1M and risk-adjusted momentum. Value and Quality "
                   "are price-based proxies (no fundamental data): value = long-term "
                   "reversal + proximity to 52-week low; quality = stability, 3Y return, "
                   "shallow drawdown, share of up days.")


def page_regime_detection(data: dict):
    """Market regime detection."""
    st.subheader("🌡️ Regime Detection")
    returns, mkt_name = market_proxy(data)
    market_index = (1 + returns).cumprod() * 100
    st.caption(f"Market series: **{mkt_name}**")

    col1, col2 = st.columns([1, 3])
    with col1:
        method = st.selectbox("Method", ["HMM", "Markov Switching", "Rule-Based"], key="reg_method")
        if method == "HMM":
            n_states = st.selectbox("HMM States", [2, 3, 4], index=1, key="reg_hmm_n")
        elif method == "Markov Switching":
            n_states = st.selectbox("Regimes", [2, 3], index=0, key="reg_mrs_n")

    with col2:
        with st.spinner("Detecting regimes…"):
            current, history, rule_stats = cached_rule_regimes(returns, market_index)
        st.success(
            f"**Rule-based consensus: {current['consensus']}**  "
            f"(agreement {current['confidence']:.0%} of {sum(current['votes'].values())} signals, "
            f"as of {current['date']})"
        )

    COLOURS = {"Bull": "#00C853", "Sideways": "#FFD600",
               "Bear": "#FF1744", "High Vol": "#FF6B35"}

    def regime_chart(labels: pd.Series, title: str, key: str):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=market_index.index, y=market_index.values,
                                 name=mkt_name, line=dict(color="#00D4FF")))
        regime_shapes(fig, labels, COLOURS)
        fig.update_layout(template="plotly_dark", height=420, title=title,
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key=key)

    if method == "HMM":
        with st.spinner(f"Fitting HMM ({n_states} states)…"):
            result, probs = cached_hmm(returns, n_states)
        regime_chart(result.regime_labels, f"Gaussian HMM — Viterbi path ({n_states} states)", "regime_chart")
        st.markdown("#### Regime Statistics")
        st.dataframe(result.state_stats.style.format({
            "Mean Return (ann.)": "{:.2%}", "Volatility (ann.)": "{:.2%}",
            "Sharpe": "{:.3f}", "% of Time": "{:.1%}", "Avg Duration (days)": "{:.1f}",
            "Transition Prob (stay)": "{:.3f}",
        }), width="stretch")
        st.caption(f"Log-likelihood {result.log_likelihood:,.1f} · AIC {result.aic:,.1f} · "
                   f"BIC {result.bic:,.1f}. Best of 5 EM restarts.")
        fig_p = px.area(probs, template="plotly_dark", height=280,
                        title="Smoothed Regime Probabilities", labels={"value": "Probability", "variable": "Regime", "index": "Date"})
        st.plotly_chart(fig_p, use_container_width=True, key="regime_probs")
        st.caption("Smoothed probabilities and the Viterbi path use the whole sample "
                   "(including future data) — they describe history and must not be "
                   "used as trading signals.")

    elif method == "Markov Switching":
        with st.spinner(f"Fitting Hamilton Markov-switching model ({n_states} regimes)…"):
            result, smoothed, filtered = cached_mrs(returns, n_states)
        regime_chart(result.regime_labels, f"Hamilton Markov Switching ({n_states} regimes)", "mrs_chart")
        st.markdown("#### Regime Statistics")
        st.dataframe(result.regime_stats.style.format({
            "Ann. Return": "{:.2%}", "Ann. Volatility": "{:.2%}", "Sharpe": "{:.3f}",
            "% of Time": "{:.1%}", "Stay Probability": "{:.3f}",
        }), width="stretch", hide_index=True)
        tm = pd.DataFrame(result.transition_matrix,
                          index=[str(result.label_map.get(i, i)) for i in range(n_states)],
                          columns=[f"→ {result.label_map.get(i, i)}" for i in range(n_states)])
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Transition matrix**")
            st.dataframe(tm.rename_axis("From").reset_index().style.format(
                {c: "{:.4f}" for c in tm.columns}), width="stretch", hide_index=True)
        with c2:
            view = st.radio("Probabilities", ["Filtered (real-time)", "Smoothed (full-sample)"],
                            horizontal=True, key="mrs_prob_view")
            probs = filtered if view.startswith("Filtered") else smoothed
            st.plotly_chart(px.area(probs, template="plotly_dark", height=260,
                                    labels={"value": "Probability", "variable": "Regime", "index": "Date"}),
                            use_container_width=True, key="mrs_probs")

    else:
        regime_chart(history["Consensus"], "Rule-based consensus regime", "rule_chart")
        recent = history.tail(20).copy()
        recent.index = pd.DatetimeIndex(recent.index).strftime("%Y-%m-%d")
        st.dataframe(recent.rename_axis("Date"), width="stretch")
        st.markdown("#### Regime Statistics")
        st.dataframe(rule_stats.rename_axis("Regime").reset_index().style.format({
            "Ann. Return": "{:.2%}", "Ann. Volatility": "{:.2%}",
            "Sharpe": "{:.3f}", "% of Time": "{:.1%}",
            "Best Day": "{:.2%}", "Worst Day": "{:.2%}",
        }), width="stretch", hide_index=True)


STRATEGIES = [
    "Cross-Sectional Momentum",
    "Time-Series Momentum",
    "Risk-Adjusted Momentum",
    "Bollinger Band Reversion",
    "RSI Mean Reversion",
    "Factor Strategy",
    "Regime Adaptive",
    "Pairs Trading (Kalman)",
    "ML Alpha (walk-forward RF)",
]


def page_strategy_backtest(data: dict, rfr: float):
    """Strategy selection, configuration, and backtesting."""
    st.subheader("⚡ Strategy Backtester")
    prices = data["close"]
    benchmark = data["benchmark"]
    n_assets = prices.shape[1]

    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_name = st.selectbox("Strategy", STRATEGIES, key="bt_strategy")
    with col2:
        initial_capital = st.number_input(
            "Capital (AUD)", min_value=10_000, value=1_000_000, step=100_000, format="%d",
        )
    with col3:
        commission = st.number_input(
            "Commission (one-way)", min_value=0.0, max_value=0.01,
            value=0.001, step=0.0005, format="%.4f",
        )

    with st.expander("⚙️ Strategy Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_long = st.slider("Long Positions", 1, max(1, n_assets),
                               max(1, n_assets // 2), key="bt_nlong")
            lookback = st.selectbox("Momentum Lookback (days)", [63, 126, 252, 504], index=2)
        with col2:
            skip = st.selectbox("Skip Period (days)", [0, 5, 21], index=2)
            rebalance = st.selectbox("Rebalance", ["monthly", "weekly", "quarterly"])
        with col3:
            signal_lag = st.selectbox("Signal Lag (days)", [1, 2, 3], index=0)
            allow_short = st.checkbox("Allow short positions", value=False,
                                      help="Mean-reversion and pairs strategies emit short "
                                           "signals; when off, short legs are dropped.")
        if strategy_name == "Pairs Trading (Kalman)":
            c1, c2 = st.columns(2)
            with c1:
                pair1 = st.selectbox("Leg 1", list(prices.columns), index=0, key="bt_p1")
            with c2:
                pair2 = st.selectbox("Leg 2", list(prices.columns),
                                     index=min(1, n_assets - 1), key="bt_p2")
        if n_long >= n_assets and strategy_name in (
            "Cross-Sectional Momentum", "Risk-Adjusted Momentum", "Factor Strategy",
            "Regime Adaptive", "ML Alpha (walk-forward RF)",
        ):
            st.warning("Long positions ≥ universe size: the ranking strategy just holds "
                       "every stock equally. Reduce 'Long Positions' or add tickers.")
        regime_confirm = 5
        if strategy_name == "Regime Adaptive":
            regime_confirm = st.slider(
                "Regime confirmation (days)", 1, 20, 5, key="bt_regime_confirm",
                help="A new regime must persist this many consecutive days before the "
                     "strategy switches. 1 = switch immediately (high turnover around "
                     "moving-average crossovers).",
            )
        if strategy_name == "ML Alpha (walk-forward RF)":
            st.info("Walk-forward Random Forest: re-trained every ~6 months on data "
                    "available at the time (21-day purge), first predictions after 2 years. "
                    "Takes roughly 30-60s for 8 stocks over 7 years.")

    run_btn = st.button("▶ Run Backtest", type="primary")
    if not run_btn:
        st.info("Configure parameters above and click **▶ Run Backtest**.")
        return

    from src.backtesting.backtester import BacktestConfig, Backtester
    from src.backtesting.performance_metrics import PerformanceMetrics

    extra_info = {}
    with st.spinner(f"Running {strategy_name} backtest…"):
        try:
            cfg_rebalance = "signal"
            bt_prices = prices
            if strategy_name == "Cross-Sectional Momentum":
                from src.strategies.momentum_strategy import CrossSectionalMomentum
                signals = CrossSectionalMomentum(
                    prices, lookback=lookback, skip=skip,
                    n_long=n_long, rebalance_freq=rebalance,
                ).generate_signals()
            elif strategy_name == "Time-Series Momentum":
                from src.strategies.momentum_strategy import TimeSeriesMomentum
                signals = TimeSeriesMomentum(prices, lookback=lookback,
                                             allow_short=allow_short).generate_signals()
                cfg_rebalance = rebalance     # TSM has no internal schedule
            elif strategy_name == "Risk-Adjusted Momentum":
                from src.strategies.momentum_strategy import RiskAdjustedMomentum
                signals = RiskAdjustedMomentum(prices, lookback=lookback, skip=skip,
                                               n_long=n_long).generate_signals()
            elif strategy_name == "Bollinger Band Reversion":
                from src.strategies.mean_reversion_strategy import BollingerBandReversion
                signals = BollingerBandReversion(prices).generate_signals()
            elif strategy_name == "RSI Mean Reversion":
                from src.strategies.mean_reversion_strategy import RSIMeanReversion
                signals = RSIMeanReversion(prices).generate_signals()
            elif strategy_name == "Factor Strategy":
                from src.strategies.factor_strategy import FactorStrategy
                signals = FactorStrategy(prices, n_long=n_long,
                                         rebalance_freq=rebalance).generate_signals()
            elif strategy_name == "Regime Adaptive":
                from src.strategies.factor_strategy import RegimeAdaptiveStrategy
                mret, _ = market_proxy(data)
                signals = RegimeAdaptiveStrategy(
                    prices, (1 + mret).cumprod(), n_positions=n_long,
                    min_regime_days=regime_confirm,
                ).generate_signals()
            elif strategy_name == "Pairs Trading (Kalman)":
                if pair1 == pair2:
                    st.error("Choose two different tickers for the pair.")
                    return
                from src.strategies.mean_reversion_strategy import PairsTrading
                pt = PairsTrading(prices, pair1, pair2, hedge_method="kalman")
                signals = pt.generate_signals()
                bt_prices = prices[[pair1, pair2]]
                eg = pt.cointegration_test()
                extra_info["Engle-Granger p-value"] = f"{eg.p_value:.3f}"
                extra_info["spread"] = pt.spread_analysis()
                if not allow_short:
                    st.warning("Pairs trading needs a short leg — enable 'Allow short "
                               "positions' for a market-neutral pair.")
            else:
                from src.machine_learning.alpha_models import AlphaModelPipeline
                pipe = AlphaModelPipeline(
                    prices, model_type="rf", horizon=21,
                    model_params={"n_estimators": 100, "max_depth": 6,
                                  "min_samples_leaf": 50},
                )
                signals = pipe.generate_signals(n_long=n_long, retrain_every=126,
                                                min_train_days=504,
                                                rebalance_freq=rebalance)
                if not pipe.oos_ic.empty:
                    extra_info["OOS rank IC (mean)"] = f"{pipe.oos_ic.mean():.3f}"
                    extra_info["IC t-stat (non-overlapping)"] = (
                        f"{pipe.ic_tstat(pipe.oos_ic, 21):.2f}"
                    )

            if signals is None or signals.empty:
                st.error("Signal generation returned no signals (not enough history?).")
                return

            config = BacktestConfig(
                initial_capital=float(initial_capital),
                commission=commission,
                slippage=0.0005,
                signal_lag=signal_lag,
                risk_free_rate=rfr,
                allow_short=allow_short,
                rebalance_freq=cfg_rebalance,
            )
            result = Backtester(prices=bt_prices, signals=signals, config=config,
                                benchmark=benchmark, strategy_name=strategy_name).run()
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    if result.returns.empty or len(result.returns) < 5:
        st.warning(
            "The strategy never took a position. It usually needs more history than "
            "the selected range provides (e.g. 12-month momentum needs ~13 months, "
            "the ML strategy ~2 years) - try an earlier start date."
        )
        return

    pm = PerformanceMetrics(result)
    c = st.columns(6)
    c[0].metric("CAGR", f"{pm.cagr():.2%}")
    c[1].metric("Sharpe", f"{pm.sharpe_ratio():.2f}")
    c[2].metric("Sortino", f"{pm.sortino_ratio():.2f}")
    c[3].metric("Max DD", f"{pm.max_drawdown():.2%}")
    c[4].metric("Ann. Turnover", f"{pm.avg_annual_turnover():.0%}")
    c[5].metric("Rebalances", f"{len(result.trades):,}")
    st.caption(
        f"Period {result.returns.index[0].date()} → {result.returns.index[-1].date()} "
        "(starts when capital is first deployed). Costs: commission + 5 bp slippage "
        "per side, charged on traded value; positions drift between rebalances."
    )
    if extra_info:
        cols = st.columns(len([k for k in extra_info if k != "spread"]) or 1)
        for i, (k, v) in enumerate((k, v) for k, v in extra_info.items() if k != "spread"):
            cols[i].metric(k, v)

    nav = result.nav
    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(x=nav.index, y=nav.values, name=strategy_name,
                                 line=dict(color="#00D4FF", width=2)))
    if benchmark is not None:
        bm = benchmark.reindex(nav.index).fillna(0)
        bm.iloc[0] = 0.0
        bm_nav = (1 + bm).cumprod() * float(nav.iloc[0])
        fig_nav.add_trace(go.Scatter(x=bm_nav.index, y=bm_nav.values, name="Benchmark",
                                     line=dict(color="#FF6B35", dash="dot", width=1.5)))
    fig_nav.update_layout(template="plotly_dark", height=400,
                          title="Strategy Equity Curve", hovermode="x unified")
    st.plotly_chart(fig_nav, use_container_width=True, key="bt_nav")

    dd = pm.drawdown_series()
    fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, name="Drawdown",
                                  line=dict(color="#FF1744"), fill="tozeroy",
                                  fillcolor="rgba(255,23,68,0.1)"))
    fig_dd.update_layout(template="plotly_dark", height=280,
                         title="Drawdown (%)", yaxis_ticksuffix="%")
    st.plotly_chart(fig_dd, use_container_width=True, key="bt_dd")

    if "spread" in extra_info:
        sp = extra_info["spread"]
        fig_sp = go.Figure(go.Scatter(x=sp.index, y=sp["z_score"], name="z-score",
                                      line=dict(color="#00D4FF", width=1)))
        for y in (2, -2):
            fig_sp.add_hline(y=y, line_dash="dash", line_color="#FF6B35")
        fig_sp.update_layout(template="plotly_dark", height=260,
                             title="Kalman spread z-score (entry ±2, exit ±0.5)")
        st.plotly_chart(fig_sp, use_container_width=True, key="bt_spread")

    with st.expander("📋 Full Metrics Table"):
        st.dataframe(result.metrics, width="stretch", height=600, hide_index=True)

    with st.expander("📅 Monthly Returns Heatmap"):
        monthly = pm.monthly_returns()
        fig_heat = px.imshow(
            monthly.drop(columns="Annual") * 100,
            color_continuous_scale="RdYlGn", text_auto=".1f",
            template="plotly_dark", zmin=-10, zmax=10,
            title="Monthly Returns (%)", aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True, key="bt_monthly")

    with st.expander("🔁 Rebalance log"):
        st.dataframe(result.trades, width="stretch", hide_index=True)


def page_portfolio_optimisation(data: dict, rfr: float):
    """Portfolio optimisation with Markowitz and Black-Litterman."""
    st.subheader("⚖️ Portfolio Optimisation")
    prices = data["close"]
    returns = prices.pct_change(fill_method=None).dropna()

    tab1, tab2 = st.tabs(["Markowitz", "Black-Litterman"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            cov_method = st.selectbox("Covariance Method", ["ledoit_wolf", "sample", "ewma"])
        with c2:
            max_w = st.slider("Max weight per stock", 0.10, 1.0, 1.0, 0.05, key="opt_maxw")

        with st.spinner("Optimising portfolios…"):
            ms, mv, comparison, frontier, mc_ports = cached_markowitz(
                returns, rfr, cov_method, max_w
            )

        st.markdown("#### Portfolio Comparison (unconstrained long-only)")
        st.dataframe(
            comparison.style.format({
                "Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.3f}",
                "Max Weight": "{:.1%}", "Min Weight": "{:.1%}",
            }),
            width="stretch",
        )
        st.caption("In-sample: expected returns are historical means, which are noisy — "
                   "this is why max-Sharpe weights are unstable and why Black-Litterman exists.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mc_ports["Volatility"] * 100, y=mc_ports["Return"] * 100,
            mode="markers", name="Random Portfolios",
            marker=dict(size=3, color=mc_ports["Sharpe"], colorscale="Viridis", opacity=0.5),
        ))
        if not frontier.empty:
            fig.add_trace(go.Scatter(
                x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
                name="Efficient Frontier", mode="lines",
                line=dict(color="#FF6B35", width=3),
            ))
        for r, port_name, color in [(ms, "Max Sharpe", "#00D4FF"), (mv, "Min Volatility", "#FFD600")]:
            fig.add_trace(go.Scatter(
                x=[r.expected_volatility * 100], y=[r.expected_return * 100],
                mode="markers+text", name=port_name, text=[port_name],
                textposition="top center",
                marker=dict(size=12, color=color, symbol="star"),
            ))
        fig.update_layout(
            template="plotly_dark", height=500, title="Efficient Frontier",
            xaxis=dict(title="Volatility (%)", ticksuffix="%"),
            yaxis=dict(title="Expected Return (%)", ticksuffix="%"),
        )
        st.plotly_chart(fig, use_container_width=True, key="opt_frontier")

        st.markdown("#### Max Sharpe Weights")
        w = ms.weights[ms.weights > 0.001].sort_values(ascending=False)
        fig_w = px.bar(x=w.index, y=w.values * 100, template="plotly_dark",
                       title="Portfolio Weights (%)", labels={"x": "Ticker", "y": "Weight (%)"})
        st.plotly_chart(fig_w, use_container_width=True, key="opt_weights")

    with tab2:
        st.markdown("#### Black-Litterman Optimisation")
        st.markdown(
            "Express views on expected returns. The model blends them with the "
            "equilibrium returns implied by an equal-weighted prior portfolio "
            "(market-cap weights are not available from Yahoo)."
        )
        from src.optimisation.black_litterman import BlackLitterman

        c1, c2 = st.columns(2)
        with c1:
            n_views = st.number_input("Number of Views", min_value=0, max_value=5, value=2)
        with c2:
            bl_max_w = st.slider("Max weight per stock", 0.10, 1.0, 0.40, 0.05, key="bl_maxw",
                                 help="Long-only max-Sharpe tends to concentrate in the few "
                                      "assets with the best posterior returns; a cap keeps "
                                      "the portfolio diversified.")
        tick_list = prices.columns.tolist()
        views = []
        for i in range(int(n_views)):
            st.markdown(f"**View {i+1}**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                vtype = st.selectbox("Type", ["Absolute", "Relative"], key=f"vtype_{i}")
            with c2:
                t1 = st.selectbox("Ticker (Long)", tick_list, index=i % len(tick_list), key=f"t1_{i}")
            with c3:
                t2 = (st.selectbox("Ticker (Short)", tick_list,
                                   index=(i + 1) % len(tick_list), key=f"t2_{i}")
                      if vtype == "Relative" else None)
            with c4:
                label = "Total return p.a. (%)" if vtype == "Absolute" else "Outperformance p.a. (%)"
                ret_view = st.number_input(label, value=12.0 if vtype == "Absolute" else 3.0,
                                           step=1.0, key=f"ret_{i}") / 100
                conf = st.slider("Confidence", 0.05, 0.95, 0.5, key=f"conf_{i}")
            views.append((vtype, t1, t2, ret_view, conf))

        if st.button("▶ Run BL Optimisation", key="bl_run"):
            bl = BlackLitterman(returns, risk_free_rate=rfr)
            errors = []
            for vtype, t1, t2, ret_view, conf in views:
                try:
                    if vtype == "Absolute":
                        bl.add_absolute_view(t1, ret_view, conf)
                    else:
                        bl.add_relative_view(t1, t2, ret_view, conf)
                except ValueError as e:
                    errors.append(str(e))
            for e in errors:
                st.warning(f"View skipped: {e}")
            n_bl = returns.shape[1]
            cap = max(bl_max_w, 1.0 / n_bl)      # a cap below 1/N is infeasible
            if cap > bl_max_w:
                st.caption(f"Max weight raised to {cap:.0%} — with {n_bl} stocks a lower cap cannot sum to 100%.")
            with st.spinner("Running Black-Litterman…"):
                bl_result = bl.optimise(max_weight=cap)

            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Return", f"{bl_result.expected_return:.2%}")
            c2.metric("Expected Vol", f"{bl_result.expected_volatility:.2%}")
            c3.metric("Sharpe Ratio", f"{bl_result.sharpe_ratio:.3f}")

            ev = bl.equilibrium_vs_posterior()
            fig_bl = go.Figure()
            fig_bl.add_trace(go.Bar(name="Equilibrium", x=ev.index,
                                    y=ev["Equilibrium Return"] * 100, marker_color="#90A4AE"))
            fig_bl.add_trace(go.Bar(name="Posterior", x=ev.index,
                                    y=ev["Posterior Return"] * 100, marker_color="#00D4FF"))
            fig_bl.update_layout(barmode="group", template="plotly_dark", height=380,
                                 title="Equilibrium vs BL Posterior Returns (%, total)",
                                 yaxis_ticksuffix="%")
            st.plotly_chart(fig_bl, use_container_width=True, key="bl_returns")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Optimal weights**")
                st.dataframe(bl_result.optimal_weights.rename("Weight").to_frame()
                             .style.format("{:.1%}"), width="stretch")
            with c2:
                if not bl_result.views_summary.empty:
                    st.markdown("**Views**")
                    st.dataframe(bl_result.views_summary.style.format(
                        {"Q (Return)": "{:.2%}", "Confidence": "{:.2f}"}),
                        width="stretch", hide_index=True)


def page_monte_carlo(data: dict):
    """Monte Carlo simulation and VaR backtesting."""
    st.subheader("🎲 Monte Carlo Simulation")
    prices = data["close"]

    ticker = st.selectbox("Asset to Simulate", prices.columns.tolist(), key="mc_ticker")
    px_series = prices[ticker].dropna()
    returns_series = px_series.pct_change().dropna()
    S0 = float(px_series.iloc[-1])

    col1, col2, col3 = st.columns(3)
    with col1:
        n_sim = st.selectbox("Simulations", [1000, 5000, 10000, 50000], index=1)
    with col2:
        horizon = st.selectbox("Horizon (days)", [21, 63, 126, 252], index=3)
    with col3:
        model = st.selectbox("Model", ["GBM", "Jump Diffusion", "Bootstrap"])

    if st.button("▶ Run Simulation", key="mc_run"):
        from src.simulation.monte_carlo_paths import MonteCarloSimulator
        mc = MonteCarloSimulator(returns_series, n_simulations=n_sim, horizon=horizon)
        with st.spinner(f"Running {n_sim:,} simulations…"):
            if model == "GBM":
                result = mc.simulate_gbm(S0=S0)
            elif model == "Jump Diffusion":
                result = mc.simulate_jump_diffusion(S0=S0)
            else:
                result = mc.simulate_bootstrap(S0=S0)

        c = st.columns(4)
        c[0].metric("Expected Return", f"{result.expected_return:.2%}")
        c[1].metric("P(Loss)", f"{result.prob_loss:.1%}")
        c[2].metric(f"VaR 95% ({horizon}d)", f"{result.var_95:.2%}")
        c[3].metric(f"CVaR 95% ({horizon}d)", f"{result.cvar_95:.2%}")
        st.caption(f"Calibrated to {ticker}'s daily history: μ = {mc.mu:.1%} p.a., "
                   f"σ = {mc.sigma:.1%} p.a. These are statistical scenarios, not forecasts.")

        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_sim, min(50, n_sim), replace=False)
        t_axis = list(range(horizon + 1))
        fig = go.Figure()
        for path in result.paths[sample_idx]:
            fig.add_trace(go.Scatter(x=t_axis, y=path, mode="lines",
                                     line=dict(color="rgba(0,212,255,0.08)", width=0.5),
                                     showlegend=False))
        for p, c_ in [(5, "#FF1744"), (25, "#FF6B35"), (50, "#00D4FF"),
                      (75, "#FF6B35"), (95, "#FF1744")]:
            fig.add_trace(go.Scatter(x=t_axis, y=np.percentile(result.paths, p, axis=0),
                                     name=f"P{p}", line=dict(width=2, color=c_)))
        fig.update_layout(template="plotly_dark", height=450,
                          title=f"{ticker} {model} Simulation ({n_sim:,} paths, {horizon} days)",
                          xaxis_title="Days", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True, key="mc_paths")

        fig_hist = px.histogram(result.final_prices, nbins=100, template="plotly_dark",
                                title="Distribution of Final Prices")
        fig_hist.add_vline(x=S0, line_dash="dash", line_color="#FFD600",
                           annotation_text=f"Start ${S0:.2f}")
        fig_hist.update_layout(showlegend=False, xaxis_title="Price ($)")
        st.plotly_chart(fig_hist, use_container_width=True, key="mc_hist")

    with st.expander("✅ VaR model backtest (Kupiec POF test)"):
        conf = st.selectbox("VaR confidence", [0.95, 0.99], key="kupiec_conf")
        window = st.selectbox("Estimation window (days)", [250, 500], key="kupiec_win")
        from src.simulation.monte_carlo_paths import StressTester
        st_obj = StressTester(returns_series.to_frame(ticker), {ticker: 1.0})
        st.dataframe(st_obj.var_backtest(confidence=conf, window=window),
                     width="stretch", hide_index=True)
        st.caption("Rolling historical-simulation VaR estimated from the previous "
                   f"{window} days only, compared with the next day's return. Kupiec's "
                   "likelihood-ratio test checks whether the exceedance rate matches "
                   "1 − confidence (H₀: the VaR model is correctly calibrated).")


def page_options_pricer(data: dict, rfr: float):
    """Black-Scholes and Monte Carlo options pricing."""
    st.subheader("💹 Options Pricer")
    from src.options.black_scholes import BlackScholes, ImpliedVolatility, MonteCarloOptionPricer

    prices = data["close"]
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.selectbox("Underlying", prices.columns.tolist(), key="opt_ticker")
        S = float(prices[ticker].dropna().iloc[-1])
        st.metric("Current Spot", f"${S:.2f}")
        K = st.number_input("Strike (K)", value=float(round(S, 0)), step=0.5)
        expiry_days = st.slider("Days to Expiry", 1, 730, 90)
        T = expiry_days / 365
    with col2:
        hist_vol = float(prices[ticker].pct_change().dropna().tail(63).std() * np.sqrt(TRADING_DAYS))
        sigma = st.slider("Volatility (%)", 5, 100, int(np.clip(round(hist_vol * 100), 5, 100))) / 100
        st.caption(f"Default = 63-day historical volatility ({hist_vol:.1%}).")
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
        q = st.number_input("Dividend Yield (continuous)", 0.0, 0.10, 0.03, step=0.005)

    bs = BlackScholes(S=S, K=K, T=T, r=rfr, sigma=sigma, q=q)
    ot = option_type.lower()
    greeks = bs.greeks(ot)
    price = bs.price(ot)

    c = st.columns(5)
    c[0].metric("Option Price", money(price))
    c[1].metric("Delta", f"{greeks.delta:.4f}")
    c[2].metric("Gamma", f"{greeks.gamma:.6f}")
    c[3].metric("Theta/day", money(greeks.theta))
    c[4].metric("Vega/1% vol", money(greeks.vega))
    with st.expander("All Greeks"):
        st.dataframe(greeks.to_dataframe().style.format({"Value": "{:.6f}"}),
                     width="stretch", hide_index=True)
        parity = bs.put_call_parity_check()
        st.caption(f"Put-call parity error: {parity['parity_error']:.2e} "
                   "(European, continuous dividend yield). ASX single-stock options are "
                   "American-style, so these are European approximations.")

    st.markdown("#### Payoff at Expiry")
    S_range = np.linspace(S * 0.5, S * 1.5, 300)
    payoff = (np.maximum(S_range - K, 0) if ot == "call" else np.maximum(K - S_range, 0)) - price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=payoff, name=f"{option_type} P&L",
                             line=dict(color="#00D4FF", width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="#90A4AE")
    # put the two labels on opposite sides so they never overlap when K ≈ S
    spot_left = S <= K
    fig.add_vline(x=S, line_dash="dot", line_color="#FFD600",
                  annotation_text=f"Spot ${S:.2f}",
                  annotation_position="top left" if spot_left else "top right",
                  annotation_font_color="#FFD600")
    fig.add_vline(x=K, line_dash="dot", line_color="#FF6B35",
                  annotation_text=f"Strike ${K:.2f}",
                  annotation_position="bottom right" if spot_left else "bottom left",
                  annotation_font_color="#FF6B35")
    fig.update_layout(template="plotly_dark", height=380,
                      title=f"Long {option_type} P&L at Expiry (premium ${price:.2f})",
                      xaxis_title="Underlying Price ($)", yaxis_title="P&L ($)")
    st.plotly_chart(fig, use_container_width=True, key="opt_payoff")

    with st.expander("📊 Greeks Sensitivity"):
        S_arr = np.linspace(S * 0.7, S * 1.3, 200)
        deltas = [BlackScholes(s, K, T, rfr, sigma, q).delta(ot) for s in S_arr]
        gammas = [BlackScholes(s, K, T, rfr, sigma, q).gamma() for s in S_arr]
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=S_arr, y=deltas, name="Delta", line=dict(color="#00D4FF")))
        fig_g.add_trace(go.Scatter(x=S_arr, y=gammas, name="Gamma",
                                   line=dict(color="#FF6B35"), yaxis="y2"))
        fig_g.update_layout(template="plotly_dark", height=340, title="Delta & Gamma vs Spot",
                            yaxis=dict(title="Delta"),
                            yaxis2=dict(title="Gamma", overlaying="y", side="right"))
        st.plotly_chart(fig_g, use_container_width=True, key="opt_greeks")

    with st.expander("🔎 Implied volatility (Brent's method)"):
        mkt = st.number_input("Observed option price ($)", min_value=0.0,
                              value=float(round(price, 2)), step=0.05, key="iv_px")
        iv = ImpliedVolatility(mkt, S, K, T, rfr, q=q, option_type=ot).calculate()
        if np.isnan(iv):
            st.error("No implied volatility: the price is outside the no-arbitrage bounds.")
        else:
            st.metric("Implied volatility", f"{iv:.2%}")

    with st.expander("🎲 Monte Carlo exotic options"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_paths = st.selectbox("Paths", [10_000, 20_000, 50_000], index=1, key="mc_opt_n")
        with c2:
            barrier = st.number_input("Barrier ($)", value=float(round(S * 0.85, 2)), key="mc_bar")
        with c3:
            btype = st.selectbox("Barrier type",
                                 ["down-and-out", "down-and-in", "up-and-out", "up-and-in"])
        if st.button("▶ Price exotics", key="mc_opt_run"):
            with st.spinner("Simulating risk-neutral paths…"):
                # daily monitoring on trading days, capped to keep memory bounded
                n_steps = int(np.clip(round(T * TRADING_DAYS), 1, TRADING_DAYS))
                mcp = MonteCarloOptionPricer(S, K, T, rfr, sigma, q=q, n_simulations=n_paths,
                                             n_steps=n_steps)
                rows = []
                eu = mcp.european_call()
                rows.append(("European call", eu["mc_price"], eu["std_error"], eu["bs_price"]))
                asian = mcp.asian_call("arithmetic")
                rows.append(("Asian call (arithmetic)", asian["mc_price"], asian["std_error"], np.nan))
                bar = mcp.barrier_call(barrier, btype)
                rows.append((f"Barrier call ({btype})", bar["mc_price"], bar["std_error"], np.nan))
                lb = mcp.lookback_call()
                rows.append(("Lookback call (floating)", lb["mc_price"], np.nan, np.nan))
                dg = mcp.digital_call()
                rows.append(("Digital call ($1)", dg["mc_price"], np.nan, dg["bs_price"]))
            st.dataframe(pd.DataFrame(rows, columns=["Option", "MC price", "Std error",
                                                     "Analytical"]).set_index("Option")
                         .style.format("{:.4f}", na_rep="–"), width="stretch")
            st.caption("Antithetic variates; barrier monitored once per day (discrete).")


def money(x: float, dp: int = 4) -> str:
    """Format currency with the sign before the dollar sign (-$0.02, not $-0.02)."""
    return f"-${abs(x):,.{dp}f}" if x < 0 else f"${x:,.{dp}f}"


def page_portfolio_analytics(data: dict, rfr: float):
    """Full embedded portfolio analytics dashboard."""
    from src.portfolio_analytics.portfolio_dashboard import run_portfolio_dashboard
    run_portfolio_dashboard(
        preloaded_prices=data["close"],
        preloaded_benchmark=data["benchmark"],
        preloaded_rfr=rfr,
    )


# ─────────────────────────────────────────────
#  Main app
# ─────────────────────────────────────────────

def main():
    st.markdown('<p class="main-header">ASX Quant Terminal</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Quantitative research platform · ASX Equities · Python</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    config = render_sidebar()

    if "data" not in st.session_state:
        st.session_state.data = None

    if config["load"]:
        # Feedback is rendered in the sidebar so the main layout above the tabs
        # is identical on every run (otherwise Streamlit resets the active tab
        # on the next interaction).
        with st.sidebar:
            if len(config["tickers"]) < 2:
                st.error("Enter at least two tickers.")
            else:
                with st.spinner("Loading market data…"):
                    try:
                        if config["source"].startswith("Synthetic"):
                            data = load_synthetic_data(tuple(config["tickers"]),
                                                       config["start"], config["end"])
                        else:
                            data = load_market_data(tuple(config["tickers"]), config["start"],
                                                    config["end"], config["benchmark"])
                        if data["close"].empty or len(data["available"]) < 2:
                            st.error("No usable data returned. Check tickers and dates.")
                        else:
                            st.session_state.data = data
                            st.session_state.missing = sorted(
                                set(config["tickers"]) - set(data["available"])
                            )
                            st.toast(f"✅ Loaded {len(data['available'])} tickers, "
                                     f"{len(data['close'])} trading days ({data['source']}).")
                    except Exception as e:
                        st.error(f"Data load failed: {e}")

    data = st.session_state.data
    if data is None:
        st.info("👈 Configure your universe in the sidebar and click **📥 Load Data** to begin.")
        st.markdown("""
        ### Platform Capabilities
        | Module | Description |
        |---|---|
        | 📊 Market | Price performance, return statistics, correlations |
        | 📉 Technical | 55+ indicator series, GARCH/EGARCH/GJR & range-based volatility, statistical test battery |
        | 🔬 Factors | CAPM + Security Market Line, Fama-French 3F, momentum/value/quality scores |
        | 🌡️ Regimes | Gaussian HMM, Hamilton Markov switching, rule-based consensus |
        | ⚡ Backtest | 9 strategies incl. Kalman pairs and walk-forward ML, with costs and tearsheet |
        | ⚖️ Optimisation | Markowitz, Ledoit-Wolf, risk parity, Black-Litterman |
        | 🎲 Monte Carlo | GBM, jump diffusion, block bootstrap, Kupiec VaR backtest |
        | 💹 Options | Black-Scholes-Merton, full Greeks, implied vol, MC exotics |
        | 📖 Portfolio | Risk decomposition, attribution, exposures, stress tests |
        """)
        return

    if data["source"].startswith("Synthetic"):
        st.warning("⚠️ **Synthetic demo data** — simulated prices, not real market data.")
    if st.session_state.get("missing"):
        st.warning(f"No data returned for: {', '.join(st.session_state.missing)}")

    rfr = config["rfr"]

    (tab_market, tab_tech, tab_factor, tab_regime, tab_backtest,
     tab_opt, tab_mc, tab_options, tab_portfolio, tab_guide) = st.tabs([
        "📊 Market", "📉 Technical", "🔬 Factors", "🌡️ Regimes", "⚡ Backtest",
        "⚖️ Optimisation", "🎲 Monte Carlo", "💹 Options", "📖 Portfolio", "❓ Guide",
    ])

    pages = [
        (tab_market, page_market_overview, (data, rfr)),
        (tab_tech, page_technical_analysis, (data,)),
        (tab_factor, page_factor_analysis, (data, rfr)),
        (tab_regime, page_regime_detection, (data,)),
        (tab_backtest, page_strategy_backtest, (data, rfr)),
        (tab_opt, page_portfolio_optimisation, (data, rfr)),
        (tab_mc, page_monte_carlo, (data,)),
        (tab_options, page_options_pricer, (data, rfr)),
        (tab_portfolio, page_portfolio_analytics, (data, rfr)),
    ]
    for tab, fn, args in pages:
        with tab:
            try:
                fn(*args)
            except Exception as e:      # one failing page must not blank the others
                st.error(f"{fn.__name__.replace('page_', '').replace('_', ' ').title()} "
                         f"error: {e}")

    with tab_guide:
        from src.portfolio_analytics.portfolio_dashboard import render_platform_guide
        render_platform_guide()

    st.divider()
    st.caption(
        "ASX Quant Terminal · Python, Streamlit, Plotly · "
        f"Data: {data['source']} · {len(data['close'])} trading days loaded"
    )


if __name__ == "__main__":
    main()
