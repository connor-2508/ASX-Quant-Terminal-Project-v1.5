# ASX Quantitative Research Terminal

**v1.5 · Personal project · Python**

A full-stack quantitative research and trading platform for Australian Securities Exchange (ASX) equities, with an interactive Streamlit dashboard. It's modelled on the research infrastructure that systematic hedge funds use. One workflow takes raw market data through signal research, factor and regime modelling, strategy backtesting, portfolio construction and risk analysis, Monte Carlo simulation and options pricing.

An automated test suite (53 tests) checks every model against closed-form results, reference libraries and synthetic data where the true answer is known. A dedicated look-ahead test confirms that no signal or prediction uses future information.

<!-- 📷 IMAGE: hero screenshot of the dashboard (e.g. the Market tab after loading data).
     On GitHub, edit this file and drag the image onto the line below. -->


## Overview

The platform brings together the main pieces of a systematic research stack:

- **Market data pipeline:** automated ingestion, caching and cleaning of historical OHLCV prices for a 60-stock ASX universe. It also loads macroeconomic variables (VIX, interest rates, AUD/USD and commodities) and has an offline synthetic-data mode.
- **Technical analysis engine:** 55+ indicator series, including RSI, MACD, Bollinger Bands, ATR, ADX, Ichimoku Cloud, Parabolic SAR, Stochastic, OBV and Keltner Channels.
- **Statistical testing suite:**
  - stationarity: ADF and KPSS
  - autocorrelation: Ljung-Box
  - heteroskedasticity: ARCH-LM
  - normality: Jarque-Bera
  - cointegration: Engle-Granger and Johansen
  - tails: Hill tail-index estimation
- **Factor models:**
  - CAPM with Jensen's Alpha and the Security Market Line
  - the Fama-French 3-Factor Model
  - cross-sectional momentum, value and quality factor signals
- **Volatility models:** GARCH(1,1), EGARCH, GJR-GARCH, EWMA (RiskMetrics) and HAR-RV, plus the Parkinson, Garman-Klass and Yang-Zhang range-based estimators.
- **Regime detection:**
  - a Gaussian Hidden Markov Model (Baum-Welch EM with Viterbi decoding)
  - Hamilton Markov Regime Switching
  - rule-based Bull/Bear/Sideways classifiers, combined by consensus voting
- **Backtesting engine:** daily simulation with a signal-execution lag. Positions drift between rebalances, transaction costs and slippage are charged, and every run produces a full tearsheet.
- **Trading strategies:**
  - cross-sectional and time-series momentum
  - Bollinger Band and RSI mean reversion
  - Kalman filter pairs trading
  - multi-factor ranking
  - regime-adaptive strategy switching
- **Portfolio optimisation:**
  - Markowitz mean-variance optimisation: max Sharpe, min volatility and risk parity
  - the full efficient frontier, with Ledoit-Wolf covariance shrinkage
  - the Black-Litterman model with absolute and relative investor views
- **Portfolio analytics:** a full institutional tearsheet.
  - returns: Sharpe, Sortino, Calmar, Information Ratio and Jensen's Alpha
  - risk: VaR, CVaR, marginal and component risk contributions (MCTR/PCTR) and PCA risk decomposition
  - attribution: Brinson-Hood-Beebower
  - diversification ratio
- **Monte Carlo simulation:**
  - Geometric Brownian Motion and Merton jump-diffusion
  - block bootstrap
  - correlated multi-asset paths via Cholesky decomposition
  - historical and hypothetical stress testing, with Kupiec VaR backtesting
- **Machine learning:**
  - Random Forest, Gradient Boosting and XGBoost return-prediction models
  - purged walk-forward training, so every backtested trade is out-of-sample
  - Information Coefficient evaluation
  - a feature pipeline covering technical, factor and macro inputs
- **Options pricing:**
  - Black-Scholes-Merton pricing with the full Greeks: Delta, Gamma, Theta, Vega, Rho, Vanna, Volga and Charm
  - implied volatility via Brent's method
  - Monte Carlo pricing for Asian, barrier, lookback and digital options
- **Interactive dashboard:** built in Streamlit, with 9 analytical modules, session-state caching, embedded portfolio analytics and a plain-English guide to interpreting every output.

## Dashboard

| Tab | What it shows |
|---|---|
| 📊 **Market** | Normalised price performance, a returns summary and a correlation matrix for the universe |
| 📉 **Technical** | Indicator charts for any ticker, GARCH-family and range-based volatility models, and the statistical test battery |
| 🔬 **Factors** | CAPM regressions and the Security Market Line, the Fama-French 3-factor model, and cross-sectional factor scores |
| 🌡️ **Regimes** | HMM, Markov-switching and rule-based regimes overlaid on price, with filtered and smoothed probabilities and per-regime statistics |
| ⚡ **Backtest** | 9 strategies, including a walk-forward ML alpha model. Lookback, rebalance frequency, signal lag and costs are configurable, and each run produces a tearsheet and rebalance log. |
| ⚖️ **Optimisation** | Markowitz efficient frontier, a portfolio comparison and max-Sharpe weights, plus Black-Litterman with user-defined views |
| 🎲 **Monte Carlo** | GBM, jump-diffusion or bootstrap paths, the terminal distribution, VaR/CVaR and a Kupiec VaR backtest |
| 💹 **Options** | Black-Scholes pricer with Greeks, payoff and Greeks-sensitivity charts, an implied-volatility solver and Monte Carlo exotics |
| 📖 **Portfolio** | Institutional tearsheet covering performance, risk and CVaR decomposition, exposures, Brinson attribution, stress scenarios, Monte Carlo, optimisation and regimes |
| ❓ **Guide** | Plain-English explanations of every metric and chart |

### Screenshots

<!-- 📷 Paste screenshots under each heading below (drag-and-drop in the GitHub editor). Delete any you don't need. -->

#### Market overview


#### Technical analysis


#### Factor analysis


#### Regime detection


#### Strategy backtest


#### Portfolio optimisation


#### Monte Carlo simulation


#### Options pricer


#### Portfolio analytics


## Technical detail

| Area | What's implemented |
|---|---|
| **Market data pipeline** | Yahoo Finance OHLCV ingestion with parquet caching, cleaning and validation for a 60-stock ASX universe. Macro proxies: VIX, US 10Y yield, AUD/USD, gold, oil and copper, plus optional US series from FRED. An offline synthetic-data mode with a known data-generating process. |
| **Technical analysis** | About 30 indicator families (55+ series): RSI, MACD, Bollinger, ATR (Wilder), ADX/DI, Ichimoku, Parabolic SAR, Stochastic, CCI, MFI, OBV, Donchian, Keltner and more |
| **Statistical tests** | ADF and KPSS stationarity, Lo-MacKinlay variance ratio, Ljung-Box, Durbin-Watson, ARCH-LM, Jarque-Bera, Shapiro-Wilk, D'Agostino, Anderson-Darling, Engle-Granger and Johansen cointegration, Hill tail index |
| **Volatility models** | GARCH(1,1), EGARCH, GJR-GARCH (`arch`), EWMA (RiskMetrics), HAR-RV, plus the Parkinson, Garman-Klass and Yang-Zhang range estimators |
| **Factor models** | CAPM with Jensen's alpha (Newey-West t-stats) and the Security Market Line. Fama-French 3-factor model with SMB and HML built from the universe. Cross-sectional momentum, value and quality signals. |
| **Regime detection** | Gaussian HMM (Baum-Welch EM with restarts, Viterbi path). Hamilton Markov-switching model by MLE, giving filtered and smoothed probabilities. A rule-based volatility, trend, drawdown and RSI classifier with consensus voting. |
| **Backtesting engine** | Daily simulation with a signal-execution lag, weight drift between rebalances, commission and slippage on traded value, trading-day rebalance calendars, a rebalance log and a full tearsheet |
| **Strategies** | Cross-sectional, time-series, dual and risk-adjusted momentum. Bollinger and RSI mean reversion. **Kalman-filter pairs trading.** Multi-factor ranking, regime-adaptive switching and factor timing. |
| **Portfolio optimisation** | Markowitz (max Sharpe, min variance, efficient frontier), risk parity, Ledoit-Wolf shrinkage (matches scikit-learn to machine precision), Black-Litterman with absolute and relative views. Grid, random, Bayesian (GP-EI) and walk-forward parameter search. |
| **Portfolio analytics** | Tearsheet: Sharpe, Sortino, Calmar, Information Ratio, Jensen's alpha, VaR/CVaR (historical, parametric, Cornish-Fisher). Risk: MCTR, CTR, PCTR (Euler) and CVaR decomposition, PCA risk. Attribution: Brinson-Hood-Beebower. Exposures: concentration, diversification ratio, stress tests. |
| **Monte Carlo** | GBM, Merton jump-diffusion (variance-matched), moving-block bootstrap, correlated multi-asset paths (Cholesky), historical and hypothetical stress tests, Kupiec POF VaR backtest |
| **Machine learning** | Random Forest, Gradient Boosting and XGBoost return prediction on a pooled (date × ticker) panel. **Purged, expanding-window walk-forward training**, so every backtested trade is out-of-sample. Cross-sectional rank IC with a non-overlapping t-stat. |
| **Options** | Black-Scholes-Merton with the full Greeks (Δ, Γ, Θ, ν, ρ, vanna, volga, charm), implied volatility via Brent's method, and Monte Carlo pricing with antithetic variates for Asian, barrier, lookback and digital options |
| **Dashboard** | Streamlit with 9 analytical tabs and a plain-English guide to every metric and chart |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.9 – 3.12
pip install -r requirements.txt
streamlit run app/terminal_app.py
```

In the sidebar, choose **Yahoo Finance** for live data. If Yahoo is rate-limiting or you're offline, choose **Synthetic demo (offline)** instead. Synthetic data is clearly labelled in the app.

```bash
pytest            # 53 tests, ~1 minute, fully offline
```

## Project layout

```
app/terminal_app.py            Streamlit entry point (9 tabs + guide)
src/data/                      Yahoo/FRED loaders, cleaning, synthetic data
src/analytics/                 technical indicators, statistical tests, volatility models
src/factor_models/             CAPM, Fama-French, momentum / value / quality signals
src/regime_detection/          HMM, Markov switching, rule-based consensus
src/strategies/                momentum, mean reversion, pairs, factor, regime-adaptive
src/backtesting/               engine, transaction costs, performance tearsheet
src/optimisation/              Markowitz, Black-Litterman, parameter optimisation
src/simulation/                Monte Carlo paths, stress tests, Kupiec backtest
src/machine_learning/          feature engineering, tree models, walk-forward pipeline
src/options/                   Black-Scholes, Greeks, implied vol, MC exotics
src/portfolio_analytics/       metrics, risk decomposition, attribution, exposures, dashboard
src/utils/                     trading-day calendars, covariance estimators, shared stats
tests/                         look-ahead tests and model-validation tests
```

## Methodology notes

**No look-ahead.** Signals observed at close *t* are traded at the close of *t+1* and first earn the return from *t+1* to *t+2*. A generic test perturbs every price after a cutoff date and asserts that no signal, feature, factor return or walk-forward ML prediction dated on or before the cutoff changes. This covers all 9 strategies, the indicators, the factor signals, the Fama-French factors and the ML pipeline. A perfect-foresight "oracle" signal earns over 100% a year with no lag and roughly nothing once lagged.

**Realistic portfolio accounting.** Between rebalances, positions drift with prices; costs are charged on the turnover needed to return to target. Rebalance dates are the last *trading* day of each week, month or quarter. Results start when capital is first deployed, so a 12-month lookback doesn't dilute CAGR and Sharpe with a year of zeros.

**Fama-French factors from available data.** Yahoo Finance has no point-in-time market caps or book values. Each month-end, stocks are sorted on information available at that close and held for the following month. SMB uses 63-day average traded value (price × volume) as a size proxy. HML uses a price-based value score: 36-12 month long-term reversal (De Bondt & Thaler) plus proximity to the 52-week low. On simulated data with no size or value premium, the factors average roughly zero Sharpe, as they should.

**Machine learning.** One pooled model is fitted on the stacked (date, ticker) panel. At each retrain date it only sees samples whose 21-day forward target was fully realised (a 21-day purge). It then predicts until the next retrain. Early stopping uses a validation slice carved from the training data, never the test set. Macro features are lagged one day, because US-hours data such as the VIX settles after the ASX close.

**Black-Litterman.** The model works in excess returns. The prior is equal-weighted, because market-cap weights aren't available. View uncertainty is Ω = (1−c)/c · diag(PτΣPᵀ), so a confidence of 0.5 weights a view equally with the prior (verified in tests).

**Choosing the risk-free rate.** The sidebar's risk-free rate is used to form excess returns everywhere: Sharpe and Sortino ratios, CAPM and Fama-French regressions, Black-Litterman, the backtest tearsheet and the options pricer. The rate should match the horizon of the returns being measured. The platform works with daily returns and holding periods of days to months, so the appropriate benchmark is a **short-term AUD rate**: the RBA cash rate, 90-day BBSW or the 3-month Treasury note yield.

The common advice to use the **10-year government bond yield** applies to long-horizon valuation, such as DCF discount rates, cost of capital and the long-run equity risk premium. Over a short holding period a 10-year bond carries interest-rate risk, and its yield includes a term premium. Using it here would understate excess returns when the yield curve slopes upward.

For the options pricer, the theoretically correct rate is the one matching the option's time to expiry, but for short-dated options the cash rate is a close approximation and has a small effect on price.

- **Caveat — a single rate for the whole history:** the platform applies one constant rate across the selected date range. Australian short rates moved from about 0.1% (2020–21) to about 4% later, so using today's rate over a 2019–2026 backtest overstates the risk-free return in the low-rate years and understates Sharpe ratios and alphas for that period.
- **Practical choice:** enter the average cash rate over the chosen window.
- **More rigorous approach (not yet implemented):** a time-varying daily risk-free series, such as historical 90-day BBSW.
- **Default:** 4.35% is the RBA cash rate that applied through 2024. Check the RBA for the current rate.

## Validation (`tests/`)

- **Reference implementations:** Ledoit-Wolf against scikit-learn. All Greeks against finite differences. Put-call parity. Implied-vol round trip. MC prices within 3 standard errors of Black-Scholes. Barrier in-out parity.
- **Planted ground truth:** CAPM recovers known betas. HMM and Markov switching each recover a planted bull/bear switch with over 97% accuracy. A Kalman pairs strategy recovers the hedge ratio of a synthetic cointegrated pair and trades it profitably.
- **Accounting identities:** Euler risk contributions sum to portfolio volatility. BHB effects sum to the active return. A buy-and-hold backtest equals the price path less entry cost.
- **Statistical sanity:** jump-diffusion matches the target variance with negative skew. A correctly specified VaR model isn't rejected by the Kupiec test.

## Known limitations

- **Survivorship bias:** the 60-stock universe is today's constituents applied to history.
- **Constant risk-free rate:** one rate is applied across the whole history rather than a time-varying series (see Methodology notes).
- **Price-based proxies:** value and quality signals use prices, not fundamental data. The Fama-French factors from a small universe are noisy.
- **Option style:** options are priced as European with a continuous dividend yield, but ASX single-stock options are American.
- **Data quality:** Yahoo Finance data is adjusted close only, with no point-in-time corporate-action audit trail.

## Tools

Python · NumPy · pandas · SciPy · statsmodels · scikit-learn · XGBoost · arch · hmmlearn · Plotly · Streamlit · yfinance · pytest

---

**Note:** This was built entirely with Claude AI - This project doesn't display an understanding of coding, but rather, displays an understanding of the models, the development, the testing of the outputs and the integration of the components into the broader research workflow. Overall, this project required me to research, understand and integrate a wide range of financial models and concepts into a complete and functional dashboard and system.
