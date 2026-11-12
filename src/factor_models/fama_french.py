"""
fama_french.py
==============
Fama-French 3-Factor Model implementation for ASX equities.

The Fama-French model extends CAPM with two additional factors:
    SMB (Small Minus Big)  : Size premium — small caps outperform large caps
    HML (High Minus Low)   : Value premium — high book/price stocks outperform

Model
-----
R_i - R_f = α + β_mkt × (R_m - R_f) + β_smb × SMB + β_hml × HML + ε

Factor Construction (from available ASX price/volume data)
----------------------------------------------------------
Yahoo Finance does not provide point-in-time market caps or book values, so
SMB and HML are built from transparent proxies. Portfolios are formed on the
last trading day of each month using only information available at that
close, then held (equal-weighted) over the following month:

    MKT  : Benchmark (ASX 200) return minus the risk-free rate
    SMB  : Bottom-30% minus top-30% by 63-day average traded value
           (price x volume) - a liquidity-based size proxy. Requires
           ``size_data``; if it is not supplied SMB is omitted rather than
           built from a meaningless proxy such as share price.
    HML  : Top-30% minus bottom-30% by the platform's price-based value
           composite (36-12M long-term reversal + distance from 52-week low;
           De Bondt & Thaler 1985).

With a small universe each leg holds only a few names, so the factors are
noisy; they are intended for exposure analysis, not as a replacement for
published Fama-French factors.

Usage
-----
    from src.factor_models.fama_french import FamaFrench

    ff = FamaFrench(asset_returns, market_returns, universe_prices)
    results = ff.fit()
    factors = ff.get_factor_returns()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ─────────────────────────────────────────────
#  Result Container
# ─────────────────────────────────────────────

@dataclass
class FamaFrenchResult:
    """Container for Fama-French regression results."""
    ticker: str = ""
    alpha: float = 0.0
    beta_mkt: float = 0.0
    beta_smb: float = 0.0
    beta_hml: float = 0.0
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    alpha_tstat: float = 0.0
    alpha_pvalue: float = 1.0
    alpha_significant: bool = False
    mkt_tstat: float = 0.0
    smb_tstat: float = 0.0
    hml_tstat: float = 0.0
    factor_contributions: dict = field(default_factory=dict)
    n_observations: int = 0

    def summary(self) -> str:
        return (
            f"\nFama-French 3-Factor: {self.ticker}\n"
            f"{'─' * 45}\n"
            f"  Alpha (ann.)  : {self.alpha:.2%}  "
            f"(t={self.alpha_tstat:.2f}, "
            f"{'✅' if self.alpha_significant else '❌'})\n"
            f"  β Market      : {self.beta_mkt:.4f}  (t={self.mkt_tstat:.2f})\n"
            f"  β SMB (Size)  : {self.beta_smb:.4f}  (t={self.smb_tstat:.2f})\n"
            f"  β HML (Value) : {self.beta_hml:.4f}  (t={self.hml_tstat:.2f})\n"
            f"  R²            : {self.r_squared:.4f}  "
            f"(Adj R²: {self.adj_r_squared:.4f})\n"
            f"  Observations  : {self.n_observations}\n"
        )


# ─────────────────────────────────────────────
#  FamaFrench
# ─────────────────────────────────────────────

class FamaFrench:
    """
    Fama-French 3-Factor Model.

    Parameters
    ----------
    asset_returns   : pd.Series or pd.DataFrame
        Daily returns for the asset(s) to analyse.
    market_returns  : pd.Series
        Daily market (benchmark) returns.
    universe_prices : pd.DataFrame, optional
        Close prices for a broad universe of ASX stocks.
        Used to construct SMB and HML factor proxies.
    factor_returns  : pd.DataFrame, optional
        Pre-computed factor return series with columns:
        ['MKT', 'SMB', 'HML']. If provided, skips factor construction.
    size_data       : pd.DataFrame, optional
        Daily traded value (close x volume) per ticker, used for the SMB
        size sort. Without it, SMB is not constructed.
    risk_free_rate  : float
        Annualised risk-free rate.
    periods_per_year: int
        Trading days per year.
    """

    FACTORS = ["MKT", "SMB", "HML"]

    def __init__(
        self,
        asset_returns: pd.Series | pd.DataFrame,
        market_returns: pd.Series,
        universe_prices: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
        size_data: Optional[pd.DataFrame] = None,
        formation_freq: str = "monthly",
        leg_fraction: float = 0.30,
    ) -> None:
        if market_returns is None or market_returns.dropna().empty:
            raise ValueError("FamaFrench requires a benchmark (market) return series.")
        self.market = market_returns.dropna()
        self.universe = universe_prices
        self.size_data = size_data
        self.formation_freq = formation_freq
        self.leg_fraction = leg_fraction
        self.rfr = risk_free_rate
        self.rfr_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        self.ppy = periods_per_year

        if isinstance(asset_returns, pd.Series):
            self.assets = asset_returns.dropna().to_frame()
        else:
            self.assets = asset_returns.dropna()

        # Build or use provided factor returns
        if factor_returns is not None:
            self._factors = factor_returns
        elif universe_prices is not None:
            self._factors = self._build_factors()
        else:
            self._factors = self._build_market_only_factors()

    # ──────────────────────────────────────
    #  Factor Construction
    # ──────────────────────────────────────

    @staticmethod
    def _long_short_returns(
        returns: pd.DataFrame,
        characteristic: pd.DataFrame,
        formation_dates: list,
        fraction: float,
    ) -> pd.Series:
        """
        Return of an equal-weighted (top - bottom) portfolio sorted on
        ``characteristic`` at each formation date and held until the next.
        Membership formed at the close of date d is first applied to the
        return of d+1, so the sort never sees the returns it is paid on.
        """
        member = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        char = characteristic.reindex(index=returns.index, columns=returns.columns)
        for d in formation_dates:
            row = char.loc[d].dropna()
            m = pd.Series(0.0, index=returns.columns)
            if len(row) >= 4:
                k = max(1, int(round(fraction * len(row))))
                m[row.nlargest(k).index] = 1.0
                m[row.nsmallest(k).index] = -1.0
            member.loc[d] = m
        member = member.ffill().shift(1)
        r_top = returns.where(member == 1.0).mean(axis=1)
        r_bot = returns.where(member == -1.0).mean(axis=1)
        return (r_top - r_bot).dropna()

    def _build_factors(self) -> pd.DataFrame:
        """
        Construct MKT, SMB and HML from universe data (see module docstring).
        """
        from src.utils.calendar import period_end_dates
        from src.factor_models.momentum_factor import ValueFactor

        prices = self.universe.copy()
        ret = prices.pct_change(fill_method=None)
        form_dates = sorted(period_end_dates(prices.index, self.formation_freq))

        factors = {}
        common_mkt = self.market.index.intersection(ret.index)
        factors["MKT"] = self.market.loc[common_mkt] - self.rfr_daily

        if self.size_data is not None:
            size = (
                self.size_data.reindex(index=prices.index, columns=prices.columns)
                .rolling(63, min_periods=21).mean()
            )
            # SMB = small minus big = -(big minus small)
            factors["SMB"] = -self._long_short_returns(
                ret, size, form_dates, self.leg_fraction
            )

        value_score = ValueFactor(prices).compute_signals(zscore=True)
        factors["HML"] = self._long_short_returns(
            ret, value_score, form_dates, self.leg_fraction
        )

        df = pd.DataFrame(factors).dropna()
        return df

    def _build_market_only_factors(self) -> pd.DataFrame:
        """
        Fallback: build factors using market return only.
        SMB and HML are set to zero (degenerates to CAPM).
        """
        mkt_excess = self.market - self.rfr_daily
        df = pd.DataFrame({"MKT": mkt_excess})
        df["SMB"] = 0.0
        df["HML"] = 0.0
        return df

    # ──────────────────────────────────────
    #  Regression
    # ──────────────────────────────────────

    def _run_regression(
        self,
        ticker: str,
        returns: pd.Series,
    ) -> FamaFrenchResult:
        """
        Run OLS regression of asset excess returns on 3 factors.
        R_i - R_f = α + β_mkt×MKT + β_smb×SMB + β_hml×HML + ε
        """
        common = returns.index.intersection(self._factors.index)
        if len(common) < 60:
            return FamaFrenchResult(ticker=ticker)

        y = returns.loc[common] - self.rfr_daily
        used = [f for f in self.FACTORS if f in self._factors.columns]
        X_df = self._factors.loc[common][used]

        if HAS_STATSMODELS:
            X = sm.add_constant(X_df.values)
            model = sm.OLS(y.values, X)
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})

            params = fit.params
            tstats = fit.tvalues
            pvals = fit.pvalues
            r2 = float(fit.rsquared)
            adj_r2 = float(fit.rsquared_adj)
            n = int(fit.nobs)
        else:
            X = np.column_stack([np.ones(len(y)), X_df.values])
            try:
                params, _, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
            except Exception:
                return FamaFrenchResult(ticker=ticker)

            y_hat = X @ params
            residuals = y.values - y_hat
            n = len(y)
            k = len(params)
            mse = np.dot(residuals, residuals) / (n - k)
            cov = mse * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            tstats = params / se
            pvals = 2 * (1 - stats.t.cdf(np.abs(tstats), df=n - k))
            ss_res = np.dot(residuals, residuals)
            ss_tot = np.dot(y.values - y.values.mean(), y.values - y.values.mean())
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        coef = dict(zip(["const"] + used, np.asarray(params, dtype=float)))
        tst = dict(zip(["const"] + used, np.asarray(tstats, dtype=float)))
        alpha_d = coef["const"]
        beta_mkt = coef.get("MKT", np.nan)
        beta_smb = coef.get("SMB", np.nan)
        beta_hml = coef.get("HML", np.nan)

        # Factor return contributions (annualised)
        factor_contribs = {}
        factor_means = self._factors.loc[common].mean() * self.ppy
        for f in used:
            factor_contribs[f] = coef[f] * float(factor_means.get(f, 0))
        factor_contribs["Alpha"] = alpha_d * self.ppy

        return FamaFrenchResult(
            ticker=ticker,
            alpha=alpha_d * self.ppy,
            beta_mkt=beta_mkt,
            beta_smb=beta_smb,
            beta_hml=beta_hml,
            r_squared=r2,
            adj_r_squared=adj_r2,
            alpha_tstat=tst["const"],
            alpha_pvalue=float(pvals[0]),
            alpha_significant=float(pvals[0]) < 0.05,
            mkt_tstat=tst.get("MKT", np.nan),
            smb_tstat=tst.get("SMB", np.nan),
            hml_tstat=tst.get("HML", np.nan),
            factor_contributions=factor_contribs,
            n_observations=n,
        )

    # ──────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────

    def fit(self) -> dict[str, FamaFrenchResult]:
        """Fit FF3 for all assets. Returns {ticker: FamaFrenchResult}."""
        return {
            ticker: self._run_regression(ticker, self.assets[ticker])
            for ticker in self.assets.columns
        }

    def fit_to_dataframe(self) -> pd.DataFrame:
        """Fit FF3 for all assets and return a comparison DataFrame."""
        results = self.fit()
        rows = []
        for ticker, r in results.items():
            rows.append({
                "Ticker": ticker,
                "Alpha (ann.)": r.alpha,
                "β MKT": r.beta_mkt,
                "β SMB": r.beta_smb,
                "β HML": r.beta_hml,
                "R²": r.r_squared,
                "Adj R²": r.adj_r_squared,
                "Alpha t-stat": r.alpha_tstat,
                "Alpha Sig.": r.alpha_significant,
                "MKT Contrib": r.factor_contributions.get("MKT", np.nan),
                "SMB Contrib": r.factor_contributions.get("SMB", np.nan),
                "HML Contrib": r.factor_contributions.get("HML", np.nan),
                "Observations": r.n_observations,
            })
        return (
            pd.DataFrame(rows)
            .set_index("Ticker")
            .sort_values("Alpha (ann.)", ascending=False)
        )

    def get_factor_returns(self) -> pd.DataFrame:
        """Return the constructed factor return time series."""
        return self._factors.copy()

    def factor_correlation(self) -> pd.DataFrame:
        """Correlation matrix between factors."""
        return self._factors.corr()

    def factor_summary(self) -> pd.DataFrame:
        """Annualised statistics for each factor."""
        f = self._factors
        rows = []
        for col in f.columns:
            s = f[col].dropna()
            rows.append({
                "Factor": col,
                "Ann. Return": s.mean() * 252,
                "Ann. Volatility": s.std() * np.sqrt(252),
                "Sharpe": s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0,
                "Skewness": float(s.skew()),
                "Kurtosis": float(s.kurtosis()),
                "Observations": len(s),
            })
        return pd.DataFrame(rows).set_index("Factor")
