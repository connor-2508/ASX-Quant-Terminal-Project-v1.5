"""
black_scholes.py  /  monte_carlo_option_pricing.py
==================================================
Options pricing models for ASX equity derivatives.

Models Implemented
------------------
    BlackScholes        : Analytical BS pricing with full Greeks
    BlackScholesBarrier : Barrier option extensions
    MonteCarloPricer    : Simulation-based pricing for exotic options
    ImpliedVolatility   : IV calculation via Brent's method
    VolatilitySurface   : IV surface construction and interpolation

Greeks Computed
---------------
    Delta   : dV/dS   — sensitivity to underlying price
    Gamma   : d²V/dS² — rate of change of delta
    Theta   : dV/dt   — time decay (per calendar day)
    Vega    : dV/dσ   — sensitivity to volatility
    Rho     : dV/dr   — sensitivity to interest rate

ASX Options Notes
-----------------
    ASX equity options are American-style (can exercise early).
    This module prices European options analytically.
    American options require binomial tree or LSM Monte Carlo.

Usage
-----
    from src.options.black_scholes import BlackScholes

    bs = BlackScholes(S=45.20, K=45.00, T=0.25, r=0.0435, sigma=0.22)
    price = bs.call_price()
    greeks = bs.greeks()
    print(bs.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq



# ─────────────────────────────────────────────
#  Result Containers
# ─────────────────────────────────────────────

@dataclass
class GreeksResult:
    """Container for option Greeks."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0      # Per calendar day
    vega: float = 0.0       # Per 1% change in vol
    rho: float = 0.0        # Per 1% change in rate
    vanna: float = 0.0      # dDelta/dSigma (per 1.00 change in vol)
    volga: float = 0.0      # d²V/dSigma² (per 1.00 change in vol)
    charm: float = 0.0      # dDelta/dt (per year of calendar time)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            ("Delta", self.delta, "Change in option price per $1 move in underlying"),
            ("Gamma", self.gamma, "Rate of change of delta per $1 move"),
            ("Theta", self.theta, "Time decay per calendar day (in option price units)"),
            ("Vega", self.vega, "Change per 1% increase in implied volatility"),
            ("Rho", self.rho, "Change per 1% increase in risk-free rate"),
            ("Vanna", self.vanna / 100, "Change in delta per 1% change in vol"),
            ("Volga", self.volga / 10_000, "Change in vega(per 1%) per 1% change in vol"),
            ("Charm", self.charm / 365, "Change in delta per calendar day"),
        ]
        return pd.DataFrame(rows, columns=["Greek", "Value", "Interpretation"])


@dataclass
class OptionResult:
    """Container for option pricing results."""
    option_type: str = "call"
    price: float = 0.0
    intrinsic_value: float = 0.0
    time_value: float = 0.0
    greeks: GreeksResult = field(default_factory=GreeksResult)
    # Inputs
    S: float = 0.0    # Spot price
    K: float = 0.0    # Strike
    T: float = 0.0    # Time to expiry (years)
    r: float = 0.0    # Risk-free rate
    sigma: float = 0.0

    def summary(self) -> str:
        moneyness = "ATM" if abs(self.S/self.K - 1) < 0.01 else (
            "ITM" if (self.option_type == "call" and self.S > self.K) or
                     (self.option_type == "put" and self.S < self.K) else "OTM"
        )
        return (
            f"\nBlack-Scholes {self.option_type.upper()} Option\n"
            f"{'─' * 45}\n"
            f"  Spot Price        : ${self.S:.2f}\n"
            f"  Strike Price      : ${self.K:.2f}\n"
            f"  Moneyness         : {moneyness}\n"
            f"  Time to Expiry    : {self.T:.4f} years ({self.T*365:.0f} days)\n"
            f"  Implied Volatility: {self.sigma:.1%}\n"
            f"  Risk-Free Rate    : {self.r:.2%}\n"
            f"{'─' * 45}\n"
            f"  Option Price      : ${self.price:.4f}\n"
            f"  Intrinsic Value   : ${self.intrinsic_value:.4f}\n"
            f"  Time Value        : ${self.time_value:.4f}\n"
            f"{'─' * 45}\n"
            f"  Delta             : {self.greeks.delta:.4f}\n"
            f"  Gamma             : {self.greeks.gamma:.6f}\n"
            f"  Theta             : ${self.greeks.theta:.4f}/day\n"
            f"  Vega              : ${self.greeks.vega:.4f}/1% vol\n"
            f"  Rho               : ${self.greeks.rho:.4f}/1% rate\n"
        )


# ─────────────────────────────────────────────
#  Black-Scholes Model
# ─────────────────────────────────────────────

class BlackScholes:
    """
    Black-Scholes-Merton European option pricing model.

    Parameters
    ----------
    S     : float   Current spot price of underlying.
    K     : float   Option strike price.
    T     : float   Time to expiry in years (e.g. 90/365 = 0.2466).
    r     : float   Annualised risk-free rate (e.g. 0.0435).
    sigma : float   Annualised implied volatility (e.g. 0.22).
    q     : float   Continuous dividend yield (default 0 — most ASX stocks
                    pay discrete dividends, which reduces accuracy slightly).

    Example
    -------
    >>> bs = BlackScholes(S=45.20, K=45.00, T=90/365, r=0.0435, sigma=0.22)
    >>> print(bs.call_price())
    >>> print(bs.greeks())
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> None:
        self.S = S
        self.K = K
        self.T = max(T, 1e-6)   # Avoid division by zero at expiry
        self.r = r
        self.sigma = max(sigma, 1e-6)
        self.q = q

    # ──────────────────────────────────────
    #  d1 and d2 factors
    # ──────────────────────────────────────

    @property
    def d1(self) -> float:
        """
        d1 = [ln(S/K) + (r - q + σ²/2) × T] / (σ√T)
        """
        return (
            np.log(self.S / self.K)
            + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

    @property
    def d2(self) -> float:
        """d2 = d1 - σ√T"""
        return self.d1 - self.sigma * np.sqrt(self.T)

    # ──────────────────────────────────────
    #  Prices
    # ──────────────────────────────────────

    def call_price(self) -> float:
        """
        European call option price.
        C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
        """
        return float(
            self.S * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)
            - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(self.d2)
        )

    def put_price(self) -> float:
        """
        European put option price.
        P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
        """
        return float(
            self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-self.d2)
            - self.S * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)
        )

    def price(self, option_type: str = "call") -> float:
        """Return call or put price."""
        return self.call_price() if option_type.lower() == "call" else self.put_price()

    # ──────────────────────────────────────
    #  Greeks
    # ──────────────────────────────────────

    def delta(self, option_type: str = "call") -> float:
        """
        Delta: dV/dS
        Call delta ∈ (0, 1), Put delta ∈ (-1, 0).
        ATM options have delta ≈ ±0.5.
        """
        if option_type.lower() == "call":
            return float(np.exp(-self.q * self.T) * stats.norm.cdf(self.d1))
        return float(np.exp(-self.q * self.T) * (stats.norm.cdf(self.d1) - 1))

    def gamma(self) -> float:
        """
        Gamma: d²V/dS²  (same for calls and puts)
        Measures how fast delta changes.
        High gamma = delta hedges need frequent adjustment.
        """
        return float(
            np.exp(-self.q * self.T)
            * stats.norm.pdf(self.d1)
            / (self.S * self.sigma * np.sqrt(self.T))
        )

    def theta(self, option_type: str = "call") -> float:
        """
        Theta: dV/dt (time decay, expressed per calendar day).
        Options lose value as time passes (negative for long options).
        """
        term1 = (
            -self.S * np.exp(-self.q * self.T)
            * stats.norm.pdf(self.d1)
            * self.sigma
            / (2 * np.sqrt(self.T))
        )
        if option_type.lower() == "call":
            term2 = (
                self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(self.d2)
            )
            term3 = (
                self.q * self.S * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)
            )
            annual = term1 - term2 + term3
        else:
            term2 = (
                self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-self.d2)
            )
            term3 = (
                self.q * self.S * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)
            )
            annual = term1 + term2 - term3

        return float(annual / 365)   # Per calendar day

    def vega(self) -> float:
        """
        Vega: dV/dσ  (same for calls and puts).
        Expressed per 1% change in volatility.
        """
        raw = float(
            self.S * np.exp(-self.q * self.T)
            * stats.norm.pdf(self.d1)
            * np.sqrt(self.T)
        )
        return raw / 100   # Per 1% change in vol

    def rho(self, option_type: str = "call") -> float:
        """
        Rho: dV/dr
        Expressed per 1% change in interest rate.
        """
        if option_type.lower() == "call":
            raw = self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(self.d2)
        else:
            raw = -self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(-self.d2)
        return float(raw) / 100   # Per 1% change

    def vanna(self) -> float:
        """Vanna: dDelta/dSigma = d²V/dSdσ"""
        return float(
            -np.exp(-self.q * self.T)
            * stats.norm.pdf(self.d1)
            * self.d2
            / self.sigma
        )

    def volga(self) -> float:
        """Volga (Vomma): d²V/dσ² — convexity of option value to vol."""
        raw_vega = float(
            self.S * np.exp(-self.q * self.T)
            * stats.norm.pdf(self.d1)
            * np.sqrt(self.T)
        )
        return float(raw_vega * self.d1 * self.d2 / self.sigma)

    def charm(self, option_type: str = "call") -> float:
        """
        Charm (delta decay): dDelta/dt as calendar time passes, i.e.
        -dDelta/dT, per YEAR. Divide by 365 for a per-day figure.
        """
        d1, d2 = self.d1, self.d2
        term = (
            np.exp(-self.q * self.T) * stats.norm.pdf(d1) *
            (2 * (self.r - self.q) * self.T - d2 * self.sigma * np.sqrt(self.T)) /
            (2 * self.T * self.sigma * np.sqrt(self.T))
        )
        if option_type.lower() == "call":
            return float(self.q * np.exp(-self.q * self.T) * stats.norm.cdf(d1) - term)
        return float(-self.q * np.exp(-self.q * self.T) * stats.norm.cdf(-d1) - term)

    def greeks(self, option_type: str = "call") -> GreeksResult:
        """Compute all Greeks for the option."""
        return GreeksResult(
            delta=self.delta(option_type),
            gamma=self.gamma(),
            theta=self.theta(option_type),
            vega=self.vega(),
            rho=self.rho(option_type),
            vanna=self.vanna(),
            volga=self.volga(),
            charm=self.charm(option_type),
        )

    def full_result(self, option_type: str = "call") -> OptionResult:
        """Return a full OptionResult with price and all Greeks."""
        price = self.price(option_type)
        intrinsic = max(
            (self.S - self.K) if option_type == "call" else (self.K - self.S),
            0.0
        )
        return OptionResult(
            option_type=option_type,
            price=price,
            intrinsic_value=intrinsic,
            time_value=price - intrinsic,
            greeks=self.greeks(option_type),
            S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma,
        )

    def summary(self, option_type: str = "call") -> str:
        """Print formatted summary."""
        return self.full_result(option_type).summary()

    # ──────────────────────────────────────
    #  Put-Call Parity
    # ──────────────────────────────────────

    def put_call_parity_check(self) -> dict:
        """
        Verify put-call parity: C - P = S·e^(-qT) - K·e^(-rT)
        """
        c = self.call_price()
        p = self.put_price()
        lhs = c - p
        rhs = (
            self.S * np.exp(-self.q * self.T)
            - self.K * np.exp(-self.r * self.T)
        )
        error = abs(lhs - rhs)
        return {
            "call_price": c,
            "put_price": p,
            "C - P": lhs,
            "S·e^(-qT) - K·e^(-rT)": rhs,
            "parity_error": error,
            "passes": error < 1e-8,
        }


# ─────────────────────────────────────────────
#  Implied Volatility
# ─────────────────────────────────────────────

class ImpliedVolatility:
    """
    Calculate implied volatility from observed market option prices.
    Uses Brent's root-finding method for speed and robustness.

    Parameters
    ----------
    market_price : float   Observed option price in the market.
    S, K, T, r   : float   Standard BSM inputs.
    q            : float   Dividend yield.
    option_type  : str     'call' or 'put'.
    """

    def __init__(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> None:
        self.market_price = market_price
        self.S = S
        self.K = K
        self.T = max(T, 1e-6)
        self.r = r
        self.q = q
        self.option_type = option_type.lower()

    def calculate(
        self,
        low: float = 0.001,
        high: float = 5.0,
        tol: float = 1e-6,
    ) -> float:
        """
        Find the implied volatility using Brent's method.

        Returns
        -------
        float   Implied volatility (annualised).
                Returns NaN if no solution found.
        """
        def objective(sigma):
            bs = BlackScholes(self.S, self.K, self.T, self.r, sigma, self.q)
            model_price = bs.price(self.option_type)
            return model_price - self.market_price

        try:
            # Check if solution exists
            f_low = objective(low)
            f_high = objective(high)

            if f_low * f_high > 0:
                # Price outside the no-arbitrage range attainable for
                # vol in [low, high] (e.g. below intrinsic value) -> no IV
                return np.nan

            iv = brentq(objective, low, high, xtol=tol, maxiter=500)
            return float(iv)

        except (ValueError, RuntimeError):
            return np.nan

    @classmethod
    def surface(
        cls,
        strikes: np.ndarray,
        expiries: np.ndarray,
        market_prices: np.ndarray,
        S: float,
        r: float,
        option_type: str = "call",
    ) -> pd.DataFrame:
        """
        Construct an implied volatility surface.

        Parameters
        ----------
        strikes  : np.ndarray   1D array of strike prices.
        expiries : np.ndarray   1D array of expiry times (years).
        prices   : np.ndarray   2D array of market prices (expiries × strikes).
        S        : float        Current spot price.
        r        : float        Risk-free rate.

        Returns
        -------
        pd.DataFrame   IV surface (index=expiry, cols=strike).
        """
        iv_matrix = np.full((len(expiries), len(strikes)), np.nan)

        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                price = float(market_prices[i, j])
                iv_calc = cls(price, S, K, T, r, option_type=option_type)
                iv_matrix[i, j] = iv_calc.calculate()

        return pd.DataFrame(
            iv_matrix,
            index=[f"{T:.4f}Y" for T in expiries],
            columns=[f"K={K:.2f}" for K in strikes],
        )


# ─────────────────────────────────────────────
#  Monte Carlo Option Pricer
# ─────────────────────────────────────────────

class MonteCarloOptionPricer:
    """
    Monte Carlo option pricing for standard and exotic options.
    Handles path-dependent options that lack analytical solutions.

    Parameters
    ----------
    S, K, T, r, sigma, q : Standard BSM inputs.
    n_simulations : int   Number of simulation paths.
    n_steps       : int   Time steps per path.
    seed          : int   Random seed.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        n_simulations: int = 100_000,
        n_steps: int = 252,
        seed: int = 42,
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_sim = n_simulations
        self.n_steps = n_steps
        self.seed = seed

    def _simulate_paths(self) -> np.ndarray:
        """
        Simulate risk-neutral GBM paths with antithetic variates
        (each normal draw Z is paired with -Z, which reduces variance for
        monotone payoffs at no extra cost).
        Returns (n_sim, n_steps+1) array of price paths.
        """
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.n_steps
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * np.sqrt(dt)

        half = (self.n_sim + 1) // 2
        Z_half = rng.standard_normal((half, self.n_steps))
        Z = np.vstack([Z_half, -Z_half])[: self.n_sim]
        log_returns = drift + vol * Z
        log_cumsum = np.cumsum(log_returns, axis=1)
        paths = self.S * np.exp(
            np.column_stack([np.zeros(self.n_sim), log_cumsum])
        )
        return paths

    def _std_error(self, payoffs: np.ndarray) -> float:
        """
        Standard error of the discounted MC price. With antithetic pairs
        (i, i + half) the pair averages are the independent samples.
        """
        disc = np.exp(-self.r * self.T)
        half = (self.n_sim + 1) // 2
        n_pairs = self.n_sim - half
        if n_pairs > 1:
            pair_avg = 0.5 * (payoffs[:n_pairs] + payoffs[half:half + n_pairs])
            return float(disc * np.std(pair_avg, ddof=1) / np.sqrt(n_pairs))
        return float(disc * np.std(payoffs, ddof=1) / np.sqrt(len(payoffs)))

    def european_call(self) -> dict:
        """Price a European call option using Monte Carlo."""
        paths = self._simulate_paths()
        S_T = paths[:, -1]
        payoffs = np.maximum(S_T - self.K, 0)
        price = float(np.exp(-self.r * self.T) * np.mean(payoffs))
        se = self._std_error(payoffs)

        # Compare to BS analytical
        bs_price = BlackScholes(self.S, self.K, self.T, self.r, self.sigma, self.q).call_price()

        return {
            "mc_price": price,
            "std_error": se,
            "95pct_interval": (price - 1.96 * se, price + 1.96 * se),
            "bs_price": bs_price,
            "pricing_error": abs(price - bs_price),
        }

    def european_put(self) -> dict:
        """Price a European put option."""
        paths = self._simulate_paths()
        S_T = paths[:, -1]
        payoffs = np.maximum(self.K - S_T, 0)
        price = float(np.exp(-self.r * self.T) * np.mean(payoffs))
        se = self._std_error(payoffs)
        bs_price = BlackScholes(self.S, self.K, self.T, self.r, self.sigma, self.q).put_price()

        return {
            "mc_price": price,
            "std_error": se,
            "95pct_interval": (price - 1.96 * se, price + 1.96 * se),
            "bs_price": bs_price,
            "pricing_error": abs(price - bs_price),
        }

    def asian_call(self, averaging: str = "arithmetic") -> dict:
        """
        Asian call option: payoff based on average price, not final price.
        Reduces volatility risk and is cheaper than vanilla options.

        Parameters
        ----------
        averaging : str   'arithmetic' | 'geometric'
        """
        paths = self._simulate_paths()

        # Average over the monitoring dates t_1..t_n (exclude S_0)
        monitored = paths[:, 1:]
        if averaging == "arithmetic":
            avg_price = monitored.mean(axis=1)
        else:
            avg_price = np.exp(np.log(monitored).mean(axis=1))

        payoffs = np.maximum(avg_price - self.K, 0)
        price = float(np.exp(-self.r * self.T) * np.mean(payoffs))
        se = self._std_error(payoffs)

        return {
            "mc_price": price,
            "std_error": se,
            "averaging": averaging,
            "vs_vanilla": price / BlackScholes(
                self.S, self.K, self.T, self.r, self.sigma, self.q
            ).call_price() - 1,
        }

    def barrier_call(
        self,
        barrier: float,
        barrier_type: str = "down-and-out",
    ) -> dict:
        """
        Barrier option: knocked out or in when price hits barrier.

        Parameters
        ----------
        barrier      : float   Barrier price level.
        barrier_type : str     'down-and-out' | 'up-and-out' |
                               'down-and-in' | 'up-and-in'

        Note: the barrier is monitored discretely at each of the n_steps
        dates; continuous-monitoring prices differ (Broadie-Glasserman-Kou
        shift the barrier by exp(±0.5826 σ √dt) to correct for this).
        """
        paths = self._simulate_paths()
        S_T = paths[:, -1]
        vanilla_payoffs = np.maximum(S_T - self.K, 0)

        min_price = paths.min(axis=1)
        max_price = paths.max(axis=1)

        if barrier_type == "down-and-out":
            active = min_price > barrier
        elif barrier_type == "up-and-out":
            active = max_price < barrier
        elif barrier_type == "down-and-in":
            active = min_price <= barrier
        elif barrier_type == "up-and-in":
            active = max_price >= barrier
        else:
            active = np.ones(self.n_sim, dtype=bool)

        payoffs = vanilla_payoffs * active
        price = float(np.exp(-self.r * self.T) * np.mean(payoffs))
        se = self._std_error(payoffs)
        knock_rate = float(1 - active.mean())

        return {
            "mc_price": price,
            "std_error": se,
            "barrier_type": barrier_type,
            "barrier_level": barrier,
            "knock_out_rate": knock_rate,
        }

    def lookback_call(self) -> dict:
        """
        Floating-strike lookback call: payoff = S_T - min(S).
        Buyer always buys at the lowest price over the period.
        """
        paths = self._simulate_paths()
        S_T = paths[:, -1]
        S_min = paths.min(axis=1)
        payoffs = S_T - S_min
        price = float(np.exp(-self.r * self.T) * np.mean(payoffs))

        return {"mc_price": price, "option_type": "lookback_call"}

    def digital_call(self) -> dict:
        """
        Binary/digital call: pays $1 if S_T > K, else $0.
        """
        paths = self._simulate_paths()
        S_T = paths[:, -1]
        payoffs = (S_T > self.K).astype(float)
        price = float(np.exp(-self.r * self.T) * np.mean(payoffs))
        bs_digital = float(
            np.exp(-self.r * self.T) *
            stats.norm.cdf(
                BlackScholes(self.S, self.K, self.T, self.r, self.sigma, self.q).d2
            )
        )
        return {"mc_price": price, "bs_price": bs_digital}

    def summary_table(self) -> pd.DataFrame:
        """Compute prices for all standard option types."""
        results = []
        for opt_type, fn in [
            ("European Call", self.european_call),
            ("European Put", self.european_put),
            ("Asian Call (Arith)", lambda: self.asian_call("arithmetic")),
            ("Digital Call", self.digital_call),
        ]:
            try:
                r = fn()
                results.append({
                    "Option Type": opt_type,
                    "MC Price": r.get("mc_price", np.nan),
                    "BS Price": r.get("bs_price", np.nan),
                    "Std Error": r.get("std_error", np.nan),
                })
            except Exception:
                pass

        return pd.DataFrame(results).set_index("Option Type")


# ─────────────────────────────────────────────
#  Options Strategy Payoff Diagrams
# ─────────────────────────────────────────────

def option_payoff_diagram(
    strategies: dict[str, list[tuple]],
    S_range: Optional[np.ndarray] = None,
    S0: float = 100.0,
) -> pd.DataFrame:
    """
    Compute payoff/P&L at expiry for options strategies.

    Parameters
    ----------
    strategies : dict
        {strategy_name: [(option_type, K, premium, qty), ...]}
        option_type: 'call', 'put', 'stock'
    S_range : np.ndarray  Spot prices at expiry. Default 50%–150% of S0.
    S0      : float       Current spot price.

    Returns
    -------
    pd.DataFrame  Columns = strategy names, index = S_T values.
    """
    if S_range is None:
        S_range = np.linspace(S0 * 0.5, S0 * 1.5, 200)

    payoffs = {}

    for name, legs in strategies.items():
        total_payoff = np.zeros(len(S_range))
        for leg in legs:
            opt_type, K, premium, qty = leg
            if opt_type == "call":
                payoff = qty * (np.maximum(S_range - K, 0) - premium)
            elif opt_type == "put":
                payoff = qty * (np.maximum(K - S_range, 0) - premium)
            elif opt_type == "stock":
                payoff = qty * (S_range - K)
            else:
                payoff = np.zeros(len(S_range))
            total_payoff += payoff
        payoffs[name] = total_payoff

    return pd.DataFrame(payoffs, index=S_range)
