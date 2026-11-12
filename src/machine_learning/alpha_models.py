"""
alpha_models.py
===============
Machine learning alpha signal generation for ASX equities.

Models Implemented
------------------
    RandomForestAlpha   : Ensemble of decision trees for return prediction
    GradientBoostingAlpha: Gradient boosted trees (scikit-learn)
    XGBoostAlpha        : XGBoost implementation with early stopping
    MLEnsemble          : Weighted ensemble of all models
    AlphaModelPipeline  : End-to-end training and signal generation

All models:
    - Output a continuous return prediction score (not binary)
    - Include feature importance analysis
    - Report rank Information Coefficient (IC) and purged time-series CV IC

The pipeline trains ONE pooled cross-sectional model on the (date, ticker)
panel and generates trading signals strictly out-of-sample with an
expanding-window walk-forward: at each retrain date the model only sees
samples whose forward-return target was fully realised before that date
(a purge of ``horizon`` days), then predicts until the next retrain.

Usage
-----
    from src.machine_learning.alpha_models import AlphaModelPipeline

    pipeline = AlphaModelPipeline(prices, macro_data=macro_df)
    signals = pipeline.generate_signals(horizon=21, n_long=10)
    result = pipeline.backtest(benchmark=benchmark_returns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.calendar import period_end_dates
from scipy import stats


try:
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import Ridge, Lasso
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  [AlphaModels] Warning: scikit-learn not installed. "
          "Run: pip install scikit-learn")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ─────────────────────────────────────────────
#  Result Container
# ─────────────────────────────────────────────

@dataclass
class ModelResult:
    """Container for ML model training results."""
    model_name: str = ""
    predictions: pd.Series = field(default_factory=pd.Series)
    feature_importance: pd.Series = field(default_factory=pd.Series)
    train_r2: float = 0.0
    test_r2: float = 0.0
    train_ic: float = 0.0      # Information Coefficient (rank correlation)
    test_ic: float = 0.0
    cv_ic_mean: float = 0.0
    cv_ic_std: float = 0.0
    n_features: int = 0
    n_train: int = 0
    n_test: int = 0


# ─────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────

class BaseAlphaModel:
    """Base class for all alpha models."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self._fitted = False
        self.feature_names_: list[str] = []
        # rows purged between CV train/test folds (set to the target horizon
        # in samples so overlapping forward-return labels cannot leak)
        self.purge: int = 21
        # diagnostic time-series CV inside fit(); switched off for the
        # repeated re-fits of the walk-forward loop
        self.run_cv: bool = True

    def _fresh_model(self):
        """Unfitted copy of the estimator (early stopping disabled)."""
        from sklearn.base import clone
        m = clone(self.model)
        if "early_stopping_rounds" in m.get_params():
            m.set_params(early_stopping_rounds=None)
        return m

    def _information_coefficient(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """
        Rank Information Coefficient: Spearman correlation between
        predicted and actual returns. IC > 0.05 is considered useful.
        """
        if len(y_true) < 5:
            return 0.0
        ic, _ = stats.spearmanr(y_pred, y_true)
        return float(ic) if not np.isnan(ic) else 0.0

    def _cross_validate_ic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> tuple[float, float]:
        """Walk-forward cross-validation of IC."""
        if not HAS_SKLEARN or not self.run_cv:
            return np.nan, np.nan

        gap = int(min(self.purge, max(len(X) // (n_splits + 1) - 1, 0)))
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        ics = []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            try:
                # fit a CLONE: refitting self.model here would silently
                # replace the fully-trained model with a fold model
                m = self._fresh_model()
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                ic = self._information_coefficient(y_te, preds)
                ics.append(ic)
            except Exception:
                pass

        return float(np.mean(ics)) if ics else 0, float(np.std(ics)) if ics else 0

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        """Extract feature importances from the fitted model."""
        if not self._fitted or self.model is None:
            return pd.Series(dtype=float)

        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_)
        else:
            return pd.Series(dtype=float)

        n = min(len(imp), len(feature_names))
        return pd.Series(imp[:n], index=feature_names[:n]).sort_values(ascending=False)


# ─────────────────────────────────────────────
#  Random Forest
# ─────────────────────────────────────────────

class RandomForestAlpha(BaseAlphaModel):
    """
    Random Forest return prediction model.

    Builds an ensemble of decision trees, each trained on a random
    subset of features and data. Averaging across trees reduces
    overfitting relative to a single tree.

    Parameters
    ----------
    n_estimators : int   Number of trees (default 200).
    max_depth    : int   Maximum tree depth (default 8).
    max_features : str   Features per split ('sqrt' = √n_features).
    min_samples_leaf : int   Minimum samples in leaf nodes.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        max_features: str = "sqrt",
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> None:
        super().__init__("Random Forest")
        if HAS_SKLEARN:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
            )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> ModelResult:
        """Fit the model and return evaluation metrics."""
        if not HAS_SKLEARN:
            return ModelResult(model_name=self.name)

        self.feature_names_ = list(X_train.columns)
        X_tr = X_train.fillna(0).values
        y_tr = y_train.values

        # Fit
        self.model.fit(X_tr, y_tr)
        self._fitted = True

        train_preds = self.model.predict(X_tr)
        train_r2 = float(r2_score(y_tr, train_preds))
        train_ic = self._information_coefficient(y_tr, train_preds)

        # CV
        cv_ic_mean, cv_ic_std = self._cross_validate_ic(X_tr, y_tr)

        # Test
        test_r2, test_ic = 0.0, 0.0
        test_predictions = pd.Series(dtype=float)

        if X_test is not None and y_test is not None:
            X_te = X_test.fillna(0).values
            y_te = y_test.values
            test_preds_arr = self.model.predict(X_te)
            test_r2 = float(r2_score(y_te, test_preds_arr))
            test_ic = self._information_coefficient(y_te, test_preds_arr)
            test_predictions = pd.Series(
                test_preds_arr, index=X_test.index, name="prediction"
            )

        return ModelResult(
            model_name=self.name,
            predictions=test_predictions,
            feature_importance=self.get_feature_importance(self.feature_names_),
            train_r2=train_r2,
            test_r2=test_r2,
            train_ic=train_ic,
            test_ic=test_ic,
            cv_ic_mean=cv_ic_mean,
            cv_ic_std=cv_ic_std,
            n_features=len(self.feature_names_),
            n_train=len(X_train),
            n_test=len(X_test) if X_test is not None else 0,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data."""
        if not self._fitted or not HAS_SKLEARN:
            return pd.Series(0.0, index=X.index)
        return pd.Series(
            self.model.predict(X.fillna(0).values),
            index=X.index,
            name="RF_prediction",
        )


# ─────────────────────────────────────────────
#  Gradient Boosting
# ─────────────────────────────────────────────

class GradientBoostingAlpha(BaseAlphaModel):
    """
    Gradient Boosting return prediction model.
    Builds trees sequentially — each tree corrects the errors of the last.

    Parameters
    ----------
    n_estimators  : int     Number of boosting rounds.
    learning_rate : float   Shrinkage factor (0.01–0.1 typical).
    max_depth     : int     Tree depth (3–5 typical for boosting).
    subsample     : float   Row sampling per tree (reduces overfitting).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        super().__init__("Gradient Boosting")
        if HAS_SKLEARN:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=random_state,
            )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> ModelResult:
        """Fit and evaluate."""
        if not HAS_SKLEARN:
            return ModelResult(model_name=self.name)

        self.feature_names_ = list(X_train.columns)
        X_tr = X_train.fillna(0).values
        y_tr = y_train.values

        self.model.fit(X_tr, y_tr)
        self._fitted = True

        train_preds = self.model.predict(X_tr)
        train_ic = self._information_coefficient(y_tr, train_preds)
        train_r2 = float(r2_score(y_tr, train_preds))
        cv_ic_mean, cv_ic_std = self._cross_validate_ic(X_tr, y_tr)

        test_r2, test_ic = 0.0, 0.0
        test_predictions = pd.Series(dtype=float)

        if X_test is not None and y_test is not None:
            X_te = X_test.fillna(0).values
            y_te = y_test.values
            test_preds_arr = self.model.predict(X_te)
            test_r2 = float(r2_score(y_te, test_preds_arr))
            test_ic = self._information_coefficient(y_te, test_preds_arr)
            test_predictions = pd.Series(
                test_preds_arr, index=X_test.index, name="prediction"
            )

        return ModelResult(
            model_name=self.name,
            predictions=test_predictions,
            feature_importance=self.get_feature_importance(self.feature_names_),
            train_r2=train_r2, test_r2=test_r2,
            train_ic=train_ic, test_ic=test_ic,
            cv_ic_mean=cv_ic_mean, cv_ic_std=cv_ic_std,
            n_features=len(self.feature_names_),
            n_train=len(X_train),
            n_test=len(X_test) if X_test is not None else 0,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted or not HAS_SKLEARN:
            return pd.Series(0.0, index=X.index)
        return pd.Series(
            self.model.predict(X.fillna(0).values),
            index=X.index, name="GB_prediction"
        )


# ─────────────────────────────────────────────
#  XGBoost
# ─────────────────────────────────────────────

class XGBoostAlpha(BaseAlphaModel):
    """
    XGBoost return prediction model with early stopping.

    XGBoost is typically the highest-performing tree model for
    tabular financial data. Key advantages:
        - L1/L2 regularisation built in
        - Handles missing values natively
        - Faster than sklearn GBM on large datasets

    Parameters
    ----------
    n_estimators  : int     Maximum boosting rounds.
    learning_rate : float   Eta parameter.
    max_depth     : int     Tree depth.
    reg_alpha     : float   L1 regularisation.
    reg_lambda    : float   L2 regularisation.
    early_stopping: int     Stop if no improvement for N rounds.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping: int = 50,
        random_state: int = 42,
    ) -> None:
        super().__init__("XGBoost")
        self.early_stopping = early_stopping
        if HAS_XGB:
            # XGBoost >= 2.0: early stopping is configured on the estimator
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        elif HAS_SKLEARN:
            # Fallback to sklearn GBM
            self.model = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, random_state=random_state,
            )
            self.name = "GradientBoosting (XGB fallback)"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> ModelResult:
        """Fit XGBoost with optional early stopping."""
        if not HAS_SKLEARN and not HAS_XGB:
            return ModelResult(model_name=self.name)

        self.feature_names_ = list(X_train.columns)
        X_tr = X_train.fillna(0).values
        y_tr = y_train.values

        # Early stopping uses a validation slice carved from the END of the
        # training data (with a purge gap) - never the test set, otherwise
        # the reported test IC would be optimistically biased.
        n_val = int(len(X_tr) * 0.2)
        if HAS_XGB and self.early_stopping and n_val >= 50 and len(X_tr) - n_val - self.purge >= 100:
            cut = len(X_tr) - n_val
            self.model.set_params(early_stopping_rounds=self.early_stopping)
            self.model.fit(
                X_tr[: cut - self.purge], y_tr[: cut - self.purge],
                eval_set=[(X_tr[cut:], y_tr[cut:])],
                verbose=False,
            )
        else:
            if HAS_XGB:
                self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X_tr, y_tr)

        self._fitted = True
        train_preds = self.model.predict(X_tr)
        train_ic = self._information_coefficient(y_tr, train_preds)
        train_r2 = float(r2_score(y_tr, train_preds))
        cv_ic_mean, cv_ic_std = self._cross_validate_ic(X_tr, y_tr)

        test_r2, test_ic = 0.0, 0.0
        test_predictions = pd.Series(dtype=float)

        if X_test is not None and y_test is not None:
            X_te = X_test.fillna(0).values
            y_te = y_test.values
            test_preds_arr = self.model.predict(X_te)
            test_r2 = float(r2_score(y_te, test_preds_arr))
            test_ic = self._information_coefficient(y_te, test_preds_arr)
            test_predictions = pd.Series(
                test_preds_arr, index=X_test.index, name="prediction"
            )

        return ModelResult(
            model_name=self.name,
            predictions=test_predictions,
            feature_importance=self.get_feature_importance(self.feature_names_),
            train_r2=train_r2, test_r2=test_r2,
            train_ic=train_ic, test_ic=test_ic,
            cv_ic_mean=cv_ic_mean, cv_ic_std=cv_ic_std,
            n_features=len(self.feature_names_),
            n_train=len(X_train),
            n_test=len(X_test) if X_test is not None else 0,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            return pd.Series(0.0, index=X.index)
        return pd.Series(
            self.model.predict(X.fillna(0).values),
            index=X.index, name="XGB_prediction"
        )


# ─────────────────────────────────────────────
#  ML Ensemble
# ─────────────────────────────────────────────

class MLEnsemble:
    """
    Weighted ensemble of multiple alpha models.
    Combines predictions from RF, GB, and XGB with configurable weights.
    Reduces model-specific overfitting and improves stability.

    Parameters
    ----------
    weights : dict   {model_name: weight} — defaults to equal weight.
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
    ) -> None:
        self.rf = RandomForestAlpha()
        self.gb = GradientBoostingAlpha()
        self.xgb_model = XGBoostAlpha()
        self.weights = weights or {"RF": 0.33, "GB": 0.33, "XGB": 0.34}
        self._results: dict[str, ModelResult] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> dict[str, ModelResult]:
        """Fit all models and return individual results."""
        self._results["RF"] = self.rf.fit(X_train, y_train, X_test, y_test)
        self._results["GB"] = self.gb.fit(X_train, y_train, X_test, y_test)
        self._results["XGB"] = self.xgb_model.fit(X_train, y_train, X_test, y_test)
        return self._results

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Weighted average of model predictions."""
        preds = {}
        total_w = 0.0

        for name, w in self.weights.items():
            model = {"RF": self.rf, "GB": self.gb, "XGB": self.xgb_model}.get(name)
            if model and model._fitted:
                preds[name] = model.predict(X) * w
                total_w += w

        if not preds:
            return pd.Series(0.0, index=X.index)

        ensemble = sum(preds.values()) / total_w
        ensemble.name = "Ensemble_prediction"
        return ensemble

    def compare_models(self) -> pd.DataFrame:
        """Compare all model metrics side by side."""
        rows = []
        for name, r in self._results.items():
            rows.append({
                "Model": r.model_name,
                "Train IC": r.train_ic,
                "Test IC": r.test_ic,
                "CV IC Mean": r.cv_ic_mean,
                "CV IC Std": r.cv_ic_std,
                "Train R²": r.train_r2,
                "Test R²": r.test_r2,
                "N Features": r.n_features,
                "N Train": r.n_train,
                "N Test": r.n_test,
            })
        return pd.DataFrame(rows).set_index("Model")

    def top_features(self, n: int = 20) -> pd.DataFrame:
        """Aggregate feature importance across all models."""
        all_imp = []
        for name, r in self._results.items():
            if not r.feature_importance.empty:
                imp = r.feature_importance.rename(name)
                all_imp.append(imp)

        if not all_imp:
            return pd.DataFrame()

        df = pd.concat(all_imp, axis=1).fillna(0)
        df["Average"] = df.mean(axis=1)
        return df.sort_values("Average", ascending=False).head(n)


# ─────────────────────────────────────────────
#  End-to-End Pipeline
# ─────────────────────────────────────────────

class AlphaModelPipeline:
    """
    End-to-end ML alpha pipeline on a pooled (date, ticker) panel.

    feature engineering -> pooled model -> purged walk-forward predictions
    -> cross-sectional ranking -> backtest.

    Parameters
    ----------
    prices         : pd.DataFrame   Adjusted close prices.
    macro_data     : pd.DataFrame   Macro variables (optional).
    model_type     : str            'rf' | 'gb' | 'xgb' | 'ensemble'
    horizon        : int            Forward return prediction horizon (days).
    model_params   : dict           Keyword arguments for the model constructor.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
        model_type: str = "rf",
        horizon: int = 21,
        model_params: Optional[dict] = None,
    ) -> None:
        from src.machine_learning.feature_engineering import FeatureEngineer

        self.model_params = model_params or {}
        self.prices = prices.copy()
        self.macro = macro_data
        self.model_type = model_type
        self.horizon = horizon
        self.fe = FeatureEngineer(prices, macro_data)

        self._model = None
        self._feature_names: list[str] = []
        self._panel: Optional[pd.DataFrame] = None
        self.oos_predictions: pd.DataFrame = pd.DataFrame()
        self.oos_ic: pd.Series = pd.Series(dtype=float)

    # ── panel construction ──
    def _get_model(self):
        """Instantiate the chosen model."""
        cls = {
            "rf": RandomForestAlpha,
            "gb": GradientBoostingAlpha,
            "xgb": XGBoostAlpha,
        }.get(self.model_type)
        model = cls(**self.model_params) if cls else MLEnsemble()
        for m in ([model.rf, model.gb, model.xgb_model] if isinstance(model, MLEnsemble) else [model]):
            m.purge = self.horizon
        return model

    def build_panel(self) -> pd.DataFrame:
        """Stacked feature panel indexed by (date, ticker), with 'target'."""
        if self._panel is None:
            panel = self.fe.build_universe_features(target_horizon=self.horizon)
            panel = panel.replace([np.inf, -np.inf], np.nan).sort_index()
            self._feature_names = [c for c in panel.columns if c != "target"]
            self._panel = panel
        return self._panel

    @staticmethod
    def cross_sectional_ic(pred: pd.Series, target: pd.Series) -> pd.Series:
        """Spearman rank IC between predictions and realised returns, per date."""
        df = pd.DataFrame({"p": pred, "y": target}).dropna()
        out = {}
        for d, g in df.groupby(level=0):
            if len(g) >= 3 and g["p"].nunique() > 1 and g["y"].nunique() > 1:
                out[d] = stats.spearmanr(g["p"], g["y"])[0]
        return pd.Series(out, name="IC").sort_index()

    @staticmethod
    def ic_tstat(ic: pd.Series, horizon: int) -> float:
        """
        t-statistic of the mean IC using NON-OVERLAPPING observations
        (every ``horizon``-th date). Daily ICs on h-day forward returns
        overlap by h-1 days and are strongly autocorrelated, so the naive
        sqrt(N) t-stat overstates significance by roughly sqrt(h).
        """
        sub = ic.dropna().iloc[:: max(int(horizon), 1)]
        if len(sub) < 3 or sub.std() == 0:
            return np.nan
        return float(sub.mean() / sub.std() * np.sqrt(len(sub)))

    # ── single hold-out evaluation ──
    def train(self, test_pct: float = 0.20) -> dict:
        """
        Fit on the first (1 - test_pct) of DATES and evaluate on the rest,
        purging ``horizon`` days between them.

        Returns
        -------
        dict with model results and the out-of-sample cross-sectional IC
        (mean, std, t-stat, hit-rate).
        """
        panel = self.build_panel().dropna(subset=["target"])
        dates = panel.index.get_level_values(0).unique().sort_values()
        split = int(len(dates) * (1 - test_pct))
        train_end = dates[max(split - self.horizon, 1) - 1]
        test_start = dates[split]

        d = panel.index.get_level_values(0)
        train = panel[d <= train_end]
        test = panel[d >= test_start]

        X_tr, y_tr = train[self._feature_names], train["target"]
        X_te, y_te = test[self._feature_names], test["target"]

        model = self._get_model()
        result = model.fit(X_tr, y_tr, X_te, y_te)
        self._model = model

        preds = model.predict(X_te.fillna(0))
        ic = self.cross_sectional_ic(preds, y_te)
        summary = {
            "results": result,
            "oos_cs_ic_mean": float(ic.mean()) if len(ic) else np.nan,
            "oos_cs_ic_std": float(ic.std()) if len(ic) > 1 else np.nan,
            "oos_cs_ic_tstat": self.ic_tstat(ic, self.horizon),
            "oos_ic_hit_rate": float((ic > 0).mean()) if len(ic) else np.nan,
            "n_train_rows": len(X_tr),
            "n_test_rows": len(X_te),
            "train_end": train_end,
            "test_start": test_start,
        }
        return summary

    # ── walk-forward predictions ──
    def walk_forward_predict(
        self,
        min_train_days: int = 504,
        retrain_every: int = 126,
    ) -> pd.DataFrame:
        """
        Expanding-window, purged walk-forward predictions.

        At each retrain date d the model is fitted on panel rows dated
        <= d - horizon (targets realised by d) and then predicts every
        date in (d, next retrain]. All predictions are out-of-sample.

        Returns
        -------
        pd.DataFrame  rows = dates, columns = tickers, values = predicted
                      forward return.
        """
        panel = self.build_panel()
        feats = self._feature_names
        dates = panel.index.get_level_values(0).unique().sort_values()
        d_level = panel.index.get_level_values(0)

        preds = []
        retrain_idx = list(range(min_train_days, len(dates), retrain_every))
        for k, i in enumerate(retrain_idx):
            d = dates[i]
            cutoff = dates[max(i - self.horizon, 0)]
            train = panel[(d_level <= cutoff)].dropna(subset=["target"])
            if len(train) < 200:
                continue
            nxt = dates[retrain_idx[k + 1]] if k + 1 < len(retrain_idx) else dates[-1] + pd.Timedelta(days=1)
            live = panel[(d_level >= d) & (d_level < nxt)]
            if live.empty:
                continue
            model = self._get_model()
            for m in ([model.rf, model.gb, model.xgb_model] if isinstance(model, MLEnsemble) else [model]):
                m.run_cv = False
            model.fit(train[feats], train["target"])
            p = model.predict(live[feats].fillna(0))
            preds.append(pd.Series(np.asarray(p), index=live.index))
            self._model = model

        if not preds:
            return pd.DataFrame()

        pred = pd.concat(preds)
        pred = pred[~pred.index.duplicated(keep="last")]
        self.oos_predictions = pred.unstack(level=1).sort_index()
        self.oos_ic = self.cross_sectional_ic(pred, panel["target"])
        return self.oos_predictions

    def generate_signals(
        self,
        n_long: int = 10,
        n_short: int = 0,
        min_train_days: int = 504,
        retrain_every: int = 126,
        rebalance_freq: str = "monthly",
    ) -> pd.DataFrame:
        """
        Cross-sectional trading signals from OUT-OF-SAMPLE predictions:
        on each rebalance date go long the n_long highest predicted
        forward returns (and optionally short the n_short lowest).
        """
        pred_df = self.walk_forward_predict(min_train_days, retrain_every)
        if pred_df.empty:
            return pd.DataFrame()

        pred_df = pred_df.reindex(columns=self.prices.columns)
        signals = pd.DataFrame(0.0, index=pred_df.index, columns=pred_df.columns)
        rebalance_dates = period_end_dates(pred_df.index, rebalance_freq)
        last_signal = pd.Series(0.0, index=pred_df.columns)

        for date in pred_df.index:
            if date in rebalance_dates:
                row = pred_df.loc[date].dropna()
                ns = min(n_short, len(row) // 2) if n_short > 0 else 0
                nl = min(n_long, len(row) - ns)
                new_sig = pd.Series(0.0, index=pred_df.columns)
                if nl >= 1:
                    new_sig[row.nlargest(nl).index] = 1.0 / nl
                    if ns > 0:
                        new_sig[row.nsmallest(ns).index] = -1.0 / ns
                last_signal = new_sig
            signals.loc[date] = last_signal

        return signals.reindex(self.prices.index).fillna(0.0)

    def backtest(
        self,
        n_long: int = 10,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
        **signal_kwargs,
    ):
        """Generate walk-forward signals and run a backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals(n_long=n_long, **signal_kwargs)
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
            strategy_name=f"ML_{self.model_type.upper()}_WalkForward",
        )
        return bt.run()
