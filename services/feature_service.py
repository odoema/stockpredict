"""
Builds the model-ready feature matrix and target variable, and implements
feature selection methods.

Feature selection methods implemented in this pass: None, Correlation,
Mutual Information, Random Forest Importance, Variance Threshold.
RFE / Boruta / PCA are flagged as a next-pass extension (see README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

TARGET_TYPES = {
    "next_day_price": "regression",
    "next_day_return": "regression",
    "next_day_direction": "classification",
}


def build_target(df: pd.DataFrame, target: str) -> pd.Series:
    """Build the prediction target series, shifted so each row's target is
    the *next* trading day's outcome relative to that row's features."""
    close = df["Close"]
    if target == "next_day_price":
        return close.shift(-1).rename("target")
    if target == "next_day_return":
        return (close.pct_change().shift(-1) * 100).rename("target")
    if target == "next_day_direction":
        direction = (close.shift(-1) > close).astype(int)
        return direction.rename("target")
    raise ValueError(f"Unknown target '{target}'")


def assemble_dataset(price_df: pd.DataFrame, indicator_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Combine raw OHLCV + indicators + target into one aligned, NaN-free
    dataset ready for feature selection and model training."""
    base_features = price_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    features = base_features.join(indicator_df)
    target_series = build_target(price_df, target)
    dataset = features.join(target_series)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    return dataset


def select_features(dataset: pd.DataFrame, method: str, top_k: int = 15) -> dict:
    """Apply a feature selection method to `dataset` (which must include a
    'target' column). Returns the selected feature names plus an
    importance/score per feature for the UI's bar chart.
    """
    X = dataset.drop(columns=["target"])
    y = dataset["target"]
    is_classification = y.nunique() <= 10 and set(np.unique(y)).issubset({0, 1})

    if method == "none" or not method:
        scores = {col: 1.0 for col in X.columns}
        selected = list(X.columns)

    elif method == "correlation":
        corr = X.apply(lambda col: col.corr(y)).abs().fillna(0)
        scores = corr.to_dict()
        selected = corr.sort_values(ascending=False).head(top_k).index.tolist()

    elif method == "mutual_information":
        mi_fn = mutual_info_classif if is_classification else mutual_info_regression
        mi = mi_fn(X, y, random_state=42)
        scores = dict(zip(X.columns, mi.tolist()))
        selected = pd.Series(scores).sort_values(ascending=False).head(top_k).index.tolist()

    elif method == "random_forest_importance":
        model = (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            if is_classification
            else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        )
        model.fit(X, y)
        scores = dict(zip(X.columns, model.feature_importances_.tolist()))
        selected = pd.Series(scores).sort_values(ascending=False).head(top_k).index.tolist()

    elif method == "variance_threshold":
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(X)
        variances = dict(zip(X.columns, vt.variances_.tolist()))
        scores = variances
        selected = pd.Series(variances).sort_values(ascending=False).head(top_k).index.tolist()

    else:
        raise ValueError(f"Unknown feature selection method '{method}'")

    return {
        "method": method,
        "selected_features": selected,
        "scores": {k: float(v) for k, v in scores.items()},
    }
