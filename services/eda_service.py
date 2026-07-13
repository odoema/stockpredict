"""
Data quality and exploratory data analysis (EDA) service.

Every function returns plain Python / JSON-serialisable structures so the
API layer can pass results straight to `jsonify` for the frontend charts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summarise dataset shape, missingness, duplicates, dtypes and coverage."""
    missing = df.isna().sum()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "date_start": df.index.min().strftime("%Y-%m-%d") if len(df) else None,
        "date_end": df.index.max().strftime("%Y-%m-%d") if len(df) else None,
        "missing_values": {col: int(v) for col, v in missing.items()},
        "total_missing": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "summary_stats": _summary_stats(df),
    }


def _summary_stats(df: pd.DataFrame) -> dict:
    desc = df.describe().to_dict()
    return {col: {k: _safe_float(v) for k, v in stats.items()} for col, stats in desc.items()}


def _safe_float(value):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return float(value)


def price_series(df: pd.DataFrame) -> dict:
    """Return OHLCV series formatted for a Plotly candlestick + volume chart."""
    return {
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "open": df["Open"].round(4).tolist(),
        "high": df["High"].round(4).tolist(),
        "low": df["Low"].round(4).tolist(),
        "close": df["Close"].round(4).tolist(),
        "volume": df["Volume"].tolist(),
    }


def returns_analysis(df: pd.DataFrame) -> dict:
    """Daily returns, cumulative returns, and rolling volatility, plus a
    histogram-ready distribution of daily returns."""
    daily_returns = df["Close"].pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    rolling_vol = daily_returns.rolling(21).std() * np.sqrt(252) * 100

    hist, bin_edges = np.histogram(daily_returns.dropna(), bins=40)

    return {
        "dates": daily_returns.index.strftime("%Y-%m-%d").tolist(),
        "daily_returns": (daily_returns * 100).round(4).tolist(),
        "cumulative_returns": (cumulative_returns * 100).round(4).tolist(),
        "rolling_volatility": rolling_vol.round(4).fillna(0).tolist(),
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.round(4).tolist(),
        },
        "boxplot_values": (daily_returns * 100).round(4).tolist(),
    }


def correlation_matrix(feature_df: pd.DataFrame) -> dict:
    """Pearson correlation matrix for a feature/indicator DataFrame."""
    corr = feature_df.corr().round(3)
    return {
        "labels": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
    }


def feature_distributions(feature_df: pd.DataFrame, max_features: int = 12) -> dict:
    """Histogram-ready distributions for up to `max_features` numeric columns."""
    out = {}
    for col in feature_df.columns[:max_features]:
        series = feature_df[col].dropna()
        if series.empty:
            continue
        counts, edges = np.histogram(series, bins=30)
        out[col] = {"counts": counts.tolist(), "bin_edges": edges.round(4).tolist()}
    return out
