"""
Portfolio service: compares multiple tickers side by side — normalised
performance, cross-asset correlation, and per-ticker risk metrics
(annualised return/volatility, Sharpe, Sortino, max drawdown, and beta
against the first ticker in the list, treated as the benchmark).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_comparison(
    price_frames: dict[str, pd.DataFrame],
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Build a full portfolio comparison from a dict of {ticker: OHLCV df}.

    The first ticker in `price_frames` is treated as the benchmark for beta.
    All series are aligned to their common overlapping date range so
    correlations and betas are computed on the same trading days.
    """
    tickers = list(price_frames.keys())
    if len(tickers) < 2:
        raise ValueError("Portfolio comparison needs at least 2 tickers.")

    closes = pd.DataFrame({t: df["Close"] for t, df in price_frames.items()})
    closes = closes.dropna(how="any")  # align to common trading days
    if len(closes) < 20:
        raise ValueError(
            "Not enough overlapping trading days across the selected tickers "
            "and date range to build a meaningful comparison."
        )

    daily_returns = closes.pct_change().dropna()
    normalized = (closes / closes.iloc[0]) * 100  # base-100 performance index

    correlation = daily_returns.corr().round(3)

    benchmark_key = tickers[0]
    benchmark_returns = daily_returns[benchmark_key]
    benchmark_var = float(np.var(benchmark_returns, ddof=1))

    risk_rows = []
    for ticker in tickers:
        ret = daily_returns[ticker]
        n_days = len(ret)
        years = max(n_days / trading_days_per_year, 1e-9)

        total_return = float(closes[ticker].iloc[-1] / closes[ticker].iloc[0] - 1)
        annual_return = float((1 + total_return) ** (1 / years) - 1)
        annual_vol = float(ret.std(ddof=1) * np.sqrt(trading_days_per_year))

        excess = ret - (risk_free_rate / trading_days_per_year)
        sharpe = float((excess.mean() / ret.std(ddof=1)) * np.sqrt(trading_days_per_year)) if ret.std(ddof=1) > 0 else 0.0

        downside = ret[ret < 0]
        downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = float((excess.mean() / downside_std) * np.sqrt(trading_days_per_year)) if downside_std > 0 else 0.0

        cum = (1 + ret).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_drawdown = float(drawdown.min())

        if ticker == benchmark_key or benchmark_var == 0:
            beta = 1.0 if ticker == benchmark_key else None
        else:
            covariance = float(np.cov(ret, benchmark_returns, ddof=1)[0, 1])
            beta = covariance / benchmark_var

        risk_rows.append(
            {
                "ticker": ticker,
                "total_return_pct": round(total_return * 100, 2),
                "annual_return_pct": round(annual_return * 100, 2),
                "annual_volatility_pct": round(annual_vol * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "sortino_ratio": round(sortino, 3),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "beta": round(beta, 3) if beta is not None else None,
            }
        )

    return {
        "tickers": tickers,
        "benchmark": benchmark_key,
        "dates": normalized.index.strftime("%Y-%m-%d").tolist(),
        "normalized_performance": {t: normalized[t].round(3).tolist() for t in tickers},
        "correlation": {
            "labels": correlation.columns.tolist(),
            "matrix": correlation.values.tolist(),
        },
        "risk_metrics": risk_rows,
    }
