"""
Backtesting service.

Turns a trained model's out-of-sample predictions into a simple long/flat
(or long/short) trading strategy simulation and computes standard
performance metrics against a buy-and-hold benchmark.

Strategy logic (kept intentionally simple and transparent for v1):
  - On each test-set day, the model's prediction for *that* day (made using
    only information available the prior day) determines the position held
    during that day's return:
      classification target -> long if predicted direction is "up", else flat
                                (or short, if `allow_short` is True)
      regression target      -> long if predicted return/price change > 0,
                                else flat (or short)
  - Returns are compounded daily. Trading costs are modelled as a flat
    per-trade fee in basis points, charged whenever the position changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    model,
    test_start_idx: int,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    trading_days_per_year: int = 252,
) -> dict:
    """Simulate a strategy driven by `model`'s predictions on the test slice
    of `dataset` (rows from `test_start_idx` onward) and compute performance
    metrics, benchmarked against buy-and-hold over the same period.
    """
    test_df = dataset.iloc[test_start_idx:]
    if len(test_df) < 5:
        raise ValueError("Not enough out-of-sample rows to run a backtest.")

    X_test = test_df[feature_cols]
    classification = target_name == "next_day_direction"

    if classification:
        preds = model.predict(X_test)
        positions = np.where(preds == 1, 1, -1 if allow_short else 0)
    else:
        preds = model.predict(X_test)
        positions = np.where(preds > 0, 1, -1 if allow_short else 0)

    # Actual next-day return realised for each row (Close[t+1]/Close[t] - 1).
    # `dataset` already carries a shifted 'target' for the direction/return
    # targets, but we recompute the realised simple return directly from
    # price so the backtest is correct regardless of which target was modeled.
    close = test_df["Close"]
    realized_return = close.pct_change().shift(-1).fillna(0).values

    strategy_return = positions * realized_return

    # Transaction costs: a fee is charged whenever the position changes.
    position_changes = np.abs(np.diff(positions, prepend=0))
    fee_per_change = fee_bps / 10_000
    costs = position_changes * fee_per_change
    strategy_return_net = strategy_return - costs

    equity_curve = (1 + strategy_return_net).cumprod()
    benchmark_curve = (1 + realized_return).cumprod()

    total_return = float(equity_curve[-1] - 1) if len(equity_curve) else 0.0
    n_days = len(strategy_return_net)
    years = max(n_days / trading_days_per_year, 1e-9)
    annual_return = float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -1.0

    daily_std = float(np.std(strategy_return_net, ddof=1)) if n_days > 1 else 0.0
    sharpe = float((np.mean(strategy_return_net) / daily_std) * np.sqrt(trading_days_per_year)) if daily_std > 0 else 0.0

    downside = strategy_return_net[strategy_return_net < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float((np.mean(strategy_return_net) / downside_std) * np.sqrt(trading_days_per_year))
        if downside_std > 0
        else 0.0
    )

    running_max = np.maximum.accumulate(equity_curve) if len(equity_curve) else np.array([1.0])
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    trades = int(position_changes.sum())
    winning_days = strategy_return_net[strategy_return_net > 0]
    losing_days = strategy_return_net[strategy_return_net < 0]
    win_rate = float(len(winning_days) / n_days) if n_days else 0.0
    gross_profit = float(winning_days.sum())
    gross_loss = float(-losing_days.sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    return {
        "metrics": {
            "total_return_pct": round(total_return * 100, 3),
            "annual_return_pct": round(annual_return * 100, 3),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown_pct": round(max_drawdown * 100, 3),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": None if profit_factor == float("inf") else round(profit_factor, 3),
            "number_of_trades": trades,
        },
        "benchmark_total_return_pct": round(float(benchmark_curve[-1] - 1) * 100, 3) if len(benchmark_curve) else 0.0,
        "dates": test_df.index.strftime("%Y-%m-%d").tolist(),
        "strategy_equity_curve": (equity_curve * 100).round(3).tolist(),
        "benchmark_equity_curve": (benchmark_curve * 100).round(3).tolist(),
        "positions": positions.tolist(),
    }
