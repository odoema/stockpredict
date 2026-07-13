"""
Basic tests for the core services, run against synthetic OHLCV data so they
don't depend on network access to Yahoo Finance.

Run with:  python -m pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from services import backtest_service, feature_service, indicator_service, model_service, portfolio_service


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(size=n))
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(scale=0.5, size=n),
            "High": close + rng.random(n) * 2,
            "Low": close - rng.random(n) * 2,
            "Close": close,
            "Volume": rng.integers(100_000, 1_000_000, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_indicators_compute_without_error(synthetic_ohlcv):
    keys = list(indicator_service.INDICATOR_REGISTRY.keys())
    out = indicator_service.compute_indicators(synthetic_ohlcv, keys)
    assert len(out) == len(synthetic_ohlcv)
    assert out.shape[1] > 0


def test_assemble_dataset_drops_nans_and_has_target(synthetic_ohlcv):
    ind_df = indicator_service.compute_indicators(synthetic_ohlcv, ["sma", "rsi"])
    dataset = feature_service.assemble_dataset(synthetic_ohlcv, ind_df, "next_day_direction")
    assert "target" in dataset.columns
    assert dataset.isna().sum().sum() == 0
    assert set(dataset["target"].unique()).issubset({0, 1})


@pytest.mark.parametrize(
    "method", ["none", "correlation", "mutual_information", "random_forest_importance", "variance_threshold"]
)
def test_feature_selection_methods(synthetic_ohlcv, method):
    ind_df = indicator_service.compute_indicators(synthetic_ohlcv, list(indicator_service.INDICATOR_REGISTRY.keys()))
    dataset = feature_service.assemble_dataset(synthetic_ohlcv, ind_df, "next_day_direction")
    result = feature_service.select_features(dataset, method, top_k=8)
    assert len(result["selected_features"]) > 0
    assert result["method"] == method


@pytest.mark.parametrize("model_key", list(model_service.MODEL_LABELS.keys()))
def test_train_and_evaluate_classification(synthetic_ohlcv, model_key):
    ind_df = indicator_service.compute_indicators(synthetic_ohlcv, ["sma", "ema", "rsi", "atr"])
    dataset = feature_service.assemble_dataset(synthetic_ohlcv, ind_df, "next_day_direction")
    feature_cols = [c for c in dataset.columns if c != "target"]

    result = model_service.train_and_evaluate(dataset, feature_cols, "next_day_direction", model_key)
    assert "accuracy" in result["metrics"]
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0

    prediction = model_service.predict_latest(result["_model_object"], dataset, feature_cols, "next_day_direction")
    assert prediction["signal"] in {"BUY", "SELL", "HOLD"}


@pytest.mark.parametrize("model_key", list(model_service.MODEL_LABELS.keys()))
def test_train_and_evaluate_regression(synthetic_ohlcv, model_key):
    ind_df = indicator_service.compute_indicators(synthetic_ohlcv, ["sma", "ema"])
    dataset = feature_service.assemble_dataset(synthetic_ohlcv, ind_df, "next_day_return")
    feature_cols = [c for c in dataset.columns if c != "target"]

    result = model_service.train_and_evaluate(dataset, feature_cols, "next_day_return", model_key)
    assert "rmse" in result["metrics"]


@pytest.mark.parametrize("allow_short", [False, True])
def test_backtest_runs_and_returns_sane_metrics(synthetic_ohlcv, allow_short):
    ind_df = indicator_service.compute_indicators(synthetic_ohlcv, ["sma", "ema", "rsi", "atr"])
    dataset = feature_service.assemble_dataset(synthetic_ohlcv, ind_df, "next_day_direction")
    feature_cols = [c for c in dataset.columns if c != "target"]

    trained = model_service.train_and_evaluate(dataset, feature_cols, "next_day_direction", "random_forest")
    result = backtest_service.run_backtest(
        dataset, feature_cols, "next_day_direction", trained["_model_object"], trained["_split_idx"],
        allow_short=allow_short, fee_bps=5.0,
    )
    metrics = result["metrics"]
    assert -100 <= metrics["total_return_pct"] <= 1000
    assert -100 <= metrics["max_drawdown_pct"] <= 0
    assert 0 <= metrics["win_rate_pct"] <= 100
    assert metrics["number_of_trades"] >= 0
    assert len(result["dates"]) == len(result["strategy_equity_curve"]) == len(result["benchmark_equity_curve"])


def test_portfolio_comparison_requires_at_least_two_tickers(synthetic_ohlcv):
    with pytest.raises(ValueError):
        portfolio_service.build_comparison({"IVV": synthetic_ohlcv})


def test_portfolio_comparison_computes_sane_metrics(synthetic_ohlcv):
    rng = np.random.default_rng(7)
    n = len(synthetic_ohlcv)
    other_close = 100 + np.cumsum(rng.normal(size=n))
    other_df = pd.DataFrame(
        {
            "Open": other_close,
            "High": other_close + 1,
            "Low": other_close - 1,
            "Close": other_close,
            "Volume": rng.integers(100_000, 1_000_000, n),
        },
        index=synthetic_ohlcv.index,
    )

    result = portfolio_service.build_comparison({"IVV": synthetic_ohlcv, "SPY": other_df})
    assert result["benchmark"] == "IVV"
    assert set(result["tickers"]) == {"IVV", "SPY"}
    assert len(result["risk_metrics"]) == 2

    benchmark_row = next(r for r in result["risk_metrics"] if r["ticker"] == "IVV")
    assert benchmark_row["beta"] == 1.0

    corr = result["correlation"]
    assert corr["labels"] == ["IVV", "SPY"]
    assert -1.0 <= corr["matrix"][0][1] <= 1.0
