"""
REST API for StockPredict.

Designed to be stateless per request: every endpoint takes the ticker,
date range, and any indicator/model choices it needs, and recomputes from
the (disk-cached) OHLCV data. This keeps the server simple to reason about
and to scale horizontally later.
"""
from __future__ import annotations

import io
import logging

from flask import Blueprint, Response, current_app, jsonify, request

from services import backtest_service, eda_service, feature_service, indicator_service, model_service
from services.data_service import DataService, TickerNotFoundError

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


def _data_service() -> DataService:
    return DataService(current_app.config["DATA_DIR"], current_app.config["CACHE_TTL_SECONDS"])


def _parse_request(payload: dict):
    ticker = (payload.get("ticker") or "").strip()
    start = payload.get("start")
    end = payload.get("end")
    if not ticker:
        raise ValueError("A 'ticker' is required.")
    if not start or not end:
        raise ValueError("Both 'start' and 'end' dates are required (YYYY-MM-DD).")
    return ticker, start, end


@api_bp.errorhandler(ValueError)
def handle_value_error(err: ValueError):
    return jsonify({"error": str(err)}), 400


@api_bp.errorhandler(TickerNotFoundError)
def handle_ticker_not_found(err: TickerNotFoundError):
    return jsonify({"error": str(err)}), 404


# ---------------------------------------------------------------------------
# Metadata endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/indicators", methods=["GET"])
def get_indicators():
    return jsonify(indicator_service.list_indicators())


@api_bp.route("/models", methods=["GET"])
def get_models():
    return jsonify(current_app.config["MODEL_REGISTRY"])


@api_bp.route("/tickers", methods=["GET"])
def get_tickers():
    return jsonify(current_app.config["POPULAR_TICKERS"])


# ---------------------------------------------------------------------------
# Data + data quality
# ---------------------------------------------------------------------------

@api_bp.route("/data", methods=["POST"])
def get_data():
    payload = request.get_json(force=True) or {}
    ticker, start, end = _parse_request(payload)

    df = _data_service().get_history(ticker, start, end)
    return jsonify(
        {
            "ticker": ticker.upper(),
            "quality_report": eda_service.data_quality_report(df),
            "price_series": eda_service.price_series(df),
        }
    )


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

@api_bp.route("/eda", methods=["POST"])
def get_eda():
    payload = request.get_json(force=True) or {}
    ticker, start, end = _parse_request(payload)

    df = _data_service().get_history(ticker, start, end)
    ohlcv_corr = eda_service.correlation_matrix(df[["Open", "High", "Low", "Close", "Volume"]])

    return jsonify(
        {
            "ticker": ticker.upper(),
            "returns": eda_service.returns_analysis(df),
            "correlation": ohlcv_corr,
        }
    )


# ---------------------------------------------------------------------------
# Feature engineering + selection
# ---------------------------------------------------------------------------

def _build_dataset(payload: dict):
    ticker, start, end = _parse_request(payload)
    indicators = payload.get("indicators") or list(indicator_service.INDICATOR_REGISTRY.keys())
    target = payload.get("target", "next_day_direction")
    if target not in feature_service.TARGET_TYPES:
        raise ValueError(f"Unknown target '{target}'.")

    price_df = _data_service().get_history(ticker, start, end)
    indicator_df = indicator_service.compute_indicators(price_df, indicators)
    dataset = feature_service.assemble_dataset(price_df, indicator_df, target)

    if len(dataset) < 60:
        raise ValueError(
            "Not enough clean rows after computing indicators and dropping "
            "warm-up NaNs. Try a longer date range or fewer long-window indicators."
        )
    return ticker, target, indicator_df, dataset


@api_bp.route("/features", methods=["POST"])
def get_features():
    payload = request.get_json(force=True) or {}
    method = payload.get("feature_selection_method", "none")

    ticker, target, indicator_df, dataset = _build_dataset(payload)
    selection = feature_service.select_features(dataset, method)
    feature_corr = eda_service.correlation_matrix(indicator_df.dropna(axis=1, how="all"))
    distributions = eda_service.feature_distributions(indicator_df)

    return jsonify(
        {
            "ticker": ticker.upper(),
            "target": target,
            "n_rows": int(len(dataset)),
            "all_features": [c for c in dataset.columns if c != "target"],
            "selection": selection,
            "correlation": feature_corr,
            "distributions": distributions,
        }
    )


# ---------------------------------------------------------------------------
# Model training, comparison, and prediction
# ---------------------------------------------------------------------------

@api_bp.route("/train", methods=["POST"])
def train_model():
    payload = request.get_json(force=True) or {}
    model_key = payload.get("model", "random_forest")
    method = payload.get("feature_selection_method", "none")

    ticker, target, _indicator_df, dataset = _build_dataset(payload)
    selection = feature_service.select_features(dataset, method)
    feature_cols = selection["selected_features"]

    result = model_service.train_and_evaluate(dataset, feature_cols, target, model_key)
    prediction = model_service.predict_latest(result.pop("_model_object"), dataset, feature_cols, target)
    result.pop("_y_test", None)
    y_pred = result.pop("_y_pred", None)
    test_dates = result.pop("_test_dates", None)

    return jsonify(
        {
            "ticker": ticker.upper(),
            "target": target,
            "selected_features": feature_cols,
            "evaluation": result,
            "prediction": prediction,
            "test_dates": test_dates,
            "test_predictions": y_pred,
        }
    )


@api_bp.route("/train/compare", methods=["POST"])
def compare_models():
    payload = request.get_json(force=True) or {}
    method = payload.get("feature_selection_method", "none")
    model_keys = payload.get("models") or list(model_service.MODEL_LABELS.keys())

    ticker, target, _indicator_df, dataset = _build_dataset(payload)
    selection = feature_service.select_features(dataset, method)
    feature_cols = selection["selected_features"]

    comparison = model_service.compare_models(dataset, feature_cols, target, model_keys)
    return jsonify({"ticker": ticker.upper(), "target": target, "comparison": comparison})


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

@api_bp.route("/backtest", methods=["POST"])
def backtest():
    payload = request.get_json(force=True) or {}
    model_key = payload.get("model", "random_forest")
    method = payload.get("feature_selection_method", "none")
    allow_short = bool(payload.get("allow_short", False))
    fee_bps = float(payload.get("fee_bps", 5.0))

    ticker, target, _indicator_df, dataset = _build_dataset(payload)
    selection = feature_service.select_features(dataset, method)
    feature_cols = selection["selected_features"]

    trained = model_service.train_and_evaluate(dataset, feature_cols, target, model_key)
    model = trained["_model_object"]
    split_idx = trained["_split_idx"]

    backtest_result = backtest_service.run_backtest(
        dataset, feature_cols, target, model, split_idx, allow_short=allow_short, fee_bps=fee_bps
    )

    return jsonify(
        {
            "ticker": ticker.upper(),
            "target": target,
            "model": trained["model_label"],
            "selected_features": feature_cols,
            "backtest": backtest_result,
        }
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@api_bp.route("/export", methods=["POST"])
def export_data():
    payload = request.get_json(force=True) or {}
    fmt = payload.get("format", "csv").lower()
    ticker, start, end = _parse_request(payload)
    indicators = payload.get("indicators") or []

    price_df = _data_service().get_history(ticker, start, end)
    indicator_df = indicator_service.compute_indicators(price_df, indicators) if indicators else None
    full_df = price_df.join(indicator_df) if indicator_df is not None else price_df

    if fmt == "csv":
        buf = io.StringIO()
        full_df.to_csv(buf)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={ticker.upper()}_export.csv"},
        )
    if fmt == "json":
        return Response(
            full_df.reset_index().to_json(orient="records", date_format="iso"),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={ticker.upper()}_export.json"},
        )

    raise ValueError(f"Unsupported export format '{fmt}'. Use 'csv' or 'json'.")
