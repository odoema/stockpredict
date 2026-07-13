"""
Page routes. These render the HTML shell for each page; the pages then
call the JSON REST API (see api/routes.py) via JavaScript/AJAX to populate
themselves with data.
"""
from flask import Blueprint, current_app, render_template

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def home():
    return render_template("index.html")


@views_bp.route("/dashboard")
def dashboard():
    cfg = current_app.config
    return render_template(
        "dashboard.html",
        popular_tickers=cfg["POPULAR_TICKERS"],
        default_ticker=cfg["DEFAULT_TICKER"],
        default_lookback_years=cfg["DEFAULT_LOOKBACK_YEARS"],
        model_registry=cfg["MODEL_REGISTRY"],
    )


@views_bp.route("/data-quality")
def data_quality():
    return render_template("data_quality.html")


@views_bp.route("/eda")
def eda():
    return render_template("eda.html")


@views_bp.route("/backtest")
def backtest():
    cfg = current_app.config
    return render_template(
        "backtest.html",
        popular_tickers=cfg["POPULAR_TICKERS"],
        default_ticker=cfg["DEFAULT_TICKER"],
        model_registry=cfg["MODEL_REGISTRY"],
    )
