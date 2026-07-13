"""
Central configuration for StockPredict.

Reads overrides from environment variables so the same codebase works
locally and on a hosted platform (e.g. Render) without code changes.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base configuration shared by all environments."""

    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-key-change-in-production")
    DEBUG: bool = os.environ.get("FLASK_DEBUG", "0") == "1"

    # Where downloaded market data is cached on disk (as parquet/csv).
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_TTL_SECONDS: int = int(os.environ.get("CACHE_TTL_SECONDS", 60 * 60))  # 1 hour

    # Defaults used when the dashboard first loads.
    DEFAULT_TICKER: str = "IVV"
    DEFAULT_LOOKBACK_YEARS: int = 10

    # Tickers shown in the quick-select dropdown (users may type any other
    # valid Yahoo Finance symbol into the search box).
    POPULAR_TICKERS = [
        "IVV", "SPY", "QQQ", "VOO", "AAPL", "MSFT", "GOOGL", "AMZN",
        "META", "TSLA", "NVDA", "BTC-USD", "ETH-USD", "EURUSD=X", "GC=F", "CL=F",
    ]

    MODEL_REGISTRY = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient Boosting",
        "adaboost": "AdaBoost",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "svm": "Support Vector Machine",
        "knn": "K-Nearest Neighbors",
        "naive_bayes": "Naive Bayes",
        "mlp": "MLP Neural Network",
        "lstm": "LSTM (Deep Learning)",
        "gru": "GRU (Deep Learning)",
    }


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
