# StockPredict

A web application version of the WQU MScFE market-prediction research
notebook — data ingestion, data quality checks, EDA, feature engineering,
feature selection, model training/evaluation, and next-day prediction —
for **any valid Yahoo Finance ticker**, not just IVV.

This is **v4 (sprints 1–4)** of the full master spec: a working end-to-end
pipeline with a real dashboard, real data quality and EDA pages, real
indicators, real feature selection, a 14-model registry (12 tabular + 2
deep learning), a full backtesting page, and multi-ticker portfolio
comparison. See **Roadmap** below for what's planned next.

---

## Features in this version

- **Any ticker**: type or pick from a shortlist (IVV, SPY, QQQ, VOO, AAPL,
  MSFT, GOOGL, AMZN, META, TSLA, NVDA, BTC-USD, ETH-USD, EURUSD=X, GC=F, CL=F,
  or any other valid Yahoo Finance symbol).
- **Data Quality page**: shape, missing values, duplicates, dtypes, date
  coverage, summary statistics.
- **EDA page**: interactive Plotly candlestick + volume, daily returns,
  return histogram, box plot, rolling volatility, cumulative returns, and an
  OHLCV correlation heatmap. All charts zoom/pan/download natively via
  Plotly's toolbar.
- **19 technical indicators** across Trend (SMA, EMA, MACD, ADX, Parabolic
  SAR), Momentum (RSI, Williams %R, CCI, ROC, Stochastic Oscillator),
  Volatility (ATR, Bollinger Bands, Historical Volatility, Rolling
  Volatility), and Volume (OBV, CMF, MFI, VWAP, Ease of Movement) —
  independently toggleable.
- **5 feature selection methods**: None, Correlation, Mutual Information,
  Random Forest Importance, Variance Threshold — each with a
  score-per-feature breakdown for the UI.
- **3 prediction targets**: next-day price, next-day return, next-day
  direction (up/down).
- **12 models fully wired** (real training + evaluation, not stubs):
  Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient
  Boosting, AdaBoost, XGBoost, LightGBM, SVM, KNN, Naive Bayes, MLP Neural
  Network. Every model works for all 3 prediction targets. Scale-sensitive
  models (Logistic/Linear Regression, SVM, KNN, Naive Bayes, MLP) are
  wrapped in a `StandardScaler` pipeline; tree/boosting models are used
  as-is (scale-invariant).
- **2 deep learning models**: LSTM and GRU, trained on a sliding window of
  sequential trading days (default 20) rather than flat feature rows.
  Feature scaling is fit on the training slice only to avoid lookahead
  leakage. Configurable `window` and `epochs` via the API; noticeably
  slower to train than the 12 tabular models, so they're excluded from the
  default "Compare All Models" run (but can be added to it explicitly) and
  aren't yet wired into the Backtesting page — see Roadmap.
- **Run All / Compare Models**: side-by-side comparison table (accuracy,
  precision, recall, F1, ROC AUC for classification; RMSE, MAE, R² for
  regression; plus train/predict time), best model highlighted.
- **Prediction panel**: BUY / SELL / HOLD signal with probability-up,
  probability-down, and confidence.
- **Backtesting page**: turns a model's out-of-sample predictions into a
  long/flat or long/short daily strategy, with total return, annualised
  return, Sharpe, Sortino, max drawdown, win rate, profit factor, number of
  trades, transaction-cost modelling (bps/trade), and a strategy-vs-buy-and-
  hold portfolio growth chart.
- **Export**: CSV and JSON export of the raw + indicator dataset.
- **Portfolio page**: compare 2–8 tickers side by side — normalised
  performance chart (base 100), cross-asset correlation heatmap, and a risk
  metrics table (total/annualised return, annualised volatility, Sharpe,
  Sortino, max drawdown, beta vs. the first ticker as benchmark).
- **Dark-mode terminal UI**: built with Bootstrap 5 + Plotly.js, no build
  step required.

---

## Architecture

```
StockPredict/
  app.py              # Flask application factory
  config.py           # Environment-driven configuration
  run.py              # Local dev entrypoint (python run.py)
  views.py            # HTML page routes (/, /dashboard, /data-quality, /eda)
  api/
    routes.py          # REST API blueprint (/api/...)
  services/
    data_service.py     # Yahoo Finance download + disk caching
    indicator_service.py# 19 technical indicators, grouped registry
    eda_service.py       # Data quality + EDA computations
    feature_service.py   # Target construction + feature selection
    model_service.py     # Train/eval/predict/compare for the 12 tabular models
    deep_learning_service.py # LSTM/GRU: sequence windowing, train/eval/predict
    backtest_service.py  # Strategy simulation + performance metrics
    portfolio_service.py # Multi-ticker performance, correlation, risk metrics
  templates/            # Jinja2 page shells (base, index, dashboard, ...)
  static/css/style.css   # Design system (dark terminal aesthetic)
  static/js/             # Per-page AJAX + Plotly logic
  data/                  # On-disk cache of downloaded OHLCV data (gitignored)
  tests/                 # Pytest suite (runs against synthetic data, no network needed)
```

The API is **stateless per request**: every endpoint takes the ticker,
date range, and any indicator/model choices it needs, and recomputes from
the (disk-cached) OHLCV data. This keeps the server simple to reason about
and to scale horizontally later — no server-side session state to manage.

### REST API

| Endpoint              | Method | Purpose                                             |
|------------------------|--------|------------------------------------------------------|
| `/api/tickers`          | GET    | Popular ticker shortlist                              |
| `/api/indicators`       | GET    | Grouped indicator registry for the multi-select UI     |
| `/api/models`           | GET    | Available model keys/labels                            |
| `/api/data`             | POST   | OHLCV history + data quality report                     |
| `/api/eda`              | POST   | Returns analysis + OHLCV correlation                     |
| `/api/features`         | POST   | Assembled dataset + feature selection results             |
| `/api/train`            | POST   | Train one model (tabular or LSTM/GRU), evaluate, predict next day |
| `/api/train/compare`    | POST   | Train several models, return a ranked comparison table (tabular by default) |
| `/api/backtest`         | POST   | Simulate a strategy from model signals, return metrics + equity curve |
| `/api/portfolio`        | POST   | Compare 2-8 tickers: performance, correlation, risk metrics    |
| `/api/export`           | POST   | Download CSV/JSON of the raw + indicator dataset               |

---

## Installation

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd StockPredict
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running locally

```bash
python run.py
```

Then open **http://localhost:5000**.

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests run against synthetic OHLCV data, so they pass without any network
access to Yahoo Finance.

---

## Deploying (get a shareable link)

The easiest free option is **Render**:

1. Push this folder to a GitHub repository.
2. On [render.com](https://render.com), click **New → Web Service** and
   connect the repo.
3. Set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn "app:create_app()" --bind 0.0.0.0:$PORT`
   - **Environment variable**: `FLASK_ENV=production`
4. Deploy. Render gives you a public `https://your-app.onrender.com` URL.

Any other host that runs a standard Flask/WSGI app (Railway, PythonAnywhere,
Fly.io, a VPS with gunicorn + nginx) will work the same way — the app has no
platform-specific code.

**Note:** the app calls the live Yahoo Finance API at request time, so the
host you choose must have outbound internet access.

---

## Example usage

1. Go to **Dashboard**.
2. Type a ticker (e.g. `NVDA`) or pick one from the list.
3. Choose a date range (defaults to the last 10 years).
4. Pick a prediction target, feature selection method, and model.
5. Toggle the indicators you want computed.
6. Click **Run Analysis** to train the model and see the prediction,
   metrics, feature importance, and confusion matrix.
7. Click **Compare All Models** to rank all 4 models on the same data.
8. Use **Export** to download the raw + indicator dataset.

---

## Roadmap (from the full master spec, not yet built)

These sprints intentionally scoped to a working core rather than a
half-finished everything. Still planned:

- Wire LSTM/GRU into the Backtesting page (currently tabular-models-only,
  since the backtest engine calls `model.predict()` on a plain feature
  DataFrame rather than a scaled sequence window).
- Transformer as a third deep-learning option.
- Additional feature selection: Recursive Feature Elimination, Boruta, PCA.
- Hyperparameter tuning UI, learning/validation curves.
- PDF/PNG/`.pkl` export.
- Ticker autocomplete backed by a real symbol search API (currently a
  static shortlist + free-text ticker input).

**Deployment note:** LSTM/GRU training can take longer than the default
30-second web request timeout on some hosts. The included `Procfile`
already sets `gunicorn --timeout 120` to accommodate this — increase
further if you raise `epochs` or `window` significantly.

---

## Notes on data & disclaimers

Market data is sourced from Yahoo Finance via `yfinance` and cached to disk
for up to 1 hour (configurable via `CACHE_TTL_SECONDS`) to reduce repeated
network calls. Predictions are for research/educational purposes only and
are not investment advice.
