"""
Deep learning service: LSTM and GRU models for the market-prediction
pipeline.

These use a fundamentally different interface from the 12 tabular models
in model_service.py — they need a sliding window of *sequential* days as
input (shape: samples x window x features) rather than one row per
prediction, plus feature scaling fit only on the training slice to avoid
lookahead leakage. That's why they live in their own module and are routed
to separately in the API rather than being added to model_service's flat
registry.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

DEEP_MODEL_LABELS = {
    "lstm": "LSTM (Deep Learning)",
    "gru": "GRU (Deep Learning)",
}

DEFAULT_WINDOW = 20


def _build_model(model_key: str, window: int, n_features: int, classification: bool):
    """Build a small 2-layer LSTM/GRU network. Imported lazily so the app
    can start (and the other 12 models can run) even in environments where
    TensorFlow isn't installed."""
    from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    RecurrentLayer = LSTM if model_key == "lstm" else GRU

    model = Sequential(
        [
            Input(shape=(window, n_features)),
            RecurrentLayer(64, return_sequences=True),
            Dropout(0.2),
            RecurrentLayer(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid" if classification else None),
        ]
    )
    if classification:
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.compile(optimizer="adam", loss="mse")
    return model


def _create_sequences(X: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a `window`-day lookback over X to predict the target at the
    row immediately following each window."""
    xs, ys = [], []
    for i in range(window, len(X)):
        xs.append(X[i - window : i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


def train_and_evaluate(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    model_key: str,
    window: int = DEFAULT_WINDOW,
    epochs: int = 20,
    test_size: float = 0.2,
) -> dict:
    """Train an LSTM/GRU on time-ordered sequences and return metrics plus
    the artefacts (model, scaler) needed for `predict_latest`."""
    if model_key not in DEEP_MODEL_LABELS:
        raise ValueError(f"Unknown deep learning model '{model_key}'")

    from tensorflow.keras.callbacks import EarlyStopping

    X_raw = dataset[feature_cols].values
    y_raw = dataset["target"].values
    classification = target_name == "next_day_direction"

    split_idx = int(len(dataset) * (1 - test_size))
    if split_idx <= window + 10:
        raise ValueError(
            f"Not enough rows for a {window}-day lookback window. Use a longer "
            f"date range or a smaller window."
        )

    # Fit the scaler on the training slice only, to avoid leaking test-period
    # statistics into the model.
    scaler = StandardScaler().fit(X_raw[:split_idx])
    X_scaled = scaler.transform(X_raw)

    X_seq, y_seq = _create_sequences(X_scaled, y_raw, window)
    seq_split_idx = split_idx - window  # shift for the rows consumed by windowing

    X_train, X_test = X_seq[:seq_split_idx], X_seq[seq_split_idx:]
    y_train, y_test = y_seq[:seq_split_idx], y_seq[seq_split_idx:]

    model = _build_model(model_key, window, len(feature_cols), classification)

    train_start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=0,
    )
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    raw_test_pred = model.predict(X_test, verbose=0).ravel()
    predict_time = time.perf_counter() - predict_start

    result = {
        "model": model_key,
        "model_label": DEEP_MODEL_LABELS[model_key],
        "target_type": "classification" if classification else "regression",
        "train_time_seconds": round(train_time, 4),
        "prediction_time_seconds": round(predict_time, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "window": window,
    }

    if classification:
        y_pred = (raw_test_pred >= 0.5).astype(int)
        result["metrics"] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, raw_test_pred)), 4) if len(set(y_test)) > 1 else None,
        }
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        if len(set(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, raw_test_pred)
            result["roc_curve"] = {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}
    else:
        y_pred = raw_test_pred
        result["metrics"] = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 6),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 6),
        }

    result["_model_object"] = model
    result["_scaler"] = scaler
    result["_test_dates"] = dataset.index[split_idx:].strftime("%Y-%m-%d").tolist()
    result["_y_test"] = y_test.tolist()
    result["_y_pred"] = np.asarray(y_pred).tolist()

    return result


def predict_latest(model, scaler, dataset: pd.DataFrame, feature_cols: list[str], target_name: str, window: int) -> dict:
    """Predict the next trading day's outcome from the most recent `window`
    days of features."""
    X_scaled = scaler.transform(dataset[feature_cols].values)
    if len(X_scaled) < window:
        raise ValueError(f"Need at least {window} rows of history to predict with this window size.")

    last_window = X_scaled[-window:].reshape(1, window, -1)
    raw_prediction = float(model.predict(last_window, verbose=0)[0][0])

    classification = target_name == "next_day_direction"
    out = {"raw_prediction": raw_prediction}

    if classification:
        prob_up = raw_prediction
        out.update(
            {
                "probability_up": round(prob_up, 4),
                "probability_down": round(1 - prob_up, 4),
                "signal": "BUY" if prob_up >= 0.55 else ("SELL" if prob_up <= 0.45 else "HOLD"),
                "confidence": round(max(prob_up, 1 - prob_up), 4),
            }
        )
    else:
        out.update({"signal": "BUY" if raw_prediction > 0 else "SELL" if raw_prediction < 0 else "HOLD"})

    return out
