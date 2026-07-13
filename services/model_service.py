"""
Model service: trains, evaluates, and predicts across the full model
registry (12 models). LSTM/GRU/Transformer are deferred — they need a
sequence-windowed training loop rather than this tabular fit/predict
interface — see README "Roadmap".
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def _scaled(estimator_factory, step_name: str):
    """Wrap a scale-sensitive estimator in a StandardScaler pipeline."""
    return lambda: Pipeline([("scaler", StandardScaler()), (step_name, estimator_factory())])


# Tree-based / boosting models are scale-invariant and used as-is.
# Linear, distance-based, and gradient-based-on-weights models (Logistic/
# Linear Regression, SVM, KNN, Naive Bayes, MLP) are scale-sensitive and
# wrapped in a StandardScaler pipeline via `_scaled`.
CLASSIFIERS = {
    "logistic_regression": _scaled(lambda: LogisticRegression(max_iter=2000), "clf"),
    "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    "adaboost": lambda: AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),
    "xgboost": lambda: XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42
    ),
    "lightgbm": lambda: LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1),
    "svm": _scaled(lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=42), "clf"),
    "knn": _scaled(lambda: KNeighborsClassifier(n_neighbors=15), "clf"),
    "naive_bayes": _scaled(lambda: GaussianNB(), "clf"),
    "mlp": _scaled(
        lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
        "clf",
    ),
}

REGRESSORS = {
    # Named "logistic_regression" for UI/key parity; behaves as (scaled)
    # linear regression when the target is a regression target.
    "logistic_regression": _scaled(lambda: LinearRegression(), "reg"),
    "decision_tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    "adaboost": lambda: AdaBoostRegressor(n_estimators=200, learning_rate=0.5, random_state=42),
    "xgboost": lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
    "lightgbm": lambda: LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1),
    "svm": _scaled(lambda: SVR(kernel="rbf", C=1.0), "reg"),
    "knn": _scaled(lambda: KNeighborsRegressor(n_neighbors=15), "reg"),
    # GaussianNB has no regression equivalent; Bayesian Ridge is the closest
    # "Bayesian" linear analogue and is used here so every model key works
    # for every target type.
    "naive_bayes": _scaled(lambda: BayesianRidge(), "reg"),
    "mlp": _scaled(
        lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
        "reg",
    ),
}

MODEL_LABELS = {
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
}


def _is_classification(target_name: str) -> bool:
    return target_name == "next_day_direction"


def train_and_evaluate(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    model_key: str,
    test_size: float = 0.2,
) -> dict:
    """Train one model on a time-ordered train/test split and return metrics
    plus artefacts needed by the UI (confusion matrix, ROC curve, importances).
    """
    if model_key not in MODEL_LABELS:
        raise ValueError(f"Unknown model '{model_key}'")

    X = dataset[feature_cols]
    y = dataset["target"]
    classification = _is_classification(target_name)

    # Time-ordered split (no shuffling) since this is sequential market data.
    split_idx = int(len(dataset) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = (CLASSIFIERS if classification else REGRESSORS)[model_key]()

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - predict_start

    result = {
        "model": model_key,
        "model_label": MODEL_LABELS[model_key],
        "target_type": "classification" if classification else "regression",
        "train_time_seconds": round(train_time, 4),
        "prediction_time_seconds": round(predict_time, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    if classification:
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        result["metrics"] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4) if y_proba is not None else None,
        }
        cm = confusion_matrix(y_test, y_pred).tolist()
        result["confusion_matrix"] = cm
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            result["roc_curve"] = {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}
    else:
        result["metrics"] = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 6),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 6),
            "r2": round(float(model.score(X_test, y_test)), 4) if hasattr(model, "score") else None,
        }

    # Pipelines (used for the scaled linear/logistic models) expose the
    # underlying estimator via named_steps; tree models expose attributes
    # directly on the model itself.
    importance_source = model
    if isinstance(model, Pipeline):
        importance_source = model.named_steps.get("clf") or model.named_steps.get("reg")

    if hasattr(importance_source, "feature_importances_"):
        result["feature_importance"] = dict(
            sorted(zip(feature_cols, importance_source.feature_importances_.tolist()), key=lambda kv: -kv[1])
        )
    elif hasattr(importance_source, "coef_"):
        coefs = np.ravel(importance_source.coef_)
        result["feature_importance"] = dict(
            sorted(zip(feature_cols, np.abs(coefs).tolist()), key=lambda kv: -kv[1])
        )

    result["_model_object"] = model
    result["_split_idx"] = split_idx
    result["_test_dates"] = dataset.index[split_idx:].strftime("%Y-%m-%d").tolist()
    result["_y_test"] = y_test.tolist()
    result["_y_pred"] = np.asarray(y_pred).tolist()

    return result


def predict_latest(model, dataset: pd.DataFrame, feature_cols: list[str], target_name: str) -> dict:
    """Predict the next trading day's outcome using the most recent row of
    features (the row whose target is still unknown)."""
    latest_features = dataset[feature_cols].iloc[[-1]]
    classification = _is_classification(target_name)

    prediction = model.predict(latest_features)[0]
    out = {"raw_prediction": float(prediction)}

    if classification:
        proba = model.predict_proba(latest_features)[0]
        prob_up = float(proba[1])
        out.update(
            {
                "probability_up": round(prob_up, 4),
                "probability_down": round(1 - prob_up, 4),
                "signal": "BUY" if prob_up >= 0.55 else ("SELL" if prob_up <= 0.45 else "HOLD"),
                "confidence": round(max(prob_up, 1 - prob_up), 4),
            }
        )
    else:
        out.update({"signal": "BUY" if prediction > 0 else "SELL" if prediction < 0 else "HOLD"})

    return out


def compare_models(
    dataset: pd.DataFrame, feature_cols: list[str], target_name: str, model_keys: list[str]
) -> list[dict]:
    """Run several models and return a comparison-table-ready list, sorted
    with the best model first."""
    classification = _is_classification(target_name)
    rows = []
    for key in model_keys:
        res = train_and_evaluate(dataset, feature_cols, target_name, key)
        row = {
            "model": res["model_label"],
            "train_time_seconds": res["train_time_seconds"],
            "prediction_time_seconds": res["prediction_time_seconds"],
            **res["metrics"],
        }
        rows.append(row)

    sort_key = "f1" if classification else "rmse"
    reverse = classification  # higher f1 is better; lower rmse is better
    rows.sort(key=lambda r: (r.get(sort_key) is None, r.get(sort_key, 0)), reverse=reverse)
    return rows
