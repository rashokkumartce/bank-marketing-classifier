"""Shared input validation and binary classification evaluation."""

import csv
import io

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

from download_data import FEATURES


CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "poutcome",
]
NUMERIC_FEATURES = [name for name in FEATURES if name not in CATEGORICAL_FEATURES]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TEST_ROWS = 10000


def validate_test_data(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The test CSV contains no data rows.")
    if len(frame) > MAX_TEST_ROWS:
        raise ValueError(f"Upload at most {MAX_TEST_ROWS:,} test rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    required = FEATURES + ["y"]
    missing = sorted(set(required) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(required))
    if missing or extra:
        raise ValueError(f"CSV columns do not match. Missing: {missing}; unexpected: {extra}.")
    frame = frame.loc[:, required].copy()
    if frame.isna().any().any():
        raise ValueError("The test CSV contains missing values; every feature and target is required.")
    for name in NUMERIC_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must be finite; NaN and infinity are not supported.")
    for name in CATEGORICAL_FEATURES + ["y"]:
        frame[name] = frame[name].astype(str).str.strip()
        if frame[name].eq("").any():
            raise ValueError(f"Column '{name}' contains empty values.")
    frame["y"] = frame["y"].str.lower()
    if not frame["y"].isin(["no", "yes"]).all():
        raise ValueError("Target column 'y' must contain only 'no' and 'yes'.")
    return frame


def read_test_csv(contents: bytes) -> pd.DataFrame:
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload a test CSV of at most 10 MiB.")
    text = contents.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text), strict=True)
    header = next(reader, [])
    if len(header) != len(set(header)):
        raise ValueError("CSV header contains duplicate column names.")
    for line, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise ValueError(f"CSV row {line} has {len(row)} fields; expected {len(header)}.")
    frame = pd.read_csv(io.StringIO(text), on_bad_lines="error")
    return validate_test_data(frame)


def evaluate_model(model, features: pd.DataFrame, target: pd.Series):
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        positive_index = list(model.classes_).index(1)
        scores = model.predict_proba(features)[:, positive_index]
    else:
        scores = model.decision_function(features)
    if not np.isfinite(scores).all():
        raise ValueError("The saved model produced non-finite prediction scores.")
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC Score": roc_auc_score(target, scores) if target.nunique() == 2 else None,
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1 Score": f1_score(target, predictions, zero_division=0),
        "MCC": (
            0.0 if len(np.unique(np.concatenate([target.to_numpy(), predictions]))) == 1
            else matthews_corrcoef(target, predictions)
        ),
    }
    diagnostics = {
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            target, predictions, labels=[0, 1], target_names=["no", "yes"],
            output_dict=True, zero_division=0,
        ),
        "predicted_yes": int(np.sum(predictions == 1)),
    }
    return metrics, diagnostics
