"""Train five complete pipelines on the genuine UCI Bank Marketing dataset."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from download_data import FEATURES, SOURCE_URL


SEED = 42
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


def evaluate_model(model, features, target):
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        positive_index = list(model.classes_).index(1)
        scores = model.predict_proba(features)[:, positive_index]
    else:
        scores = model.decision_function(features)
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC Score": roc_auc_score(target, scores),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1 Score": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
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


def train_models(data_path, output_dir):
    dataset = pd.read_csv(data_path)
    if list(dataset.columns) != FEATURES + ["y"]:
        raise ValueError("Expected the genuine additional dataset; run download_data.py first")
    if len(dataset) != 41188 or dataset["y"].value_counts().to_dict() != {
        "no": 36548, "yes": 4640,
    }:
        raise ValueError("Dataset row count or target distribution differs from verified UCI data")
    if dataset.isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not np.isfinite(dataset[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must contain finite numbers")

    target = dataset["y"].map({"no": 0, "yes": 1})
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=SEED, stratify=target,
    )
    train_features = dataset.iloc[train_indices][FEATURES]
    test_features = dataset.iloc[test_indices][FEATURES]
    train_target = target.iloc[train_indices]
    test_target = target.iloc[test_indices]
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=SEED,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced", probability=False,
            cache_size=128, random_state=SEED,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    results = []
    diagnostics = {}
    print(f"Train: {len(train_indices):,}; held-out test: {len(test_indices):,}", flush=True)
    baseline_accuracy = float((test_target == 0).mean())
    print(f"Always-no baseline accuracy: {baseline_accuracy:.4f}", flush=True)
    for name, classifier in classifiers.items():
        print(f"Training {name}...", flush=True)
        started = time.monotonic()
        model = Pipeline([
            ("preprocessing", clone(preprocessing)), ("classifier", classifier),
        ])
        model.fit(train_features, train_target)
        metrics, report = evaluate_model(model, test_features, test_target)
        model_path = model_dir / MODEL_FILES[name]
        joblib.dump(model, model_path, compress=3)
        restored = joblib.load(model_path)
        np.testing.assert_array_equal(
            model.predict(test_features.iloc[:20]),
            restored.predict(test_features.iloc[:20]),
        )
        results.append({"Model": name, **metrics})
        diagnostics[name] = {**report, "elapsed_seconds": time.monotonic() - started}
        print(
            f"  Accuracy={metrics['Accuracy']:.4f}, AUC={metrics['AUC Score']:.4f}, "
            f"F1={metrics['F1 Score']:.4f}, MCC={metrics['MCC']:.4f}", flush=True,
        )
    results_table = pd.DataFrame(results)
    results_table.to_csv(output_dir / "model_results.csv", index=False)
    dataset.iloc[test_indices].to_csv(output_dir / "test_data.csv", index=False)
    metadata = {
        "source_url": SOURCE_URL,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_rows": len(dataset),
        "features": FEATURES,
        "target_mapping": {"no": 0, "yes": 1},
        "random_seed": SEED,
        "split": "80/20 stratified random split; preprocessing fitted only on training rows",
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "test_class_counts": dataset.iloc[test_indices]["y"].value_counts().to_dict(),
        "always_no_baseline_accuracy": baseline_accuracy,
        "winner_by_test_f1": results_table.loc[results_table["F1 Score"].idxmax(), "Model"],
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]
        },
        "limitations": [
            "Duration is only known after the call; this is post-call analysis.",
            "Random splitting is for coursework comparison, not a future-time deployment estimate.",
            "Hyperparameters are fixed before evaluation; no held-out test tuning is performed.",
            "SVM AUC uses decision scores; probabilities are not calibrated or required.",
        ],
        "diagnostics": diagnostics,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print("\n" + results_table.to_string(index=False), flush=True)
    print(f"\nSaved five pipelines, held-out CSV and results to {output_dir.resolve()}", flush=True)
    print("The old Streamlit app must be corrected before using these new pipelines.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=root / "data" / "bank_additional_full.csv")
    parser.add_argument("--output-dir", type=Path, default=root)
    arguments = parser.parse_args()
    train_models(arguments.data, arguments.output_dir)
