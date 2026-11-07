"""Interactive evaluation of saved Bank Marketing classifiers; never trains."""

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st

from download_data import FEATURES
from evaluation import CATEGORICAL_FEATURES, MODEL_FILES, evaluate_model, read_test_csv


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Bank Campaign Model Lab", page_icon="🏦", layout="wide")
st.title("Bank Campaign Model Lab")
st.caption("Compare five classifiers on labeled, held-out customer records.")
st.info(
    "Post-call analysis only: call duration is included and is unavailable before a call. "
    "Uploaded data is evaluated with saved pipelines; no training occurs in this app."
)


@st.cache_resource
def load_model(name: str, fingerprint):
    model = joblib.load(ROOT / "model" / MODEL_FILES[name])
    if list(model.feature_names_in_) != FEATURES or list(model.classes_) != [0, 1]:
        raise ValueError(f"{name} has an incompatible schema. Retrain with train_models.py.")
    return model


@st.cache_data
def evaluate_saved_model(name: str, contents: bytes, fingerprint):
    data = read_test_csv(contents)
    model = load_model(name, fingerprint)
    metrics, diagnostics = evaluate_model(
        model, data[FEATURES], data["y"].map({"no": 0, "yes": 1}),
    )
    return metrics, diagnostics


try:
    summary = json.loads((ROOT / "training_summary.json").read_text(encoding="utf-8"))
    if summary["package_versions"]["scikit-learn"] != sklearn.__version__:
        raise ValueError("scikit-learn differs from the training environment. Install requirements.txt.")
    fingerprint = tuple(
        ((ROOT / "model" / filename).stat().st_mtime_ns,
         (ROOT / "model" / filename).stat().st_size)
        for filename in MODEL_FILES.values()
    )
    sample_contents = (ROOT / "test_data.csv").read_bytes()
except (OSError, ValueError, KeyError) as error:
    st.error(f"Required model artifacts are missing or incompatible: {error}")
    st.stop()

selected_model = st.sidebar.selectbox("Classification model", list(MODEL_FILES))
st.sidebar.markdown("### Evaluation settings")
st.sidebar.caption("Positive class: yes (subscribed). Decision thresholds are fixed by each model.")
source = st.radio("Test dataset", ["Included held-out test CSV", "Upload test CSV"])
st.download_button(
    "Download held-out test CSV", sample_contents, "test_data.csv", mime="text/csv",
)
if source == "Upload test CSV":
    upload = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
    if upload is None:
        st.info("Upload a CSV containing all 20 input features and target 'y' (no/yes).")
        st.stop()
    contents = upload.getvalue()
else:
    contents = sample_contents

try:
    data = read_test_csv(contents)
    metrics, diagnostics = evaluate_saved_model(selected_model, contents, fingerprint)
except (ValueError, UnicodeError, csv.Error, pd.errors.ParserError, OSError, EOFError) as error:
    st.error(f"Cannot evaluate this test CSV: {error}")
    st.stop()

if data["y"].nunique() != 2:
    st.warning("AUC is undefined because this CSV has only one target class; it is shown as N/A.")
if diagnostics["predicted_yes"] == 0:
    st.warning("This model predicted no positive cases. Positive-class precision is recorded as 0.")
if not data["y"].eq("yes").any():
    st.warning("There are no actual positive cases. Positive-class recall and F1 are recorded as 0.")
encoder = load_model(selected_model, fingerprint).named_steps["preprocessing"].named_transformers_[
    "categorical"
]
unseen = [
    column for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_)
    if not data[column].isin(categories).all()
]
if unseen:
    st.warning(
        f"Unseen categories in {', '.join(unseen)} use the saved encoder's all-zero representation."
    )

overview = st.columns(3)
overview[0].metric("Test records", f"{len(data):,}")
overview[1].metric("Actual subscriptions", int(data["y"].eq("yes").sum()))
overview[2].metric("Always-no baseline accuracy", f"{data['y'].eq('no').mean():.4f}")
evaluation_tab, comparison_tab, details_tab = st.tabs(
    ["Selected model", "Compare all five", "Dataset and method"]
)

with evaluation_tab:
    st.subheader(f"{selected_model}: results on current test CSV")
    cards = st.columns(3)
    for index, (label, value) in enumerate(metrics.items()):
        cards[index % 3].metric(label, "N/A" if value is None else f"{value:.4f}")
    st.caption("Precision, recall and F1 refer to subscriptions (yes). MCC ranges from -1 to 1.")
    matrix = diagnostics["confusion_matrix"]
    st.subheader("Confusion matrix")
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center",
                      color="black", bbox={"facecolor": "white", "alpha": 0.8})
    axis.set_xticks([0, 1], labels=["no", "yes"])
    axis.set_yticks([0, 1], labels=["no", "yes"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.subheader("Classification report")
    report = diagnostics["classification_report"]
    st.dataframe(
        pd.DataFrame({key: report[key] for key in ["no", "yes", "macro avg", "weighted avg"]}).T,
        width="stretch",
    )
    with st.expander("Test records and predictions"):
        preview = data.head(100).copy()
        model = load_model(selected_model, fingerprint)
        preview["prediction"] = pd.Series(
            model.predict(preview[FEATURES]), index=preview.index,
        ).map({0: "no", 1: "yes"})
        st.dataframe(preview, width="stretch", hide_index=True)
        st.caption("Preview is limited to 100 rows; all metrics above use the entire test CSV.")

with comparison_tab:
    st.subheader("Same test data, five saved classifiers")
    rows = []
    try:
        for name in MODEL_FILES:
            model_metrics, _ = evaluate_saved_model(name, contents, fingerprint)
            rows.append({"Model": name, **model_metrics})
    except (ValueError, OSError, EOFError) as error:
        st.error(f"Cannot compare saved models: {error}")
        st.stop()
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison, width="stretch", hide_index=True)
    chart, chart_axis = plt.subplots(figsize=(8, 4))
    comparison.set_index("Model")[["F1 Score", "Recall", "Precision"]].plot.barh(
        ax=chart_axis,
    )
    chart_axis.set_xlim(0, 1)
    chart_axis.set_xlabel("Score")
    chart.tight_layout()
    st.pyplot(chart)
    plt.close(chart)
    winner = comparison.loc[comparison["F1 Score"].idxmax(), "Model"]
    st.write(f"Highest F1 on this CSV: **{winner}** (first listed model breaks ties).")
    st.caption("This is a descriptive comparison, not hyperparameter tuning on test data.")
    st.download_button(
        "Download current evaluation metrics", comparison.to_csv(index=False),
        "evaluation_metrics.csv", mime="text/csv",
    )

with details_tab:
    st.markdown(
        "Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), "
        "additional-full variant: 41,188 rows, 20 features. The included test CSV has "
        "8,238 rows excluded from model fitting. Numeric scaling and one-hot encoding "
        "are fitted on training data and saved inside each pipeline."
    )
    st.write("Required CSV columns:", FEATURES + ["y"])
    st.caption(
        "UTF-8, comma-separated CSV; no missing values, duplicate/extra columns or non-finite "
        "numeric values. Targets are no/yes. Up to 10 MiB and 10,000 test rows. "
        "Unknown categories are accepted with an explicit warning."
    )
    st.write(
        "AUC uses P(yes) for four models and uncalibrated decision scores for SVM. "
        "Single-class AUC is N/A. Undefined precision/recall/F1 is 0. "
        "The random stratified split is a coursework comparison, not a future-time validation."
    )
    st.caption(f"Training Python version: {summary['python_version']}; random seed: 42.")

st.divider()
st.markdown("[GitHub repository](https://github.com/rashokkumartce/bank-marketing-classifier)")
