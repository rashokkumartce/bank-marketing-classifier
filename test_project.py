"""Regression checks using unittest and Streamlit's bundled AppTest."""

import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from download_data import FEATURES
from evaluation import (
    MAX_TEST_ROWS, MAX_UPLOAD_BYTES, MODEL_FILES, evaluate_model,
    read_test_csv, validate_test_data,
)


ROOT = Path(__file__).resolve().parent


class InputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = pd.read_csv(ROOT / "test_data.csv").head(30)

    def test_valid_reordered_bom_csv(self):
        reordered = self.sample.iloc[:, ::-1].copy()
        reordered["y"] = reordered["y"].str.upper()
        result = read_test_csv(b"\xef\xbb\xbf" + reordered.to_csv(index=False).encode())
        self.assertEqual(list(result.columns), FEATURES + ["y"])
        pd.testing.assert_frame_equal(result, self.sample, check_dtype=False)

    def test_reject_invalid_data(self):
        cases = {
            "missing target": self.sample.drop(columns="y"),
            "extra column": self.sample.assign(extra=1),
            "empty": self.sample.iloc[:0],
            "null": self.sample.assign(age=np.nan),
            "infinity": self.sample.assign(age=np.inf),
            "invalid numeric": self.sample.assign(age="not-a-number"),
            "invalid target": self.sample.assign(y="maybe"),
            "blank category": self.sample.assign(job=" "),
            "too many rows": pd.concat([self.sample.iloc[:1]] * (MAX_TEST_ROWS + 1)),
        }
        for label, frame in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_test_data(frame)

    def test_reject_duplicate_header(self):
        duplicate = self.sample.copy()
        duplicate.columns = ["age"] + list(duplicate.columns[1:-1]) + ["age"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_test_csv(duplicate.to_csv(index=False).encode())

    def test_reject_ragged_csv(self):
        text = self.sample.head(1).to_csv(index=False)
        with self.assertRaisesRegex(ValueError, "fields"):
            read_test_csv((text.rstrip("\n") + ",extra\n").encode())

    def test_reject_invalid_encoding_and_oversize(self):
        with self.assertRaises(UnicodeError):
            read_test_csv(b"\xff\xff")
        with self.assertRaisesRegex(ValueError, "10 MiB"):
            read_test_csv(b"x" * (MAX_UPLOAD_BYTES + 1))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = read_test_csv((ROOT / "test_data.csv").read_bytes())
        cls.models = {
            name: joblib.load(ROOT / "model" / filename) for name, filename in MODEL_FILES.items()
        }
        cls.results = pd.read_csv(ROOT / "model_results.csv").set_index("Model")
        cls.summary = json.loads((ROOT / "training_summary.json").read_text())

    def test_artifact_shapes_and_versions(self):
        self.assertEqual(self.data.shape, (8238, 21))
        self.assertEqual(self.results.shape, (5, 6))
        self.assertEqual(self.summary["train_rows"], 32950)
        self.assertEqual(self.summary["test_class_counts"], {"no": 7310, "yes": 928})
        from importlib.metadata import version
        for name, expected in self.summary["package_versions"].items():
            self.assertEqual(version(name), expected)

    def test_all_models_reproduce_saved_metrics_without_fit(self):
        for name, model in self.models.items():
            with self.subTest(model=name), patch.object(
                model, "fit", side_effect=AssertionError("Evaluation must not fit")
            ):
                metrics, report = evaluate_model(
                    model, self.data[FEATURES], self.data.y.map({"no": 0, "yes": 1})
                )
                np.testing.assert_allclose(
                    [metrics[key] for key in self.results.columns],
                    self.results.loc[name].to_numpy(dtype=float), atol=1e-12,
                )
                self.assertEqual(sum(map(sum, report["confusion_matrix"])), 8238)
                self.assertEqual(report, {
                    key: self.summary["diagnostics"][name][key] for key in report
                })

    def test_one_row_and_unseen_categories(self):
        sample = self.data[FEATURES].iloc[:1].copy()
        sample["job"] = "new-category"
        for name, model in self.models.items():
            with self.subTest(model=name):
                self.assertEqual(model.predict(sample).shape, (1,))

    def test_single_class_auc_is_unavailable(self):
        sample = self.data[self.data.y == "no"].head(10)
        for name, model in self.models.items():
            with self.subTest(model=name):
                metrics, report = evaluate_model(model, sample[FEATURES], pd.Series([0] * 10))
                self.assertIsNone(metrics["AUC Score"])
                self.assertEqual(metrics["Recall"], 0)
                self.assertEqual(sum(map(sum, report["confusion_matrix"])), 10)

    def test_test_rows_excluded_from_preprocessing_fit(self):
        path = ROOT / "data" / "bank_additional_full.csv"
        if not path.exists():
            self.skipTest("Raw dataset not downloaded; run download_data.py for split verification")
        dataset = pd.read_csv(path)
        train_indices, test_indices = train_test_split(
            np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.y,
        )
        self.assertFalse(set(train_indices) & set(test_indices))
        pd.testing.assert_frame_equal(
            self.data, dataset.iloc[test_indices].reset_index(drop=True), check_dtype=False
        )
        from evaluation import NUMERIC_FEATURES
        for model in self.models.values():
            scaler = model.named_steps["preprocessing"].named_transformers_["numeric"]
            np.testing.assert_allclose(
                scaler.mean_, dataset.iloc[train_indices][NUMERIC_FEATURES].mean().to_numpy()
            )


class StreamlitTests(unittest.TestCase):
    def new_app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(list(app.error), [])
        return app

    def test_sample_and_model_switching(self):
        app = self.new_app()
        self.assertEqual(app.metric[0].value, "8,238")
        comparison = next(item.value for item in app.dataframe if "Model" in item.value.columns)
        self.assertEqual(len(comparison), 5)
        results = pd.read_csv(ROOT / "model_results.csv").set_index("Model")
        for name in MODEL_FILES:
            app.sidebar.selectbox[0].select(name).run()
            self.assertEqual(list(app.exception), [])
            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(metrics["F1 Score"], f"{results.loc[name, 'F1 Score']:.4f}")
            self.assertTrue(any(item.value == "Confusion matrix" for item in app.subheader))
            self.assertTrue(any(item.value == "Classification report" for item in app.subheader))
            self.assertEqual(len(app.dataframe[0].value), 4)

    def test_uploaded_metrics_and_single_class_warning(self):
        sample = pd.read_csv(ROOT / "test_data.csv")
        sample = sample[sample.y == "no"].head(10)
        app = self.new_app()
        with patch("streamlit.file_uploader", return_value=io.BytesIO(
            sample.to_csv(index=False).encode()
        )):
            app.radio[0].set_value("Upload test CSV").run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(list(app.error), [])
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Test records"], "10")
        self.assertEqual(metrics["AUC Score"], "N/A")
        self.assertTrue(any("AUC is undefined" in item.value for item in app.warning))
        comparison = next(item.value for item in app.dataframe if "Model" in item.value.columns)
        self.assertTrue(comparison["AUC Score"].isna().all())

    def test_invalid_upload_is_visible_error(self):
        sample = pd.read_csv(ROOT / "test_data.csv").head(10).drop(columns="y")
        app = self.new_app()
        with patch("streamlit.file_uploader", return_value=io.BytesIO(
            sample.to_csv(index=False).encode()
        )):
            app.radio[0].set_value("Upload test CSV").run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("Missing: ['y']" in item.value for item in app.error))

    def test_upload_without_file_waits(self):
        app = self.new_app()
        app.radio[0].set_value("Upload test CSV").run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(any("Upload a CSV" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
