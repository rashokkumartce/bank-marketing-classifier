# Bank Campaign Model Lab

BITS Pilani WILP M.Tech (AIML), Machine Learning Assignment 2.
Deadline: **8 November 2026, 23:59 IST**. Total: **15 marks**.

Development and local validation are complete. Live deployment, final BITS lab
execution evidence and the single submission PDF remain to be completed.
Earlier synthetic data and its model results have been replaced. Current saved
artifacts were regenerated locally; their provenance is recorded in
[training_summary.json](training_summary.json). They must not be represented as
the artifacts from the final lab execution.

## a. Problem statement

Predict whether a customer subscribed to a term deposit (`y = yes/no`) from a
Portuguese bank's marketing-campaign records. Compare five classifiers using
the same held-out data and demonstrate their performance in Streamlit.

This is **post-call analysis**: `duration` is only known after a call ends.
Including it is unsuitable for pre-call targeting. These models are coursework
demonstrations, not a validated production marketing system.

## b. Dataset description

- Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- Variant: `bank-additional-full.csv`, from the official downloadable archive.
- Instances: **41,188**; input features: **20**; target: `y`.
- Target distribution: **36,548 no (88.73%)**, **4,640 yes (11.27%)**.
- Ten numeric and ten categorical features.
- No empty CSV values in the downloaded file. Several categories contain
  `unknown`, which is retained as an explicit category, not a missing value.
- `pdays = 999` indicates no previous contact; it is retained as provided.
- Deterministic 80/20 stratified split, seed 42: **32,950 training** and
  **8,238 test** rows (7,310 no; 928 yes).
- Every row of [test_data.csv](test_data.csv) belongs to the held-out split.
  Training data is downloaded locally into ignored `data/`, not uploaded to the app.

| Feature | Type | Meaning |
|---|---|---|
| age | Numeric | Customer age |
| job | Categorical | Job type |
| marital | Categorical | Marital status |
| education | Categorical | Education level |
| default | Categorical | Credit in default |
| housing | Categorical | Housing loan |
| loan | Categorical | Personal loan |
| contact | Categorical | Communication type |
| month | Categorical | Last contact month |
| day_of_week | Categorical | Last contact weekday |
| duration | Numeric | Last call duration in seconds; post-call only |
| campaign | Numeric | Contacts during this campaign |
| pdays | Numeric | Days since previous contact (999: not previously contacted) |
| previous | Numeric | Previous campaign contacts |
| poutcome | Categorical | Previous campaign outcome |
| emp.var.rate | Numeric | Employment variation rate |
| cons.price.idx | Numeric | Consumer price index |
| cons.conf.idx | Numeric | Consumer confidence index |
| euribor3m | Numeric | Euribor three-month rate |
| nr.employed | Numeric | Number of employees indicator |

## c. GitHub repository link

[https://github.com/rashokkumartce/bank-marketing-classifier](https://github.com/rashokkumartce/bank-marketing-classifier)

**Live Streamlit link:** pending deployment. Replace this text with the actual
tested `https://...streamlit.app` URL before submission.

## d. Models used and evaluation

Each model is saved as a complete scikit-learn pipeline. Numeric features use
`StandardScaler`; categorical features use dense one-hot encoding with
`handle_unknown="ignore"`. All preprocessing is fitted **only on training rows**.
Neither the app nor the test evaluation fits encoders or retrains classifiers.

| Model | Fixed configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, balanced class weights, seed 42 |
| Decision Tree | Depth 8, minimum 20 samples per leaf, balanced weights, seed 42 |
| KNN | 15 neighbors, distance-weighted voting |
| Naive Bayes | GaussianNB defaults |
| Support Vector Machine | RBF kernel, balanced weights, `probability=False`, seed 42 |

Hyperparameters were fixed before evaluation; the held-out test data was not
used for tuning. Gaussian NB is used because it accepts the continuous scaled
features. SVM uses decision-function scores for ROC AUC, not probabilities.

### Comparison table

Results on all 8,238 held-out rows; rounded to four decimal places. Full
precision values are in [model_results.csv](model_results.csv).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8651 | 0.9438 | 0.4512 | 0.9116 | 0.6036 | 0.5813 |
| Decision Tree | 0.8562 | 0.9452 | 0.4344 | 0.9170 | 0.5895 | 0.5684 |
| KNN | 0.9101 | 0.9216 | 0.6667 | 0.4030 | 0.5024 | 0.4737 |
| Naive Bayes | 0.8203 | 0.8393 | 0.3495 | 0.6907 | 0.4642 | 0.4009 |
| Support Vector Machine | 0.8507 | 0.9417 | 0.4263 | 0.9418 | 0.5870 | 0.5711 |

Accuracy is the fraction of correct predictions. AUC measures ranking quality.
Precision, recall and F1 use **yes as the positive class**. MCC uses all four
confusion-matrix cells and ranges from -1 to 1.
The always-no baseline has **0.8874 accuracy** but zero positive recall and F1.
This makes accuracy alone a poor criterion for subscription detection.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest F1 (0.6036) and MCC (0.5813), with 91.16% positive recall. Balanced weights improve subscription detection at the cost of false positives; precision is 45.12%. |
| Decision Tree | Highest AUC (0.9452); recall is 91.70%. Its depth/leaf constraints provide a regularized tree, but positive F1 is below Logistic Regression. |
| KNN | Highest accuracy (91.01%) and precision (66.67%), but only 40.30% recall: many subscribers are missed. Scaling is essential for distance-based classification. |
| Naive Bayes | Lowest AUC, F1 and MCC in this comparison. Correlated economic variables and Gaussian/conditional-independence assumptions are plausible limitations, not proven causes. |
| Support Vector Machine | Highest positive recall (94.18%); lower precision (42.63%) and F1 than Logistic Regression. Decision scores allow AUC without probability calibration. |
| Overall winner for this dataset | **Logistic Regression**, using held-out F1 as the declared primary criterion; it also has the highest MCC. KNN would be preferred if positive precision were the primary criterion. |

### Limitations

- Random stratified splitting is a coursework comparison; it does not estimate
  performance on future campaigns. A time-based evaluation would be needed.
- Call duration is unavailable before a call. Remove it and retrain for pre-call use.
- One-hot encoding does not eliminate correlated-feature effects.
- No cross-validation, threshold optimization, or confidence intervals are claimed.
- Platform differences may alter the last decimal places. After final lab
  training, update this table and observations if the rounded values change.

## Repository structure

```text
app.py                       # Streamlit UI; no model fitting
download_data.py             # Official UCI downloader with checks
train_models.py              # Reproducible training and artifact export
evaluation.py                # Shared CSV validation and six metrics
test_project.py              # Regression and Streamlit interaction tests
requirements.txt             # Version-pinned runtime dependencies
README.md
test_data.csv                # Labeled held-out test rows
model_results.csv
training_summary.json        # Provenance and evaluation diagnostics
model/                       # Five compressed pipelines (.joblib)
.streamlit/config.toml
.gitignore
```

## Setup, retraining and running

Use **Python 3.12** for the final lab run and Streamlit deployment. Local
validation also ran on Python 3.9.6 with the same pinned packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

On Windows, activate using `.venv\Scripts\Activate.ps1`.
Do not install into a shared lab environment if activation fails.

The saved artifacts already allow the app to run:

```bash
python -m streamlit run app.py
```

To regenerate the entire experiment:

```bash
python download_data.py
python train_models.py
python -m unittest -v test_project
```

Training replaces the five pipelines, test CSV, metrics CSV and summary JSON.
It prints all six metrics and verifies predictions after serialization.
Failures are explicit; downloading never substitutes synthetic data.
The raw archive's CSV hash is printed by the downloader. The normalized
comma-separated dataset hash and package versions are recorded in the summary.
Only load trusted, repository-generated joblib files; pickle-based model
formats must not be accepted from user uploads.

## Streamlit features and upload contract

1. **CSV upload:** choose "Upload test CSV", or use the included held-out file.
2. **Model dropdown:** select any of the five saved classifiers.
3. **Six metrics:** recalculated on the current test CSV, not copied from the
   training results table.
4. **Confusion matrix and classification report:** actual rows, predicted
   columns; fixed label order no/yes.
5. **All-five comparison:** same current CSV, table and precision/recall/F1 chart.
6. Downloads for the test CSV and current evaluation metrics.

CSV must be UTF-8 (optional BOM), comma-separated, and contain exactly the
20 features listed above plus target `y`. Column order may differ. Missing,
duplicate or extra columns, malformed rows, blank fields, invalid numeric
values and invalid targets are rejected with visible errors.
Targets are normalized to lowercase no/yes. Up to **10 MiB and 10,000 rows**
are supported for this demonstration.

The saved encoder handles previously unseen categories as all-zero vectors;
the app explicitly warns about affected columns. SVM scores are not displayed
as probabilities. AUC is **N/A** for a single-class CSV, with a warning.
Undefined precision/recall/F1 is recorded as zero, with warnings when positive
predictions or positive examples are absent. Preview predictions are limited
to 100 rows; metrics and comparison use every test row.
MCC follows scikit-learn's zero convention when its denominator is zero.

### Development validation

Fourteen regression tests cover all five models, serialized artifact metrics,
train/test separation, unseen categories, single-class AUC, CSV errors,
model switching and actual uploaded-data evaluation. All passed locally with
runtime numerical warnings treated as errors. `pip check` passed.
An actual browser CSV upload and all-five comparison were also exercised.
These are local checks, not a claim that live deployment or final lab evidence
is complete. The corrected NumPy pin is 1.26.4; NumPy 2.0.2 produced numerical
warnings on the local macOS runtime, so artifacts were regenerated with 1.26.4.

## Deployment and submission

Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) and
[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).
Deploy `app.py` from the main branch on Streamlit Community Cloud, selecting
Python 3.12. The cloud app loads artifacts; it does not download training data
or train on startup.

The final assignment must be performed in BITS Virtual Lab. Capture **one
genuine screenshot** of assignment execution. The final submission is **one
PDF**, in this order:

1. Clickable GitHub repository link.
2. Clickable live Streamlit Community Cloud link.
3. BITS Virtual Lab execution screenshot.
4. The required README content itself, including both tables, not just excerpts
   or a README link.

## Academic integrity and references

The dataset is public UCI data; algorithms are standard scikit-learn
implementations. Development used AI assistance. Review and understand the
code, follow course rules about permitted assistance, and disclose assistance
where required. Do not claim unaided authorship or fabricate execution proof.
Commit history and UI/model originality may be checked. No grade is guaranteed.

- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
