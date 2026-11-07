# Deployment guide

## 1. Verify and push the developed project

From the cloned repository, use its isolated Python environment:

```bash
source .venv/bin/activate
python -m pip check
python -m unittest -v test_project
git status --short
```

Review changed files before committing. Do not include credentials, raw data
or virtual environments. Commit the current source and generated model
artifacts together; the application requires `training_summary.json`.
Push to the existing main branch without force-pushing.
The repository already exists; do not run `git init` again.

## 2. Final BITS lab execution

Pull only when lab changes are safely preserved:

```bash
git status --short
git pull --ff-only
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python download_data.py
python train_models.py
python -m unittest -v test_project
python -m streamlit run app.py
```

Run commands one at a time; stop on errors. Do not install in shared Anaconda
if virtual-environment activation fails. Lab port/browser access depends on
the provided environment; use its authorized preview or port forwarding.
Do not expose the service publicly or disable security settings as a shortcut.

Capture one genuine screenshot showing the lab environment and successful
assignment execution. Preserve the five models, test CSV, results CSV and
summary JSON before the lab restarts. Update the README table if rounded
results differ, and publish those lab artifacts together with their metadata.
For unresolved lab issues use the support contacts in the assignment PDF.

## 3. Streamlit Community Cloud

1. Sign in at https://streamlit.io/cloud with your personal GitHub account.
2. Create a new app from `rashokkumartce/bank-marketing-classifier`.
3. Choose branch `main`, entrypoint `app.py`, and Python **3.12** in settings.
4. Deploy and inspect build/runtime logs for errors.
5. Copy the actual assigned `https://...streamlit.app` URL.

Use the same pinned packages as training. No secrets are required.
Models load from `model/`; only test CSVs are uploaded by users.
The app never retrains on startup or uploaded data.

## 4. Verify live behavior

- Included held-out CSV displays 8,238 records and six metrics.
- Every dropdown option updates the selected model's metrics.
- Upload the downloaded labeled CSV and compare all five models.
- Confusion matrix and classification report are visible.
- Missing target/invalid CSV produces a clear error.
- A single-class CSV gives N/A AUC and a warning.
- Confirm the app link works in a signed-out/private browser session.

Replace the pending app link in README with this tested URL, then follow
[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).
No deployment, screenshot or submission is claimed complete until verified.
