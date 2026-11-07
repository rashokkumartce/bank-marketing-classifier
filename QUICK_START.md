# Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

Saved genuine-data models are included. To retrain:

```bash
python download_data.py
python train_models.py
python -m unittest -v test_project
```

Use Python 3.12 for final lab execution and deployment. See [README.md](README.md)
for upload requirements and results. Deployment and lab evidence are not
claimed complete; see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).
