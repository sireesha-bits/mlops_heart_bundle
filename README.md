
# Heart Disease Risk Prediction — MLOps End-to-End (UCI)

Production-ready machine learning pipeline to predict heart disease risk using the UCI Cleveland dataset. Includes:

- Reproducible **preprocessing pipeline** (scikit-learn `Pipeline` + `ColumnTransformer`)
- **Model training** for Logistic Regression & Random Forest with cross-validation
- **Experiment tracking** (optional) with MLflow
- **FastAPI** model serving (`/predict`, `/health`, `/metrics`)
- **Docker** containerization
- **Kubernetes** deployment manifests (Minikube-friendly)
- **CI/CD** with GitHub Actions (lint, tests, train, upload artifacts)
- Basic **monitoring** via Prometheus metrics

> Author: **Sireesha Yalla**  
> Course: **MLOps (S1-25_AIMLCZG523) – Assignment I**

---

## 1. Dataset
- Source: UCI Heart Disease — Cleveland subset. Download script in `src/data.py`.
- Target: Binary label (0 = no disease, 1 = disease). We map original values (0–4) to binary.

## 2. Quick Start
```bash
# Create & activate virtual env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train model (RandomForest by default)
python src/train.py --model rf --mlflow False

# Run API
uvicorn app:app --host 0.0.0.0 --port 8000

# Test prediction
curl -X POST http://localhost:8000/predict   -H 'Content-Type: application/json'   -d @sample_request.json
```

## 3. Docker
```bash
# Build
docker build -t heart-api:latest .
# Run
docker run --rm -p 8000:8000 heart-api:latest
```

## 4. Kubernetes (Minikube)
```bash
minikube start
kubectl apply -f deployment.yaml
# Access NodePort
minikube service heart-api-svc --url
```

## 5. CI/CD (GitHub Actions)
- Workflow in `.github/workflows/ci.yml` runs on push/PR to `main`:
  - `flake8` lint
  - `pytest` unit tests
  - (optional) training step
  - uploads artifacts: `artifacts/model.joblib`, `artifacts/metrics.json`

## 6. Repository Layout
```
.
├── app.py
├── artifacts/
├── data/
├── deployment.yaml
├── Dockerfile
├── notebooks/
│   └── 01_eda.ipynb
├── requirements.txt
├── sample_request.json
├── screenshots/
├── src/
│   ├── data.py
│   ├── preprocess.py
│   ├── train.py
│   └── infer.py
├── tests/
│   ├── test_preprocess.py
│   └── test_training.py
└── .github/workflows/ci.yml
```

## 7. API Contract
- `POST /predict` — JSON body:
```json
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}
```
- Response:
```json
{ "prediction": 1, "confidence": 0.87 }
```

## 8. Notes
- MLflow logging is optional (enable with `--mlflow True`). If MLflow is not installed or running, the script will skip gracefully.
- For cloud registries, replace the image in `deployment.yaml` with `ghcr.io/<org>/<repo>:<tag>` and set `imagePullPolicy: Always`.

## 9. License
Educational use for assignment submission.
