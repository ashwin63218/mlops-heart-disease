# Heart Disease Classification — MLOps Pipeline


An end-to-end, production-grade MLOps pipeline for binary heart disease classification using the UCI Heart Disease dataset. Covers data ingestion, preprocessing, model training with experiment tracking, FastAPI serving, CI/CD, Docker, Kubernetes, and Prometheus/Grafana monitoring.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Prerequisites](#2-prerequisites)
3. [Quick Start](#3-quick-start)
4. [Step-by-Step Setup](#4-step-by-step-setup)
5. [Running the Full Pipeline](#5-running-the-full-pipeline)
6. [API Usage](#6-api-usage)
7. [Running Tests](#7-running-tests)
8. [Docker — Containerised API](#8-docker--containerised-api)
9. [Kubernetes Deployment](#9-kubernetes-deployment)
10. [Monitoring Stack](#10-monitoring-stack)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Environment Variables](#12-environment-variables)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Structure

```
mlops-heart-disease/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions — lint → test → train
├── data/
│   ├── raw/                        # download_data.py output
│   └── processed/                  # preprocess.py output
├── k8s/
│   ├── deployment.yaml             # Kubernetes Deployment + Secret template
│   └── service.yaml                # ClusterIP Service + HPA
├── models/                         # Trained model artifacts
│   ├── best_model.joblib           # sklearn Pipeline (preprocessor + classifier)
│   ├── best_model.onnx             # ONNX export
│   ├── best_model_summary.json     # Metrics and MLflow run ID
│   └── roc_comparison.png          # ROC curves for all three models
├── monitoring/
│   ├── prometheus.yml              # Prometheus scrape config
│   └── grafana/
│       └── provisioning/           # Auto-provisioned datasource + dashboard
├── notebooks/
│   └── 01_EDA.ipynb                # Exploratory Data Analysis
├── src/
│   ├── api/
│   │   └── app.py                  # FastAPI inference service
│   ├── data/
│   │   ├── download_data.py        # Dataset acquisition (3-fallback)
│   │   └── preprocess.py           # Feature engineering + train/test split
│   └── models/
│       ├── train.py                # Model training + MLflow tracking
│       └── predict.py              # Inference class + ONNX export
├── tests/
│   ├── conftest.py                 # Shared pytest fixtures
│   ├── test_data.py                # Preprocessing tests (22 tests)
│   └── test_model.py               # Model inference tests (30 tests)
├── .dockerignore
├── .env.example                    # Environment variable template
├── docker-compose.yml              # FastAPI + Prometheus + Grafana stack
├── Dockerfile                      # Multi-stage production image
├── requirements.txt                # Pinned Python dependencies
└── README.md
```

---

## 2. Prerequisites

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| Python | 3.10+ | Runtime | [python.org](https://www.python.org/downloads/) |
| pip | 23+ | Package manager | Bundled with Python |
| Git | Any | Version control | [git-scm.com](https://git-scm.com) |
| Docker Desktop | 4.x+ | Containerisation + Kubernetes | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) |
| minikube | 1.32+ | Local Kubernetes | [minikube.sigs.k8s.io](https://minikube.sigs.k8s.io/docs/start/) |
| kubectl | 1.28+ | Kubernetes CLI | Bundled with Docker Desktop |

> **Apple Silicon (M1/M2/M3):** Add `--platform linux/amd64` when building Docker images.

---

## 3. Quick Start


```
# Clone the repository
git clone https://github.com/ashwin63218/mlops-heart-disease.git
cd mlops-heart-disease

# Install dependencies
pip install setuptools
pip install -r requirements.txt

# Run the full pipeline
python -m src.data.download_data
python -m src.data.preprocess
python -m src.models.train
python -m src.models.predict --export

# Start the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Test a prediction (new terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":4,"trestbps":125,"chol":212,
       "fbs":0,"restecg":1,"thalach":168,"exang":0,
       "oldpeak":1.0,"slope":2,"ca":2,"thal":7}'
```

Expected response:
```json
{
  "prediction": 1,
  "confidence": 0.8731,
  "probability": 0.8731,
  "risk_label": "High",
  "latency_ms": 4.2
}
```

---

## 4. Step-by-Step Setup

### 4.1 Clone and Install


```
git clone https://github.com/ashwin63218/mlops-heart-disease.git
cd mlops-heart-disease

# Install setuptools first — required on Python 3.12+ for pkg_resources
pip install setuptools
pip install -r requirements.txt
```

### 4.2 Environment Variables (Optional — for DagsHub tracking)


```
cp .env.example .env
# Open .env and fill in your DagsHub credentials
```

`.env` format:
```
DAGSHUB_USERNAME=your_dagshub_username
DAGSHUB_TOKEN=your_dagshub_access_token
DAGSHUB_REPO=mlops-heart-disease
```

> **Get your token:** DagsHub repo → Remote → Experiments → copy MLFLOW_TRACKING_PASSWORD.
> If `.env` is left empty, MLflow falls back to local `mlflow.db` automatically.

---

## 5. Running the Full Pipeline

Run all commands from the **project root directory**.

### Step 1 — Download Dataset


```
python -m src.data.download_data
```

Downloads via three fallbacks: ucimlrepo → direct URL → Kaggle API.

**Output:**
```
data/raw/heart_disease_raw.csv      ← 303 rows, 14 columns
data/raw/download_report.json       ← method used, shape, missing values
```

---

### Step 2 — Preprocess


```
python -m src.data.preprocess
```

Applies RobustScaler + OneHotEncoder + imputation. Stratified 80/20 split.

**Output:**
```
data/processed/X_train.csv          ← 242 rows × 20 features
data/processed/X_test.csv           ← 61 rows × 20 features
data/processed/y_train.csv
data/processed/y_test.csv
data/processed/preprocessor.joblib
```

---

### Step 3 — Train Models


```
# Full training with GridSearchCV (recommended)
python -m src.models.train

# Fast run — no tuning, 2-fold CV (for quick iteration / CI)
python -m src.models.train --no-tune --cv-folds 2
```

Trains Logistic Regression, Random Forest, and XGBoost. Prints a leaderboard on completion.

**Output:**
```
models/best_model.joblib            ← Pipeline(preprocessor + classifier)
models/best_model_summary.json      ← metrics + MLflow run ID
models/roc_comparison.png           ← ROC curves for all three models
```

---

### Step 4 — Export to ONNX


```
python -m src.models.predict --export
```

**Output:**
```
models/best_model.onnx
models/onnx_validation_report.json  ← sklearn vs ONNX match rate
```

---

### Step 5 — Smoke Test (Optional)


```
python -m src.models.predict --smoke-test
```

Runs sklearn → ONNX export → ONNX inference → batch predict end-to-end.

---

## 6. API Usage

### Start the Server


```
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe — 200 if process is running |
| `GET` | `/ready` | Readiness probe — 200 when model is loaded |
| `GET` | `/info` | Model metadata (name, AUC, pipeline steps) |
| `POST` | `/predict` | Single patient prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 100 patients) |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/docs` | Interactive Swagger UI (browser) |

### Single Prediction


```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "sex": 1, "cp": 4, "trestbps": 125,
    "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 7
  }'
```

### Batch Prediction


```
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {"age":52,"sex":1,"cp":4,"trestbps":125,"chol":212,
       "fbs":0,"restecg":1,"thalach":168,"exang":0,
       "oldpeak":1.0,"slope":2,"ca":2,"thal":7},
      {"age":35,"sex":0,"cp":1,"trestbps":110,"chol":180,
       "fbs":0,"restecg":0,"thalach":185,"exang":0,
       "oldpeak":0.0,"slope":1,"ca":0,"thal":3}
    ]
  }'
```

### Feature Reference

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `age` | float | 1–120 | Age in years |
| `sex` | int | 0–1 | 0 = Female, 1 = Male |
| `cp` | int | 1–4 | Chest pain type |
| `trestbps` | float | 50–300 | Resting blood pressure (mmHg) |
| `chol` | float | 50–700 | Serum cholesterol (mg/dl) |
| `fbs` | int | 0–1 | Fasting blood sugar > 120 |
| `restecg` | int | 0–2 | Resting ECG results |
| `thalach` | float | 50–250 | Max heart rate achieved |
| `exang` | int | 0–1 | Exercise-induced angina |
| `oldpeak` | float | 0–10 | ST depression |
| `slope` | int | 1–3 | Slope of peak exercise ST |
| `ca` | float\|null | 0–3 | Major vessels (nullable) |
| `thal` | float\|null | 3, 6, 7 | Thalassemia type (nullable) |

> **Tip:** Open `http://localhost:8000/docs` in your browser to test all endpoints interactively via the built-in Swagger UI — no curl required.

---

## 7. Running Tests


```
# Full suite (60 tests)
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Data tests only (22 tests — no model required)
pytest tests/test_data.py -v

# Model tests only (30 tests)
pytest tests/test_model.py -v

# Skip ONNX tests
pytest tests/test_model.py -v -k "not onnx"
```

Expected result: `60 passed, 1 skipped in ~2.5s`

---

## 8. Docker — Containerised API

> **Prerequisite:** Docker Desktop must be running.


```
# Build the image
docker build -t heart-disease-api:latest .

# Run the container
docker run -d \
  --name heart-api \
  -p 8000:8000 \
  heart-disease-api:latest

# Verify it is healthy
docker ps
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":4,"trestbps":125,"chol":212,
       "fbs":0,"restecg":1,"thalach":168,"exang":0,
       "oldpeak":1.0,"slope":2,"ca":2,"thal":7}'

# View logs
docker logs heart-api --tail 50

# Stop and clean up
docker stop heart-api && docker rm heart-api
```

---

## 9. Kubernetes Deployment

> **Prerequisites:** Docker Desktop with Kubernetes enabled, or minikube.

### Using minikube


```
# 1. Start minikube
minikube start

# 2. Point Docker CLI at minikube's daemon
eval $(minikube docker-env)

# 3. Build image inside minikube
docker build -t heart-disease-api:latest .

# 4. Apply manifests (Secret + Deployment first, then Service)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 5. Watch pods come up (wait for STATUS=Running, READY=2/2)
kubectl get pods -l app=heart-disease-api --watch

# 6. Get the service URL (open new terminal if step 5 is still watching)
minikube service heart-disease-api --url

# 7. Test
curl <minikube-url>/health
curl -X POST <minikube-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":4,"trestbps":125,"chol":212,
       "fbs":0,"restecg":1,"thalach":168,"exang":0,
       "oldpeak":1.0,"slope":2,"ca":2,"thal":7}'
```

### DagsHub Secret


```
# Create secret directly (credentials never touch a file)
kubectl create secret generic dagshub-credentials \
  --from-literal=username="your_dagshub_username" \
  --from-literal=token="your_dagshub_token"
```

### Useful Commands


```
kubectl get pods -l app=heart-disease-api        # pod status
kubectl logs -l app=heart-disease-api --tail=50  # pod logs
kubectl describe pod <pod-name>                  # debug probe failures
kubectl get hpa heart-disease-api                # autoscaler status
kubectl delete -f k8s/                           # tear everything down
```

---

## 10. Monitoring Stack

> **Prerequisite:** Docker Desktop must be running.


```
# Start API + Prometheus + Grafana
docker-compose up --build

# Start in background
docker-compose up --build -d

# Generate traffic to populate the dashboard
for i in {1..50}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"age":52,"sex":1,"cp":4,"trestbps":125,"chol":212,
         "fbs":0,"restecg":1,"thalach":168,"exang":0,
         "oldpeak":1.0,"slope":2,"ca":2,"thal":7}' > /dev/null
  echo "Request $i"
done

# Stop everything
docker-compose down
```

### Dashboard URLs

| Service | URL | Login |
|---------|-----|-------|
| FastAPI Swagger UI | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

The Grafana dashboard **"Heart Disease API — MLOps Monitor"** loads automatically on first boot with 8 panels covering request count, latency percentiles, error rate, and status code breakdown.

---

## 11. CI/CD Pipeline

The GitHub Actions workflow triggers automatically on push to `main` and on pull requests:

```
lint (flake8) → test (pytest, 60 tests, coverage ≥ 70%) → train (main only, AUC ≥ 0.80)
```

### Setup

1. Push the repository to GitHub
2. Add secrets under **Settings → Secrets and variables → Actions:**

| Secret | Description |
|--------|-------------|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | Your DagsHub access token |
| `DAGSHUB_REPO` | `mlops-heart-disease` |

3. Push any commit to `main` — all three jobs run automatically

### Artifacts

The `train` job uploads the following artifacts (downloadable from the Actions run page):

- `best-model` — joblib, ONNX, summary JSON, ROC chart (30 days)
- `training-metrics` — summary JSON only (90 days)
- `coverage-report` — coverage.xml (14 days)

---

## 12. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DAGSHUB_USERNAME` | No | — | DagsHub username |
| `DAGSHUB_TOKEN` | No | — | DagsHub access token |
| `DAGSHUB_REPO` | No | `mlops-heart-disease` | DagsHub repo name |
| `MODEL_BACKEND` | No | `sklearn` | `sklearn` or `onnx` |
| `PYTHONUNBUFFERED` | No | `1` | Unbuffered stdout (set automatically in Docker) |

---

