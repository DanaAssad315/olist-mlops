# Olist MLOps — Late Delivery Prediction

An end-to-end MLOps project for predicting whether an Olist order will be delivered late.

The project covers the complete workflow from data ingestion and ML development to experiment tracking, model versioning, validation, testing, API serving, Docker deployment, CI/CD, and monitoring.

---

## 1. Project Overview

The system predicts whether an Olist order will be delivered late based on information available at prediction time.

### Main components

* PostgreSQL for data storage
* Pandas / Scikit-learn for ML development
* Random Forest for late-delivery prediction
* DVC for dataset and artifact versioning
* Great Expectations for data validation
* MLflow for experiment tracking and model registry
* FastAPI for model serving
* Docker / Docker Compose for deployment
* Pytest for automated testing
* GitHub Actions for CI/CD
* Pre-commit for code quality
* Logging and prediction monitoring

---

## 2. Project Structure

```text
olist-mlops/
│
├── app/
│   └── ...
│
├── artifacts/
│   ├── feature_names.csv
│   ├── final_model.joblib
│   ├── final_results.csv
│   ├── labeled_table.csv
│   ├── ml_table.csv
│   ├── preprocessor.joblib
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── config/
│   └── config.yaml
│
├── database/
│   └── ...
│
├── ingestion/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   ├── 01_read_join.ipynb
│   ├── 02_create_labels.ipynb
│   ├── 03_split_data.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_train_evaluate.ipynb
│
├── src/
│   ├── config.py
│   ├── inference.py
│   ├── mlflow_tracking.py
│   └── ...
│
├── tests/
│   └── ...
│
├── .github/
│   └── workflows/
│
├── .dvc/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 3. Prerequisites

Install the following before starting:

* Git
* Python 3.11+
* Docker Desktop
* Docker Compose
* DVC

Verify:

```powershell
git --version
python --version
docker --version
docker compose version
dvc --version
```

---

# 4. Clone the Repository

```powershell
git clone https://github.com/DanaAssad315/olist-mlops.git
cd olist-mlops
```

---

# 5. Python Environment

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 6. Environment Variables

Create a local `.env` file based on `.env.example`.

```powershell
Copy-Item .env.example .env
```

Fill in the required values according to the local environment.

Do not commit `.env` or any credentials.

---

# 7. Download Versioned Data and Artifacts

The repository uses DVC to version datasets and ML artifacts.

Run:

```powershell
dvc pull
```

The tracked artifacts include:

* processed datasets
* train / validation / test splits
* preprocessing object
* feature names
* trained model
* evaluation results

### DVC credentials

The DVC remote is hosted on DAGsHub.

Credentials are intentionally **not stored in Git**.

If the remote requires authentication, configure DVC credentials locally before running `dvc pull`.

For example, authenticate with DAGsHub using its supported authentication method.

Never commit:

```text
.dvc/config.local
```

or any token/password.

---

# 8. Start the Services

Start PostgreSQL, MLflow, and the API:

```powershell
docker compose up --build -d
```

Check the containers:

```powershell
docker compose ps
```

Expected services include:

```text
olist-postgres
olist-mlflow
olist-api
```

Default ports:

```text
PostgreSQL : 5432
MLflow      : 5000
FastAPI     : 8000
```

---

# 9. Initialize the MLflow Model Registry

A fresh environment may have an empty MLflow Model Registry.

The trained model artifact is versioned with DVC, but the MLflow Registry itself is initialized separately.

Run:

```powershell
python -m src.mlflow_tracking
```

This will:

1. Configure the MLflow tracking server.
2. Create the experiment if it does not exist.
3. Log the final model and evaluation metrics.
4. Register the model.
5. Create/update the `champion` alias.
6. Add model metadata and validation tags.

The registered model name is:

```text
olist-delivery-random-forest
```

The serving alias is:

```text
champion
```

---

# 10. Restart the API After Model Registration

The API loads the serving model when the application starts.

Therefore, after initializing or updating the MLflow Registry, restart the API so it loads the current `champion` model:

```powershell
docker compose restart api
```

Then verify that the API is running:

```powershell
curl.exe http://localhost:8000/health
```

Expected:

```json
{
  "status": "ok",
  "model_status": "available"
}
```

If the API was restarted before the model was registered, it may report:

```json
{
  "status": "degraded",
  "model_status": "unavailable"
}
```

In that case:

1. Run `python -m src.mlflow_tracking`
2. Restart the API again
3. Check `/health`

---

# 11. Check the Loaded Model

Run:

```powershell
curl.exe http://localhost:8000/model-info
```

The response should contain information similar to:

```json
{
  "model_name": "olist-delivery-random-forest",
  "model_version": "1",
  "model_alias": "champion",
  "serving_stage": "production"
}
```

The model version number may be different depending on the MLflow Registry state.

---

# 12. Prediction API

### Endpoint

```text
POST /predict
```

### Required input

The prediction endpoint expects:

```text
order_purchase_timestamp
customer_zip_code_prefix
customer_state
item_count
total_price
total_freight
unique_products
unique_sellers
payment_total
payment_count
```

Example:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "order_purchase_timestamp": "2018-01-01 10:00:00",
    "customer_zip_code_prefix": 14409,
    "customer_state": "SP",
    "item_count": 1,
    "total_price": 100.0,
    "total_freight": 20.0,
    "unique_products": 1,
    "unique_sellers": 1,
    "payment_total": 120.0,
    "payment_count": 1
  }'
```

The response contains the prediction, probability, and model version.

---

# 13. ML Development Workflow

The ML development process is implemented through six notebooks.

### Notebook 1 — Read and Join

Loads the Olist tables and creates the main ML dataset.

### Notebook 2 — Create Labels

Creates the late-delivery target using the actual delivery date and estimated delivery date.

Orders without the required delivery information are excluded.

### Notebook 3 — Train / Validation / Test Split

Uses a time-based split to avoid using future information when training the model.

### Notebook 4 — EDA

Explores:

* target imbalance
* missing values
* categorical variables
* delivery patterns
* temporal patterns
* potential leakage
* feature selection

### Notebook 5 — Feature Engineering

Builds the model features and preprocessing pipeline.

The final processed feature set contains 54 features after removing identifiers and leakage-prone columns.

The fitted preprocessing object is saved as:

```text
artifacts/preprocessor.joblib
```

### Notebook 6 — Train and Evaluate

Trains and evaluates the final model.

The final trained model is saved as:

```text
artifacts/final_model.joblib
```

Evaluation results are saved as:

```text
artifacts/final_results.csv
```

---

# 14. Final Model

The final production model is a Random Forest classifier.

The model uses a probability threshold selected during validation for the late-delivery classification task.

The final model and preprocessing objects are saved and reused during inference.

**The production inference pipeline does not retrain the model.**

It loads:

```text
preprocessor.joblib
```

and the registered MLflow model.

This ensures that training and inference use the same preprocessing workflow.

---

# 15. Data Validation

Great Expectations is used to validate data quality before it is used by the pipeline.

The validation process checks expected properties of the input data and helps detect invalid or unexpected data before model processing.

This provides an additional protection layer between raw data and model inference.

---

# 16. DVC

DVC is used to version large datasets and ML artifacts without storing them directly in Git.

Tracked artifacts include:

```text
artifacts/feature_names.csv
artifacts/final_model.joblib
artifacts/final_results.csv
artifacts/labeled_table.csv
artifacts/ml_table.csv
artifacts/preprocessor.joblib
artifacts/test.csv
artifacts/train.csv
artifacts/validation.csv
```

Check DVC status:

```powershell
dvc status
```

Expected:

```text
Data and pipelines are up to date.
```

Download artifacts:

```powershell
dvc pull
```

---

# 17. MLflow

MLflow is used for:

* experiment tracking
* parameters
* metrics
* artifacts
* model registration
* model versioning
* model aliases
* model metadata

The production-serving alias is:

```text
champion
```

The API loads the registered model through MLflow rather than directly loading an arbitrary model file.

This allows the served model version to be controlled through the registry.

---

# 18. Testing

The project includes unit and integration tests.

Run all tests with:

```powershell
pytest -q
```

The project currently contains 36 passing tests.

The tests cover areas including:

* preprocessing
* feature handling
* inference
* API behavior
* validation
* configuration
* model-related functionality

---

# 19. API Endpoints

### Health

```text
GET /health
```

Checks API and model availability.

### Model Information

```text
GET /model-info
```

Returns:

* registered model name
* model version
* model alias
* serving stage

### Prediction

```text
POST /predict
```

Returns a late-delivery prediction and probability.

---

# 20. Docker

The project uses Docker Compose to run the main infrastructure.

Start:

```powershell
docker compose up --build -d
```

Stop:

```powershell
docker compose down
```

View logs:

```powershell
docker compose logs api
```

View all service logs:

```powershell
docker compose logs
```

Restart the API:

```powershell
docker compose restart api
```

---

# 21. Monitoring and Logging

The API includes application logging and prediction logging.

Prediction-related information is recorded to support monitoring and troubleshooting.

Logs can be inspected using:

```powershell
docker compose logs api
```

---

# 22. CI/CD

GitHub Actions is configured to automate project checks.

The CI workflow includes automated testing and Docker build validation.

The purpose is to catch problems before changes are merged into the repository.

---

# 23. Pre-commit

Pre-commit hooks are configured for code-quality checks.

Install:

```powershell
pre-commit install
```

Run manually:

```powershell
pre-commit run --all-files
```

---

# 24. Reproducibility

To reproduce the project from a fresh machine:

```powershell
git clone https://github.com/DanaAssad315/olist-mlops.git
cd olist-mlops

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

dvc pull

docker compose up --build -d
```

Then initialize the MLflow Registry:

```powershell
python -m src.mlflow_tracking
```

Restart the API so it loads the registered model:

```powershell
docker compose restart api
```

Verify:

```powershell
curl.exe http://localhost:8000/health
```

Then:

```powershell
curl.exe http://localhost:8000/model-info
```

Finally, send a request to:

```text
POST /predict
```

---

# 25. Troubleshooting

## API says model is unavailable

Check:

```powershell
docker compose logs api --tail=100
```

Make sure MLflow is running:

```powershell
docker compose ps
```

Then initialize the registry:

```powershell
python -m src.mlflow_tracking
```

Restart the API:

```powershell
docker compose restart api
```

---

## MLflow says the registered model does not exist

Example:

```text
RESOURCE_DOES_NOT_EXIST:
Registered Model with name=olist-delivery-random-forest not found
```

Initialize the registry:

```powershell
python -m src.mlflow_tracking
```

Then restart the API:

```powershell
docker compose restart api
```

---

## Prediction refers to an old or missing MLflow Run

Example:

```text
RESOURCE_DOES_NOT_EXIST:
Run with id=... not found
```

This can happen when the API process was started before the current MLflow model version was registered.

Run:

```powershell
python -m src.mlflow_tracking
```

Then:

```powershell
docker compose restart api
```

Then verify:

```powershell
curl.exe http://localhost:8000/health
```

and:

```powershell
curl.exe http://localhost:8000/model-info
```

Do not repeatedly run the MLflow registration command unless a new model version is intentionally required, because each registration can create a new model version.

---

## DVC pull fails

Make sure DVC is installed:

```powershell
dvc --version
```

Make sure the machine has valid credentials for the configured DAGsHub DVC remote.

Then retry:

```powershell
dvc pull
```

Credentials must remain local and must not be committed to Git.

---

# 26. Expected End-to-End Flow

The intended production-oriented workflow is:

```text
Olist Data
    ↓
PostgreSQL
    ↓
Data Validation
    ↓
Feature Engineering
    ↓
ML Training
    ↓
Model Evaluation
    ↓
DVC Versioning
    ↓
MLflow Tracking
    ↓
MLflow Model Registry
    ↓
Champion Model
    ↓
FastAPI
    ↓
Docker
    ↓
Prediction + Monitoring
```

---

# 27. Project Goal

This project demonstrates an end-to-end MLOps workflow rather than only a machine-learning model.

The focus is on making the model:

* reproducible
* versioned
* validated
* testable
* deployable
* observable
* accessible through an API
* manageable through a model registry
* reproducible on a clean environment
