# Olist MLOps — Delivery Delay Predictor

An end-to-end MLOps project for predicting whether an Olist order will be delivered late or on time.

The project covers the complete machine learning lifecycle, from data preparation and validation to model training, experiment tracking, model registry, production inference, API serving, testing, monitoring, and CI/CD.

> **Status:** Complete MLOps pipeline with a trained Random Forest model, DVC data/artifact versioning, MLflow experiment tracking and model registry, FastAPI inference service, Docker Compose deployment, automated tests, monitoring, and CI/CD.

---

## Project Structure

```text
olist-mlops/
├── app/                         # FastAPI inference service
├── artifacts/                   # DVC-tracked datasets and ML artifacts
├── config/
│   └── config.yaml              # Project and model configuration
├── data/                        # Project data
├── database/                    # Database-related files
├── ingestion/                   # Data ingestion utilities
├── notebooks/                   # End-to-end ML development workflow
│   ├── 01_read_join
│   ├── 02_create_labels
│   ├── 03_split_data
│   ├── 04_eda
│   ├── 05_feature_engineering
│   └── 06_train_evaluate
├── scripts/                     # Utility and operational scripts
├── src/                         # Reusable production modules
├── tests/                       # Automated tests
├── great_expectations/          # Data validation configuration
├── .github/workflows/           # GitHub Actions CI/CD
├── Dockerfile                   # FastAPI application image
├── Dockerfile.mlflow            # MLflow service image
├── docker-compose.yml           # PostgreSQL + MLflow + FastAPI
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development and testing dependencies
├── .env.example                 # Environment variable template
├── .pre-commit-config.yaml      # Pre-commit configuration
└── README.md
```

---

## Prerequisites

* Python 3.11
* Git
* Docker Desktop
* DVC

Docker Desktop must be running before starting the services.

---

## Quickstart

### 1. Clone the repository

```powershell
git clone https://github.com/DanaAssad315/olist-mlops.git
cd olist-mlops
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

Create the local environment file:

```powershell
copy .env.example .env
```

Edit `.env` if needed.

> `.env` is local-only and must not be committed to Git.

### 5. Configure DVC credentials

The project uses DAGsHub as the DVC remote.

On a new machine, configure your DAGsHub credentials locally:

```powershell
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user <DAGSHUB_USERNAME>
dvc remote modify dagshub --local password <DAGSHUB_TOKEN>
```

Then pull the versioned datasets and ML artifacts:

```powershell
dvc pull
```

Verify the data state:

```powershell
dvc status
```

Expected:

```text
Data and pipelines are up to date.
```

> DVC credentials are stored locally in `.dvc/config.local` and are not committed to the repository.

### 6. Start the application services

Build and start PostgreSQL, MLflow, and FastAPI:

```powershell
docker compose up --build -d
```

Check the services:

```powershell
docker compose ps
```

The application uses:

| Service    |   Port |
| ---------- | -----: |
| PostgreSQL | `5432` |
| MLflow     | `5000` |
| FastAPI    | `8000` |

Open the API documentation:

```text
http://localhost:8000/docs
```

Open MLflow:

```text
http://localhost:5000
```

---

## MLflow Model Registry Initialization

The MLflow tracking server and model registry are runtime components.

When the project is started on a completely new environment, the MLflow backend may not contain the registered model yet. The saved model and preprocessing artifacts are versioned separately with DVC.

Initialize the MLflow experiment and register the final model with:

```powershell
python -m src.mlflow_tracking
```

This step:

* logs the final model run to MLflow
* registers the production model
* creates a model version
* assigns the configured serving alias
* stores model-version metadata
* makes the model available to the FastAPI inference service

Verify the registered model:

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

Then:

```powershell
curl.exe http://localhost:8000/model-info
```

Example:

```json
{
  "model_name": "olist-delivery-random-forest",
  "model_version": "1",
  "model_alias": "champion",
  "serving_stage": "production"
}
```

> The MLflow Registry is runtime state and is not stored in Git or DVC. Therefore, a fresh MLflow backend must be initialized from the versioned project artifacts.

---

## MLflow Troubleshooting

If `/health` reports:

```json
{
  "status": "degraded",
  "model_status": "unavailable"
}
```

first initialize or refresh the model registration:

```powershell
python -m src.mlflow_tracking
```

Then verify:

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/model-info
```

### Stale or invalid MLflow model version

If `/predict` returns an error similar to:

```text
RESOURCE_DOES_NOT_EXIST:
Run with id=<run_id> not found
```

the MLflow Registry may contain a model version whose artifact points to a run that is no longer available in the current MLflow backend.

This can happen when using a fresh or reset MLflow backend together with previously persisted registry metadata.

Re-register the final model:

```powershell
python -m src.mlflow_tracking
```

Then verify the newly registered model:

```powershell
curl.exe http://localhost:8000/model-info
```

After registration, retry the prediction request.

> The source model and preprocessing artifacts are preserved through DVC. Re-running `src.mlflow_tracking` recreates the MLflow runtime registration from those versioned artifacts.

---

## Running the API Locally

The API can also be started directly from the Python environment for development.

Make sure PostgreSQL and MLflow are running first:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Docker Compose is recommended when running the complete application because the API depends on the project services.

---

## API Endpoints

| Method | Route            | Description                                           |
| ------ | ---------------- | ----------------------------------------------------- |
| GET    | `/health`        | Health check                                          |
| GET    | `/model-info`    | Returns information about the loaded registered model |
| POST   | `/predict`       | Predicts whether an order will be delivered late      |
| POST   | `/predict/batch` | Performs batch predictions                            |
| GET    | `/metrics`       | Returns application metrics                           |
| GET    | `/monitoring`    | Returns monitoring information                        |

### Prediction Request

The `/predict` endpoint accepts an order feature payload matching the production inference schema.

The current request schema contains:

```json
{
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
}
```

The exact validation rules are defined by the FastAPI request schema in:

```text
app/main.py
```

### Example PowerShell Request

PowerShell users can send a prediction request with:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"order_purchase_timestamp":"2018-01-01 10:00:00","customer_zip_code_prefix":14409,"customer_state":"SP","item_count":1,"total_price":100.0,"total_freight":20.0,"unique_products":1,"unique_sellers":1,"payment_total":120.0,"payment_count":1}'
```

A successful prediction returns the predicted class, probability, and model version used by the inference service.

---

## Machine Learning Workflow

The ML development process is implemented in six notebooks:

```text
01_read_join
      ↓
02_create_labels
      ↓
03_split_data
      ↓
04_eda
      ↓
05_feature_engineering
      ↓
06_train_evaluate
```

### Notebook 1 — Read and Join

Loads the Olist tables and creates the main ML dataset.

### Notebook 2 — Create Labels

Creates the target variable indicating whether an order was delivered late.

### Notebook 3 — Train / Validation / Test Split

Creates time-based training, validation, and test datasets.

### Notebook 4 — EDA

Analyzes:

* Target distribution
* Missing values
* Feature distributions
* Late-delivery patterns
* Potential leakage
* Feature selection

### Notebook 5 — Feature Engineering

Handles:

* Missing values
* Categorical encoding
* Feature transformation
* Leakage prevention

Preprocessing is fitted only on the training data and then applied to validation and test data.

### Notebook 6 — Train and Evaluate

Trains and evaluates the Random Forest model and selects the final prediction threshold.

---

## Model

The production model is a Random Forest classifier.

The final model and preprocessing artifacts are versioned with DVC.

MLflow is used to track experiments and manage the registered production model.

The API loads the configured model from the MLflow Model Registry rather than retraining during inference.

Registered model:

```text
olist-delivery-random-forest
```

Production serving alias:

```text
champion
```

The model version is exposed through the `/model-info` endpoint and used by the inference service.

---

## Data Validation

Great Expectations is used to validate data quality before the data enters the ML workflow.

Configuration is stored under:

```text
great_expectations/
```

Data validation helps detect invalid or unexpected data before it reaches downstream ML processing.

---

## DVC

DVC is used to version large datasets and generated ML artifacts that should not be stored directly in Git.

Tracked artifacts include:

```text
artifacts/
├── ml_table.csv
├── labeled_table.csv
├── train.csv
├── validation.csv
├── test.csv
├── preprocessor.joblib
├── feature_names.csv
├── final_model.joblib
└── final_results.csv
```

To check the DVC state:

```powershell
dvc status
```

To pull the tracked data and artifacts:

```powershell
dvc pull
```

The project uses DAGsHub as the configured DVC remote.

---

## MLflow

MLflow provides:

* Experiment tracking
* Parameter and metric logging
* Model artifact tracking
* Model Registry
* Model versioning
* Model aliases
* Model-version metadata

The registered model is:

```text
olist-delivery-random-forest
```

The production model is served using the configured MLflow model alias.

MLflow UI:

```text
http://localhost:5000
```

The registration workflow is implemented in:

```text
src/mlflow_tracking.py
```

---

## Testing

Run the complete test suite:

```powershell
pytest
```

For more detailed output:

```powershell
pytest -v
```

The test suite covers the main data, feature, model, and API functionality.

The project test suite was verified with:

```text
36 passed
```

---

## Docker

Build and start the complete application:

```powershell
docker compose up --build -d
```

Check running containers:

```powershell
docker compose ps
```

View logs:

```powershell
docker compose logs
```

Follow logs:

```powershell
docker compose logs -f
```

Stop the application:

```powershell
docker compose down
```

---

## Monitoring and Logging

The application includes logging and prediction monitoring.

Available monitoring endpoints:

```text
GET /metrics
GET /monitoring
```

Prediction and application logs are used to support production monitoring and troubleshooting.

The monitoring components are implemented in the application source code and expose operational information through the API.

---

## CI/CD

GitHub Actions automatically runs the project's quality and testing checks.

The workflow is located at:

```text
.github/workflows/
```

The CI pipeline includes:

1. Environment setup
2. Dependency installation
3. Code quality checks
4. Formatting checks
5. Automated tests
6. Docker build validation

---

## Pre-commit

The project uses pre-commit hooks for local code-quality checks.

Install the hooks:

```powershell
pre-commit install
```

Run them manually:

```powershell
pre-commit run --all-files
```

---

## Reproducibility

The project uses:

* **Git** — source code versioning
* **DVC** — dataset and ML artifact versioning
* **MLflow** — experiment tracking and model registry
* **Docker** — reproducible application services
* **Great Expectations** — data validation
* **Pytest** — automated testing
* **GitHub Actions** — CI/CD

A new environment can reproduce the project by:

1. Cloning the repository.
2. Installing the required dependencies.
3. Configuring local DVC credentials.
4. Running `dvc pull`.
5. Starting the Docker Compose services.
6. Initializing the MLflow Model Registry with:

   ```powershell
   python -m src.mlflow_tracking
   ```
7. Verifying `/health` and `/model-info`.
8. Sending a request to `/predict`.

The source code and ML artifacts are versioned through Git and DVC. MLflow Registry state is initialized in the target environment from the versioned model artifacts.

---

## Stop the Application

```powershell
docker compose down
```

To start it again later:

```powershell
docker compose up -d
```

---

## License

For educational use as part of the Qafza MLOps training program.
