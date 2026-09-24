# Olist MLOps

An end-to-end MLOps project for predicting whether an Olist order will be delivered late.

The project covers data ingestion, data validation, feature engineering, model training and evaluation, experiment tracking, model versioning, production inference, API serving, testing, monitoring, and CI/CD.

## Project Structure

```text
olist-mlops/
├── app/
├── artifacts/
├── config/
│   └── config.yaml
├── data/
├── database/
├── ingestion/
├── notebooks/
├── scripts/
├── src/
├── tests/
├── .github/
├── great_expectations/
├── Dockerfile
├── Dockerfile.mlflow
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .pre-commit-config.yaml
└── README.md
```

## Prerequisites

* Python 3.11
* Docker Desktop
* Git
* DVC

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/DanaAssad315/olist-mlops.git
cd olist-mlops
```

### 2. Create and activate a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

Create a local `.env` file from the provided example:

```powershell
copy .env.example .env
```

Update the PostgreSQL password if needed.

Environment variables are used for local service configuration and are not committed to the repository.

## Data and DVC

Large data and machine learning artifacts are tracked using DVC.

The project uses a DVC remote hosted on DAGsHub for portable storage of tracked data and artifacts.

The repository contains the DVC remote URL, while authentication credentials are stored locally and are not committed to Git.

Before pulling the tracked artifacts, configure your DAGsHub DVC credentials locally.

Then run:

```powershell
dvc pull
```

After a successful pull, the DVC-tracked artifacts will be restored under:

```text
artifacts/
```

You can verify the DVC state with:

```powershell
dvc status
```

## Database

PostgreSQL 16 is used for the Olist dataset.

To start only the PostgreSQL service:

```powershell
docker compose up -d postgres
```

To stop the PostgreSQL service:

```powershell
docker compose stop postgres
```

The database configuration is defined through Docker Compose and environment variables.

## Machine Learning Workflow

The ML development workflow is organized into the following notebooks:

```text
01_read_join
02_create_labels
03_split_data
04_eda
05_feature_engineering
06_train_evaluate
```

The workflow includes:

1. Loading and joining the Olist datasets.
2. Creating the late-delivery target.
3. Creating time-based train, validation, and test splits.
4. Exploratory data analysis.
5. Feature engineering and preprocessing.
6. Model training and evaluation.

The final trained model is a Random Forest classifier.

The production inference workflow uses the saved preprocessing pipeline and the registered trained model. Inference does not retrain the model.

## Model Artifacts

Important artifacts are stored under:

```text
artifacts/
```

Examples include:

* `final_model.joblib`
* `preprocessor.joblib`
* `feature_names.csv`
* `ml_table.csv`
* `labeled_table.csv`
* `train.csv`
* `validation.csv`
* `test.csv`
* `final_results.csv`

These artifacts are versioned using DVC.

## Data Validation

The project includes input and data quality validation to detect invalid or unexpected data before prediction.

Validation functionality is implemented in the `src/` modules and covered by automated tests.

Great Expectations is also used as part of the data quality and validation workflow.

## Experiment Tracking and Model Registry

MLflow is used to track experiments, parameters, metrics, and model information.

The project also uses the MLflow Model Registry to manage the model used for production inference.

The inference service loads the model from the MLflow Model Registry using the configured model name and `champion` alias rather than loading or retraining a model during application startup.

## API

The project provides a FastAPI application for production-oriented inference.

The API provides endpoints for:

* Health checks
* Model information
* Single prediction
* Batch prediction
* Prometheus metrics
* Prediction monitoring

The main endpoints are:

```text
GET  /health
GET  /model-info
POST /predict
POST /predict/batch
GET  /metrics
GET  /monitoring
```

### Run the API directly

Make sure the MLflow server is running and the required registered model is available.

Then run:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

## Testing

The project includes automated tests covering configuration, preprocessing, validation, prediction, monitoring, and API behavior.

Run the complete test suite with:

```powershell
pytest
```

For a more detailed output:

```powershell
pytest -v
```

## Docker

The project includes Docker support for PostgreSQL, MLflow, and the FastAPI application.

To build and start the complete stack:

```powershell
docker compose up --build
```

To run the stack in detached mode:

```powershell
docker compose up --build -d
```

To stop the services:

```powershell
docker compose down
```

The services use the following ports:

```text
PostgreSQL: 5432
MLflow:     5000
FastAPI:    8000
```

## MLflow

The MLflow tracking server is included as a Docker service.

When the Docker stack is running, MLflow is available at:

```text
http://localhost:5000
```

The MLflow service stores its tracking database and artifacts in a Docker volume.

The production API uses the configured MLflow tracking URI and loads the registered model through the configured model alias.

## Logging and Monitoring

The application includes structured logging for important application and inference events.

Prediction logs are stored as JSON Lines and are used to support basic production monitoring.

Monitoring functionality includes prediction distribution, latency, error-rate, and prediction-drift checks.

Monitoring functionality is implemented under:

```text
src/monitoring.py
```

The API also exposes Prometheus-compatible metrics through:

```text
GET /metrics
```

## CI/CD

GitHub Actions is used to automate project quality checks.

The CI workflow includes:

* Ruff linting
* Black formatting checks
* Automated tests
* Docker image build

The workflow is triggered on pushes and pull requests to the configured `main` and `master` branches.

A Docker Hub push job is also configured for pushes to the `main` branch when the required Docker Hub secrets are available.

Pre-commit hooks are configured for development-time code quality checks.

## Reproducibility

The project is designed to support reproducibility from data preparation to production inference.

The main components supporting reproducibility are:

* Git for source-code versioning.
* DVC for data and ML artifact versioning.
* MLflow for experiment tracking and model registry.
* Configuration files and environment variables.
* Docker for service/container reproducibility.
* Automated tests and CI checks.

A clean clone can restore the DVC-tracked artifacts using:

```powershell
dvc pull
```

and verify the project state using:

```powershell
dvc status
```

## Production Inference Workflow

The production workflow follows this sequence:

```text
Input Data
    ↓
Input/Data Validation
    ↓
Preprocessing
    ↓
MLflow Registered Model
    ↓
Prediction
    ↓
Prediction Logging
    ↓
Monitoring
```

The production application uses the existing preprocessing objects and the registered trained model. Model training is kept separate from inference.

## Development

For development, install both runtime and development dependencies:

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run tests:

```powershell
pytest
```

Run pre-commit checks:

```powershell
pre-commit run --all-files
```

## Project Goal

The goal of this project is to demonstrate an end-to-end MLOps workflow, from data preparation and model development to reproducible, tested, and production-oriented ML inference.
