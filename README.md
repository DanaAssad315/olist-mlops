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
├── models/
├── notebooks/
├── scripts/
├── src/
├── tests/
├── .github/
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
git clone <repository-url>
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

Create a `.env` file based on `.env.example`.

```powershell
copy .env.example .env
```

Environment variables are used for configuration that should not be hardcoded, such as service credentials and connection settings.

## Data and DVC

Large data and ML artifacts are tracked using DVC.

The project uses a DVC remote hosted on DAGsHub for portable storage of tracked data and artifacts.

After cloning the repository, configure DVC credentials locally and pull the tracked data:

```powershell
dvc pull
```

DVC credentials are stored locally and are not committed to the repository.

## Database

PostgreSQL is used for the Olist dataset.

Start the database with:

```powershell
docker compose up -d
```

The database configuration is defined through Docker Compose and environment variables.

## Machine Learning Workflow

The ML development workflow is organized into notebooks:

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

The final model is a Random Forest classifier.

The production inference workflow uses the saved preprocessing pipeline and trained model. **Inference does not retrain the model.**

## Model Artifacts

Important artifacts are stored under:

```text
artifacts/
```

Examples include:

* `final_model.joblib`
* `preprocessor.joblib`
* `feature_names.csv`
* processed feature arrays
* evaluation results
* EDA outputs

These artifacts are versioned with DVC where appropriate.

## Data Validation

The project includes data validation checks to detect invalid or unexpected input data before inference.

Validation is implemented in the `src/` modules and covered by automated tests.

Great Expectations is also used as part of the data quality and validation workflow.

## Experiment Tracking and Model Registry

MLflow is used to track experiments, parameters, metrics, and model information.

The project also uses the MLflow Model Registry to manage the production model version.

The inference service loads the required model from the registry rather than retraining a model during application startup.

## API

The project provides a FastAPI application for production-oriented inference.

The API is responsible for:

* Receiving prediction requests.
* Validating input data.
* Loading the registered model and preprocessing pipeline.
* Generating predictions.
* Returning prediction results.
* Logging prediction information.

The application can be started through Docker Compose or directly from the project environment.

## Testing

The project includes automated tests covering configuration, preprocessing, validation, data validation, prediction, monitoring, and API behavior.

Run the test suite with:

```powershell
pytest
```

## Docker

The application and supporting services can be run using Docker.

Start the services with:

```powershell
docker compose up --build
```

To stop them:

```powershell
docker compose down
```

## Logging and Monitoring

The application includes structured logging for important application and inference events.

Prediction logs are used to support basic production monitoring and allow prediction behavior to be reviewed over time.

Monitoring functionality is implemented under:

```text
src/monitoring.py
```

## CI/CD

GitHub Actions is used to automate project checks.

The CI workflow runs automated validation and testing to help ensure that changes do not break the project.

Pre-commit hooks are also configured for development-time checks.

## Reproducibility

The project is designed to reproduce the ML workflow from data preparation to production inference.

The main components supporting reproducibility are:

* Git for source-code versioning.
* DVC for data and ML artifact versioning.
* MLflow for experiment and model tracking.
* Configuration files and environment variables.
* Docker for service/container reproducibility.
* Automated tests and CI checks.

## Production Inference Workflow

The production workflow follows this general sequence:

```text
Input Data
    ↓
Data Validation
    ↓
Preprocessing
    ↓
Registered Model
    ↓
Prediction
    ↓
Prediction Logging
    ↓
Monitoring
```

The production application uses the existing trained model and preprocessing objects. Model training is kept separate from inference.

## Development

For development, install both runtime and development dependencies:

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run tests with:

```powershell
pytest
```

Run pre-commit checks with:

```powershell
pre-commit run --all-files
```

## Project Goal

The goal of this project is to demonstrate an end-to-end MLOps workflow, from data preparation and model development to reproducible, tested, and production-oriented ML inference.
