import subprocess

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from src.config import PROJECT_ROOT, config


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def setup_mlflow() -> None:
    """Configure MLflow tracking and experiment settings."""
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def load_final_results() -> dict:
    """Load the final Task 2 evaluation results."""
    results_path = PROJECT_ROOT / "artifacts" / "final_results.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"Final results file not found: {results_path}")

    results = pd.read_csv(results_path)

    if results.empty:
        raise ValueError("final_results.csv is empty.")

    row = results.iloc[0]

    return {
        "model": str(row["model"]),
        "threshold": float(row["threshold"]),
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
    }


def log_final_model() -> None:
    """
    Log the selected Task 2 model to MLflow,
    register it, and assign the champion alias.
    """

    setup_mlflow()

    model_path = PROJECT_ROOT / config["paths"]["model"]
    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor"]
    feature_names_path = PROJECT_ROOT / config["paths"]["feature_names"]
    results_path = PROJECT_ROOT / "artifacts" / "final_results.csv"

    for path in [
        model_path,
        preprocessor_path,
        feature_names_path,
        results_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    results = load_final_results()

    model_name = config["mlflow"]["registered_model_name"]
    alias = config["mlflow"]["model_alias"]

    # Load the exact model selected in Task 2.
    model = joblib.load(model_path)

    params = {
        "model_type": results["model"],
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
        "prediction_threshold": results["threshold"],
    }

    metrics = {
        "test_accuracy": results["accuracy"],
        "test_precision": results["precision"],
        "test_recall": results["recall"],
        "test_f1": results["f1"],
    }

    with mlflow.start_run(run_name="task2-final-random-forest") as run:
        # -----------------------------
        # Parameters
        # -----------------------------
        mlflow.log_params(params)

        # -----------------------------
        # Metrics
        # -----------------------------
        mlflow.log_metrics(metrics)

        # -----------------------------
        # Traceability metadata
        # -----------------------------
        mlflow.set_tags(
            {
                "project": config["project"]["name"],
                "model_selection": "final_model",
                "dataset_source": "Olist",
                "dataset_versioning": "DVC",
                "git_commit": get_git_commit(),
                "serving_stage": config["mlflow"]["serving_stage"],
                "validation_status": "passed",
            }
        )

        # -----------------------------
        # Supporting artifacts
        # -----------------------------
        mlflow.log_artifact(
            str(preprocessor_path),
            artifact_path="preprocessing",
        )

        mlflow.log_artifact(
            str(feature_names_path),
            artifact_path="features",
        )

        mlflow.log_artifact(
            str(results_path),
            artifact_path="evaluation",
        )

        # -----------------------------
        # Log + register model
        # -----------------------------
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            registered_model_name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        run_id = run.info.run_id

        print(f"MLflow run ID: {run_id}")
        print(f"Registered model: {model_name}")
        print(f"Model URI: {model_info.model_uri}")

    # -----------------------------
    # Model Registry
    # -----------------------------
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise RuntimeError(f"No registered versions found for '{model_name}'.")

    latest_version = max(
        versions,
        key=lambda version: int(version.version),
    )

    version_number = latest_version.version

    # Assign the serving alias.
    client.set_registered_model_alias(
        model_name,
        alias,
        version_number,
    )

    # Add model-version metadata.
    client.set_model_version_tag(
        model_name,
        version_number,
        "serving_stage",
        config["mlflow"]["serving_stage"],
    )

    client.set_model_version_tag(
        model_name,
        version_number,
        "validation_status",
        "passed",
    )

    print(f"Registered model version: {version_number}")
    print(f"Model alias: {alias}")
    print(f"Serving stage tag: {config['mlflow']['serving_stage']}")


if __name__ == "__main__":
    log_final_model()
