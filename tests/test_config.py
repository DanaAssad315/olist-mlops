from src.config import config


def test_config_contains_required_sections():
    """Project configuration should contain the required sections."""

    required_sections = [
        "project",
        "paths",
        "prediction",
        "logging",
        "mlflow",
    ]

    for section in required_sections:
        assert section in config


def test_mlflow_configuration_is_complete():
    """MLflow configuration should contain the required settings."""

    mlflow_config = config["mlflow"]

    assert mlflow_config["tracking_uri"]
    assert mlflow_config["experiment_name"]
    assert mlflow_config["registered_model_name"]
    assert mlflow_config["model_alias"]
    assert mlflow_config["serving_stage"]


def test_prediction_threshold_is_valid():
    """Prediction threshold must be between 0 and 1."""

    threshold = config["prediction"]["threshold"]

    assert 0 <= threshold <= 1
