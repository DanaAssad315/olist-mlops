import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import config
from src.predictor import load_model, predict


def test_model_loads_from_mlflow_registry():
    """The production model should load successfully from MLflow Registry."""

    model = load_model()

    assert model is not None
    assert type(model).__name__ == "RandomForestClassifier"


def test_model_registry_uri_uses_champion_alias():
    """The predictor should use the configured MLflow model alias."""

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    model_name = config["mlflow"]["registered_model_name"]
    alias = config["mlflow"]["model_alias"]

    model_uri = f"models:/{model_name}@{alias}"

    model = mlflow.sklearn.load_model(model_uri)

    assert model is not None


def test_predict_returns_expected_columns():
    """Prediction should return probability and binary prediction columns."""

    df = pd.read_csv("artifacts/test.csv").head(5)

    result = predict(df)

    assert list(result.columns) == [
        "late_probability",
        "prediction",
    ]


def test_predict_returns_correct_number_of_rows():
    """Prediction should return one result for each input row."""

    df = pd.read_csv("artifacts/test.csv").head(5)

    result = predict(df)

    assert len(result) == len(df)


def test_predict_returns_valid_values():
    """Probabilities should be between 0 and 1 and predictions should be binary."""

    df = pd.read_csv("artifacts/test.csv").head(5)

    result = predict(df)

    assert result["late_probability"].between(0, 1).all()
    assert result["prediction"].isin([0, 1]).all()


def test_predict_is_deterministic_for_known_input():
    """The same known input should produce the same prediction repeatedly."""

    df = pd.read_csv("artifacts/test.csv").head(5)

    first_result = predict(df)
    second_result = predict(df)

    pd.testing.assert_frame_equal(first_result, second_result)
