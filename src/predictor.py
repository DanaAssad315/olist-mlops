import logging
import time

import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import config
from src.data_validation import validate_dataframe
from src.logging_config import setup_logging
from src.preprocessing import preprocess_input
from src.validation import validate_input


logger = logging.getLogger(__name__)


def load_model():
    """Load the production model from the MLflow Model Registry."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    model_name = config["mlflow"]["registered_model_name"]
    model_alias = config["mlflow"]["model_alias"]

    model_uri = f"models:/{model_name}@{model_alias}"

    logger.info("Loading model from MLflow: %s", model_uri)

    model = mlflow.sklearn.load_model(model_uri)

    logger.info(
        "Model loaded successfully from MLflow: %s",
        model_uri,
    )

    return model


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input, preprocess it, and generate predictions."""
    start_time = time.perf_counter()

    try:
        logger.info(
            "Prediction request received: rows=%d, columns=%d",
            len(df),
            len(df.columns),
        )

        # Basic input validation
        validate_input(df)

        # Great Expectations validation
        validate_dataframe(df)

        # Load model from MLflow Registry
        model = load_model()

        # Preprocess without fitting
        X = preprocess_input(df)

        # Generate probabilities
        probabilities = model.predict_proba(X)[:, 1]

        threshold = config["prediction"]["threshold"]

        predictions = (probabilities >= threshold).astype(int)

        result = pd.DataFrame(
            {
                "late_probability": probabilities,
                "prediction": predictions,
            }
        )

        duration = time.perf_counter() - start_time

        logger.info(
            "Prediction completed: rows=%d, threshold=%.2f, "
            "predicted_late=%d, duration=%.4fs, model=%s@%s",
            len(df),
            threshold,
            int(predictions.sum()),
            duration,
            config["mlflow"]["registered_model_name"],
            config["mlflow"]["model_alias"],
        )

        return result

    except Exception:
        duration = time.perf_counter() - start_time

        logger.exception(
            "Prediction failed after %.4f seconds.",
            duration,
        )

        raise


if __name__ == "__main__":
    setup_logging()
