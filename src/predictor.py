import logging
import time

import joblib
import pandas as pd

from src.config import PROJECT_ROOT, config
from src.preprocessing import load_preprocessor, preprocess_input
from src.validation import validate_input
from src.data_validation import validate_dataframe

logger = logging.getLogger(__name__)


def load_model():
    """
    Load the trained model saved during model development.
    """

    model_path = (
        PROJECT_ROOT
        / config["paths"]["model"]
    )

    logger.info(
        "Loading model from %s",
        model_path
    )

    try:
        model = joblib.load(model_path)

    except Exception:
        logger.exception(
            "Failed to load model from %s",
            model_path
        )
        raise

    logger.info("Model loaded successfully.")

    return model


def predict(df: pd.DataFrame):
    """
    Generate predictions for new order data.

    The saved preprocessor is used to transform the input,
    and the saved model is used for prediction.

    No fitting or retraining is performed.

    Logs include:
    - Input summary
    - Model version
    - Prediction output summary
    - Prediction duration
    """

    start_time = time.perf_counter()

    logger.info(
        "Starting prediction request. Input rows: %d | Input columns: %d",
        len(df),
        len(df.columns) if isinstance(df, pd.DataFrame) else 0
    )

    # Validate the input before loading the model or preprocessing.
    validate_input(df)
    validate_dataframe(df)
    
    try:
        model_version = config["project"]["model_version"]

        logger.info(
            "Using model version: %s",
            model_version
        )

        # Load the saved preprocessing pipeline.
        preprocessor = load_preprocessor()

        # Load the saved trained model.
        model = load_model()

        # Apply the same feature engineering and preprocessing
        # used during model development.
        processed_features = preprocess_input(
            df,
            preprocessor=preprocessor
        )

        # Probability of the order being late.
        probability = model.predict_proba(
            processed_features
        )[:, 1]

        # Use the threshold selected during model evaluation.
        threshold = config["prediction"]["threshold"]

        prediction = (
            probability >= threshold
        ).astype(int)

    except Exception:
        logger.exception(
            "Prediction failed."
        )
        raise

    duration = time.perf_counter() - start_time

    result = pd.DataFrame({
        "late_probability": probability,
        "prediction": prediction
    })

    logger.info(
        "Prediction output: %d rows | Late predictions: %d | "
        "Threshold: %.2f | Model version: %s",
        len(result),
        int(prediction.sum()),
        threshold,
        model_version
    )

    logger.info(
        "Prediction duration: %.4f seconds",
        duration
    )

    logger.info(
        "Prediction request completed successfully."
    )

    return result