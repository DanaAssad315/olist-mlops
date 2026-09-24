import logging

import holidays
import joblib
import numpy as np
import pandas as pd

from src.config import config
import mlflow

logger = logging.getLogger(__name__)


EXCLUDED_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "late",
    "order_status",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "customer_city",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the same features used during model development.

    The prediction point is the order purchase time, so
    future delivery information is excluded.
    """

    logger.info(
        "Starting feature creation for %d rows.",
        len(df),
    )

    df = df.copy()

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    purchase_time = df["order_purchase_timestamp"]

    # Time-based features
    # order_year is derived from the purchase timestamp.
    # It is NOT required from the API user.
    df["order_year"] = purchase_time.dt.year
    df["purchase_year"] = purchase_time.dt.year
    df["purchase_month"] = purchase_time.dt.month
    df["purchase_day"] = purchase_time.dt.day
    df["purchase_weekday"] = purchase_time.dt.weekday
    df["purchase_hour"] = purchase_time.dt.hour

    # Weekend indicator
    df["is_weekend"] = (purchase_time.dt.weekday >= 5).astype(int)

    # Brazilian holidays
    years = purchase_time.dt.year.dropna().astype(int).unique()

    brazil_holidays = holidays.Brazil(years=years)

    df["is_holiday"] = purchase_time.dt.date.isin(brazil_holidays).astype(int)

    # Ratio-based features
    item_count_safe = df["item_count"].replace(
        0,
        np.nan,
    )

    df["price_per_item"] = df["total_price"] / item_count_safe

    df["freight_per_item"] = df["total_freight"] / item_count_safe

    df["freight_price_ratio"] = df["total_freight"] / df["total_price"].replace(
        0, np.nan
    )

    df["payment_per_item"] = df["payment_total"] / item_count_safe

    # Log-transformed numerical features
    log_columns = [
        "total_price",
        "total_freight",
        "payment_total",
        "item_count",
        "unique_products",
        "unique_sellers",
        "payment_count",
    ]

    for column in log_columns:
        df[f"log_{column}"] = np.log1p(df[column].clip(lower=0))

    # Remove IDs, leakage columns, and other excluded fields
    df = df.drop(
        columns=[column for column in EXCLUDED_COLUMNS if column in df.columns],
        errors="ignore",
    )

    # The timestamp has already been used to create
    # time-based features.
    df = df.drop(
        columns=["order_purchase_timestamp"],
        errors="ignore",
    )

    logger.info(
        "Feature creation completed. Output shape: %s",
        df.shape,
    )

    return df


def load_preprocessor():
    """
    Load the fitted preprocessing pipeline from MLflow artifacts.
    """

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    run_id = config["mlflow"]["run_id"]

    logger.info(
        "Loading preprocessor from MLflow run: %s",
        run_id,
    )

    try:
        preprocessor_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessor.joblib",
        )

        preprocessor = joblib.load(preprocessor_path)

    except Exception:
        logger.exception(
            "Failed to load preprocessor from MLflow run: %s",
            run_id,
        )
        raise

    logger.info("Preprocessor loaded successfully from MLflow.")

    return preprocessor


def preprocess_input(
    df: pd.DataFrame,
    preprocessor=None,
):
    """
    Create model features and apply the already-fitted
    preprocessing pipeline.

    The preprocessor is only transformed here.
    It is never fitted again.
    """

    logger.info(
        "Starting preprocessing for %d input rows.",
        len(df),
    )

    if preprocessor is None:
        preprocessor = load_preprocessor()

    features = create_features(df)

    try:
        processed_features = preprocessor.transform(features)

    except Exception:
        logger.exception("Preprocessing failed.")
        raise

    logger.info(
        "Preprocessing completed. Output shape: %s",
        processed_features.shape,
    )

    return processed_features
