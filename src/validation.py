import logging

import pandas as pd


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = [
    "order_purchase_timestamp",
    "item_count",
    "total_price",
    "total_freight",
    "payment_total",
    "unique_products",
    "unique_sellers",
    "payment_count",
]


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate the structure of the input data before prediction.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        ValueError: If required columns are missing or the input is empty.
    """

    logger.info("Validating prediction input.")

    if not isinstance(df, pd.DataFrame):
        logger.error(
            "Invalid input type: expected pandas DataFrame, got %s.", type(df).__name__
        )
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        logger.error("Prediction input is empty.")
        raise ValueError("Input data cannot be empty.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        logger.error("Input is missing required columns: %s", missing_columns)
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info("Input validation passed for %d rows.", len(df))
