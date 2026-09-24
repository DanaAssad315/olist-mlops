import pandas as pd

from src.preprocessing import (
    EXCLUDED_COLUMNS,
    create_features,
    preprocess_input,
)


def test_create_features_builds_expected_features():
    """Feature engineering should create the expected prediction-time features."""

    df = pd.read_csv("artifacts/test.csv").head(5)

    features = create_features(df)

    expected_features = [
        "purchase_year",
        "purchase_month",
        "purchase_day",
        "purchase_weekday",
        "purchase_hour",
        "is_weekend",
        "is_holiday",
        "price_per_item",
        "freight_per_item",
        "freight_price_ratio",
        "payment_per_item",
    ]

    for feature in expected_features:
        assert feature in features.columns


def test_create_features_excludes_ids_and_future_columns():
    """Future information and identifier columns must not enter model features."""

    df = pd.read_csv("artifacts/test.csv").head(5)

    features = create_features(df)

    leakage_columns = [
        "order_id",
        "customer_id",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "order_status",
        "delivery_days",
        "is_late",
    ]

    for column in leakage_columns:
        assert column not in features.columns

    for column in EXCLUDED_COLUMNS:
        assert column not in features.columns


def test_preprocess_input_shape():
    """Preprocessing should produce the same 54 features used by the model."""

    df = pd.read_csv("artifacts/test.csv").head(5)

    processed_features = preprocess_input(df)

    assert processed_features.shape == (5, 54)


def test_preprocess_input_is_deterministic():
    """The same input should produce the same processed features."""

    df = pd.read_csv("artifacts/test.csv").head(5)

    first_result = preprocess_input(df)
    second_result = preprocess_input(df)

    assert (first_result == second_result).all()
