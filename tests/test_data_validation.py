import pandas as pd
import pytest

from src.data_validation import validate_dataframe


def make_valid_dataframe():
    return pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2018-01-01 10:00:00",
                    "2018-01-02 12:00:00",
                ]
            ),
            "item_count": [1, 2],
            "total_price": [100.0, 200.0],
            "total_freight": [10.0, 20.0],
            "payment_total": [110.0, 220.0],
            "unique_products": [1, 2],
            "unique_sellers": [1, 2],
            "payment_count": [1, 1],
        }
    )


def test_valid_prediction_data():
    df = make_valid_dataframe()

    assert validate_dataframe(df) is True


def test_missing_required_column():
    df = make_valid_dataframe()

    df = df.drop(columns=["total_price"])

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_negative_price_rejected():
    df = make_valid_dataframe()

    df.loc[0, "total_price"] = -10

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_zero_item_count_rejected():
    df = make_valid_dataframe()

    df.loc[0, "item_count"] = 0

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_invalid_numeric_type_rejected():
    df = make_valid_dataframe()

    df["total_price"] = ["invalid", "invalid"]

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_invalid_timestamp_rejected():
    df = make_valid_dataframe()

    df["order_purchase_timestamp"] = [
        "not-a-date",
        "also-not-a-date",
    ]

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_excessive_missing_values_rejected():
    df = make_valid_dataframe()

    df["total_price"] = [None, None]

    with pytest.raises(ValueError):
        validate_dataframe(df)