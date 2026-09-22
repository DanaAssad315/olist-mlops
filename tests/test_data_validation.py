import pandas as pd
import pytest

from src.data_validation import validate_dataframe


def test_valid_prediction_data():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2018-01-01 10:00:00",
                "2018-01-02 12:00:00",
            ],
            "item_count": [1, 2],
            "total_price": [100.0, 200.0],
            "total_freight": [10.0, 20.0],
            "payment_total": [110.0, 220.0],
            "unique_products": [1, 2],
            "unique_sellers": [1, 2],
            "payment_count": [1, 1],
        }
    )

    assert validate_dataframe(df) is True


def test_missing_required_column():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2018-01-01 10:00:00"],
            "item_count": [1],
            "total_price": [100.0],
        }
    )

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_invalid_negative_price():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2018-01-01 10:00:00"],
            "item_count": [1],
            "total_price": [-10.0],
            "total_freight": [10.0],
            "payment_total": [20.0],
            "unique_products": [1],
            "unique_sellers": [1],
            "payment_count": [1],
        }
    )

    with pytest.raises(ValueError):
        validate_dataframe(df)