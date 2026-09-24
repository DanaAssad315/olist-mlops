import pandas as pd
import pytest

from src.validation import validate_input


def test_validate_input_accepts_valid_data():
    """Valid input containing all required columns should pass validation."""

    valid_data = {
        "order_purchase_timestamp": ["2018-01-01 10:00:00"],
        "item_count": [1],
        "total_price": [100.0],
        "total_freight": [20.0],
        "payment_total": [120.0],
        "unique_products": [1],
        "unique_sellers": [1],
        "payment_count": [1],
    }

    df = pd.DataFrame(valid_data)

    validate_input(df)


def test_validate_input_rejects_empty_data():
    """Empty input should raise a ValueError."""

    df = pd.DataFrame()

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_input(df)


def test_validate_input_rejects_missing_columns():
    """Input with missing required columns should raise a ValueError."""

    df = pd.DataFrame({"total_price": [100.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)


def test_validate_input_rejects_wrong_input_type():
    """Non-DataFrame input should raise a TypeError."""

    invalid_input = {"total_price": [100.0]}

    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        validate_input(invalid_input)
