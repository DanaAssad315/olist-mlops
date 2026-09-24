from pathlib import Path
import logging

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
import yaml


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_validation.yaml"
GX_DIR = PROJECT_ROOT / "great_expectations"

SUITE_NAME = "olist_prediction_input_suite"
DATA_SOURCE_NAME = "olist_prediction_data"
ASSET_NAME = "prediction_input"
BATCH_DEFINITION_NAME = "whole_dataframe"


def load_validation_config():
    """Load Great Expectations validation rules."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_expectations(suite, config):
    """Add all configured expectations to the suite."""

    required_columns = config["required_columns"]
    max_missing_rate = config["max_missing_rate"]

    # ---------------------------------------------------------
    # 1. Schema: required columns
    # ---------------------------------------------------------
    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(
            column_set=required_columns,
            exact_match=False,
        )
    )

    # ---------------------------------------------------------
    # 2. Missing-value rate
    # ---------------------------------------------------------
    for column in required_columns:
        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(
                column=column,
                mostly=1.0 - max_missing_rate,
            )
        )

    # ---------------------------------------------------------
    # 3. Numeric ranges
    # ---------------------------------------------------------
    for column, constraints in config["numeric_constraints"].items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=constraints.get("min"),
                max_value=constraints.get("max"),
            )
        )

    # ---------------------------------------------------------
    # 4. Allowed categories
    # ---------------------------------------------------------
    for column, allowed_values in config.get("allowed_categories", {}).items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=allowed_values,
            )
        )

    return suite


def build_expectation_suite():
    """
    Create or retrieve the project's Great Expectations suite.
    """
    config = load_validation_config()["validation"]

    context = gx.get_context(
        mode="file",
        project_root_dir=str(GX_DIR),
    )

    try:
        suite = context.suites.get(name=SUITE_NAME)
        logger.info(
            "Loaded existing GX expectation suite: %s",
            SUITE_NAME,
        )
        return context, suite

    except Exception:
        logger.info(
            "Creating GX expectation suite: %s",
            SUITE_NAME,
        )

        suite = gx.ExpectationSuite(name=SUITE_NAME)
        suite = _build_expectations(suite, config)

        context.suites.add(suite)

        logger.info(
            "Created GX suite '%s' with %d expectations.",
            SUITE_NAME,
            len(suite.expectations),
        )

        return context, suite


def _get_batch(context, df):
    """Create a GX batch from an in-memory pandas DataFrame."""

    try:
        data_source = context.data_sources.get(DATA_SOURCE_NAME)
    except Exception:
        data_source = context.data_sources.add_pandas(name=DATA_SOURCE_NAME)

    try:
        data_asset = data_source.get_asset(ASSET_NAME)
    except Exception:
        data_asset = data_source.add_dataframe_asset(name=ASSET_NAME)

    try:
        batch_definition = data_asset.get_batch_definition(BATCH_DEFINITION_NAME)
    except Exception:
        batch_definition = data_asset.add_batch_definition_whole_dataframe(
            BATCH_DEFINITION_NAME
        )

    return batch_definition.get_batch(batch_parameters={"dataframe": df})


def _validate_column_types(df, config):
    """
    Validate pandas dtypes explicitly.

    GX's type expectations are backend-dependent, while this
    service receives pandas DataFrames. Therefore datetime and
    numeric contracts are checked explicitly before GX execution.
    """
    failures = []

    for column, expected_type in config["column_types"].items():
        if column not in df.columns:
            continue

        series = df[column]

        if expected_type == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(series):
                try:
                    pd.to_datetime(series, errors="raise")
                except (TypeError, ValueError):
                    failures.append(f"{column}: expected datetime-compatible values")

        elif expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(series):
                failures.append(f"{column}: expected numeric values")

    if failures:
        raise ValueError("Column type validation failed: " + "; ".join(failures))


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate prediction input before model inference.

    Validation failure policy:
        REJECT the request.

    Returns:
        True when all validation checks pass.

    Raises:
        TypeError: invalid input type.
        ValueError: validation failure.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input data cannot be empty.")

    config = load_validation_config()["validation"]

    # ---------------------------------------------------------
    # Explicit schema/type checks
    # ---------------------------------------------------------
    missing_columns = [
        column for column in config["required_columns"] if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    _validate_column_types(df, config)

    # ---------------------------------------------------------
    # Great Expectations validation
    # ---------------------------------------------------------
    context, suite = build_expectation_suite()
    batch = _get_batch(context, df)

    validation_result = batch.validate(suite)

    if validation_result["success"]:
        logger.info(
            "Great Expectations validation passed for %d rows.",
            len(df),
        )
        return True

    failed_expectations = []

    for result in validation_result["results"]:
        if not result["success"]:
            failed_expectations.append(result["expectation_config"]["type"])

    logger.error(
        "Great Expectations validation failed: %s",
        failed_expectations,
    )

    raise ValueError(
        f"Data validation failed. Failed expectations: {failed_expectations}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
    )

    test_path = PROJECT_ROOT / "artifacts" / "test.csv"

    test_df = pd.read_csv(test_path)

    # Convert timestamp exactly as the production pipeline expects.
    test_df["order_purchase_timestamp"] = pd.to_datetime(
        test_df["order_purchase_timestamp"],
        errors="raise",
    )

    validate_dataframe(test_df)

    print("Great Expectations validation passed.")
