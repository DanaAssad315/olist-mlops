from pathlib import Path
import logging

import great_expectations as gx
import pandas as pd
import yaml


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_validation.yaml"
GX_DIR = PROJECT_ROOT / "great_expectations"


def load_validation_config():
    """Load data validation configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_expectation_suite():
    """
    Create and persist the Great Expectations suite used
    to validate prediction input data.
    """
    config = load_validation_config()["validation"]

    context = gx.get_context(
        mode="file",
        project_root_dir=str(GX_DIR),
    )

    suite_name = "olist_prediction_input_suite"

    try:
        suite = context.suites.get(name=suite_name)
        logger.info("Loaded existing GX expectation suite: %s", suite_name)
        return context, suite

    except Exception:
        logger.info("Creating GX expectation suite: %s", suite_name)

    suite = gx.ExpectationSuite(name=suite_name)

    # Required columns
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=config["required_columns"],
            exact_match=False,
        )
    )

    # Required columns should not exceed the configured missing-value rate.
    for column in config["required_columns"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=column,
                mostly=1.0 - config["max_missing_rate"],
            )
        )

    # Numeric constraints
    for column, constraints in config["numeric_constraints"].items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=constraints["min"],
            )
        )

    suite = context.suites.add(suite)

    logger.info(
        "Created GX expectation suite '%s' with %d expectations.",
        suite_name,
        len(suite.expectations),
    )

    return context, suite


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate prediction input using Great Expectations.

    Returns:
        True when validation passes.

    Raises:
        ValueError when validation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input data cannot be empty.")

    context, suite = build_expectation_suite()

    data_source_name = "olist_prediction_data"

    try:
        data_source = context.data_sources.get(data_source_name)
    except Exception:
        data_source = context.data_sources.add_pandas(
            name=data_source_name
        )

    asset_name = "prediction_input"

    try:
        data_asset = data_source.get_asset(asset_name)
    except Exception:
        data_asset = data_source.add_dataframe_asset(
            name=asset_name
        )

    batch_definition_name = "whole_dataframe"

    try:
        batch_definition = data_asset.get_batch_definition(
            batch_definition_name
        )
    except Exception:
        batch_definition = (
            data_asset.add_batch_definition_whole_dataframe(
                batch_definition_name
            )
        )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    validation_result = batch.validate(suite)

    success = bool(validation_result["success"])

    if success:
        logger.info(
            "Great Expectations validation passed for %d rows.",
            len(df),
        )
        return True

    failed_expectations = [
        result["expectation_config"]["type"]
        for result in validation_result["results"]
        if not result["success"]
    ]

    logger.error(
        "Great Expectations validation failed: %s",
        failed_expectations,
    )

    raise ValueError(
        f"Data validation failed. Failed expectations: "
        f"{failed_expectations}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    test_path = PROJECT_ROOT / "artifacts" / "test.csv"
    test_df = pd.read_csv(test_path)

    validate_dataframe(test_df)

    print("Great Expectations validation passed.")