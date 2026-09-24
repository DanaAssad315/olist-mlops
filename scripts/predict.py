import argparse

import pandas as pd

from src.predictor import predict


def main():
    parser = argparse.ArgumentParser(description="Run Olist delivery prediction.")

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = predict(df)

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
