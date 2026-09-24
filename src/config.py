import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config():
    """Load project configuration from config.yaml."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_tracking_uri:
        loaded_config["mlflow"]["tracking_uri"] = mlflow_tracking_uri

    return loaded_config


config = load_config()
