import logging

from src.config import PROJECT_ROOT, config


def setup_logging():
    """
    Configure application-wide logging.

    Logs are written both to the console and to the
    log file specified in config.yaml.
    """

    log_file = PROJECT_ROOT / config["logging"]["file"]

    # Create the logs directory if it does not exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
