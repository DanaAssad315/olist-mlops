import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import Counter, Histogram

from src.config import config

logger = logging.getLogger(__name__)


# Prometheus metrics
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests.",
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests.",
)

prediction_total = Counter(
    "prediction_total",
    "Total number of predictions.",
    ["prediction"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds.",
)


def get_prediction_log_path() -> Path:
    """Return the configured path for prediction logs."""
    log_file = config["monitoring"]["prediction_log_file"]
    path = Path(log_file)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record_prediction_metrics(
    prediction: int,
    duration: float,
) -> None:
    """Record Prometheus metrics for a completed prediction."""
    prediction_total.labels(prediction=str(prediction)).inc()

    prediction_latency_seconds.observe(duration)


def record_prediction_error() -> None:
    """Record a failed prediction request."""
    prediction_errors_total.inc()


def record_prediction_request() -> None:
    """Record an incoming prediction request."""
    prediction_requests_total.inc()


def log_prediction(
    input_data: dict,
    prediction: int,
    probability: float,
    model_version: str,
    duration: float,
) -> None:
    """
    Store prediction details as one JSON object per line.

    The stored records can later be joined with actual outcomes
    to evaluate model performance and monitor drift.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": int(prediction),
        "late_probability": float(probability),
        "model_version": model_version,
        "duration_seconds": float(duration),
        "actual_outcome": None,
    }

    log_path = get_prediction_log_path()

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def load_prediction_logs() -> list[dict]:
    """Load stored prediction records."""
    log_path = get_prediction_log_path()

    if not log_path.exists():
        return []

    records = []

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid prediction log line.")

    return records


def get_prediction_distribution() -> dict:
    """
    Calculate the current prediction distribution.

    Returns the total number of predictions and the percentage
    predicted as late.
    """
    records = load_prediction_logs()

    if not records:
        return {
            "total_predictions": 0,
            "late_predictions": 0,
            "late_prediction_rate": 0.0,
        }

    late_predictions = sum(record.get("prediction") == 1 for record in records)

    total_predictions = len(records)

    return {
        "total_predictions": total_predictions,
        "late_predictions": late_predictions,
        "late_prediction_rate": (late_predictions / total_predictions),
    }


def check_prediction_drift() -> dict:
    """
    Compare the current prediction rate with the configured baseline.

    Drift is measured as the absolute difference between the
    current late-prediction rate and the baseline rate.
    """
    distribution = get_prediction_distribution()

    baseline = config["monitoring"]["prediction_rate_baseline"]
    alert_threshold = config["monitoring"]["drift_alert_threshold"]

    drift = abs(distribution["late_prediction_rate"] - baseline)

    alert = drift > alert_threshold

    if alert:
        logger.warning(
            "Prediction drift alert: current_rate=%.4f, "
            "baseline=%.4f, drift=%.4f, threshold=%.4f",
            distribution["late_prediction_rate"],
            baseline,
            drift,
            alert_threshold,
        )

    return {
        **distribution,
        "baseline_rate": baseline,
        "drift": drift,
        "alert": alert,
    }


def check_latency_alert(duration: float) -> bool:
    """Check whether a prediction request exceeded the latency threshold."""
    threshold = config["monitoring"]["latency_alert_seconds"]

    if duration > threshold:
        logger.warning(
            "Latency alert: duration=%.4fs, threshold=%.4fs",
            duration,
            threshold,
        )
        return True

    return False


def check_error_rate_alert() -> bool:
    """Check whether the prediction error rate exceeded the threshold."""
    requests = prediction_requests_total._value.get()
    errors = prediction_errors_total._value.get()

    if requests == 0:
        return False

    error_rate = errors / requests
    threshold = config["monitoring"]["error_rate_alert_threshold"]

    if error_rate > threshold:
        logger.warning(
            "Error rate alert: error_rate=%.4f, threshold=%.4f",
            error_rate,
            threshold,
        )
        return True

    return False
