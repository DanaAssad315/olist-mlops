import json

from src import monitoring


def test_record_prediction_request():
    before = monitoring.prediction_requests_total._value.get()

    monitoring.record_prediction_request()

    after = monitoring.prediction_requests_total._value.get()

    assert after == before + 1


def test_record_prediction_error():
    before = monitoring.prediction_errors_total._value.get()

    monitoring.record_prediction_error()

    after = monitoring.prediction_errors_total._value.get()

    assert after == before + 1


def test_log_prediction(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"

    monkeypatch.setitem(
        monitoring.config["monitoring"],
        "prediction_log_file",
        str(log_file),
    )

    monitoring.log_prediction(
        input_data={
            "item_count": 1,
            "total_price": 100.0,
        },
        prediction=0,
        probability=0.025,
        model_version="1",
        duration=0.5,
    )

    assert log_file.exists()

    with log_file.open("r", encoding="utf-8") as file:
        record = json.loads(file.readline())

    assert record["prediction"] == 0
    assert record["late_probability"] == 0.025
    assert record["model_version"] == "1"
    assert record["duration_seconds"] == 0.5
    assert record["actual_outcome"] is None


def test_prediction_distribution(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"

    monkeypatch.setitem(
        monitoring.config["monitoring"],
        "prediction_log_file",
        str(log_file),
    )

    records = [
        {
            "prediction": 0,
        },
        {
            "prediction": 1,
        },
        {
            "prediction": 0,
        },
        {
            "prediction": 1,
        },
    ]

    with log_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    distribution = monitoring.get_prediction_distribution()

    assert distribution["total_predictions"] == 4
    assert distribution["late_predictions"] == 2
    assert distribution["late_prediction_rate"] == 0.5


def test_prediction_drift_alert(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"

    monkeypatch.setitem(
        monitoring.config["monitoring"],
        "prediction_log_file",
        str(log_file),
    )

    monkeypatch.setitem(
        monitoring.config["monitoring"],
        "prediction_rate_baseline",
        0.10,
    )

    monkeypatch.setitem(
        monitoring.config["monitoring"],
        "drift_alert_threshold",
        0.05,
    )

    records = [
        {"prediction": 1},
        {"prediction": 1},
        {"prediction": 1},
        {"prediction": 0},
    ]

    with log_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    result = monitoring.check_prediction_drift()

    assert result["late_prediction_rate"] == 0.75
    assert result["baseline_rate"] == 0.10
    assert result["drift"] == 0.65
    assert result["alert"] is True
