from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_REQUEST = {
    "order_purchase_timestamp": "2018-01-01 10:00:00",
    "customer_zip_code_prefix": 14409,
    "customer_state": "SP",
    "item_count": 1,
    "total_price": 100.0,
    "total_freight": 20.0,
    "payment_total": 120.0,
    "unique_products": 1,
    "unique_sellers": 1,
    "payment_count": 1,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] in ["ok", "degraded"]
    assert "model_status" in data


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "olist-delivery-random-forest"
    assert data["model_version"] == "1"
    assert data["model_alias"] == "champion"
    assert data["serving_stage"] == "production"


def test_single_prediction_endpoint():
    response = client.post(
        "/predict",
        json=VALID_REQUEST,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "late_probability" in data
    assert "model_version" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["late_probability"] <= 1
    assert data["model_version"] == "1"


def test_batch_prediction_endpoint():
    batch_request = {
        "requests": [
            VALID_REQUEST,
            {
                **VALID_REQUEST,
                "customer_state": "RJ",
                "customer_zip_code_prefix": 20040,
                "item_count": 2,
                "total_price": 250.0,
                "total_freight": 35.0,
                "payment_total": 285.0,
                "unique_products": 2,
                "unique_sellers": 2,
            },
        ]
    }

    response = client.post(
        "/predict/batch",
        json=batch_request,
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert "model_version" in data

    assert len(data["predictions"]) == 2
    assert data["model_version"] == "1"

    for prediction in data["predictions"]:
        assert prediction["prediction"] in [0, 1]
        assert 0 <= prediction["late_probability"] <= 1
        assert prediction["model_version"] == "1"


def test_invalid_prediction_input_is_rejected():
    invalid_request = {
        **VALID_REQUEST,
        "item_count": 0,
    }

    response = client.post(
        "/predict",
        json=invalid_request,
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data


def test_missing_required_field_is_rejected():
    invalid_request = VALID_REQUEST.copy()
    del invalid_request["total_price"]

    response = client.post(
        "/predict",
        json=invalid_request,
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data


def test_invalid_batch_request_is_rejected():
    invalid_batch = {"requests": []}

    response = client.post(
        "/predict/batch",
        json=invalid_batch,
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data
