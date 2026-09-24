import time
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.config import config
from src.logging_config import setup_logging
from src.monitoring import (
    check_error_rate_alert,
    check_latency_alert,
    check_prediction_drift,
    log_prediction,
    record_prediction_error,
    record_prediction_metrics,
    record_prediction_request,
)
from src.predictor import load_model, predict


setup_logging()

app = FastAPI(
    title="Olist Delivery Prediction API",
    description=("API for predicting whether an Olist order will be delivered late."),
    version=config["project"]["model_version"],
)


def get_model_version() -> str:
    """Get the current model version assigned to the configured MLflow alias."""
    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])

    model_name = config["mlflow"]["registered_model_name"]
    alias = config["mlflow"]["model_alias"]

    model_version = client.get_model_version_by_alias(
        model_name,
        alias,
    )

    return str(model_version.version)


def prediction_request_to_dataframe(
    request: PredictionRequest,
) -> pd.DataFrame:
    """Convert a single API request into a DataFrame."""
    return pd.DataFrame(
        [
            {
                "order_purchase_timestamp": request.order_purchase_timestamp,
                "customer_zip_code_prefix": request.customer_zip_code_prefix,
                "customer_state": request.customer_state,
                "item_count": request.item_count,
                "total_price": request.total_price,
                "total_freight": request.total_freight,
                "payment_total": request.payment_total,
                "unique_products": request.unique_products,
                "unique_sellers": request.unique_sellers,
                "payment_count": request.payment_count,
            }
        ]
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check():
    """Check whether the API and MLflow model are available."""
    try:
        load_model()

        return {
            "status": "ok",
            "model_status": "available",
        }

    except Exception:
        return {
            "status": "degraded",
            "model_status": "unavailable",
        }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["System"],
)
def model_info():
    """Return information about the currently served model."""
    try:
        model_version = get_model_version()

        return {
            "model_name": config["mlflow"]["registered_model_name"],
            "model_version": model_version,
            "model_alias": config["mlflow"]["model_alias"],
            "serving_stage": config["mlflow"]["serving_stage"],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to retrieve model information: {str(exc)}",
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict_single(request: PredictionRequest):
    """Generate a delivery-delay prediction for one order."""
    start_time = time.perf_counter()
    record_prediction_request()

    try:
        df = prediction_request_to_dataframe(request)

        result = predict(df)

        model_version = get_model_version()

        prediction = int(result.iloc[0]["prediction"])
        probability = float(result.iloc[0]["late_probability"])

        duration = time.perf_counter() - start_time

        record_prediction_metrics(
            prediction=prediction,
            duration=duration,
        )

        log_prediction(
            input_data=request.model_dump(),
            prediction=prediction,
            probability=probability,
            model_version=model_version,
            duration=duration,
        )

        check_latency_alert(duration)
        check_error_rate_alert()
        check_prediction_drift()

        return {
            "prediction": prediction,
            "late_probability": probability,
            "model_version": model_version,
        }

    except ValueError as exc:
        record_prediction_error()
        check_error_rate_alert()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        record_prediction_error()
        check_error_rate_alert()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(request: BatchPredictionRequest):
    """Generate delivery-delay predictions for multiple orders."""
    start_time = time.perf_counter()
    record_prediction_request()

    try:
        rows = [
            {
                "order_purchase_timestamp": item.order_purchase_timestamp,
                "customer_zip_code_prefix": item.customer_zip_code_prefix,
                "customer_state": item.customer_state,
                "item_count": item.item_count,
                "total_price": item.total_price,
                "total_freight": item.total_freight,
                "payment_total": item.payment_total,
                "unique_products": item.unique_products,
                "unique_sellers": item.unique_sellers,
                "payment_count": item.payment_count,
            }
            for item in request.requests
        ]

        df = pd.DataFrame(rows)

        result = predict(df)

        model_version = get_model_version()

        predictions: List[PredictionResponse] = []

        duration = time.perf_counter() - start_time

        for index, row in result.iterrows():
            prediction = int(row["prediction"])
            probability = float(row["late_probability"])

            predictions.append(
                PredictionResponse(
                    prediction=prediction,
                    late_probability=probability,
                    model_version=model_version,
                )
            )

            log_prediction(
                input_data=request.requests[index].model_dump(),
                prediction=prediction,
                probability=probability,
                model_version=model_version,
                duration=duration,
            )

            record_prediction_metrics(
                prediction=prediction,
                duration=duration,
            )

        check_latency_alert(duration)
        check_error_rate_alert()
        check_prediction_drift()

        return BatchPredictionResponse(
            predictions=predictions,
            model_version=model_version,
        )

    except ValueError as exc:
        record_prediction_error()
        check_error_rate_alert()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        record_prediction_error()
        check_error_rate_alert()

        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(exc)}",
        )


@app.get(
    "/metrics",
    tags=["Monitoring"],
)
def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get(
    "/monitoring",
    tags=["Monitoring"],
)
def monitoring():
    """Return current prediction monitoring statistics."""
    return {
        "prediction_distribution": check_prediction_drift(),
    }
