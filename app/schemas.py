from typing import List

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Input schema for a single delivery prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    order_purchase_timestamp: str

    customer_zip_code_prefix: int = Field(
        ...,
        ge=0,
    )

    customer_state: str = Field(
        ...,
        min_length=2,
        max_length=2,
    )

    item_count: int = Field(
        ...,
        ge=1,
    )

    total_price: float = Field(
        ...,
        ge=0,
    )

    total_freight: float = Field(
        ...,
        ge=0,
    )

    unique_products: int = Field(
        ...,
        ge=1,
    )

    unique_sellers: int = Field(
        ...,
        ge=1,
    )

    payment_total: float = Field(
        ...,
        ge=0,
    )

    payment_count: int = Field(
        ...,
        ge=1,
    )


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""

    requests: List[PredictionRequest] = Field(
        ...,
        min_length=1,
    )


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    prediction: int
    late_probability: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse]
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_status: str


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    model_version: str
    model_alias: str
    serving_stage: str
