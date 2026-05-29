from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    label: str
    raw_label: str
    confidence: float = Field(ge=0, le=100)


class PredictionResponse(BaseModel):
    prediction: str
    raw_prediction: str
    confidence: float = Field(ge=0, le=100)
    top_predictions: list[TopPrediction]
    uncertain: bool
    model_version: str
    image_url: str | None = None
    gradcam_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: int
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
