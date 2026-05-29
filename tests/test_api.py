from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from plant_disease_api.config import Settings
from plant_disease_api.main import create_app
from plant_disease_api.schemas import PredictionResponse, TopPrediction


class FakePredictor:
    class_names = ["Apple___Apple_scab", "Tomato___healthy"]
    model_version = "test-model"

    def predict(self, img_path: Path, explain: bool = False) -> PredictionResponse:
        return PredictionResponse(
            prediction="Apple : Apple scab",
            raw_prediction="Apple___Apple_scab",
            confidence=98.5,
            top_predictions=[
                TopPrediction(label="Apple : Apple scab", raw_label="Apple___Apple_scab", confidence=98.5),
                TopPrediction(label="Tomato : healthy", raw_label="Tomato___healthy", confidence=1.5),
            ],
            uncertain=False,
            model_version=self.model_version,
            image_url=f"/static/uploads/{img_path.name}",
            gradcam_url="/static/uploads/gradcam.jpg" if explain else None,
        )


def make_settings(tmp_path, rate_limit_per_minute=30):
    return Settings(
        base_dir=Path(__file__).resolve().parents[1],
        upload_dir=tmp_path,
        model_path=tmp_path / "model.h5",
        class_index_path=tmp_path / "class_indices.json",
        class_dir=tmp_path / "train",
        rate_limit_per_minute=rate_limit_per_minute,
    )


def make_image_bytes() -> BytesIO:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(34, 120, 70)).save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_readiness_uses_injected_predictor(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path), predictor=FakePredictor()))

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["classes"] == 2
    assert response.json()["model_version"] == "test-model"


def test_predict_rejects_invalid_extension(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path), predictor=FakePredictor()))

    response = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400


def test_predict_returns_top_predictions(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path), predictor=FakePredictor()))

    response = client.post(
        "/api/v1/predict",
        data={"explain": "true"},
        files={"file": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Apple : Apple scab"
    assert body["confidence"] == 98.5
    assert body["gradcam_url"] == "/static/uploads/gradcam.jpg"
    assert len(body["top_predictions"]) == 2


def test_rate_limit_is_enforced(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path, rate_limit_per_minute=1), predictor=FakePredictor()))

    first = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
    )
    second = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
    )

    assert first.status_code == 200
    assert second.status_code == 429
