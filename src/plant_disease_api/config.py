import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parents[2]
    app_name: str = os.getenv("APP_NAME", "Plant Disease Detection API")
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_path: Path = Path(
        os.getenv(
            "PLANT_MODEL_PATH",
            Path(__file__).resolve().parents[2] / "Model" / "plant_disease_model.h5",
        )
    )
    class_index_path: Path = Path(
        os.getenv(
            "PLANT_CLASS_INDEX_PATH",
            Path(__file__).resolve().parents[2] / "Model" / "class_indices.json",
        )
    )
    class_dir: Path = Path(
        os.getenv(
            "PLANT_CLASS_DIR",
            Path(__file__).resolve().parents[2]
            / "Dataset"
            / "New Plant Diseases Dataset(Augmented)"
            / "New Plant Diseases Dataset(Augmented)"
            / "train",
        )
    )
    upload_dir: Path = Path(
        os.getenv("UPLOAD_DIR", Path(__file__).resolve().parents[2] / "static" / "uploads")
    )
    allowed_extensions: frozenset[str] = frozenset(
        ext.strip().lower()
        for ext in os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp").split(",")
        if ext.strip()
    )
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "8"))
    top_k_predictions: int = int(os.getenv("TOP_K_PREDICTIONS", "3"))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    enable_gradcam: bool = _bool_env("ENABLE_GRADCAM", True)
    enable_docs: bool = _bool_env("ENABLE_DOCS", True)
    prediction_confidence_threshold: float = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.0"))


def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
