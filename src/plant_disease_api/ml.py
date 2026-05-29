import hashlib
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np

from plant_disease_api.config import Settings
from plant_disease_api.labels import format_class_name, load_class_names
from plant_disease_api.schemas import PredictionResponse, TopPrediction


class Predictor(Protocol):
    class_names: list[str]
    model_version: str

    def predict(self, img_path: Path, explain: bool = False) -> PredictionResponse:
        ...


def model_fingerprint(model_path: Path) -> str:
    hasher = hashlib.sha256()
    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


class PlantDiseasePredictor:
    def __init__(self, settings: Settings) -> None:
        from tensorflow.keras.models import load_model

        self.settings = settings
        self.model = load_model(settings.model_path)
        self.class_names = load_class_names(settings.class_index_path, settings.class_dir)
        self.model_version = model_fingerprint(settings.model_path)

    def _prepare_image(self, img_path: Path) -> np.ndarray:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.preprocessing import image

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def _last_conv_layer_name(self) -> str:
        for layer in reversed(self.model.layers):
            output = getattr(layer, "output", None)
            if output is not None and len(output.shape) == 4:
                return layer.name
        raise RuntimeError("Could not find a convolutional feature map layer for Grad-CAM.")

    def _gradcam_overlay(self, img_path: Path) -> str:
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image

        last_conv_layer_name = self._last_conv_layer_name()
        preprocessed = self._prepare_image(img_path)
        grad_model = tf.keras.models.Model(
            self.model.inputs,
            [self.model.get_layer(last_conv_layer_name).output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed)
            predicted_index = tf.argmax(predictions[0])
            class_score = predictions[:, predicted_index]

        gradients = tape.gradient(class_score, conv_outputs)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        original = image.load_img(img_path).convert("RGB")
        heatmap_img = image.array_to_img(np.uint8(255 * heatmap)).resize(original.size)
        heatmap_array = np.array(heatmap_img)
        red_overlay = np.zeros((*heatmap_array.shape, 3), dtype=np.uint8)
        red_overlay[..., 0] = heatmap_array

        result = image.blend(original, image.array_to_img(red_overlay), alpha=0.35)
        filename = f"gradcam_{uuid4().hex}.jpg"
        result.save(self.settings.upload_dir / filename, quality=90)
        return f"/static/uploads/{filename}"

    def predict(self, img_path: Path, explain: bool = False) -> PredictionResponse:
        probabilities = self.model.predict(self._prepare_image(img_path), verbose=0)[0]
        top_indices = probabilities.argsort()[-self.settings.top_k_predictions :][::-1]
        top_predictions = [
            TopPrediction(
                label=format_class_name(self.class_names[index]),
                raw_label=self.class_names[index],
                confidence=round(float(probabilities[index]) * 100, 2),
            )
            for index in top_indices
        ]
        best = top_predictions[0]
        confidence_ratio = best.confidence / 100

        return PredictionResponse(
            prediction=best.label,
            raw_prediction=best.raw_label,
            confidence=best.confidence,
            top_predictions=top_predictions,
            uncertain=confidence_ratio < self.settings.prediction_confidence_threshold,
            model_version=self.model_version,
            image_url=f"/static/uploads/{img_path.name}",
            gradcam_url=self._gradcam_overlay(img_path) if explain and self.settings.enable_gradcam else None,
        )
