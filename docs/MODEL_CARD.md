# Model Card

## Model Overview

- Task: Plant leaf disease image classification
- Current artifact: `Model/plant_disease_model.h5`
- Input size: 224 x 224 RGB image
- Output: 38 PlantVillage-style disease or healthy classes
- Inference preprocessing: MobileNetV2 preprocessing

## Intended Use

This model is intended for educational, portfolio, and prototype use. It can help demonstrate image classification, explainability, and API deployment workflows for plant disease prediction.

## Not Intended For

- Medical, legal, financial, or agronomic decision-making without expert review
- Replacing advice from an agricultural specialist
- Field deployment without additional real-world validation

## Dataset Notes

The project uses the PlantVillage-style "New Plant Diseases Dataset (Augmented)" layout. These images are useful for benchmarking but may not represent real field conditions such as occlusion, mixed lighting, soil backgrounds, multiple leaves, multiple diseases, or low-quality mobile photos.

## Known Limitations

- The model may overestimate confidence on out-of-distribution images.
- Disease severity is not estimated.
- Multiple simultaneous diseases are not modeled.
- Leaf segmentation is not performed before classification.
- Current evaluation depends on the available local validation folder.

## Recommended Validation Before Production

- Evaluate on a field-collected held-out test set.
- Track precision, recall, F1 score, confusion matrix, and calibration.
- Add out-of-distribution and non-leaf rejection tests.
- Review Grad-CAM heatmaps for shortcut learning.
- Version every model artifact and class-label mapping together.
